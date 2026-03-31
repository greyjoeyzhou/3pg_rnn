"""PyTorch RNN wrapper for the 3-PG forest growth model.

Exposes the 3-PG model as an ``nn.Module`` whose parameters can be
optimised via back-propagation against observed tree-ring and delta-13C
time series.
"""

import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Model3PG import run3pg, prepare
from const import StateIndex, InputIndex, NODATA


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed):
    """Set random seeds across all backends for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Site parameters container
# ---------------------------------------------------------------------------

class SiteParams:
    """Site-level parameters extracted from the input tensor each time step.

    Attributes correspond to input channels
    ``InputIndex.MAX_ASW`` .. ``InputIndex.ELEV``.
    """

    def __init__(self, MaxASW, MinASW, SWconst0, SWpower0, FR, MaxAge, elev):
        self.MaxASW = MaxASW
        self.MinASW = MinASW
        self.SWconst0 = SWconst0
        self.SWpower0 = SWpower0
        self.FR = FR
        self.MaxAge = MaxAge
        self.elev = elev


# ---------------------------------------------------------------------------
# RNN model
# ---------------------------------------------------------------------------

class RNN_3PG(nn.Module):
    """Recurrent wrapper around the 3-PG model.

    Each learnable 3-PG parameter is registered as an ``nn.Parameter``
    so that PyTorch can compute gradients and optimise them.

    Args:
        initial_weights: dict mapping parameter name to
            ``{"value": float, "trainable": bool}``.
    """

    def __init__(self, initial_weights):
        super().__init__()
        for name, cfg in initial_weights.items():
            setattr(
                self, name,
                nn.Parameter(
                    torch.tensor(cfg["value"], dtype=torch.float32),
                    requires_grad=cfg["trainable"],
                ),
            )

    def forward(self, input, h0):
        """Run the full simulation over all time steps.

        Args:
            input: Tensor of shape ``(batch, time_steps, InputIndex.COUNT)``.
            h0: Initial state tensor of shape ``(batch, StateIndex.COUNT)``.

        Returns:
            Tensor of shape ``(batch, time_steps, StateIndex.COUNT)``.
        """
        outputs = []
        h = h0
        for t in range(input.size(1)):
            h = self.step(input[:, t, :], h)
            outputs.append(h)
        return torch.stack(outputs, dim=outputs[0].dim() - 1)

    def step(self, x, hidden):
        """Advance the model by one monthly time step.

        Args:
            x: Input tensor ``(batch, InputIndex.COUNT)``.
            hidden: State tensor ``(batch, StateIndex.COUNT)``.

        Returns:
            Updated state tensor ``(batch, StateIndex.COUNT)``.
        """
        # Unpack state variables by name
        SI = StateIndex
        StandVol_prev = hidden[..., [SI.STAND_VOL]]
        LAI_prev = hidden[..., [SI.LAI]]
        ASW_prev = hidden[..., [SI.ASW]]
        StemNo_prev = hidden[..., [SI.STEM_NO]]
        PAR_prev = hidden[..., [SI.PAR]]
        stand_age_prev = hidden[..., [SI.STAND_AGE]]
        WF_prev = hidden[..., [SI.WF]]
        WR_prev = hidden[..., [SI.WR]]
        WS_prev = hidden[..., [SI.WS]]
        TotalLitter_prev = hidden[..., [SI.TOTAL_LITTER]]
        avDBH_prev = hidden[..., [SI.AV_DBH]]
        delStemNo_prev = hidden[..., [SI.DEL_STEM_NO]]
        D13CTissue_prev = hidden[..., [SI.D13C_TISSUE]]

        # Unpack input channels by name
        II = InputIndex
        T_av = x[..., [II.T_AV]]
        VPD = x[..., [II.VPD]]
        rain = x[..., [II.RAIN]]
        solar_rad = x[..., [II.SOLAR_RAD]]
        rain_days = x[..., [II.RAIN_DAYS]]
        frost_days = x[..., [II.FROST_DAYS]]
        CaMonthly = x[..., [II.CA_MONTHLY]]
        D13Catm = x[..., [II.D13C_ATM]]
        day_length = x[..., [II.DAY_LENGTH]]
        days_in_month = x[..., [II.DAYS_IN_MONTH]]

        site_paras = SiteParams(
            MaxASW=x[..., [II.MAX_ASW]],
            MinASW=x[..., [II.MIN_ASW]],
            SWconst0=x[..., [II.SW_CONST]],
            SWpower0=x[..., [II.SW_POWER]],
            FR=x[..., [II.FR]],
            MaxAge=x[..., [II.MAX_AGE]],
            elev=x[..., [II.ELEV]],
        )

        list_out = run3pg(
            LAI_prev, ASW_prev, StemNo_prev, stand_age_prev,
            WF_prev, WR_prev, WS_prev, TotalLitter_prev,
            avDBH_prev, delStemNo_prev,
            T_av, VPD, rain, solar_rad, frost_days,
            CaMonthly, D13Catm, day_length, days_in_month,
            site_paras, self,
        )
        return torch.cat(list_out, dim=list_out[0].dim() - 1)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class MaskedMSELoss(nn.Module):
    """MSE loss that ignores entries equal to a no-data sentinel."""

    def __init__(self, no_data_value=NODATA):
        super().__init__()
        self.no_data_value = no_data_value

    def forward(self, input, target):
        mask = target.ne(self.no_data_value)
        masked_input = torch.masked_select(input, mask)
        masked_target = torch.masked_select(target, mask)
        return F.mse_loss(masked_input, masked_target, reduction="mean")


class MaskedMSESelectedVarsLoss(nn.Module):
    """Weighted MSE loss on selected output variables, with masking.

    Args:
        idx_var: Indices of output variables to include in the loss
            (default: [avDBH, D13CTissue]).
        nodata: No-data sentinel value.
        weight_var: Per-variable loss weights (default: uniform).
        b_annual: If True, aggregate monthly outputs to annual before loss.
    """

    def __init__(
        self,
        idx_var=None,
        nodata=NODATA,
        weight_var=None,
        b_annual=True,
    ):
        super().__init__()
        self.nodata = nodata
        self.idx_var = idx_var if idx_var is not None else [
            StateIndex.AV_DBH, StateIndex.D13C_TISSUE,
        ]
        if weight_var is None:
            self.weight_var = [1.0] * len(self.idx_var)
        else:
            assert len(weight_var) == len(self.idx_var)
            self.weight_var = weight_var
        self.b_annual = b_annual

    def forward(self, input, target):
        lossfunc = MaskedMSELoss(no_data_value=self.nodata)
        if self.b_annual:
            input = agg_dbh_d13c(
                input, idx_var=self.idx_var,
                idx_weight=StateIndex.PAR, lib=torch,
            )
        else:
            input = input[..., self.idx_var]

        total_loss = 0.0
        for i in range(len(self.idx_var)):
            total_loss += lossfunc(input[..., i], target[..., i]) * self.weight_var[i]
        return total_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def agg_dbh_d13c(input, idx_var=None, idx_weight=StateIndex.PAR, lib=torch):
    """Extract DBH and d13C from monthly 3-PG output.

    Args:
        input: Monthly output tensor ``(batch, months, StateIndex.COUNT)``.
        idx_var: Variable indices to extract (default: [avDBH, D13CTissue]).
        idx_weight: Index of weighting variable (unused, kept for API compat).
        lib: Array library (torch or numpy).

    Returns:
        Tensor of shape ``(batch, months, len(idx_var))``.
    """
    if idx_var is None:
        idx_var = [StateIndex.AV_DBH, StateIndex.D13C_TISSUE]
    dbh = input[:, :, idx_var[0]]
    d13c = input[..., idx_var[1]]
    return lib.stack([dbh, d13c], axis=-1)


def train(model, inputs, target, criterion, optimizer, n_epochs):
    """Training loop with periodic loss reporting.

    Args:
        model: RNN_3PG instance.
        inputs: Dict passed as ``model(**inputs)``.
        target: Target tensor.
        criterion: Loss function.
        optimizer: PyTorch optimizer.
        n_epochs: Number of training epochs.

    Returns:
        Dict mapping epoch -> loss value.
    """
    dict_loss = {}
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}")
        dict_loss[epoch] = loss.item()
    return dict_loss


def inspect_params(model):
    """Return a DataFrame of trainable parameters and their gradients."""
    rows = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            rows[name] = {"value": param.item(), "grad": param.grad}
    return pd.DataFrame(rows).T


def plot_result(ts_pred, ts_target, nodata=NODATA):
    """Plot predicted vs observed DBH and d13C with R2 and RMSE."""
    titles = ["DBH", "\u03b413C"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for i in range(2):
        arr_pred = ts_pred[0, :, i].detach().cpu().numpy()
        arr_target = ts_target[0, :, i].detach().cpu().numpy()
        valid = arr_target != nodata
        r2 = np.corrcoef(arr_pred[valid], arr_target[valid])[0, 1] ** 2
        rmse = np.sqrt(((arr_pred[valid] - arr_target[valid]) ** 2).mean())
        axes[i].set_title(titles[i])
        axes[i].plot(arr_pred)
        arr_target_plot = arr_target.copy()
        arr_target_plot[~valid] = np.nan
        axes[i].plot(arr_target_plot)
        axes[i].annotate(
            f"$r^2={r2:.4f}$\n$rmse={rmse:.2f}$",
            xy=(0.8, 0.15), xycoords="axes fraction",
            ha="right", va="bottom",
        )
        axes[i].legend(["prediction", "label"])
    return fig


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEED = 101
    set_seed(SEED)

    fpath_setting = "./data_files/exp.yaml"
    (
        initial, arr_day_length, arr_days_in_month,
        arr_site_paras, arr_input, arr_target, dict_initial_weights,
    ) = prepare(fpath_setting)

    arr_input = np.concatenate(
        [arr_input, arr_day_length, arr_days_in_month], axis=1,
    )
    arr_input = np.concatenate(
        [arr_input, np.tile(arr_site_paras[None], [arr_input.shape[0], 1])],
        axis=1,
    )

    n_rep = 1
    device = "cuda"

    ts_input = torch.tensor(
        np.stack([arr_input] * n_rep, axis=0), device=device,
    )
    ts_target = torch.tensor(
        np.stack([arr_target] * n_rep, axis=0), device=device,
    )
    ts_initial = torch.tensor(np.array([initial] * n_rep), device=device)

    model = RNN_3PG(initial_weights=dict_initial_weights).to(device=device)
    ts_output = model(ts_input, ts_initial)

    cols = StateIndex.NAMES
    arr_output = ts_output[0].detach().cpu().numpy()
    df_output = pd.DataFrame(arr_output, columns=cols)

    df_output["StandVol"].plot(c="navy")
    plt.show()

    df_params = inspect_params(model)
    print(df_params)

    model = RNN_3PG(initial_weights=dict_initial_weights).to(device=device)
    criterion = MaskedMSESelectedVarsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    n_epochs = 1

    dict_loss = train(
        model=model,
        inputs={"input": ts_input, "h0": ts_initial},
        target=ts_target,
        criterion=criterion,
        optimizer=optimizer,
        n_epochs=n_epochs,
    )

    df_res = pd.DataFrame({"loss": dict_loss})
    df_res["loss"].plot()

    df_params = inspect_params(model)
    print(df_params)

    ts_pred = model(ts_input, ts_initial)
    ts_pred_sub = agg_dbh_d13c(ts_pred)
    fig = plot_result(ts_pred_sub, ts_target)
