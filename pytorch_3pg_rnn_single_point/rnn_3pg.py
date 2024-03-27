# %%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os

from Model3PG import run3pg, prepare


def set_seed(seed):
    """Set the random seed for reproducible experiments."""
    torch.manual_seed(seed)  # Set the random seed for PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Set the seed for current GPU
        torch.cuda.manual_seed_all(seed)  # Set the seed for all GPUs
    np.random.seed(seed)  # Set the random seed for NumPy
    random.seed(seed)  # Set the random seed for Python's Random
    os.environ["PYTHONHASHSEED"] = str(seed)  # Set the Python hash seed

    # If you are using cudnn, the following ensures deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 101
set_seed(SEED)


# %%
class SITE_PARAS(object):
    def __init__(
        self,
        MaxASW,  # 12. in input x
        MinASW,
        SWconst0,
        SWpower0,
        FR,
        MaxAge,
        elev,  # 18.
    ):
        self.MaxASW = MaxASW
        self.MinASW = MinASW
        self.SWconst0 = SWconst0
        self.SWpower0 = SWpower0
        self.FR = FR
        self.MaxAge = MaxAge
        self.elev = elev


class RNN_3PG(nn.Module):
    def __init__(self, initial_weights):
        super(RNN_3PG, self).__init__()
        for para in initial_weights:
            # print(initial_weights[para])
            setattr(
                self,
                para,
                nn.Parameter(
                    torch.tensor(initial_weights[para]["value"], dtype=torch.float32),
                    requires_grad=initial_weights[para]["trainable"],
                ),
            )

    def forward(self, input, h0):
        """
        input: (batch_size, time_steps, channels=19)
        h0: list of (batch_size, 1)
        """
        list_out = []
        h = h0
        # Process each time step in the input sequence
        for i in range(input.size(1)):
            h = self.step(input[:, i, :], h)
            list_out.append(h)
        return torch.stack(list_out, dim=list_out[0].dim() - 1)

    def step(self, x, hidden):
        """
        x: (batch_size, channels=19)
        hidden: list of (batch_size, 1)
        """
        [
            StandVol_prev,
            LAI_prev,
            ASW_prev,
            StemNo_prev,
            PAR_prev,
            stand_age_prev,
            WF_prev,
            WR_prev,
            WS_prev,
            TotalLitter_prev,
            avDBH_prev,
            delStemNo_prev,
            D13CTissue_prev,
        ] = [hidden[..., [i]] for i in range(hidden.shape[-1])]

        [
            _,  # 0.
            _,  # 1.
            T_av,  # 2.
            VPD,
            rain,
            solar_rad,
            rain_days,
            frost_days,
            CaMonthly,
            D13Catm,
            day_length,  # 10. calculated from utils.get_day_length
            days_in_month,  # 11. calculated from utils.get_days_in_month
            # site parameters as input:
            MaxASW,  # 12.
            MinASW,
            SWconst0,
            SWpower0,
            FR,
            MaxAge,
            elev,  # 18.
        ] = [x[..., [i]] for i in range(x.shape[-1])]

        site_paras = SITE_PARAS(
            MaxASW,
            MinASW,
            SWconst0,
            SWpower0,
            FR,
            MaxAge,
            elev,
        )

        list_out = run3pg(
            # previous step:
            LAI_prev,
            ASW_prev,
            StemNo_prev,
            stand_age_prev,
            WF_prev,
            WR_prev,
            WS_prev,
            TotalLitter_prev,
            avDBH_prev,
            delStemNo_prev,
            # current input:
            T_av,
            VPD,
            rain,
            solar_rad,
            frost_days,
            CaMonthly,
            D13Catm,
            day_length,
            days_in_month,
            # site parameters:
            site_paras,
            # model parameters:
            self,
        )
        return torch.cat(list_out, dim=list_out[0].dim() - 1)


# general masked loss function
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    def __init__(self, no_data_value):
        super(MaskedMSELoss, self).__init__()
        self.no_data_value = no_data_value

    def forward(self, input, target):
        # Create a mask that selects data where the target is not equal to no_data_value
        mask = target.ne(self.no_data_value)

        # Apply the mask to input and target to get only relevant data points
        masked_input = torch.masked_select(input, mask)
        masked_target = torch.masked_select(target, mask)

        # Calculate the MSE loss on the masked data
        loss = F.mse_loss(masked_input, masked_target, reduction="mean")
        return loss


class MaskedMSESelectedVarsLoss(nn.Module):
    def __init__(
        self,
        idx_var=[10, 12],
        nodata=-9999.0,
        weight_var=None,
        b_annual=True,
    ):
        super(MaskedMSESelectedVarsLoss, self).__init__()
        self.nodata = nodata
        self.idx_var = idx_var
        if weight_var is None:
            self.weight_var = [1.0] * len(idx_var)
        else:
            assert len(weight_var) == len(idx_var)
            self.weight_var = weight_var
        self.b_annual = b_annual

    def forward(self, input, target):
        lossv = 0.0
        lossfunc = MaskedMSELoss(no_data_value=self.nodata)
        if self.b_annual:
            input = agg_dbh_d13c(
                input,
                idx_var=self.idx_var,
                idx_weight=4,
                lib=torch,
            )
        else:
            input = input[..., self.idx_var]
        for i in range(len(self.idx_var)):
            lossv += lossfunc(input[..., i], target[..., i]) * self.weight_var[i]
        return lossv

    # ts_output.reshape([1, -1, 12, 13])


def agg_dbh_d13c(
    input,
    idx_var=[10, 12],
    idx_weight=4,  # PAR
    lib=torch,
):
    """input: monthly output from 3pg"""
    dbh = input[:, :, idx_var[0]]
    d13c_month = input[..., idx_var[1]]
    input = lib.stack([dbh, d13c_month], axis=-1)
    return input


# general training function
def train(model, inputs, target, criterion, optimizer, n_epochs):
    dict_loss = {}
    model.train()  # set the model to training mode
    for epoch in range(n_epochs):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(**inputs)
        loss = criterion(outputs, target)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        if (epoch + 1) % 10 == 0:  # print every 10 epochs
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")
        dict_loss[epoch] = loss.item()
    return dict_loss


def inspect_params(model):
    dict_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            dict_params[name] = {
                "value": param.item(),
                "grad": param.grad,
            }
    return pd.DataFrame(dict_params).T


def plot_result(ts_pred, ts_target, nodata=-9999):
    titles = ["DBH", "δ13C"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for i in range(2):
        arr_pred = ts_pred[0, :, i].detach().cpu().numpy()
        arr_target = ts_target[0, :, i].detach().cpu().numpy()
        indi_nodata = arr_target == nodata
        r2 = np.corrcoef(arr_pred[~indi_nodata], arr_target[~indi_nodata])[0, 1] ** 2
        rmse = np.sqrt(
            ((arr_pred[~indi_nodata] - arr_target[~indi_nodata]) ** 2).mean()
        )
        axes[i].set_title(titles[i])
        axes[i].plot(arr_pred)
        arr_target[arr_target == nodata] = np.nan
        axes[i].plot(arr_target)
        axes[i].annotate(
            f"$r^2={r2:.4f}$\n$rmse={rmse:.2f}$",
            xy=(0.8, 0.15),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
        )
        axes[i].legend(["prediction", "label"])
    return fig


# %%
if __name__ == "__main__":
    fpath_setting = "./data_files/exp.yaml"
    (
        initial,
        arr_day_length,
        arr_days_in_month,
        arr_site_paras,
        arr_input,
        arr_target,
        dict_initial_weights,
    ) = prepare(fpath_setting)

    arr_input = np.concatenate([arr_input, arr_day_length, arr_days_in_month], axis=1)
    arr_input = np.concatenate(
        [arr_input, np.tile(arr_site_paras[None], [arr_input.shape[0], 1])],
        axis=1,
    )

    n_rep = 1
    device = "cuda"
    nodata = -9999.0

    ts_input = torch.tensor(np.stack([arr_input] * n_rep, axis=0), device=device)
    ts_target = torch.tensor(np.stack([arr_target] * n_rep, axis=0), device=device)

    ts_initial = torch.tensor(np.array([initial] * n_rep), device=device)

    model = RNN_3PG(initial_weights=dict_initial_weights).to(device=device)
    ts_output = model(ts_input, ts_initial)
    # %%
    cols = [
        "StandVol",
        "LAI",
        "ASW",
        "StemNo",
        "PAR",
        "stand_age",
        "WF",
        "WR",
        "WS",
        "TotalLitter",
        "avDBH",
        "delStemNo",
        "D13CTissue",
    ]
    arr_output = ts_output[0].detach().cpu().numpy()
    df_output = pd.DataFrame(arr_output, columns=cols)

    # %%
    df_output["StandVol"].plot(c="navy")
    plt.show()

    # %%
    df_params = inspect_params(model)
    df_params

    # %%
    model = RNN_3PG(initial_weights=dict_initial_weights).to(device=device)
    criterion = MaskedMSESelectedVarsLoss(nodata=nodata)
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
    # %%
    df_res = pd.DataFrame({"loss": dict_loss})
    df_res["loss"].plot()
    # %%
    df_params = inspect_params(model)
    df_params

    # %%
    ts_pred = model(ts_input, ts_initial)
    ts_pred_sub = agg_dbh_d13c(ts_pred)
    fig = plot_result(ts_pred_sub, ts_target, nodata=nodata)
# %%
