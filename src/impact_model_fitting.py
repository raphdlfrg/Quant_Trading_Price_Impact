import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

################# Define the impact state

def get_row_scaling(panel_df, scaling_df):
    """
    Align stock-level ADV and sigma with a stock-date x time panel.

    panel_df index:
        MultiIndex(stock, date)

    scaling_df index:
        stock

    Returns
    -------
    adv : pd.Series indexed like panel_df
    sigma : pd.Series indexed like panel_df
    """
    stocks = panel_df.index.get_level_values("stock")

    adv = pd.Series(
        stocks.map(scaling_df["ADV"]),
        index=panel_df.index,
        name="ADV"
    )

    sigma = pd.Series(
        stocks.map(scaling_df["sigma"]),
        index=panel_df.index,
        name="sigma"
    )

    return adv, sigma


def impact_state(
    traded_volume_df,
    scaling_df,
    half_life_seconds=3600,
    dt_seconds=10,
    model_type="ow",
    eps=1e-12
):
    """
    Compute normalized impact state with lambda = 1.

    Parameters
    ----------
    traded_volume_df:
        stock-date x time matrix of signed public traded volume.

    scaling_df:
        stock-level dataframe with columns:
        - ADV
        - sigma

    half_life_seconds:
        Impact half-life.

    dt_seconds:
        Bin size. For this coursework, bin files are 10-second bins.

    model_type:
        "ow"              : linear OW model.
        "sqrt_propagator" : nonlinear square-root propagator.
        "afs"             : AFS-style model.
        "reduced_form"    : dynamic-liquidity reduced-form model.

    eps:
        Small number used to avoid division by zero.

    Returns
    -------
    impact_df:
        stock-date x time matrix of impact feature with lambda = 1.
    """

    adv, sigma = get_row_scaling(traded_volume_df, scaling_df)

    beta = np.log(2) / half_life_seconds
    decay_factor = np.exp(-beta * dt_seconds)
    alpha = 1 - decay_factor

    q = traded_volume_df.copy()

    # ------------------------------------------------------------
    # 1. OW model
    # ------------------------------------------------------------
    if model_type == "ow":
        # I_{t+dt} = exp(-beta dt) I_t + sigma * q_t / ADV
        pre_ewm = q.divide(adv, axis=0)
        pre_ewm = pre_ewm.multiply(sigma, axis=0)

        pre_ewm.iloc[:, 1:] = pre_ewm.iloc[:, 1:] / alpha

        impact_df = pre_ewm.T.ewm(alpha=alpha, adjust=False).mean().T

    # ------------------------------------------------------------
    # 2. Square-root propagator
    # ------------------------------------------------------------
    elif model_type == "sqrt_propagator":
        # I_{t+dt} = exp(-beta dt) I_t
        #            + sigma * sign(q_t) * sqrt(|q_t| / ADV)
        pre_ewm = q.divide(adv, axis=0)
        pre_ewm = np.sign(pre_ewm) * np.sqrt(np.abs(pre_ewm))
        pre_ewm = pre_ewm.multiply(sigma, axis=0)

        pre_ewm.iloc[:, 1:] = pre_ewm.iloc[:, 1:] / alpha

        impact_df = pre_ewm.T.ewm(alpha=alpha, adjust=False).mean().T

    # ------------------------------------------------------------
    # 3. AFS-style model
    # ------------------------------------------------------------
    elif model_type == "afs":
        # First compute volume-space state J_t:
        # J_{t+dt} = exp(-beta dt) J_t + sigma * q_t / ADV
        pre_ewm = q.divide(adv, axis=0)
        pre_ewm = pre_ewm.multiply(sigma, axis=0)

        pre_ewm.iloc[:, 1:] = pre_ewm.iloc[:, 1:] / alpha

        j_state = pre_ewm.T.ewm(alpha=alpha, adjust=False).mean().T

        # Then apply AFS square-root transformation:
        # I_t = sign(J_t) * sqrt(|J_t|)
        impact_df = np.sign(j_state) * np.sqrt(np.abs(j_state))

    # ------------------------------------------------------------
    # 4. Reduced-form dynamic-liquidity model
    # ------------------------------------------------------------
    elif model_type == "reduced_form":
        # Step 1:
        # Compute local market volume estimate v_t using unsigned volume.
        #
        # v_{t+dt} = exp(-beta dt) v_t + |q_t|
        local_volume_input = q.abs()

        local_volume_input.iloc[:, 1:] = local_volume_input.iloc[:, 1:] / alpha

        local_volume_state = (
            local_volume_input
            .T
            .ewm(alpha=alpha, adjust=False)
            .mean()
            .T
        )

        # Avoid division by zero
        local_volume_state = local_volume_state.clip(lower=eps)

        # Step 2:
        # Reduced-form impact input:
        #
        # sigma * q_t / sqrt(ADV * v_t)
        denominator = np.sqrt(
            local_volume_state.multiply(adv, axis=0)
        )

        pre_ewm = q.divide(denominator)
        pre_ewm = pre_ewm.multiply(sigma, axis=0)

        # Step 3:
        # Apply impact decay
        pre_ewm.iloc[:, 1:] = pre_ewm.iloc[:, 1:] / alpha

        impact_df = pre_ewm.T.ewm(alpha=alpha, adjust=False).mean().T

    else:
        raise ValueError(
            "model_type must be one of: "
            "'ow', 'sqrt_propagator', 'afs', 'reduced_form'"
        )

    return impact_df



############## Convert impact states into regression data

def impact_regression_statistics(
    impact_df,
    px_df,
    horizon_periods=6,
    start_time="10:00:00"
):
    """
    Build regression statistics for:
        returns = intercept + lambda * impact_change + error

    Parameters
    ----------
    impact_df:
        stock-date x time impact state.

    px_df:
        stock-date x time price matrix.

    horizon_periods:
        Number of 10-second periods in the return horizon.
        Example:
            6   = 1 minute
            90  = 15 minutes
            360 = 60 minutes

    start_time:
        Drop early bins to avoid unstable opening behaviour.

    Returns
    -------
    reg_df:
        Long dataframe with x, y, xy, xx, yy, count.
    """

    impact_changes = impact_df.diff(horizon_periods, axis=1)
    returns = px_df.pct_change(horizon_periods, axis=1)

    x_long = (
        impact_changes
        .stack()
        .rename("x")
        .reset_index()
        .rename(columns={"level_2": "time"})
    )

    y_long = (
        returns
        .stack()
        .rename("y")
        .reset_index()
        .rename(columns={"level_2": "time"})
    )

    reg_df = pd.merge(
        x_long,
        y_long,
        on=["stock", "date", "time"],
        how="inner"
    )

    reg_df = reg_df[reg_df["time"] >= start_time].dropna().copy()

    reg_df["xy"] = reg_df["x"] * reg_df["y"]
    reg_df["xx"] = reg_df["x"] * reg_df["x"]
    reg_df["yy"] = reg_df["y"] * reg_df["y"]
    reg_df["count"] = 1

    return reg_df


################# Fit λ and compute R

def summarize_regression_data(reg_df):
    """
    Aggregate regression sufficient statistics by stock.

    This follows the exercise idea:
    compute xy, xx, yy, x, y, count once,
    then use formulas for regression.
    """
    summary = (
        reg_df
        .groupby("stock")[["xy", "xx", "yy", "x", "y", "count"]]
        .sum()
    )

    return summary


def fit_lambda_and_r2(train_summary, test_summary):
    """
    Estimate lambda on train data and evaluate both train and test R^2.

    Regression:
        y = alpha + lambda * x + error
    """

    df = train_summary.add_prefix("is_").merge(
        test_summary.add_prefix("oos_"),
        left_index=True,
        right_index=True,
        how="inner"
    )

    # Estimate slope and intercept using training data
    numerator = df["is_xy"] - df["is_x"] * df["is_y"] / df["is_count"]
    denominator = df["is_xx"] - df["is_x"] ** 2 / df["is_count"]

    df["lambda_hat"] = numerator / denominator

    df["intercept_hat"] = (
        df["is_y"] / df["is_count"]
        - df["lambda_hat"] * df["is_x"] / df["is_count"]
    )

    # IS total sum of squares
    df["is_tss"] = df["is_yy"] - df["is_y"] ** 2 / df["is_count"]

    # IS residual sum of squares
    df["is_sse"] = (
        df["is_yy"]
        - 2 * df["intercept_hat"] * df["is_y"]
        - 2 * df["lambda_hat"] * df["is_xy"]
        + 2 * df["intercept_hat"] * df["lambda_hat"] * df["is_x"]
        + df["lambda_hat"] ** 2 * df["is_xx"]
        + df["intercept_hat"] ** 2 * df["is_count"]
    )

    df["is_r2"] = 1 - df["is_sse"] / df["is_tss"]

    # OOS total sum of squares
    df["oos_tss"] = df["oos_yy"] - df["oos_y"] ** 2 / df["oos_count"]

    # OOS residual sum of squares, using train estimates
    df["oos_sse"] = (
        df["oos_yy"]
        - 2 * df["intercept_hat"] * df["oos_y"]
        - 2 * df["lambda_hat"] * df["oos_xy"]
        + 2 * df["intercept_hat"] * df["lambda_hat"] * df["oos_x"]
        + df["lambda_hat"] ** 2 * df["oos_xx"]
        + df["intercept_hat"] ** 2 * df["oos_count"]
    )

    df["oos_r2"] = 1 - df["oos_sse"] / df["oos_tss"]

    return df


############## Fit one model for one half-life


def fit_impact_model_once(
    train_traded_volume_df,
    test_traded_volume_df,
    train_px_df,
    test_px_df,
    scaling_df,
    model_type="ow",
    half_life_seconds=3600,
    horizon_periods=6,
    dt_seconds=10,
    start_time="10:00:00"
):
    """
    Fit one impact model for one half-life and one horizon.
    """

    train_impact = impact_state(
        train_traded_volume_df,
        scaling_df,
        half_life_seconds=half_life_seconds,
        dt_seconds=dt_seconds,
        model_type=model_type
    )

    test_impact = impact_state(
        test_traded_volume_df,
        scaling_df,
        half_life_seconds=half_life_seconds,
        dt_seconds=dt_seconds,
        model_type=model_type
    )

    train_reg_df = impact_regression_statistics(
        train_impact,
        train_px_df,
        horizon_periods=horizon_periods,
        start_time=start_time
    )

    test_reg_df = impact_regression_statistics(
        test_impact,
        test_px_df,
        horizon_periods=horizon_periods,
        start_time=start_time
    )

    train_summary = summarize_regression_data(train_reg_df)
    test_summary = summarize_regression_data(test_reg_df)

    result_df = fit_lambda_and_r2(train_summary, test_summary)

    result_df["model_type"] = model_type
    result_df["half_life_seconds"] = half_life_seconds
    result_df["horizon_periods"] = horizon_periods
    result_df["horizon_minutes"] = horizon_periods * dt_seconds / 60

    return result_df



############# Summary
def model_summary_table(result_df):
    return pd.Series({
        "mean_lambda": result_df["lambda_hat"].mean(),
        "std_lambda": result_df["lambda_hat"].std(),
        "tstat_lambda": result_df["lambda_hat"].mean() / result_df["lambda_hat"].std(),
        "mean_is_r2": result_df["is_r2"].mean(),
        "mean_oos_r2": result_df["oos_r2"].mean(),
        "median_is_r2": result_df["is_r2"].median(),
        "median_oos_r2": result_df["oos_r2"].median(),
        "n_stocks": result_df.shape[0]
    })