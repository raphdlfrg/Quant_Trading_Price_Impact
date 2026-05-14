import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.backtest_engine import *
from src.synthetic_alphas import *


##########################LOADING HELPERS######################
def load_panel_csv(path):
    """Load a stock-date x intraday-time panel saved by previous notebooks."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index(["stock", "date"]).sort_index()
    df.columns = df.columns.astype(str)
    return df

def load_stock_level_csv(path):
    df = pd.read_csv(path)
    if "stock" in df.columns:
        df = df.set_index("stock").sort_index()
    return df


# ============================================================
# OW dynamic-liquidity optimal strategy
# ============================================================

def ow_target_impact_from_alpha(alpha, beta, dt_seconds):
    """
    Compute the OW target impact state:

        I*_t = 1/2 (alpha_t - beta^{-1} mu_t)

    where:

        mu_t ≈ (alpha_{t+dt} - alpha_t) / dt
    """

    alpha = (
        alpha.astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    mu = (alpha.shift(-1) - alpha) / dt_seconds
    mu = (
        mu.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    I_star = 0.5 * (alpha - mu / beta)

    # Terminal condition from the OW formula.
    I_star.iloc[-1] = alpha.iloc[-1]

    return I_star, mu


def recover_ow_trades_from_target_impact(
    I_star,
    lambda_hat,
    ADV,
    sigma,
    half_life_seconds,
    dt_seconds=10,
    eps=1e-12,
):
    """
    Recover trades using the discrete fitted OW impact equation:
        I_n = decay * I_{n-1} + lambda_hat * sigma * q_n / ADV.
    """
    beta = np.log(2) / half_life_seconds
    decay = np.exp(-beta * dt_seconds)

    lambda_eff = lambda_hat * sigma / ADV
    if abs(lambda_eff) < eps:
        return pd.Series(0.0, index=I_star.index)

    trades = []
    prev_I = 0.0

    for target_I in I_star.values.astype(float):
        q = (target_I - decay * prev_I) / lambda_eff
        trades.append(q)
        prev_I = target_I

    return pd.Series(trades, index=I_star.index)


def make_ow_optimal_trade_df(
    alpha_df,
    test_px_df,
    scaling_df,
    fit_df,
    dt_seconds=10,
    normalize_abs_volume=True,
    target_participation=0.20,
):
    """
    Build a stock-date x time trade panel from an alpha panel using the
    lecture-note OW target-impact formula and fitted OW parameters.

    For each stock-day, the function computes:

        alpha_t
            -> alpha'_t
            -> I*_t = 0.5 * (alpha_t - alpha'_t / beta)
            -> q_t from the fitted OW impact equation.

    If normalize_abs_volume=True, each stock-day is rescaled so that

        sum_t |q_t| = target_participation * ADV.

    In that case, both the trades and the target impact are rescaled by
    the same factor. Therefore, target_impact_df represents the effective
    target impact corresponding to the actual normalized trades.

    Parameters
    ----------
    alpha_df : pd.DataFrame
        Alpha panel indexed by (stock, date), with intraday time columns.

    test_px_df : pd.DataFrame
        Test price panel. Used for the stock-date index and intraday columns.

    scaling_df : pd.DataFrame
        Stock-level table indexed by stock. Must contain ADV and sigma.

    fit_df : pd.DataFrame
        Fitted OW parameter table indexed by stock.
        Must contain lambda_hat and half_life_seconds.

    dt_seconds : int
        Intraday time-bin length in seconds.

    normalize_abs_volume : bool
        If True, normalize total absolute daily traded volume to
        target_participation * ADV.

    target_participation : float
        Target daily participation rate as a fraction of ADV.

    Returns
    -------
    trades_df : pd.DataFrame
        Trade panel q_{i,d,j}, same shape as test_px_df.

    target_impact_df : pd.DataFrame
        Effective target impact panel I*_{i,d,j}, same shape as test_px_df.

    alpha_mu_df : pd.DataFrame
        Alpha mu panel alpha'_{i,d,j}, same shape as test_px_df.

    strategy_scale_df : pd.DataFrame
        Stock-day diagnostics for the normalization step.
    """

    trades_df = pd.DataFrame(
        0.0,
        index=test_px_df.index,
        columns=test_px_df.columns,
    )

    target_impact_df = pd.DataFrame(
        0.0,
        index=test_px_df.index,
        columns=test_px_df.columns,
    )

    alpha_mu_df = pd.DataFrame(
        0.0,
        index=test_px_df.index,
        columns=test_px_df.columns,
    )

    scale_rows = []

    for stock, date in test_px_df.index:
        if stock not in fit_df.index:
            continue

        if stock not in scaling_df.index:
            continue

        alpha = (
            alpha_df
            .loc[(stock, date)]
            .reindex(test_px_df.columns)
            .fillna(0.0)
            .astype(float)
        )

        lambda_hat = float(fit_df.loc[stock, "lambda_hat"])
        half_life_seconds = float(fit_df.loc[stock, "half_life_seconds"])

        ADV = float(scaling_df.loc[stock, "ADV"])
        sigma = float(scaling_df.loc[stock, "sigma"])

        beta = np.log(2) / half_life_seconds

        I_star, alpha_mu = ow_target_impact_from_alpha(
            alpha=alpha,
            beta=beta,
            dt_seconds=dt_seconds,
        )

        trades = recover_ow_trades_from_target_impact(
            I_star=I_star,
            lambda_hat=lambda_hat,
            ADV=ADV,
            sigma=sigma,
            half_life_seconds=half_life_seconds,
            dt_seconds=dt_seconds,
        )

        raw_abs_volume = trades.abs().sum()
        target_abs_volume = target_participation * ADV

        scale_factor = 1.0

        if normalize_abs_volume and raw_abs_volume > 0:
            scale_factor = target_abs_volume / raw_abs_volume

            trades = trades * scale_factor
            I_star = I_star * scale_factor

        trades_df.loc[(stock, date)] = trades.values
        target_impact_df.loc[(stock, date)] = I_star.values
        alpha_mu_df.loc[(stock, date)] = alpha_mu.values

        scale_rows.append({
            "stock": stock,
            "date": date,
            "ADV": ADV,
            "sigma": sigma,
            "lambda_hat": lambda_hat,
            "half_life_seconds": half_life_seconds,
            "beta": beta,
            "target_participation": target_participation,
            "target_abs_volume": target_abs_volume,
            "raw_abs_volume": raw_abs_volume,
            "actual_abs_volume": trades.abs().sum(),
            "net_traded": trades.sum(),
            "scale_factor": scale_factor,
            "normalize_abs_volume": normalize_abs_volume,
        })

    strategy_scale_df = pd.DataFrame(scale_rows)

    return trades_df, target_impact_df, alpha_mu_df, strategy_scale_df

def run_strategy_backtests_for_model(
    strategy_trade_dfs,
    model_type,
    test_px_df,
    test_traded_volume_df,
    scaling_df,
    fit_dfs,
    dt_seconds=10,
):
    results = []

    for strategy_name, trades_df in strategy_trade_dfs.items():
        bt = run_backtest_from_trade_df(
            test_px_df=test_px_df,
            strategy_trades_df=trades_df,
            test_traded_volume_df=test_traded_volume_df,
            scaling_df=scaling_df,
            fit_df=fit_dfs[model_type],
            model_type=model_type,
            dt_seconds=dt_seconds,
        )

        bt = add_bps_metrics(bt)
        bt["strategy"] = strategy_name
        bt["backtest_model"] = model_type

        results.append(bt)

    return pd.concat(results, ignore_index=True)






# ============================================================
# AFS dynamic-liquidity optimal strategy
# ============================================================

def afs_target_impact_from_alpha_and_mu(
    alpha,
    alpha_mu,
    beta,
    c=0.5,
    apply_terminal_condition=True,
):
    """
    Compute the AFS target impact state:

        I*_t = 1 / (1 + c) * (alpha_t - beta^{-1} mu_t)

    where:

        mu_t = (alpha_{t+dt} - alpha_t) / dt

    For square-root AFS, c = 0.5, so:

        I*_t = 2/3 * (alpha_t - beta^{-1} mu_t)
    """

    alpha = (
        alpha.astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    alpha_mu = (
        alpha_mu.astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    I_star = (1.0 / (1.0 + c)) * (alpha - alpha_mu / beta)

    if apply_terminal_condition:
        I_star.iloc[-1] = alpha.iloc[-1]

    return I_star


def recover_afs_trades_from_target_impact(
    I_star,
    lambda_hat,
    ADV,
    sigma,
    half_life_seconds,
    c=0.5,
    dt_seconds=10,
    eps=1e-12,
):
    """
    Recover AFS trades from a target impact path.

    Fitted AFS convention:

        I_t = lambda_hat * sign(J_t) * |J_t|^c

    where the volume-space state evolves as:

        J_t = decay * J_{t-dt} + sigma * q_t / ADV

    Therefore:

        J*_t = sign(I*_t / lambda_hat) * |I*_t / lambda_hat|^{1/c}

    and:

        q_t = (J*_t - decay * J*_{t-dt}) / (sigma / ADV)
    """

    if (
        not np.isfinite(lambda_hat)
        or lambda_hat <= eps
        or not np.isfinite(ADV)
        or not np.isfinite(sigma)
        or ADV <= eps
        or sigma <= eps
    ):
        return pd.Series(0.0, index=I_star.index)

    beta = np.log(2) / half_life_seconds
    decay = np.exp(-beta * dt_seconds)

    flow_coeff = sigma / ADV

    if not np.isfinite(flow_coeff) or abs(flow_coeff) < eps:
        return pd.Series(0.0, index=I_star.index)

    # Convert target return-impact into target normalized AFS feature.
    afs_feature_star = I_star / lambda_hat

    # Invert sign(J) * |J|^c.
    J_star = (
        np.sign(afs_feature_star)
        * np.abs(afs_feature_star) ** (1.0 / c)
    )

    trades = []
    prev_J = 0.0

    for target_J in J_star.values.astype(float):
        q = (target_J - decay * prev_J) / flow_coeff
        trades.append(q)
        prev_J = target_J

    return pd.Series(trades, index=I_star.index)


def make_afs_optimal_trade_df(
    alpha_df,
    test_px_df,
    scaling_df,
    fit_df,
    dt_seconds=10,
    c=0.5,
    normalize_abs_volume=True,
    target_participation=0.20,
    apply_terminal_condition=True,
):
    """
    Build a stock-date x time trade panel using the AFS optimal strategy.

    For each stock-day:

        alpha_t
            -> mu_t = (alpha_{t+dt} - alpha_t) / dt
            -> I*_t = 1/(1+c) * (alpha_t - mu_t / beta)
            -> J*_t from AFS inverse
            -> q_t from the fitted J-state recurrence

    Returns
    -------
    trades_df:
        Strategy trade panel.

    target_impact_df:
        Effective target impact panel after optional volume normalization.

    alpha_mu_df:
        Forward alpha drift / decay input mu_t.

    strategy_scale_df:
        Stock-day diagnostics.
    """

    # Compute mu_t once for the full alpha panel.
    alpha_mu_df_full = generate_synthetic_alpha_decay_df(
        synthetic_alpha_df=alpha_df,
        dt_seconds=dt_seconds,
    )

    trades_df = pd.DataFrame(
        0.0,
        index=test_px_df.index,
        columns=test_px_df.columns,
    )

    target_impact_df = pd.DataFrame(
        0.0,
        index=test_px_df.index,
        columns=test_px_df.columns,
    )

    alpha_mu_df = pd.DataFrame(
        0.0,
        index=test_px_df.index,
        columns=test_px_df.columns,
    )

    scale_rows = []

    for stock, date in test_px_df.index:

        if stock not in fit_df.index:
            continue

        if stock not in scaling_df.index:
            continue

        if (stock, date) not in alpha_df.index:
            continue

        alpha = (
            alpha_df
            .loc[(stock, date)]
            .reindex(test_px_df.columns)
            .fillna(0.0)
            .astype(float)
        )

        alpha_mu = (
            alpha_mu_df_full
            .loc[(stock, date)]
            .reindex(test_px_df.columns)
            .fillna(0.0)
            .astype(float)
        )

        lambda_hat = float(fit_df.loc[stock, "lambda_hat"])
        half_life_seconds = float(fit_df.loc[stock, "half_life_seconds"])

        ADV = float(scaling_df.loc[stock, "ADV"])
        sigma = float(scaling_df.loc[stock, "sigma"])

        beta = np.log(2) / half_life_seconds

        I_star = afs_target_impact_from_alpha_and_mu(
            alpha=alpha,
            alpha_mu=alpha_mu,
            beta=beta,
            c=c,
            apply_terminal_condition=apply_terminal_condition,
        )

        trades = recover_afs_trades_from_target_impact(
            I_star=I_star,
            lambda_hat=lambda_hat,
            ADV=ADV,
            sigma=sigma,
            half_life_seconds=half_life_seconds,
            c=c,
            dt_seconds=dt_seconds,
        )

        raw_abs_volume = trades.abs().sum()
        target_abs_volume = target_participation * ADV

        scale_factor = 1.0

        if normalize_abs_volume and raw_abs_volume > 0:
            scale_factor = target_abs_volume / raw_abs_volume

            trades = trades * scale_factor

            # AFS is nonlinear:
            # if trades scale by s, J scales by s,
            # and impact scales by s^c.
            I_star = I_star * (scale_factor ** c)

        trades_df.loc[(stock, date)] = trades.values
        target_impact_df.loc[(stock, date)] = I_star.values
        alpha_mu_df.loc[(stock, date)] = alpha_mu.values

        scale_rows.append({
            "stock": stock,
            "date": date,
            "ADV": ADV,
            "sigma": sigma,
            "lambda_hat": lambda_hat,
            "half_life_seconds": half_life_seconds,
            "beta": beta,
            "c": c,
            "target_participation": target_participation,
            "target_abs_volume": target_abs_volume,
            "raw_abs_volume": raw_abs_volume,
            "actual_abs_volume": trades.abs().sum(),
            "net_traded": trades.sum(),
            "scale_factor": scale_factor,
            "normalize_abs_volume": normalize_abs_volume,
            "apply_terminal_condition": apply_terminal_condition,
        })

    strategy_scale_df = pd.DataFrame(scale_rows)

    return trades_df, target_impact_df, alpha_mu_df, strategy_scale_df


# ============================================================
# Reduced-form dynamic-liquidity optimal strategy
# ============================================================

def reduced_form_local_volume_state(q_public, half_life_seconds, dt_seconds=10, eps=1e-12):
    """
    Compute the reduced-form local volume state v_t for one stock-day.

    This matches the convention used in impact_model_fitting.impact_state
    for model_type='reduced_form':

        v_t = EMA(|q_public,t|)

    with the same ewm trick used in the fitting code so that the state
    recursion corresponds to

        v_n = decay * v_{n-1} + |q_public,n|.

    Parameters
    ----------
    q_public : pd.Series
        Public signed traded volume for one stock-day, indexed by intraday time.
    half_life_seconds : float
        Half-life used by the reduced-form impact model.
    dt_seconds : int
        Bin size in seconds.
    eps : float
        Lower bound to avoid division by zero.

    Returns
    -------
    v : pd.Series
        Local market volume state indexed like q_public.
    """
    q_public = (
        pd.Series(q_public)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    beta = np.log(2) / half_life_seconds
    decay = np.exp(-beta * dt_seconds)
    ewm_alpha = 1.0 - decay

    local_input = q_public.abs().copy()

    # Same convention as impact_model_fitting.py:
    # y_0 = input_0, y_n = decay*y_{n-1} + input_n for n >= 1.
    if len(local_input) > 1:
        local_input.iloc[1:] = local_input.iloc[1:] / ewm_alpha

    v = local_input.ewm(alpha=ewm_alpha, adjust=False).mean()
    v = v.clip(lower=eps)

    return v


def reduced_form_gamma_prime_from_volume(
    local_volume_state,
    dt_seconds=10,
    method="log_backward",
    clip_abs=None,
):
    """
    Compute gamma'_t for the dynamic-liquidity reduced-form model.

    Reduced-form liquidity is lambda_t = lambda / sqrt(v_t), so

        gamma_t = log(lambda_t) = const - 0.5 log(v_t),
        gamma'_t = -0.5 d log(v_t) / dt.

    The default method uses a backward log-difference, which is stable and
    uses information available at time t.

    Parameters
    ----------
    local_volume_state : pd.Series
        Positive local volume state v_t.
    dt_seconds : int
        Bin size in seconds.
    method : str
        'log_backward' or 'level_backward'.
    clip_abs : float or None
        Optional cap for |gamma_prime|. Useful to prevent unstable denominators
        2*beta + gamma_prime.

    Returns
    -------
    gamma_prime : pd.Series
        Liquidity growth term indexed like local_volume_state.
    """
    v = (
        pd.Series(local_volume_state)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
    )
    v = v.clip(lower=1e-12)

    if method == "log_backward":
        gamma_prime = -0.5 * (np.log(v) - np.log(v.shift(1))) / dt_seconds
    elif method == "level_backward":
        gamma_prime = -0.5 * (v - v.shift(1)) / (dt_seconds * v)
    else:
        raise ValueError("method must be 'log_backward' or 'level_backward'.")

    gamma_prime = gamma_prime.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if clip_abs is not None:
        gamma_prime = gamma_prime.clip(lower=-abs(clip_abs), upper=abs(clip_abs))

    return gamma_prime


def reduced_form_target_impact_from_alpha(
    alpha,
    beta,
    dt_seconds=10,
    gamma_prime=None,
    use_dynamic_target=True,
    denominator_floor=1e-12,
):
    """
    Compute the reduced-form target impact state.

    Exact dynamic-liquidity target:

        I*_t = ((beta + gamma'_t) / (2 beta + gamma'_t)) alpha_t
               - (1 / (2 beta + gamma'_t)) mu_t,

    where mu_t ~= (alpha_{t+dt} - alpha_t) / dt.

    If use_dynamic_target=False, use the slow-moving-liquidity heuristic:

        I*_t = 0.5 * (alpha_t - mu_t / beta),

    and only use v_t when translating target impact into trades.

    Returns
    -------
    I_star : pd.Series
        Target actual impact in return units.
    mu : pd.Series
        Alpha drift/derivative.
    """
    alpha = (
        pd.Series(alpha)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    mu = (alpha.shift(-1) - alpha) / dt_seconds
    mu = mu.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if (gamma_prime is None) or (not use_dynamic_target):
        I_star = 0.5 * (alpha - mu / beta)
    else:
        gamma_prime = pd.Series(gamma_prime, index=alpha.index).astype(float).fillna(0.0)
        denom = 2.0 * beta + gamma_prime

        # Avoid division by numbers too close to zero; preserve sign where possible.
        small = denom.abs() < denominator_floor
        denom = denom.where(~small, np.sign(denom).replace(0.0, 1.0) * denominator_floor)

        I_star = ((beta + gamma_prime) / denom) * alpha - (mu / denom)

    # Terminal condition analogous to the OW deterministic-alpha formula.
    I_star.iloc[-1] = alpha.iloc[-1]

    return I_star, mu


def recover_reduced_form_trades_from_target_impact(
    I_star,
    lambda_hat,
    ADV,
    sigma,
    local_volume_state,
    half_life_seconds,
    dt_seconds=10,
    eps=1e-12,
):
    """
    Recover trades from a reduced-form target impact path.

    The fitted reduced-form impact recursion is

        I_n = decay * I_{n-1}
              + lambda_hat * sigma * q_n / sqrt(ADV * v_n),

    where v_n is the exogenous local market-volume state computed from
    the public tape.

    Hence

        q_n = (I*_n - decay * I*_{n-1})
              * sqrt(ADV * v_n) / (lambda_hat * sigma).
    """
    I_star = pd.Series(I_star).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    v = pd.Series(local_volume_state, index=I_star.index).astype(float).clip(lower=eps)

    beta = np.log(2) / half_life_seconds
    decay = np.exp(-beta * dt_seconds)

    denom = lambda_hat * sigma
    if abs(denom) < eps or ADV <= eps:
        return pd.Series(0.0, index=I_star.index)

    trades = []
    prev_I = 0.0

    for target_I, v_t in zip(I_star.values.astype(float), v.values.astype(float)):
        lambda_eff_t = denom / np.sqrt(ADV * max(v_t, eps))
        if abs(lambda_eff_t) < eps:
            q = 0.0
        else:
            q = (target_I - decay * prev_I) / lambda_eff_t
        trades.append(q)
        prev_I = target_I

    return pd.Series(trades, index=I_star.index)


def make_reduced_form_optimal_trade_df(
    alpha_df,
    test_px_df,
    test_traded_volume_df,
    scaling_df,
    fit_df,
    dt_seconds=10,
    normalize_abs_volume=True,
    target_participation=0.20,
    use_dynamic_target=True,
    gamma_method="log_backward",
    gamma_clip_multiple=0.5,
):
    """
    Build optimal strategy trades for the reduced-form dynamic-liquidity model.

    Compared with OW, there are two changes:

    1. The target impact may use gamma'_t:

        I*_t = ((beta + gamma'_t)/(2 beta + gamma'_t)) alpha_t
               - mu_t/(2 beta + gamma'_t).

       If use_dynamic_target=False, this falls back to the OW/slow-liquidity
       target I*_t = 0.5(alpha_t - mu_t / beta).

    2. The inverse impact equation uses time-varying liquidity:

        q_t = (I*_t - decay I*_{t-1}) sqrt(ADV v_t) / (lambda_hat sigma).

    Parameters
    ----------
    alpha_df : pd.DataFrame
        Stock-date x time alpha panel.
    test_px_df : pd.DataFrame
        Stock-date x time price panel, used for index/columns.
    test_traded_volume_df : pd.DataFrame
        Public signed trade panel, used to compute v_t.
    scaling_df : pd.DataFrame
        Stock-level ADV/sigma table indexed by stock.
    fit_df : pd.DataFrame
        Reduced-form fitted parameters indexed by stock.
    normalize_abs_volume : bool
        If True, rescale each stock-day so sum |q_t| = target_participation * ADV.
    use_dynamic_target : bool
        If True, use gamma'_t in the target impact formula. If False, use the
        slow-moving-liquidity target and only use v_t in the trade inversion.
    gamma_clip_multiple : float or None
        Optional cap |gamma'_t| <= gamma_clip_multiple * beta. This prevents
        unstable target denominators. Set to None to disable.

    Returns
    -------
    trades_df : pd.DataFrame
    target_impact_df : pd.DataFrame
    alpha_mu_df : pd.DataFrame
    local_volume_df : pd.DataFrame
    gamma_prime_df : pd.DataFrame
    strategy_scale_df : pd.DataFrame
    """
    trades_df = pd.DataFrame(0.0, index=test_px_df.index, columns=test_px_df.columns)
    target_impact_df = pd.DataFrame(0.0, index=test_px_df.index, columns=test_px_df.columns)
    alpha_mu_df = pd.DataFrame(0.0, index=test_px_df.index, columns=test_px_df.columns)
    local_volume_df = pd.DataFrame(0.0, index=test_px_df.index, columns=test_px_df.columns)
    gamma_prime_df = pd.DataFrame(0.0, index=test_px_df.index, columns=test_px_df.columns)

    scale_rows = []

    for stock, date in test_px_df.index:
        if stock not in fit_df.index or stock not in scaling_df.index:
            continue
        if (stock, date) not in alpha_df.index or (stock, date) not in test_traded_volume_df.index:
            continue

        alpha = alpha_df.loc[(stock, date)].reindex(test_px_df.columns).fillna(0.0).astype(float)
        q_public = test_traded_volume_df.loc[(stock, date)].reindex(test_px_df.columns).fillna(0.0).astype(float)

        lambda_hat = float(fit_df.loc[stock, "lambda_hat"])
        half_life_seconds = float(fit_df.loc[stock, "half_life_seconds"])
        ADV = float(scaling_df.loc[stock, "ADV"])
        sigma = float(scaling_df.loc[stock, "sigma"])

        beta = np.log(2) / half_life_seconds

        v = reduced_form_local_volume_state(
            q_public=q_public,
            half_life_seconds=half_life_seconds,
            dt_seconds=dt_seconds,
        )

        gamma_clip = None if gamma_clip_multiple is None else gamma_clip_multiple * beta
        gamma_prime = reduced_form_gamma_prime_from_volume(
            local_volume_state=v,
            dt_seconds=dt_seconds,
            method=gamma_method,
            clip_abs=gamma_clip,
        )

        I_star, alpha_mu = reduced_form_target_impact_from_alpha(
            alpha=alpha,
            beta=beta,
            dt_seconds=dt_seconds,
            gamma_prime=gamma_prime,
            use_dynamic_target=use_dynamic_target,
        )

        trades = recover_reduced_form_trades_from_target_impact(
            I_star=I_star,
            lambda_hat=lambda_hat,
            ADV=ADV,
            sigma=sigma,
            local_volume_state=v,
            half_life_seconds=half_life_seconds,
            dt_seconds=dt_seconds,
        )

        raw_abs_volume = trades.abs().sum()
        target_abs_volume = target_participation * ADV
        scale_factor = 1.0

        if normalize_abs_volume and raw_abs_volume > 0:
            scale_factor = target_abs_volume / raw_abs_volume
            trades = trades * scale_factor
            I_star = I_star * scale_factor

        trades_df.loc[(stock, date)] = trades.values
        target_impact_df.loc[(stock, date)] = I_star.values
        alpha_mu_df.loc[(stock, date)] = alpha_mu.values
        local_volume_df.loc[(stock, date)] = v.values
        gamma_prime_df.loc[(stock, date)] = gamma_prime.values

        scale_rows.append({
            "stock": stock,
            "date": date,
            "model_type": "reduced_form",
            "ADV": ADV,
            "sigma": sigma,
            "lambda_hat": lambda_hat,
            "half_life_seconds": half_life_seconds,
            "beta": beta,
            "use_dynamic_target": use_dynamic_target,
            "gamma_method": gamma_method,
            "gamma_clip_multiple": gamma_clip_multiple,
            "target_participation": target_participation,
            "target_abs_volume": target_abs_volume,
            "raw_abs_volume": raw_abs_volume,
            "actual_abs_volume": trades.abs().sum(),
            "net_traded": trades.sum(),
            "scale_factor": scale_factor,
            "normalize_abs_volume": normalize_abs_volume,
            "mean_local_volume": float(v.mean()),
            "mean_gamma_prime": float(gamma_prime.mean()),
            "max_abs_gamma_prime": float(gamma_prime.abs().max()),
        })

    strategy_scale_df = pd.DataFrame(scale_rows)

    return (
        trades_df,
        target_impact_df,
        alpha_mu_df,
        local_volume_df,
        gamma_prime_df,
        strategy_scale_df,
    )


def reduced_form_recompute_impact_from_trades(
    strategy_trades,
    q_public,
    lambda_hat,
    ADV,
    sigma,
    half_life_seconds,
    dt_seconds=10,
):
    """
    Diagnostic helper for one stock-day.

    Recompute the actual reduced-form impact generated by strategy_trades,
    using the public tape q_public to compute the exogenous local-volume
    state v_t. This should track target_impact_df if no trade normalization
    or clipping mismatch exists.
    """
    strategy_trades = pd.Series(strategy_trades).astype(float).fillna(0.0)
    q_public = pd.Series(q_public, index=strategy_trades.index).astype(float).fillna(0.0)

    v = reduced_form_local_volume_state(q_public, half_life_seconds, dt_seconds)

    beta = np.log(2) / half_life_seconds
    decay = np.exp(-beta * dt_seconds)

    impact = []
    prev_I = 0.0

    for q_t, v_t in zip(strategy_trades.values, v.values):
        lambda_eff_t = lambda_hat * sigma / np.sqrt(ADV * max(v_t, 1e-12))
        I_t = decay * prev_I + lambda_eff_t * q_t
        impact.append(I_t)
        prev_I = I_t

    return pd.Series(impact, index=strategy_trades.index)
