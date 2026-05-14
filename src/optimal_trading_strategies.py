import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from src.backtest_engine import *  # includes make_twap_trade_df and make_round_trip_twap_trade_df
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

################################################################

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


def run_one_stock_day_paths_for_strategies(
    stock,
    date,
    strategy_trade_dfs,
    model_type,
    test_px_df,
    test_traded_volume_df,
    scaling_df,
    fit_dfs,
    dt_seconds=10,
):
    paths = {}
    summaries = {}

    for strategy_name, trades_df in strategy_trade_dfs.items():

        path_df, summary = run_one_stock_day_diagnostic(
            stock=stock,
            date=date,
            model_type=model_type,
            test_px_df=test_px_df,
            strategy_trades_df=trades_df,
            test_traded_volume_df=test_traded_volume_df,
            scaling_df=scaling_df,
            fit_dfs=fit_dfs,
            dt_seconds=dt_seconds,
            plot=False,
        )

        paths[strategy_name] = path_df
        summaries[strategy_name] = summary

    return paths, pd.DataFrame(summaries).T


########################## AFS #################################

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



##############################################################

def plot_one_stock_day_pnl_comparison(paths, stock, date):
    plt.figure(figsize=(12, 5))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (strategy_name, path_df) in enumerate(paths.items()):
        plt.plot(
            np.arange(len(path_df)),
            path_df["portfolio_value"],
            label=strategy_name,
            color=colors[i % len(colors)],
        )

    tick_positions = np.linspace(0, len(path_df) - 1, 8, dtype=int)
    tick_labels = path_df.index[tick_positions]

    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"{stock} {date} - Intraday PnL by strategy")
    plt.xlabel("Time")
    plt.ylabel("Portfolio value / PnL")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def plot_one_stock_day_final_pnl_bar(summary_df, stock, date):
    plot_df = summary_df.copy()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plt.figure(figsize=(9, 4))
    plt.bar(plot_df.index, plot_df["daily_pnl"], color=[colors[i % len(colors)] for i in range(len(plot_df))])
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"{stock} {date} - Final PnL by strategy")
    plt.ylabel("Daily PnL")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_one_stock_day_impact_cost_bar(summary_df, stock, date):
    plot_df = summary_df.copy()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plt.figure(figsize=(9, 4))
    plt.bar(
        plot_df.index,
        plot_df["impact_cost"],
        color=[colors[i % len(colors)] for i in range(len(plot_df))],
    )
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"{stock} {date} - Impact cost by strategy")
    plt.ylabel("Impact cost")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_stock_daily_pnl_by_strategy(backtest_results_df, stock):
    stock_df = backtest_results_df[
        backtest_results_df["stock"] == stock
    ].copy()

    stock_df["date"] = pd.to_datetime(stock_df["date"])

    pivot = stock_df.pivot_table(
        index="date",
        columns="strategy",
        values="pnl_bps",
        aggfunc="mean",
    ).sort_index()

    plt.figure(figsize=(12, 5))

    for strategy in pivot.columns:
        plt.plot(
            pivot.index,
            pivot[strategy],
            marker="o",
            label=strategy,
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"{stock} - Daily PnL by strategy")
    plt.xlabel("Date")
    plt.ylabel("PnL / traded notional, bps")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def plot_stock_cumulative_pnl_by_strategy(backtest_results_df, stock):
    stock_df = backtest_results_df[
        backtest_results_df["stock"] == stock
    ].copy()

    stock_df["date"] = pd.to_datetime(stock_df["date"])

    pivot = stock_df.pivot_table(
        index="date",
        columns="strategy",
        values="pnl_bps",
        aggfunc="mean",
    ).sort_index()

    cum_pnl = pivot.cumsum()

    plt.figure(figsize=(12, 5))

    for strategy in cum_pnl.columns:
        plt.plot(
            cum_pnl.index,
            cum_pnl[strategy],
            marker="o",
            label=strategy,
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"{stock} - Cumulative PnL by strategy")
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL, bps")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def strategy_summary_table(backtest_results_df):
    return (
        backtest_results_df
        .groupby("strategy")
        [["pnl_bps", "impact_cost_bps", "daily_pnl", "impact_cost"]]
        .agg(["mean", "median", "std"])
        .round(4)
    )
    

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

        try:
            scaling_row = _get_scaling_row(scaling_df, stock, date)
        except KeyError:
            continue

        alpha = (
            alpha_used_df
            .loc[(stock, date)]
            .reindex(test_px_df.columns)
            .fillna(0.0)
            .astype(float)
        )

        ADV = float(scaling_row["ADV"])
        sigma = float(scaling_row["sigma"])

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
            "max_fraction_adv_per_bin_input": max_fraction_adv_per_bin,
            "turnover_penalty": turnover_penalty,
            "terminal_inventory_penalty": terminal_inventory_penalty,
            "impact_penalty_multiplier": impact_penalty_multiplier,
        })

        diagnostic_rows.append(diagnostics)

    diagnostics_df = pd.DataFrame(diagnostic_rows)

    return trades_df, impact_state_df, diagnostics_df



# ============================================================
# FIXED / ROBUST BENCHMARK AND PERFORMANCE HELPERS
# ============================================================

# Performance helpers use TWAP schedules imported from src.backtest_engine.

def _aggregate_strategy_daily(backtest_results_df, pnl_col="pnl_bps", daily_pnl_col="daily_pnl"):
    """
    Aggregate stock-day backtest rows to a strategy-date portfolio panel.

    Dollar PnL/costs are summed across stocks. Bps metrics and impact states are
    averaged across stocks because they are normalized quantities.
    """
    df = backtest_results_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    agg_spec = {}
    for col in df.columns:
        if col in ["strategy", "date", "stock", "backtest_model"]:
            continue
        if col in ["daily_pnl", "impact_cost", "total_abs_traded", "net_traded"]:
            agg_spec[col] = "sum"
        elif col in ["pnl_bps", "impact_cost_bps", "max_abs_impact"]:
            agg_spec[col] = "mean"

    group_cols = ["strategy", "date"]
    if "backtest_model" in df.columns:
        group_cols = ["backtest_model"] + group_cols

    if not agg_spec:
        return df[group_cols].drop_duplicates()

    return df.groupby(group_cols, as_index=False).agg(agg_spec)


def compute_drawdown(cumulative_pnl):
    """Drawdown from a cumulative PnL series."""
    running_max = cumulative_pnl.cummax()
    return cumulative_pnl - running_max


def strategy_performance_summary(
    backtest_results_df,
    pnl_col="pnl_bps",
    daily_pnl_col="daily_pnl",
    impact_cost_col="impact_cost_bps",
    turnover_col="total_abs_traded",
    max_impact_col="max_abs_impact",
    annualization=252,
    aggregate_by_date=True,
):
    """
    Compile performance metrics for each strategy.

    By default, stock-day rows are first aggregated to date-level portfolio rows.
    This makes expected daily PnL, Sharpe, drawdown, and hit-rate true daily
    strategy metrics rather than pooled stock-day statistics.
    """
    df = backtest_results_df.copy()
    if aggregate_by_date:
        df = _aggregate_strategy_daily(
            df,
            pnl_col=pnl_col,
            daily_pnl_col=daily_pnl_col,
        )

    df["date"] = pd.to_datetime(df["date"])

    group_cols = ["strategy"]
    if "backtest_model" in df.columns:
        group_cols = ["backtest_model", "strategy"]

    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(group_cols, keys))

        g = g.sort_values("date")
        daily_pnl = g[daily_pnl_col].astype(float)
        pnl_bps = g[pnl_col].astype(float)

        cum_pnl = daily_pnl.cumsum()
        drawdown = compute_drawdown(cum_pnl)

        mean_pnl_bps = pnl_bps.mean()
        vol_pnl_bps = pnl_bps.std(ddof=1)
        sharpe = np.nan if (pd.isna(vol_pnl_bps) or vol_pnl_bps <= 0) else mean_pnl_bps / vol_pnl_bps * np.sqrt(annualization)

        row = {
            **key_dict,
            "n_days": g["date"].nunique(),
            "mean_daily_pnl": daily_pnl.mean(),
            "median_daily_pnl": daily_pnl.median(),
            "std_daily_pnl": daily_pnl.std(ddof=1),
            "mean_pnl_bps": mean_pnl_bps,
            "median_pnl_bps": pnl_bps.median(),
            "std_pnl_bps": vol_pnl_bps,
            "sharpe_bps": sharpe,
            "hit_rate": (daily_pnl > 0).mean(),
            "max_daily_drawdown": drawdown.min(),
            "final_cumulative_pnl": cum_pnl.iloc[-1] if len(cum_pnl) else np.nan,
        }

        for col, out_prefix in [
            (impact_cost_col, "impact_cost_bps"),
            (turnover_col, "turnover"),
            (max_impact_col, "max_abs_impact"),
        ]:
            if col in g.columns:
                row[f"mean_{out_prefix}"] = g[col].mean()
                row[f"median_{out_prefix}"] = g[col].median()
                row[f"std_{out_prefix}"] = g[col].std(ddof=1)
                if col == max_impact_col:
                    row["max_impact_dislocation"] = g[col].max()

        rows.append(row)

    out = pd.DataFrame(rows)
    index_cols = [c for c in ["backtest_model", "strategy"] if c in out.columns]
    if index_cols:
        out = out.set_index(index_cols)
    if "sharpe_bps" in out.columns:
        out = out.sort_values("sharpe_bps", ascending=False)
    return out


def plot_cumulative_pnl(backtest_results_df, value_col="pnl_bps", title="Cumulative PnL by strategy", aggregate_by_date=True):
    df = _aggregate_strategy_daily(backtest_results_df) if aggregate_by_date else backtest_results_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="strategy", values=value_col, aggfunc="mean").sort_index()
    cumulative = pivot.cumsum()
    plt.figure(figsize=(12, 5))
    for strategy in cumulative.columns:
        plt.plot(cumulative.index, cumulative[strategy], label=strategy)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(f"Cumulative {value_col}")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
    return cumulative


def plot_drawdowns(backtest_results_df, value_col="daily_pnl", title="Drawdowns by strategy", aggregate_by_date=True):
    df = _aggregate_strategy_daily(backtest_results_df) if aggregate_by_date else backtest_results_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="strategy", values=value_col, aggfunc="sum").sort_index()
    cumulative = pivot.cumsum()
    drawdowns = cumulative.apply(compute_drawdown)
    plt.figure(figsize=(12, 5))
    for strategy in drawdowns.columns:
        plt.plot(drawdowns.index, drawdowns[strategy], label=strategy)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
    return drawdowns


def plot_cumulative_impact_cost(backtest_results_df, value_col="impact_cost_bps", title="Cumulative impact cost by strategy", aggregate_by_date=True):
    df = _aggregate_strategy_daily(backtest_results_df) if aggregate_by_date else backtest_results_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="strategy", values=value_col, aggfunc="mean").sort_index()
    cumulative = pivot.cumsum()
    plt.figure(figsize=(12, 5))
    for strategy in cumulative.columns:
        plt.plot(cumulative.index, cumulative[strategy], label=strategy)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(f"Cumulative {value_col}")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
    return cumulative


def plot_daily_pnl_distribution(backtest_results_df, value_col="pnl_bps", bins=30, title="Daily PnL distribution", aggregate_by_date=True):
    df = _aggregate_strategy_daily(backtest_results_df) if aggregate_by_date else backtest_results_df.copy()
    plt.figure(figsize=(10, 5))
    for strategy, g in df.groupby("strategy"):
        plt.hist(g[value_col].dropna(), bins=bins, alpha=0.5, density=True, label=strategy)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel(value_col)
    plt.ylabel("Density")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def trade_schedule_summary(strategy_trade_dfs, scaling_df):
    """Summarise turnover, net trading, and max trade size as fractions of ADV."""
    rows = []
    for strategy, trades_df in strategy_trade_dfs.items():
        for stock, date in trades_df.index:
            trades = trades_df.loc[(stock, date)].astype(float)
            try:
                scaling_row = _get_scaling_row(scaling_df, stock, date)
                ADV = float(scaling_row["ADV"])
            except KeyError:
                continue
            rows.append({
                "strategy": strategy,
                "stock": stock,
                "date": date,
                "ADV": ADV,
                "abs_volume_over_ADV": trades.abs().sum() / ADV,
                "net_traded_over_ADV": trades.sum() / ADV,
                "max_abs_trade_over_ADV": trades.abs().max() / ADV,
            })
    return pd.DataFrame(rows)


def plot_turnover_boxplot(schedule_summary_df, value_col="abs_volume_over_ADV", title="Turnover over ADV by strategy"):
    df = schedule_summary_df.copy()
    strategies = df["strategy"].dropna().unique()
    data = [df.loc[df["strategy"] == s, value_col].dropna() for s in strategies]
    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=strategies, showfliers=True)
    plt.axhline(0.2, linestyle="--", linewidth=1, label="20% ADV")
    plt.title(title)
    plt.ylabel(value_col)
    plt.xticks(rotation=30)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def compute_forward_returns(price_df):
    """Compute one-step forward simple returns from an intraday price panel."""
    returns = price_df.pct_change(axis=1).shift(-1, axis=1)
    return returns.replace([np.inf, -np.inf], np.nan)


def alpha_return_correlation(alpha_df, price_df, method="pearson"):
    """Compute pooled alpha correlation with next-bin returns."""
    fwd_returns = compute_forward_returns(price_df)
    common_index = alpha_df.index.intersection(fwd_returns.index)
    common_cols = alpha_df.columns.intersection(fwd_returns.columns)
    alpha = alpha_df.loc[common_index, common_cols]
    returns = fwd_returns.loc[common_index, common_cols]
    stacked = pd.DataFrame({"alpha": alpha.stack(), "forward_return": returns.stack()}).dropna()
    ic = stacked["alpha"].corr(stacked["forward_return"], method=method) if len(stacked) else np.nan
    return ic, stacked


def alpha_ic_by_day(alpha_df, price_df, method="pearson"):
    """Compute daily alpha IC pooled across stocks and intraday bins for each date."""
    fwd_returns = compute_forward_returns(price_df)
    common_index = alpha_df.index.intersection(fwd_returns.index)
    common_cols = alpha_df.columns.intersection(fwd_returns.columns)
    alpha = alpha_df.loc[common_index, common_cols]
    returns = fwd_returns.loc[common_index, common_cols]

    rows = []
    dates = sorted(alpha.index.get_level_values("date").unique()) if isinstance(alpha.index, pd.MultiIndex) else []
    for date in dates:
        a = alpha.xs(date, level="date")
        r = returns.xs(date, level="date")
        tmp = pd.DataFrame({"alpha": a.stack(), "forward_return": r.stack()}).dropna()
        ic = tmp["alpha"].corr(tmp["forward_return"], method=method) if len(tmp) > 2 else np.nan
        rows.append({"date": date, "ic": ic, "n_obs": len(tmp)})
    return pd.DataFrame(rows)
