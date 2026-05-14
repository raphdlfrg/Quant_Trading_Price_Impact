import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from src.backtest_engine import *
from src.synthetic_alphas import *



########################## LOADING HELPERS ######################
def load_panel_csv(path):
    """Load a stock-date x intraday-time panel saved by previous notebooks."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index(["stock", "date"]).sort_index()
    df.columns = df.columns.astype(str)
    return df

def load_stock_level_csv(path):
    """Load a stock-level CSV and index by stock when possible."""
    df = pd.read_csv(path)
    if "stock" in df.columns:
        df = df.set_index("stock").sort_index()
    return df


########################## PANEL HELPERS ######################
def smooth_panel_rows(panel_df, window=None):
    """
    Smooth each stock-day row across intraday time bins.

    A small amount of smoothing is useful because the alpha decay signal is
    estimated from finite differences and is therefore much noisier than the
    alpha level itself.
    """
    if window is None or window <= 1:
        return panel_df.copy()

    return (
        panel_df
        .T
        .rolling(window=window, min_periods=1)
        .mean()
        .T
    )


def _get_scaling_row(scaling_df, stock, date):
    """
    Return ADV and sigma for a stock-date.

    Supports both:
    - scaling_df indexed by stock;
    - scaling_df indexed by (stock, date).
    """
    if isinstance(scaling_df.index, pd.MultiIndex):
        return scaling_df.loc[(stock, date)]
    return scaling_df.loc[stock]


########################## OW STRATEGY ######################
def make_alpha_decay_df_from_alpha(alpha_df, dt_seconds=10):
    """
    Compute the synthetic alpha decay signal from an alpha level panel.

    Lecture convention:

        d alpha_t = mu_t dt + sigma_t dW_t,
        decay_t = -mu_t.

    Discrete forward-difference approximation:

        decay_j = -(alpha_{j+1} - alpha_j) / dt.

    This helper is mainly a fallback. In the final pipeline, prefer passing
    the synthetic_alpha_decay_df created in Section 2.4.
    """
    decay_df = -(
        alpha_df.shift(-1, axis=1)
        - alpha_df
    ) / dt_seconds

    return decay_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def ow_target_impact_from_alpha_and_decay(
    alpha,
    alpha_decay,
    beta,
    smooth_alpha_window=None,
    smooth_decay_window=10,
    target_impact_cap=None,
    force_zero_close=False,
):
    """
    Compute the lecture-note OW target impact state.

    The lecture writes:

        alpha_t = E[S_T - S_t | F_t],
        d alpha_t = mu_t dt + sigma_t dW_t,

    and, for the simple OW case,

        I*_t = 1/2 (alpha_t - mu_t / beta).

    Since Section 2.4 stores the alpha decay as decay_t = -mu_t, this becomes:

        I*_t = 1/2 (alpha_t + decay_t / beta).

    Parameters
    ----------
    alpha : pd.Series
        Alpha level alpha_t in return units.

    alpha_decay : pd.Series
        Alpha decay signal -mu_t in return units per second.

    beta : float
        OW impact decay speed in seconds^{-1}.

    smooth_alpha_window : int or None
        Optional rolling window for alpha level smoothing.

    smooth_decay_window : int or None
        Optional rolling window for alpha decay smoothing.

    target_impact_cap : float or None
        Optional cap on abs(I*_t), in return units.

    force_zero_close : bool
        If True, force the final target impact to zero. Default is False,
        because this function returns the raw OW target implied by alpha and
        decay rather than an execution schedule forced to finish flat.
    """
    alpha = alpha.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    alpha_decay = alpha_decay.reindex(alpha.index).astype(float)
    alpha_decay = alpha_decay.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if smooth_alpha_window is not None and smooth_alpha_window > 1:
        alpha = alpha.rolling(smooth_alpha_window, min_periods=1).mean()

    if smooth_decay_window is not None and smooth_decay_window > 1:
        alpha_decay = alpha_decay.rolling(smooth_decay_window, min_periods=1).mean()

    I_star = 0.5 * (alpha + alpha_decay / beta)

    if target_impact_cap is not None:
        I_star = I_star.clip(lower=-target_impact_cap, upper=target_impact_cap)

    if force_zero_close and len(I_star) > 0:
        I_star.iloc[-1] = 0.0

    return I_star, alpha, alpha_decay


def recover_ow_trades_from_target_impact(
    I_star,
    lambda_hat,
    ADV,
    sigma,
    half_life_seconds,
    dt_seconds=10,
    max_trade_fraction_adv_per_bin=None,
    eps=1e-12,
):
    """
    Recover trades using the discrete fitted OW impact equation.

    Your fitted/backtest OW model separates normalized impact feature F from
    fitted lambda:

        F_j = exp(-beta dt) F_{j-1} + sigma * q_j / ADV,
        I_j = lambda_hat * F_j.

    Therefore, for a target impact I*_j:

        F*_j = I*_j / lambda_hat,
        q_j = ADV / sigma * (F*_j - exp(-beta dt) F*_{j-1}).

    Equivalently:

        q_j = ADV / (lambda_hat sigma)
              * (I*_j - exp(-beta dt) I*_{j-1}).
    """
    I_star = I_star.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

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

    trades = pd.Series(trades, index=I_star.index)

    if max_trade_fraction_adv_per_bin is not None:
        cap = max_trade_fraction_adv_per_bin * ADV
        trades = trades.clip(lower=-cap, upper=cap)

    return trades


def make_ow_optimal_trade_df(
    alpha_df,
    alpha_decay_df,
    test_px_df,
    scaling_df,
    fit_df,
    dt_seconds=10,
    smooth_alpha_window=None,
    smooth_decay_window=10,
    target_impact_cap=None,
    max_trade_fraction_adv_per_bin=None,
    force_zero_close=False,
):
    """
    Build a stock-date x time trade panel from alpha and alpha decay panels
    using the raw closed-form OW target-impact strategy.

    For each stock-day, the function computes:

        alpha_t, decay_t = -mu_t
            -> I*_t = 1/2 (alpha_t + decay_t / beta)
            -> q_t from the fitted OW impact equation.

    No daily absolute-volume normalization is applied. The output is the raw
    OW strategy implied by the fitted OW parameters and the alpha signals.

    Parameters
    ----------
    alpha_df : pd.DataFrame
        Alpha level panel indexed by (stock, date), with intraday time columns.

    alpha_decay_df : pd.DataFrame
        Alpha decay panel from Section 2.4, same shape as alpha_df. Values are
        decay_t = -mu_t, not an alpha level and not another finite difference.

    test_px_df : pd.DataFrame
        Test price panel. Used for the stock-date index and intraday columns.

    scaling_df : pd.DataFrame
        Stock-level or stock-date table. Must contain ADV and sigma.

    fit_df : pd.DataFrame
        Fitted OW parameter table indexed by stock. Must contain lambda_hat and
        half_life_seconds.

    Returns
    -------
    trades_df : pd.DataFrame
        Raw OW trade panel q_{i,d,j}, same shape as test_px_df.

    target_impact_df : pd.DataFrame
        OW target impact panel I*_{i,d,j}, same shape as test_px_df.

    used_alpha_df : pd.DataFrame
        Alpha level after optional smoothing.

    used_alpha_decay_df : pd.DataFrame
        Alpha decay after optional smoothing.

    strategy_diagnostics_df : pd.DataFrame
        Stock-day diagnostics. There is no scale factor because the raw OW
        strategy is not normalized to a target participation rate.
    """
    trades_df = pd.DataFrame(0.0, index=test_px_df.index, columns=test_px_df.columns)
    target_impact_df = pd.DataFrame(0.0, index=test_px_df.index, columns=test_px_df.columns)
    used_alpha_df = pd.DataFrame(0.0, index=test_px_df.index, columns=test_px_df.columns)
    used_alpha_decay_df = pd.DataFrame(0.0, index=test_px_df.index, columns=test_px_df.columns)

    diagnostics_rows = []

    for stock, date in test_px_df.index:
        if stock not in fit_df.index:
            continue

        try:
            scaling_row = _get_scaling_row(scaling_df, stock, date)
        except KeyError:
            continue

        alpha = (
            alpha_df
            .loc[(stock, date)]
            .reindex(test_px_df.columns)
            .fillna(0.0)
            .astype(float)
        )

        alpha_decay = (
            alpha_decay_df
            .loc[(stock, date)]
            .reindex(test_px_df.columns)
            .fillna(0.0)
            .astype(float)
        )

        lambda_hat = float(fit_df.loc[stock, "lambda_hat"])
        half_life_seconds = float(fit_df.loc[stock, "half_life_seconds"])

        ADV = float(scaling_row["ADV"])
        sigma = float(scaling_row["sigma"])

        beta = np.log(2) / half_life_seconds

        I_star, alpha_used, alpha_decay_used = ow_target_impact_from_alpha_and_decay(
            alpha=alpha,
            alpha_decay=alpha_decay,
            beta=beta,
            smooth_alpha_window=smooth_alpha_window,
            smooth_decay_window=smooth_decay_window,
            target_impact_cap=target_impact_cap,
            force_zero_close=force_zero_close,
        )

        trades = recover_ow_trades_from_target_impact(
            I_star=I_star,
            lambda_hat=lambda_hat,
            ADV=ADV,
            sigma=sigma,
            half_life_seconds=half_life_seconds,
            dt_seconds=dt_seconds,
            max_trade_fraction_adv_per_bin=max_trade_fraction_adv_per_bin,
        )

        trades_df.loc[(stock, date)] = trades.values
        target_impact_df.loc[(stock, date)] = I_star.values
        used_alpha_df.loc[(stock, date)] = alpha_used.values
        used_alpha_decay_df.loc[(stock, date)] = alpha_decay_used.values

        diagnostics_rows.append({
            "stock": stock,
            "date": date,
            "ADV": ADV,
            "sigma": sigma,
            "lambda_hat": lambda_hat,
            "half_life_seconds": half_life_seconds,
            "beta": beta,
            "smooth_alpha_window": smooth_alpha_window,
            "smooth_decay_window": smooth_decay_window,
            "target_impact_cap": target_impact_cap,
            "max_trade_fraction_adv_per_bin": max_trade_fraction_adv_per_bin,
            "force_zero_close": force_zero_close,
            "total_abs_traded": trades.abs().sum(),
            "net_traded": trades.sum(),
            "max_abs_trade": trades.abs().max(),
            "max_abs_target_impact": I_star.abs().max(),
        })

    strategy_diagnostics_df = pd.DataFrame(diagnostics_rows)

    return (
        trades_df,
        target_impact_df,
        used_alpha_df,
        used_alpha_decay_df,
        strategy_diagnostics_df,
    )


########################## BACKTEST HELPERS ######################
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


########################## PLOTTING HELPERS ######################
def plot_one_stock_day_pnl_comparison(paths, stock, date):
    plt.figure(figsize=(12, 5))

    for strategy_name, path_df in paths.items():
        plt.plot(
            np.arange(len(path_df)),
            path_df["portfolio_value"],
            label=strategy_name,
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
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_df)))

    plt.figure(figsize=(9, 4))
    plt.bar(plot_df.index, plot_df["daily_pnl"], color=colors)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"{stock} {date} - Final PnL by strategy")
    plt.ylabel("Daily PnL")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_one_stock_day_impact_cost_bar(summary_df, stock, date):
    plot_df = summary_df.copy()
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_df)))

    plt.figure(figsize=(9, 4))
    plt.bar(plot_df.index, plot_df["impact_cost"], color=colors)
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




####This part is for the nonlinear impact models. The code is generated with ChatGPT, so someone should check it.

def simulate_fitted_impact_from_trade_fraction(
    trade_fraction,
    sigma,
    lambda_hat,
    half_life_seconds,
    model_type,
    dt_seconds=10,
    eps=1e-12,
):
    """
    Simulate fitted impact state in return units.

    trade_fraction[j] = q_j / ADV.
    Output impact_state[j] = lambda_hat * impact_feature[j].
    """

    beta = np.log(2) / half_life_seconds
    decay = np.exp(-beta * dt_seconds)

    trade_fraction = np.asarray(trade_fraction, dtype=float)

    impact_feature = np.zeros_like(trade_fraction)

    if model_type == "sqrt_propagator":
        I = 0.0
        for j, u in enumerate(trade_fraction):
            input_term = sigma * np.sign(u) * np.sqrt(abs(u))
            I = decay * I + input_term
            impact_feature[j] = I

    elif model_type == "afs":
        J = 0.0
        for j, u in enumerate(trade_fraction):
            input_term = sigma * u
            J = decay * J + input_term
            impact_feature[j] = np.sign(J) * np.sqrt(abs(J))

    elif model_type == "reduced_form":
        I = 0.0
        v_fraction = 0.0

        for j, u in enumerate(trade_fraction):
            v_fraction = decay * v_fraction + abs(u)
            v_fraction = max(v_fraction, eps)

            input_term = sigma * u / np.sqrt(v_fraction)
            I = decay * I + input_term
            impact_feature[j] = I

    else:
        raise ValueError(
            "model_type must be one of: "
            "'sqrt_propagator', 'afs', 'reduced_form'."
        )

    return lambda_hat * impact_feature


def nonlinear_strategy_objective(
    trade_fraction,
    alpha,
    sigma,
    lambda_hat,
    half_life_seconds,
    model_type,
    dt_seconds=10,
    turnover_penalty=1e-4,
    terminal_inventory_penalty=1e-2,
    impact_penalty_multiplier=1.0,
):
    """
    Minimize negative utility.

    Approximate utility:
        alpha gain
        - impact cost
        - turnover penalty
        - terminal inventory penalty

    Everything is expressed in return units and ADV fractions.
    """

    trade_fraction = np.asarray(trade_fraction, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    impact = simulate_fitted_impact_from_trade_fraction(
        trade_fraction=trade_fraction,
        sigma=sigma,
        lambda_hat=lambda_hat,
        half_life_seconds=half_life_seconds,
        model_type=model_type,
        dt_seconds=dt_seconds,
    )

    alpha_gain = np.sum(alpha * trade_fraction)
    impact_cost = np.sum(impact * trade_fraction)

    turnover_cost = turnover_penalty * np.sum(trade_fraction ** 2)

    final_inventory = np.sum(trade_fraction)
    terminal_cost = terminal_inventory_penalty * final_inventory ** 2

    utility = (
        alpha_gain
        - impact_penalty_multiplier * impact_cost
        - turnover_cost
        - terminal_cost
    )

    return -utility


def optimize_nonlinear_strategy_one_day(
    alpha,
    ADV,
    sigma,
    lambda_hat,
    half_life_seconds,
    model_type,
    dt_seconds=10,
    max_fraction_adv_per_bin=0.002,
    turnover_penalty=1e-4,
    terminal_inventory_penalty=1e-2,
    impact_penalty_multiplier=1.0,
    initial_trade_fraction=None,
    maxiter=300,
):
    """
    Compute one-day optimal trades for a nonlinear fitted impact model.

    The optimizer chooses u_j = q_j / ADV.
    Final output is q_j in shares.
    """

    alpha = alpha.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n = len(alpha)

    if initial_trade_fraction is None:
        x0 = np.zeros(n)
    else:
        x0 = np.asarray(initial_trade_fraction, dtype=float)
        if len(x0) != n:
            raise ValueError("initial_trade_fraction must have same length as alpha.")

    bounds = [
        (-max_fraction_adv_per_bin, max_fraction_adv_per_bin)
        for _ in range(n)
    ]

    result = minimize(
        fun=nonlinear_strategy_objective,
        x0=x0,
        args=(
            alpha.values,
            sigma,
            lambda_hat,
            half_life_seconds,
            model_type,
            dt_seconds,
            turnover_penalty,
            terminal_inventory_penalty,
            impact_penalty_multiplier,
        ),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": maxiter,
            "ftol": 1e-10,
        },
    )

    optimal_fraction = pd.Series(
        result.x,
        index=alpha.index,
        name="trade_fraction_ADV",
    )

    optimal_trades = ADV * optimal_fraction

    impact_state = simulate_fitted_impact_from_trade_fraction(
        trade_fraction=optimal_fraction.values,
        sigma=sigma,
        lambda_hat=lambda_hat,
        half_life_seconds=half_life_seconds,
        model_type=model_type,
        dt_seconds=dt_seconds,
    )

    impact_state = pd.Series(
        impact_state,
        index=alpha.index,
        name="impact_state",
    )

    diagnostics = {
        "success": result.success,
        "message": result.message,
        "objective": result.fun,
        "total_abs_fraction_adv": float(np.abs(optimal_fraction).sum()),
        "net_fraction_adv": float(optimal_fraction.sum()),
        "max_abs_fraction_per_bin": float(np.abs(optimal_fraction).max()),
    }

    return optimal_trades, optimal_fraction, impact_state, diagnostics


def make_nonlinear_optimal_trade_df(
    alpha_df,
    test_px_df,
    scaling_df,
    fit_df,
    model_type,
    dt_seconds=10,
    smooth_alpha_window=5,
    max_fraction_adv_per_bin=0.002,
    turnover_penalty=1e-4,
    terminal_inventory_penalty=1e-2,
    impact_penalty_multiplier=1.0,
    maxiter=300,
):
    """
    Build optimal strategy for a fitted nonlinear impact model:
        'afs', 'reduced_form', or 'sqrt_propagator'.
    """

    if model_type not in ["afs", "reduced_form", "sqrt_propagator"]:
        raise ValueError(
            "model_type must be 'afs', 'reduced_form', or 'sqrt_propagator'."
        )

    # Optional smoothing of alpha
    if smooth_alpha_window is not None and smooth_alpha_window > 1:
        alpha_used_df = (
            alpha_df
            .T
            .rolling(smooth_alpha_window, min_periods=1)
            .mean()
            .T
        )
    else:
        alpha_used_df = alpha_df.copy()

    trades_df = pd.DataFrame(
        0.0,
        index=test_px_df.index,
        columns=test_px_df.columns,
    )

    impact_state_df = pd.DataFrame(
        0.0,
        index=test_px_df.index,
        columns=test_px_df.columns,
    )

    diagnostic_rows = []

    for stock, date in test_px_df.index:

        if stock not in fit_df.index:
            continue

        if stock not in scaling_df.index:
            continue

        alpha = (
            alpha_used_df
            .loc[(stock, date)]
            .reindex(test_px_df.columns)
            .fillna(0.0)
            .astype(float)
        )

        ADV = float(scaling_df.loc[stock, "ADV"])
        sigma = float(scaling_df.loc[stock, "sigma"])

        lambda_hat = float(fit_df.loc[stock, "lambda_hat"])
        half_life_seconds = float(fit_df.loc[stock, "half_life_seconds"])

        trades, trade_fraction, impact_state, diagnostics = (
            optimize_nonlinear_strategy_one_day(
                alpha=alpha,
                ADV=ADV,
                sigma=sigma,
                lambda_hat=lambda_hat,
                half_life_seconds=half_life_seconds,
                model_type=model_type,
                dt_seconds=dt_seconds,
                max_fraction_adv_per_bin=max_fraction_adv_per_bin,
                turnover_penalty=turnover_penalty,
                terminal_inventory_penalty=terminal_inventory_penalty,
                impact_penalty_multiplier=impact_penalty_multiplier,
                maxiter=maxiter,
            )
        )

        trades_df.loc[(stock, date)] = trades.values
        impact_state_df.loc[(stock, date)] = impact_state.values

        diagnostics.update({
            "stock": stock,
            "date": date,
            "model_type": model_type,
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

