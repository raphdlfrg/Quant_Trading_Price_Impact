import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from src.backtest_engine import *  # includes make_twap_trade_df and make_round_trip_twap_trade_df
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


def nonlinear_trade_summary(
    trades_df,
    impact_state_df,
    diag_df=None,
):
    """
    Summarize nonlinear optimal strategy outputs.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Trade panel indexed by (stock, date).

    impact_state_df : pd.DataFrame
        Impact-state panel with same shape.

    diag_df : pd.DataFrame or None
        Optional optimizer diagnostics dataframe.

    Returns
    -------
    summary_df : pd.DataFrame
        Stock-day summary statistics.
    """

    rows = []

    for stock, date in trades_df.index:

        trades = trades_df.loc[(stock, date)].astype(float)

        impact = (
            impact_state_df
            .loc[(stock, date)]
            .astype(float)
        )

        row = {
            "stock": stock,
            "date": date,
            "total_abs_traded": trades.abs().sum(),
            "net_traded": trades.sum(),
            "max_abs_trade": trades.abs().max(),
            "mean_abs_trade": trades.abs().mean(),
            "max_abs_impact": impact.abs().max(),
            "mean_abs_impact": impact.abs().mean(),
            "impact_std": impact.std(),
        }

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    if diag_df is not None and len(diag_df) > 0:

        merge_cols = [
            c for c in ["stock", "date"]
            if c in diag_df.columns
        ]

        if len(merge_cols) == 2:

            summary_df = summary_df.merge(
                diag_df,
                on=["stock", "date"],
                how="left",
            )

    return summary_df


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
    control_block_size=30,   # 30 x 10s = 5 minutes
    max_fraction_adv_per_bin=0.002,
    turnover_penalty=1e-4,
    terminal_inventory_penalty=1e-2,
    impact_penalty_multiplier=1.0,
    initial_trade_fraction=None,
    maxiter=50,
):
    """
    Compute one-day optimal trades for a nonlinear fitted impact model.

    Optimization is done on a coarse control grid:
        one decision every `control_block_size` 10-second bins.

    The optimized coarse trade is then spread evenly across the
    corresponding 10-second bins.
    """

    # ------------------------------------------------------------
    # Clean alpha
    # ------------------------------------------------------------
    alpha = alpha.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n_fine = len(alpha)

    # ------------------------------------------------------------
    # Smooth and aggregate alpha to coarse 5-minute blocks
    # ------------------------------------------------------------
    alpha_smooth = alpha.rolling(
        control_block_size,
        min_periods=1
    ).mean()

    block_id = np.arange(n_fine) // control_block_size

    alpha_coarse = (
        alpha_smooth
        .groupby(block_id)
        .mean()
    )

    n_coarse = len(alpha_coarse)

    # ------------------------------------------------------------
    # Initial guess on coarse grid
    # ------------------------------------------------------------
    if initial_trade_fraction is None:
        x0 = np.zeros(n_coarse)
    else:
        x0 = np.asarray(initial_trade_fraction, dtype=float)

        if len(x0) == n_fine:
            # Convert fine initial guess to coarse totals
            x0 = pd.Series(x0).groupby(block_id).sum().values

        elif len(x0) != n_coarse:
            raise ValueError(
                "initial_trade_fraction must have length equal to "
                "either the fine grid or the coarse grid."
            )

    # ------------------------------------------------------------
    # Bounds on coarse trades
    # ------------------------------------------------------------
    # max_fraction_adv_per_bin is a 10-second limit.
    # A 5-minute block contains control_block_size bins, so the
    # coarse block can trade up to control_block_size times more.
    max_fraction_adv_per_block = (
        max_fraction_adv_per_bin * control_block_size
    )

    bounds = [
        (-max_fraction_adv_per_block, max_fraction_adv_per_block)
        for _ in range(n_coarse)
    ]

    # ------------------------------------------------------------
    # Optimize on coarse grid
    # ------------------------------------------------------------
    result = minimize(
        fun=nonlinear_strategy_objective,
        x0=x0,
        args=(
            alpha_coarse.values,
            sigma,
            lambda_hat,
            half_life_seconds,
            model_type,
            dt_seconds * control_block_size,
            turnover_penalty,
            terminal_inventory_penalty,
            impact_penalty_multiplier,
        ),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": maxiter,
            "ftol": 1e-8,
        },
    )

    coarse_fraction = result.x

    # ------------------------------------------------------------
    # Expand coarse trades back to fine 10-second grid
    # ------------------------------------------------------------
    fine_fraction = np.repeat(
        coarse_fraction / control_block_size,
        control_block_size
    )[:n_fine]

    optimal_fraction = pd.Series(
        fine_fraction,
        index=alpha.index,
        name="trade_fraction_ADV",
    )

    optimal_trades = ADV * optimal_fraction

    # ------------------------------------------------------------
    # Compute fine-grid impact state from the fine trades
    # ------------------------------------------------------------
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
        "n_fine_controls": n_fine,
        "n_coarse_controls": n_coarse,
        "control_block_size": control_block_size,
        "total_abs_fraction_adv": float(np.abs(optimal_fraction).sum()),
        "net_fraction_adv": float(optimal_fraction.sum()),
        "max_abs_fraction_per_bin": float(np.abs(optimal_fraction).max()),
        "max_abs_fraction_per_block": float(np.abs(coarse_fraction).max()),
    }

    return optimal_trades, optimal_fraction, impact_state, diagnostics


def make_nonlinear_optimal_trade_df(
    alpha_df,
    test_px_df,
    scaling_df,
    fit_df,
    model_type,
    dt_seconds=10,
    control_block_size=30,
    smooth_alpha_window=None,
    max_fraction_adv_per_bin=0.002,
    turnover_penalty=1e-4,
    terminal_inventory_penalty=1e-2,
    impact_penalty_multiplier=1.0,
    maxiter=50,
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
                control_block_size=control_block_size,
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
            "control_block_size": control_block_size,
            "smooth_alpha_window": smooth_alpha_window,
            "maxiter": maxiter,
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



# ============================================================
# NONLINEAR DIAGNOSTIC HELPERS
# ============================================================

def summarize_nonlinear_diagnostics(*diagnostics_dfs):
    """
    Summarise nonlinear optimizer diagnostics.

    Accepts either one diagnostics dataframe or several diagnostics dataframes.
    Each dataframe should contain at least `model_type`, `success`, and the
    trade fraction diagnostics returned by make_nonlinear_optimal_trade_df.
    """
    if len(diagnostics_dfs) == 1:
        df = diagnostics_dfs[0].copy()
    else:
        df = pd.concat([d.copy() for d in diagnostics_dfs if d is not None and len(d) > 0], ignore_index=True)

    if df.empty:
        return pd.DataFrame()

    group_cols = ["model_type"] if "model_type" in df.columns else []

    agg_cols = [
        "success",
        "objective",
        "total_abs_fraction_adv",
        "net_fraction_adv",
        "max_abs_fraction_per_bin",
        "max_abs_fraction_per_block",
    ]
    agg_cols = [c for c in agg_cols if c in df.columns]

    if not group_cols:
        return df[agg_cols].describe().T

    out = []
    for model_type, g in df.groupby(group_cols):
        row = {"model_type": model_type if not isinstance(model_type, tuple) else model_type[0]}
        if "success" in g.columns:
            row["success_rate"] = g["success"].mean()
            row["n_failures"] = int((~g["success"].astype(bool)).sum())
        for col in agg_cols:
            if col == "success":
                continue
            row[f"mean_{col}"] = g[col].mean()
            row[f"median_{col}"] = g[col].median()
            row[f"max_{col}"] = g[col].max()
        row["n_stock_days"] = len(g)
        out.append(row)
    return pd.DataFrame(out).set_index("model_type")


def find_failed_nonlinear_optimizations(diagnostics_df):
    """Return rows where the scipy optimizer failed."""
    if diagnostics_df is None or diagnostics_df.empty or "success" not in diagnostics_df.columns:
        return pd.DataFrame()
    return diagnostics_df.loc[~diagnostics_df["success"].astype(bool)].copy()


def nonlinear_trade_summary(trades_df, scaling_df, strategy_name=None):
    """
    Summarise a nonlinear trade panel in ADV units.

    Returns one row per stock-day with total absolute turnover, net trading,
    and maximum bin trade, all normalised by ADV.
    """
    rows = []
    for stock, date in trades_df.index:
        try:
            scaling_row = _get_scaling_row(scaling_df, stock, date)
        except KeyError:
            continue
        ADV = float(scaling_row["ADV"])
        trades = trades_df.loc[(stock, date)].astype(float)
        row = {
            "stock": stock,
            "date": date,
            "ADV": ADV,
            "total_abs_traded": trades.abs().sum(),
            "net_traded": trades.sum(),
            "max_abs_trade": trades.abs().max(),
            "total_abs_traded_over_ADV": trades.abs().sum() / ADV,
            "net_traded_over_ADV": trades.sum() / ADV,
            "max_abs_trade_over_ADV": trades.abs().max() / ADV,
        }
        if strategy_name is not None:
            row["strategy"] = strategy_name
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# GENERAL STRATEGY COMPARISON PLOTS
# ============================================================

def compare_strategy_metric_by_model(
    backtest_results_df,
    metric="pnl_bps",
    title=None,
    by_model=True,
    showfliers=True,
):
    """
    Boxplot comparing a metric by strategy, optionally split by backtest model.
    """
    df = backtest_results_df.copy()
    if title is None:
        title = f"{metric} by strategy"

    if by_model and "backtest_model" in df.columns:
        models = df["backtest_model"].dropna().unique()
        for model in models:
            g = df[df["backtest_model"] == model]
            strategies = g["strategy"].dropna().unique()
            data = [g.loc[g["strategy"] == s, metric].dropna() for s in strategies]
            plt.figure(figsize=(10, 5))
            plt.boxplot(data, labels=strategies, showfliers=showfliers)
            plt.axhline(0, linestyle="--", linewidth=1)
            plt.title(f"{title} - {model}")
            plt.ylabel(metric)
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.show()
    else:
        strategies = df["strategy"].dropna().unique()
        data = [df.loc[df["strategy"] == s, metric].dropna() for s in strategies]
        plt.figure(figsize=(10, 5))
        plt.boxplot(data, labels=strategies, showfliers=showfliers)
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.title(title)
        plt.ylabel(metric)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()


def plot_strategy_cumulative_metric(
    backtest_results_df,
    metric="daily_pnl",
    title=None,
    aggregate_by_date=True,
):
    """
    Plot cumulative metric by strategy.

    If aggregate_by_date=True, first aggregates stock-day rows to portfolio-day
    rows using _aggregate_strategy_daily.
    """
    df = _aggregate_strategy_daily(backtest_results_df) if aggregate_by_date else backtest_results_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    group_cols = ["strategy"]
    if "backtest_model" in df.columns:
        group_cols = ["backtest_model", "strategy"]

    if title is None:
        title = f"Cumulative {metric} by strategy"

    plt.figure(figsize=(12, 5))
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            label = keys
        else:
            label = " | ".join(map(str, keys))
        g = g.sort_values("date")
        plt.plot(g["date"], g[metric].cumsum(), label=label)

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(f"Cumulative {metric}")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


# ============================================================
# LECTURE-STYLE ONE-DAY DIAGNOSTICS
# ============================================================

def compute_pre_post_execution_impact_one_day(
    my_trades,
    ADV,
    sigma,
    half_life_seconds,
    model_type="ow",
    dt_seconds=10,
    eps=1e-12,
):
    """
    Compute pre-trade impact, post-trade impact, and execution impact.

    The execution impact is approximated by the midpoint between pre- and
    post-trade impact, consistent with the OW flat-book lecture intuition.
    """
    my_trades = my_trades.fillna(0.0).astype(float)

    beta = np.log(2) / half_life_seconds
    decay = np.exp(-beta * dt_seconds)

    pre_features = []
    post_features = []
    exec_features = []

    if model_type == "ow":
        I_prev = 0.0
        for q in my_trades:
            I_pre = decay * I_prev
            input_term = sigma * q / ADV
            I_post = I_pre + input_term
            I_exec = 0.5 * (I_pre + I_post)
            pre_features.append(I_pre)
            post_features.append(I_post)
            exec_features.append(I_exec)
            I_prev = I_post

    elif model_type == "sqrt_propagator":
        I_prev = 0.0
        for q in my_trades:
            I_pre = decay * I_prev
            u = q / ADV
            input_term = sigma * np.sign(u) * np.sqrt(abs(u))
            I_post = I_pre + input_term
            I_exec = 0.5 * (I_pre + I_post)
            pre_features.append(I_pre)
            post_features.append(I_post)
            exec_features.append(I_exec)
            I_prev = I_post

    elif model_type == "afs":
        J_prev = 0.0
        for q in my_trades:
            J_pre = decay * J_prev
            J_post = J_pre + sigma * q / ADV
            I_pre = np.sign(J_pre) * np.sqrt(abs(J_pre))
            I_post = np.sign(J_post) * np.sqrt(abs(J_post))
            I_exec = 0.5 * (I_pre + I_post)
            pre_features.append(I_pre)
            post_features.append(I_post)
            exec_features.append(I_exec)
            J_prev = J_post

    elif model_type == "reduced_form":
        I_prev = 0.0
        v_prev = 0.0
        for q in my_trades:
            I_pre = decay * I_prev
            v_pre = decay * v_prev
            v_post = max(v_pre + abs(q), eps)
            input_term = sigma * q / np.sqrt(ADV * v_post)
            I_post = I_pre + input_term
            I_exec = 0.5 * (I_pre + I_post)
            pre_features.append(I_pre)
            post_features.append(I_post)
            exec_features.append(I_exec)
            I_prev = I_post
            v_prev = v_post

    else:
        raise ValueError("model_type must be one of: 'ow', 'sqrt_propagator', 'afs', 'reduced_form'.")

    return pd.DataFrame(
        {
            "pre_impact_feature": pre_features,
            "post_impact_feature": post_features,
            "execution_impact_feature": exec_features,
        },
        index=my_trades.index,
    )


def simulate_one_stock_day_with_naive_and_realistic_pnl(
    stock,
    date,
    strategy_name,
    model_type,
    strategy_trade_dfs,
    test_px_df,
    test_traded_volume_df,
    scaling_df,
    fit_dfs,
    dt_seconds=10,
):
    """
    Lecture-style one-stock-day diagnostic.

    Computes impact-free price, impact-resultant price, execution price,
    cumulative impact, trade size, naive/accounting PnL, and realistic /
    fundamental PnL.
    """
    fit_df = fit_dfs[model_type]

    raw_prices = test_px_df.loc[(stock, date)].astype(float)
    public_trades = test_traded_volume_df.loc[(stock, date)].astype(float)
    my_trades = strategy_trade_dfs[strategy_name].loc[(stock, date)].astype(float)

    lambda_hat = float(fit_df.loc[stock, "lambda_hat"])
    half_life_seconds = float(fit_df.loc[stock, "half_life_seconds"])

    scaling_row = _get_scaling_row(scaling_df, stock, date)
    ADV = float(scaling_row["ADV"])
    sigma = float(scaling_row["sigma"])

    impact_free_price = make_impact_adjusted_prices(
        prices=raw_prices,
        public_trades=public_trades,
        lambda_hat=lambda_hat,
        ADV=ADV,
        sigma=sigma,
        half_life_seconds=half_life_seconds,
        model_type=model_type,
        dt_seconds=dt_seconds,
    )

    impact_features = compute_pre_post_execution_impact_one_day(
        my_trades=my_trades,
        ADV=ADV,
        sigma=sigma,
        half_life_seconds=half_life_seconds,
        model_type=model_type,
        dt_seconds=dt_seconds,
    )

    impact_features = lambda_hat * impact_features

    pre_impact_in_price = impact_free_price.iloc[0] * impact_features["pre_impact_feature"]
    post_impact_in_price = impact_free_price.iloc[0] * impact_features["post_impact_feature"]
    execution_impact_in_price = impact_free_price.iloc[0] * impact_features["execution_impact_feature"]

    impact_resultant_price = impact_free_price + post_impact_in_price
    execution_price = impact_free_price + execution_impact_in_price

    position = my_trades.cumsum()
    cash = -(my_trades * execution_price).cumsum()

    naive_pnl = cash + position * impact_resultant_price
    realistic_pnl = cash + position * impact_free_price
    footprint_pnl = naive_pnl - realistic_pnl

    path_df = pd.DataFrame(
        {
            "raw_mid_price": raw_prices,
            "impact_free_price": impact_free_price,
            "trade": my_trades,
            "position": position,
            "pre_impact_in_price": pre_impact_in_price,
            "post_impact_in_price": post_impact_in_price,
            "execution_impact_in_price": execution_impact_in_price,
            "impact_in_price": post_impact_in_price,
            "impact_resultant_price": impact_resultant_price,
            "execution_price": execution_price,
            "cash": cash,
            "naive_pnl": naive_pnl,
            "realistic_pnl": realistic_pnl,
            "footprint_pnl": footprint_pnl,
        }
    )

    summary = {
        "stock": stock,
        "date": date,
        "strategy": strategy_name,
        "model_type": model_type,
        "lambda_hat": lambda_hat,
        "half_life_seconds": half_life_seconds,
        "ADV": ADV,
        "sigma": sigma,
        "final_naive_pnl": naive_pnl.iloc[-1],
        "final_realistic_pnl": realistic_pnl.iloc[-1],
        "final_footprint_pnl": footprint_pnl.iloc[-1],
        "impact_cost": (my_trades * execution_impact_in_price).sum(),
        "total_abs_traded": my_trades.abs().sum(),
        "net_traded": my_trades.sum(),
        "max_abs_impact": post_impact_in_price.abs().max(),
    }

    return path_df, summary


def plot_lecture_style_strategy_diagnostics(
    path_df,
    stock,
    date,
    strategy_name,
    model_type,
    sample_every=10,
):
    """
    Lecture-style plots:
    1. impact-free price vs impact-resultant price,
    2. cumulative impact,
    3. trade size,
    4. naive/accounting PnL vs realistic/fundamental PnL.
    """
    plot_df = path_df.iloc[::sample_every].copy()

    x = np.arange(len(plot_df))
    tick_positions = np.linspace(0, len(plot_df) - 1, 8, dtype=int)
    tick_labels = plot_df.index[tick_positions]

    plt.figure(figsize=(12, 4))
    plt.plot(x, plot_df["impact_free_price"].values, label="Impact-free price")
    plt.plot(x, plot_df["impact_resultant_price"].values, label="Impact-resultant price")
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.title(f"{stock} {date} - {strategy_name} under {model_type}: impact-free vs impact-resultant price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(x, 100 * plot_df["impact_in_price"].values, label="Cumulative impact")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.title(f"{stock} {date} - {strategy_name} under {model_type}: cumulative impact")
    plt.xlabel("Time")
    plt.ylabel("Impact, cents")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.bar(x, plot_df["trade"].values, width=1.0, label="Trade size")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.title(f"{stock} {date} - {strategy_name} under {model_type}: trade size")
    plt.xlabel("Time")
    plt.ylabel("Shares")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(x, plot_df["naive_pnl"].values, label="Naive/accounting PnL")
    plt.plot(x, plot_df["realistic_pnl"].values, label="Realistic/fundamental PnL")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.title(f"{stock} {date} - {strategy_name} under {model_type}: PnL comparison")
    plt.xlabel("Time")
    plt.ylabel("PnL")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
