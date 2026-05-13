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



##################################################

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

    plt.figure(figsize=(9, 4))
    plt.bar(plot_df.index, plot_df["daily_pnl"])
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"{stock} {date} - Final PnL by strategy")
    plt.ylabel("Daily PnL")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_one_stock_day_impact_cost_bar(summary_df, stock, date):
    plot_df = summary_df.copy()

    plt.figure(figsize=(9, 4))
    plt.bar(plot_df.index, plot_df["impact_cost"])
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