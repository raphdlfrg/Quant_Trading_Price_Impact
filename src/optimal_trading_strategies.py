import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.backtest_engine import *

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

def make_next_open_overnight_alpha_panel(test_px_df, overnight_alpha_level=1.0):
    """
    Build a stock-date x time panel of synthetic overnight alpha.

    For each stock-date d, the alpha is the next open return:
        (P_{d+1, open} - P_{d, close}) / P_{d, close}

    The value is repeated over all intraday time bins.
    The final available day for each stock has no next open and is set to zero.
    """
    first_col = test_px_df.columns[0]
    last_col = test_px_df.columns[-1]

    current_close = test_px_df[last_col].astype(float)
    next_open = (
        test_px_df[first_col]
        .astype(float)
        .groupby(level="stock")
        .shift(-1)
    )

    overnight_return = (next_open - current_close) / current_close
    overnight_return = overnight_return.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    overnight_return = overnight_alpha_level * overnight_return

    overnight_alpha_df = pd.DataFrame(
        np.repeat(overnight_return.values[:, None], test_px_df.shape[1], axis=1),
        index=test_px_df.index,
        columns=test_px_df.columns,
    )

    return overnight_alpha_df, overnight_return.rename("overnight_alpha")

def ow_target_impact_from_alpha(alpha, beta, dt_seconds):
    """
    Compute the lecture-note OW target impact state:
        I*_t = 1/2 (alpha_t - beta^{-1} alpha'_t).

    alpha is a pd.Series indexed by intraday time, in return units.
    beta is the impact decay speed in seconds^{-1}.
    """
    alpha = alpha.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    alpha_prime = alpha.diff() / dt_seconds
    if len(alpha_prime) > 1:
        alpha_prime.iloc[0] = alpha_prime.iloc[1]
    else:
        alpha_prime.iloc[0] = 0.0

    I_star = 0.5 * (alpha - alpha_prime / beta)

    # Terminal condition from the deterministic-alpha OW formula.
    I_star.iloc[-1] = alpha.iloc[-1]

    return I_star, alpha_prime


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

    If normalize_abs_volume=True, each stock-day is rescaled so that
    total absolute traded volume equals target_participation * ADV.
    This preserves the OW trading shape but makes scenarios comparable.
    """
    trades_df = pd.DataFrame(0.0, index=test_px_df.index, columns=test_px_df.columns)
    target_impact_df = pd.DataFrame(0.0, index=test_px_df.index, columns=test_px_df.columns)
    alpha_prime_df = pd.DataFrame(0.0, index=test_px_df.index, columns=test_px_df.columns)

    for stock, date in test_px_df.index:
        if stock not in fit_df.index or stock not in scaling_df.index:
            continue

        alpha = alpha_df.loc[(stock, date)].reindex(test_px_df.columns).fillna(0.0)

        lambda_hat = float(fit_df.loc[stock, "lambda_hat"])
        half_life_seconds = float(fit_df.loc[stock, "half_life_seconds"])
        beta = np.log(2) / half_life_seconds

        ADV = float(scaling_df.loc[stock, "ADV"])
        sigma = float(scaling_df.loc[stock, "sigma"])

        I_star, alpha_prime = ow_target_impact_from_alpha(
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

        if normalize_abs_volume:
            target_abs_volume = target_participation * ADV
            current_abs_volume = trades.abs().sum()

            if current_abs_volume > 0:
                trades = trades * target_abs_volume / current_abs_volume

        trades_df.loc[(stock, date)] = trades.values
        target_impact_df.loc[(stock, date)] = I_star.values
        alpha_prime_df.loc[(stock, date)] = alpha_prime.values

    return trades_df, target_impact_df, alpha_prime_df

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