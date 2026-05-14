import numpy as np
import pandas as pd
from src.optimal_trading_strategy import *


##########------- Alpha Correlation ---------############

def alpha_forward_return_corr(alpha_df, px_df, horizon_bins=1):
    """
    Compute alpha-forward-return correlation per stock.

    alpha_df:
        index   = (stock, date)
        columns = intraday time bins
        values  = alpha signal used by the strategy

    px_df:
        index   = (stock, date)
        columns = intraday time bins
        values  = prices

    horizon_bins:
        number of intraday bins ahead used to compute future return
    """

    common_index = alpha_df.index.intersection(px_df.index)
    common_cols = alpha_df.columns.intersection(px_df.columns)

    alpha = alpha_df.loc[common_index, common_cols].astype(float)
    px = px_df.loc[common_index, common_cols].astype(float)

    future_px = px.shift(-horizon_bins, axis=1)
    fwd_return = (future_px - px) / px

    rows = []

    stocks = alpha.index.get_level_values(0).unique()

    for stock in stocks:
        alpha_stock = alpha.loc[stock].to_numpy().ravel()
        ret_stock = fwd_return.loc[stock].to_numpy().ravel()

        valid = (
            np.isfinite(alpha_stock)
            & np.isfinite(ret_stock)
        )

        alpha_valid = alpha_stock[valid]
        ret_valid = ret_stock[valid]

        if len(alpha_valid) > 2:
            corr = np.corrcoef(alpha_valid, ret_valid)[0, 1]
        else:
            corr = np.nan

        rows.append({
            "stock": stock,
            "horizon_bins": horizon_bins,
            "alpha_return_corr": corr,
            "n_obs": len(alpha_valid),
        })

    corr_df = pd.DataFrame(rows).set_index("stock").sort_index()

    return corr_df

## Correlation by stock

def plot_alpha_corr_by_stock(corr_df, title=None):
    """
    Bar plot of alpha-return correlation by stock.
    """

    ax = corr_df["alpha_return_corr"].plot(
        kind="bar",
        figsize=(12, 5)
    )

    ax.axhline(0, color="black", linewidth=1)

    if title is None:
        horizon_bins = corr_df["horizon_bins"].iloc[0]
        title = f"Alpha-forward-return correlation by stock, horizon={horizon_bins} bins"

    ax.set_title(title)
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Stock")

    plt.tight_layout()
    plt.show()

## Correlation across horizons
def plot_alpha_corr_by_horizon(alpha_df, px_df, horizon_bins_list):
    """
    Compute and plot average alpha-return correlation across horizons.
    """

    rows = []

    for horizon_bins in horizon_bins_list:
        corr_df = alpha_forward_return_corr(
            alpha_df=alpha_df,
            px_df=px_df,
            horizon_bins=horizon_bins
        )

        rows.append({
            "horizon_bins": horizon_bins,
            "mean_corr": corr_df["alpha_return_corr"].mean(),
            "median_corr": corr_df["alpha_return_corr"].median(),
            "min_corr": corr_df["alpha_return_corr"].min(),
            "max_corr": corr_df["alpha_return_corr"].max(),
            "n_stocks": corr_df["alpha_return_corr"].notna().sum(),
        })

    horizon_df = pd.DataFrame(rows)

    ax = horizon_df.plot(
        x="horizon_bins",
        y=["mean_corr", "median_corr"],
        marker="o",
        figsize=(9, 5)
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Alpha-forward-return correlation across prediction horizons")
    ax.set_xlabel("Horizon bins")
    ax.set_ylabel("Correlation")

    plt.tight_layout()
    plt.show()

    return horizon_df


#---------------------------------------------------------#


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




def build_daily_portfolio_df(backtests_df):
    """
    Convert stock-date strategy backtest rows into daily portfolio rows.
    One row = one backtest_model, one strategy, one date.
    """
    df = backtests_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    daily_df = (
        df
        .groupby(["backtest_model", "strategy", "date"])
        .agg(
            daily_pnl=("daily_pnl", "sum"),
            impact_cost=("impact_cost", "sum"),
            traded_notional=("traded_notional", "sum"),
            total_abs_traded=("total_abs_traded", "sum"),
            net_traded=("net_traded", "sum"),
            max_abs_impact=("max_abs_impact", "max"),
        )
        .reset_index()
    )

    daily_df["pnl_bps"] = (
        1e4 * daily_df["daily_pnl"] / daily_df["traded_notional"]
    )

    daily_df["impact_cost_bps"] = (
        1e4 * daily_df["impact_cost"] / daily_df["traded_notional"]
    )

    return daily_df

def max_drawdown(x):
    cumulative = x.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    return drawdown.min()

def summarize_strategy_performance(daily_portfolio_df):
    rows = []

    for (model, strategy), g in daily_portfolio_df.groupby(["backtest_model", "strategy"]):
        g = g.sort_values("date")

        mean_daily_pnl = g["daily_pnl"].mean()
        std_daily_pnl = g["daily_pnl"].std()

        mean_pnl_bps = g["pnl_bps"].mean()
        std_pnl_bps = g["pnl_bps"].std()

        daily_sharpe = (
            mean_pnl_bps / std_pnl_bps
            if std_pnl_bps > 0
            else np.nan
        )

        rows.append({
            "backtest_model": model,
            "strategy": strategy,

            "expected_daily_pnl": mean_daily_pnl,
            "expected_daily_pnl_bps": mean_pnl_bps,
            "daily_pnl_vol": std_daily_pnl,
            "daily_pnl_vol_bps": std_pnl_bps,

            "daily_sharpe": daily_sharpe,
            "annualized_sharpe": np.sqrt(252) * daily_sharpe,

            "mean_transaction_cost": g["impact_cost"].mean(),
            "mean_transaction_cost_bps": g["impact_cost_bps"].mean(),

            "max_daily_drawdown": max_drawdown(g["daily_pnl"]),
            "max_daily_drawdown_bps": max_drawdown(g["pnl_bps"]),

            "max_impact_dislocation": g["max_abs_impact"].max(),
            "mean_impact_dislocation": g["max_abs_impact"].mean(),

            "mean_total_abs_traded": g["total_abs_traded"].mean(),
            "mean_net_traded": g["net_traded"].mean(),

            "n_days": g["date"].nunique(),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["backtest_model", "strategy"])
        .reset_index(drop=True)
    )
        
    
def plot_cumulative_pnl_by_model(daily_portfolio_df, backtest_model):
    plot_df = daily_portfolio_df[
        daily_portfolio_df["backtest_model"] == backtest_model
    ].copy()

    pivot = (
        plot_df
        .pivot_table(
            index="date",
            columns="strategy",
            values="pnl_bps",
            aggfunc="sum",
        )
        .sort_index()
    )

    cumulative = pivot.cumsum()

    plt.figure(figsize=(12, 5))

    for strategy in cumulative.columns:
        plt.plot(
            cumulative.index,
            cumulative[strategy],
            marker="o",
            label=strategy,
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"Cumulative P&L under {backtest_model} simulator")
    plt.xlabel("Date")
    plt.ylabel("Cumulative P&L, bps")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

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











