import numpy as np
import pandas as pd

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