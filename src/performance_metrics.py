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


#################### P&L #######################

#-------------------- Final P&L by stock---------------------
def plot_matched_final_pnl_by_stock(
    backtest_results_df,
    matched_pairs=None,
    use_bps=True,
    title="Final P&L by stock for matched strategies"
):
    """
    Plot final P&L by stock for matched model-strategy pairs.

    backtest_results_df:
        stock-day level dataframe, e.g. all_model_comparison_df

    matched_pairs:
        dict mapping backtest_model -> strategy
        example:
            {
                "ow": "ow_optimal",
                "afs": "afs_optimal",
                "reduced_form": "reduced_form_optimal",
            }

    use_bps:
        If True, plot P&L normalized by total traded notional in bps.
        If False, plot raw final P&L.
    """

    if matched_pairs is None:
        matched_pairs = {
            "ow": "ow_optimal",
            "afs": "afs_optimal",
            "reduced_form": "reduced_form_optimal",
        }

    df = backtest_results_df.copy()

    matched_rows = []

    for model, strategy in matched_pairs.items():
        temp = df[
            (df["backtest_model"] == model)
            & (df["strategy"] == strategy)
        ].copy()

        matched_rows.append(temp)

    matched_df = pd.concat(matched_rows, ignore_index=True)

    stock_pnl_df = (
        matched_df
        .groupby(["stock", "backtest_model", "strategy"])
        .agg(
            final_pnl=("daily_pnl", "sum"),
            total_traded_notional=("traded_notional", "sum"),
            total_impact_cost=("impact_cost", "sum"),
            n_days=("date", "nunique"),
        )
        .reset_index()
    )

    stock_pnl_df["final_pnl_bps"] = (
        1e4
        * stock_pnl_df["final_pnl"]
        / stock_pnl_df["total_traded_notional"]
    )

    value_col = "final_pnl_bps" if use_bps else "final_pnl"

    stock_pnl_df["label"] = (
        stock_pnl_df["backtest_model"]
        + " / "
        + stock_pnl_df["strategy"]
    )

    pivot = stock_pnl_df.pivot_table(
        index="stock",
        columns="label",
        values=value_col,
        aggfunc="sum"
    ).sort_index()

    ax = pivot.plot(
        kind="bar",
        figsize=(13, 5)
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Stock")

    if use_bps:
        ax.set_ylabel("Final P&L / total traded notional, bps")
    else:
        ax.set_ylabel("Final P&L")

    plt.xticks(rotation=45)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return stock_pnl_df

#------------------------- Final P&L by strategy ------------------------------
def plot_final_pnl_by_strategy(
    daily_df,
    use_bps=True,
    title="Final P&L by matched strategy"
):
    """
    Plot final cumulative P&L by strategy.

    daily_df should already contain only the strategies you want to compare,
    e.g. matched_daily_df.

    If use_bps=True, final P&L is normalized by total traded notional.
    """

    final_pnl_df = (
        daily_df
        .groupby(["backtest_model", "strategy"])
        .agg(
            final_pnl=("daily_pnl", "sum"),
            total_traded_notional=("traded_notional", "sum"),
            total_impact_cost=("impact_cost", "sum"),
        )
        .reset_index()
    )

    final_pnl_df["final_pnl_bps"] = (
        1e4
        * final_pnl_df["final_pnl"]
        / final_pnl_df["total_traded_notional"]
    )

    value_col = "final_pnl_bps" if use_bps else "final_pnl"

    labels = final_pnl_df["strategy"].values
    values = final_pnl_df[value_col].values

    colors = plt.cm.tab10(range(len(labels)))

    plt.figure(figsize=(8, 4))

    plt.bar(
        labels,
        values,
        color=colors
    )

    plt.axhline(0, color="black", linewidth=1)
    plt.title(title)

    if use_bps:
        plt.ylabel("Final P&L / total traded notional, bps")
    else:
        plt.ylabel("Final P&L")

    plt.xlabel("Strategy")
    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.show()

    return final_pnl_df

#------------------ Hourly portfolio value -----------------------#

def plot_hourly_portfolio_value_matched_strategies(
    strategy_trade_dfs,
    test_px_df,
    test_traded_volume_df,
    scaling_df,
    fit_dfs,
    dt_seconds=10,
    matched_pairs=None,
    title="Average hourly portfolio value by matched strategy",
):
    """
    Plot average hourly portfolio value across the whole test batch.

    Aggregates across:
        all stocks
        all dates

    Each strategy is evaluated under its matching simulator.
    """

    if matched_pairs is None:
        matched_pairs = {
            "ow_optimal": "ow",
            "afs_optimal": "afs",
            "reduced_form_optimal": "reduced_form",
        }

    rows = []

    for strategy_name, model_type in matched_pairs.items():

        trades_df = strategy_trade_dfs[strategy_name]

        for stock, date in test_px_df.index:

            if (stock, date) not in trades_df.index:
                continue

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

            temp = pd.DataFrame({
                "time": path_df.index.astype(str),
                "portfolio_value": path_df["portfolio_value"].values,
            })

            temp["strategy"] = strategy_name
            temp["stock"] = stock
            temp["date"] = date

            rows.append(temp)

    intraday_df = pd.concat(rows, ignore_index=True)

    intraday_df["hour"] = (
        pd.to_datetime(intraday_df["time"], errors="coerce")
        .dt.hour
    )

    hourly_df = (
        intraday_df
        .dropna(subset=["hour"])
        .groupby(["strategy", "hour"])
        .agg(
            avg_portfolio_value=("portfolio_value", "mean"),
            total_portfolio_value=("portfolio_value", "sum"),
            n_obs=("portfolio_value", "count"),
        )
        .reset_index()
    )

    pivot = hourly_df.pivot_table(
        index="hour",
        columns="strategy",
        values="avg_portfolio_value",
    ).sort_index()

    plt.figure(figsize=(10, 5))

    for strategy in pivot.columns:
        plt.plot(
            pivot.index,
            pivot[strategy],
            marker="o",
            label=strategy,
        )

    plt.axhline(0, color="black", linewidth=1)
    plt.title(title)
    plt.xlabel("Hour of day")
    plt.ylabel("Average portfolio value / P&L")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return hourly_df


###################### IMPACT ###########################

def plot_target_vs_realized_impact(
    target_impact,
    path_df,
    stock,
    date,
    strategy_name,
    model_type,
):
    """
    Plot target impact vs realized impact for one stock-day.

    target_impact:
        Series indexed by intraday time, usually in return/alpha units.

    path_df:
        Output from run_one_stock_day_diagnostic.
        Must contain impact_in_price and mid_price.
    """

    realized_impact = path_df["impact_in_price"] / path_df["mid_price"]

    common_index = target_impact.index.intersection(realized_impact.index)

    target = target_impact.loc[common_index].astype(float)
    realized = realized_impact.loc[common_index].astype(float)

    x = np.arange(len(common_index))
    tick_positions = np.linspace(0, len(common_index) - 1, 8, dtype=int)
    tick_labels = common_index[tick_positions]

    plt.figure(figsize=(12, 5))

    plt.plot(x, target.values, label="Target impact")
    plt.plot(x, realized.values, label="Realized impact")

    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(tick_positions, tick_labels, rotation=45)

    plt.title(
        f"{stock} {date} - target vs realized impact\n"
        f"{strategy_name} under {model_type} simulator"
    )

    plt.xlabel("Time")
    plt.ylabel("Impact, return units")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

def plot_impact_tracking_error(
    target_impact,
    path_df,
    stock,
    date,
    strategy_name,
    model_type,
):
    """
    Plot realized impact minus target impact for one stock-day.
    Both are compared in return units.
    """

    realized_impact = path_df["impact_in_price"] / path_df["mid_price"]

    common_index = target_impact.index.intersection(realized_impact.index)

    target = target_impact.loc[common_index].astype(float)
    realized = realized_impact.loc[common_index].astype(float)

    tracking_error = realized - target

    x = np.arange(len(common_index))
    tick_positions = np.linspace(0, len(common_index) - 1, 8, dtype=int)
    tick_labels = common_index[tick_positions]

    plt.figure(figsize=(12, 4))

    plt.plot(x, tracking_error.values, label="Realized - target impact")
    plt.axhline(0, color="black", linewidth=1)

    plt.xticks(tick_positions, tick_labels, rotation=45)

    plt.title(
        f"{stock} {date} - impact tracking error\n"
        f"{strategy_name} under {model_type} simulator"
    )

    plt.xlabel("Time")
    plt.ylabel("Tracking error, return units")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    stats = {
        "mean_error": tracking_error.mean(),
        "mean_abs_error": tracking_error.abs().mean(),
        "rmse": np.sqrt((tracking_error ** 2).mean()),
        "max_abs_error": tracking_error.abs().max(),
    }

    return tracking_error, stats

def compare_impact_tracking_by_strategy(
    strategy_trade_dfs,
    strategy_target_impact_dfs,
    test_px_df,
    test_traded_volume_df,
    scaling_df,
    fit_dfs,
    matched_pairs=None,
    dt_seconds=10,
    plot=True,
):
    """
    Compare target-vs-realized impact tracking across matched strategies.

    Computes:
        error = realized impact - target impact

    where realized impact is converted to return units:
        impact_in_price / mid_price
    """

    if matched_pairs is None:
        matched_pairs = {
            "ow_optimal": "ow",
            "afs_optimal": "afs",
            "reduced_form_optimal": "reduced_form",
        }

    rows = []

    for strategy_name, model_type in matched_pairs.items():

        trades_df = strategy_trade_dfs[strategy_name]
        target_df = strategy_target_impact_dfs[strategy_name]

        for stock, date in test_px_df.index:

            key = (stock, date)

            if key not in trades_df.index:
                continue

            if key not in target_df.index:
                continue

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

            target_impact = target_df.loc[key].astype(float)
            realized_impact = (
                path_df["impact_in_price"]
                / path_df["mid_price"]
            )

            common_index = target_impact.index.intersection(realized_impact.index)

            target = target_impact.loc[common_index].astype(float)
            realized = realized_impact.loc[common_index].astype(float)

            error = realized - target

            rows.append({
                "strategy": strategy_name,
                "backtest_model": model_type,
                "stock": stock,
                "date": date,
                "mean_error": error.mean(),
                "mae": error.abs().mean(),
                "rmse": np.sqrt((error ** 2).mean()),
                "max_abs_error": error.abs().max(),
            })

    tracking_df = pd.DataFrame(rows)

    summary_df = (
        tracking_df
        .groupby(["backtest_model", "strategy"])
        .agg(
            mean_error=("mean_error", "mean"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            max_abs_error=("max_abs_error", "max"),
            n_stock_days=("date", "count"),
        )
        .reset_index()
    )

    if plot:
        plot_df = summary_df.copy()

        x = np.arange(len(plot_df))
        width = 0.35

        plt.figure(figsize=(9, 4))

        plt.bar(
            x - width / 2,
            plot_df["mean_mae"],
            width,
            label="MAE",
        )

        plt.bar(
            x + width / 2,
            plot_df["mean_rmse"],
            width,
            label="RMSE",
        )

        plt.xticks(x, plot_df["strategy"], rotation=30)
        plt.axhline(0, color="black", linewidth=1)

        plt.title("Impact tracking error by matched strategy")
        plt.ylabel("Tracking error, return units")
        plt.xlabel("Strategy")
        plt.legend(frameon=False)

        plt.tight_layout()
        plt.show()

    return tracking_df, summary_df

def plot_alpha_and_target_impact(
    alpha_df,
    target_impact_df,
    stock,
    date,
    strategy_name,
    normalize=True,
    bucket_size=60,
):
    """
    Plot alpha signal and target impact for one stock-day.

    Aggregates intraday values into buckets to make the plot readable.
    Default bucket_size=60 means 10-minute buckets if dt=10 seconds.
    """

    alpha = alpha_df.loc[(stock, date)].astype(float)
    target = target_impact_df.loc[(stock, date)].astype(float)

    common_cols = alpha.index.intersection(target.index)

    alpha = alpha.loc[common_cols]
    target = target.loc[common_cols]

    temp = pd.DataFrame({
        "alpha": alpha.values,
        "target_impact": target.values,
        "time": common_cols,
    })

    temp["bucket"] = np.arange(len(temp)) // bucket_size

    plot_df = (
        temp
        .groupby("bucket")
        .agg(
            alpha=("alpha", "mean"),
            target_impact=("target_impact", "mean"),
            time=("time", "first"),
        )
        .reset_index(drop=True)
    )

    if normalize:
        plot_df["alpha_plot"] = plot_df["alpha"] / plot_df["alpha"].abs().max()
        plot_df["target_plot"] = (
            plot_df["target_impact"]
            / plot_df["target_impact"].abs().max()
        )
        ylabel = "Normalized value"
    else:
        plot_df["alpha_plot"] = plot_df["alpha"]
        plot_df["target_plot"] = plot_df["target_impact"]
        ylabel = "Signal / target impact"

    x = np.arange(len(plot_df))

    plt.figure(figsize=(12, 5))

    plt.plot(
        x,
        plot_df["alpha_plot"],
        marker="o",
        linewidth=2,
        label="Alpha signal",
    )

    plt.plot(
        x,
        plot_df["target_plot"],
        marker="o",
        linewidth=2,
        linestyle="--",
        label="Target impact",
    )

    plt.axhline(0, color="black", linewidth=1)

    plt.xticks(
        x,
        plot_df["time"],
        rotation=45,
    )

    plt.title(
        f"{stock} {date} - alpha and target impact\n"
        f"{strategy_name}, {bucket_size * 10 // 60}-minute buckets"
    )

    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.show()

    return plot_df

def plot_trades_and_position(
    trades_df,
    stock,
    date,
    strategy_name,
    bucket_size=60,
    scaling_df=None,
):
    """
    Plot 10-minute bucketed trades as bars and cumulative position as a line.

    trades_df:
        index = (stock, date)
        columns = intraday time bins
        values = q_t trades

    bucket_size=60 means 10-minute buckets if dt=10 seconds.
    """

    trades = trades_df.loc[(stock, date)].astype(float)

    temp = pd.DataFrame({
        "time": trades.index,
        "trade": trades.values,
    })

    temp["position"] = temp["trade"].cumsum()
    temp["bucket"] = np.arange(len(temp)) // bucket_size

    plot_df = (
        temp
        .groupby("bucket")
        .agg(
            bucket_trade=("trade", "sum"),
            position=("position", "last"),
            time=("time", "first"),
        )
        .reset_index(drop=True)
    )

    ylabel_trade = "Bucket trade quantity"
    ylabel_position = "Cumulative position"

    if scaling_df is not None:
        ADV = float(scaling_df.loc[stock, "ADV"])
        plot_df["bucket_trade"] = plot_df["bucket_trade"] / ADV
        plot_df["position"] = plot_df["position"] / ADV
        ylabel_trade = "Bucket trade / ADV"
        ylabel_position = "Position / ADV"

    x = np.arange(len(plot_df))

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.bar(
        x,
        plot_df["bucket_trade"],
        alpha=0.7,
        label="10-min net trade",
    )

    ax1.axhline(0, color="black", linewidth=1)
    ax1.set_ylabel(ylabel_trade)
    ax1.set_xlabel("Time")

    ax2 = ax1.twinx()

    ax2.plot(
        x,
        plot_df["position"],
        marker="o",
        linewidth=2,
        label="Position",
    )

    ax2.set_ylabel(ylabel_position)

    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_df["time"], rotation=45)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=False,
        loc="best",
    )

    plt.title(
        f"{stock} {date} - trades and position\n"
        f"{strategy_name}, {bucket_size * 10 // 60}-minute buckets"
    )

    plt.tight_layout()
    plt.show()

    return plot_df

def plot_cumulative_pnl_and_impact_cost(
    path_df,
    stock,
    date,
    strategy_name,
    trades_series=None,
    bucket_size=60,
):
    """
    Plot cumulative P&L and cumulative impact cost for one stock-day.

    bucket_size=60 means 10-minute buckets if dt=10 seconds.
    """

    df = path_df.copy()

    if "portfolio_value" not in df.columns:
        raise ValueError("path_df must contain 'portfolio_value'.")

    # Get trades
    if trades_series is not None:
        trades = trades_series.astype(float).values
    elif "trade" in df.columns:
        trades = df["trade"].astype(float).values
    elif "my_trade" in df.columns:
        trades = df["my_trade"].astype(float).values
    else:
        raise ValueError("Pass trades_series, or path_df must contain 'trade' or 'my_trade'.")

    # Per-bin impact cost approximation
    df["impact_cost_step"] = trades * df["impact_in_price"].astype(float)
    df["cum_impact_cost"] = df["impact_cost_step"].cumsum()

    temp = pd.DataFrame({
        "time": df.index,
        "portfolio_value": df["portfolio_value"].values,
        "cum_impact_cost": df["cum_impact_cost"].values,
    })

    temp["bucket"] = np.arange(len(temp)) // bucket_size

    plot_df = (
        temp
        .groupby("bucket")
        .agg(
            portfolio_value=("portfolio_value", "last"),
            cum_impact_cost=("cum_impact_cost", "last"),
            time=("time", "first"),
        )
        .reset_index(drop=True)
    )

    x = np.arange(len(plot_df))

    plt.figure(figsize=(12, 5))

    plt.plot(
        x,
        plot_df["portfolio_value"],
        marker="o",
        linewidth=2,
        label="Cumulative P&L / portfolio value",
    )

    plt.plot(
        x,
        -plot_df["cum_impact_cost"],
        marker="o",
        linewidth=2,
        linestyle="--",
        label="Negative cumulative impact cost",
    )

    plt.axhline(0, color="black", linewidth=1)

    plt.xticks(x, plot_df["time"], rotation=45)

    plt.title(
        f"{stock} {date} - cumulative P&L and impact cost\n"
        f"{strategy_name}, {bucket_size * 10 // 60}-minute buckets"
    )

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.show()

    return plot_df



def run_backtest_from_trade_df(
    test_px_df,
    strategy_trades_df,
    test_traded_volume_df,
    scaling_df,
    fit_df,
    model_type,
    dt_seconds=10
):
    """
    Run a complete backtest of a trading strategy under an impact model.
    
    Loops through all stock-days, adjusting prices for public market impact,
    then simulating the strategy's trades under the fitted impact model.
    Collects performance metrics for each stock-day.
    
    Args:
        test_px_df: DataFrame with test period prices indexed by (stock, date)
        strategy_trades_df: DataFrame with strategy trade volumes to backtest
        test_traded_volume_df: DataFrame with market-wide traded volumes
        scaling_df: DataFrame with ADV and sigma for each stock
        fit_df: DataFrame with fitted model parameters (lambda_hat, half_life_seconds) by stock
        model_type: Impact model type to use ('ow', 'afs', 'reduced_form')
        dt_seconds: Time step in seconds (default 10)
    
    Returns:
        DataFrame with one row per stock-day containing:
        - stock, date: Identifier columns
        - daily_pnl, impact_cost: Performance metrics
        - max_abs_impact: Maximum realized price impact
        - final_position, total_abs_traded, net_traded: Trade execution details
        - avg_price, traded_notional: Trade statistics
        - lambda_hat, ADV, sigma, half_life_seconds: Model parameters
        - model_type: Identifier for this backtest run
    """
    summaries = []

    # Simulate each stock-day independently
    for stock, date in test_px_df.index:
        # Extract data for this stock-day
        prices = test_px_df.loc[(stock, date)]
        my_trades = strategy_trades_df.loc[(stock, date)]
        public_trades = test_traded_volume_df.loc[(stock, date)]

        # Get fitted parameters for this stock
        lambda_hat = float(fit_df.loc[stock, "lambda_hat"])
        half_life_seconds = float(fit_df.loc[stock, "half_life_seconds"])

        # Get market-specific parameters
        ADV = float(scaling_df.loc[stock, "ADV"])
        sigma = float(scaling_df.loc[stock, "sigma"])

        # Step 1: Adjust prices for public market impact
        # This isolates the trading environment for our strategy
        prices = make_impact_adjusted_prices(
            prices=prices,
            public_trades=public_trades,
            lambda_hat=lambda_hat,
            ADV=ADV,
            sigma=sigma,
            half_life_seconds=half_life_seconds,
            model_type=model_type,
            dt_seconds=dt_seconds
        )

        # Step 2: Simulate our strategy trading on adjusted prices
        path_df, summary = simulate_one_stock_day(
            prices=prices,
            my_trades=my_trades,
            lambda_hat=lambda_hat,
            ADV=ADV,
            sigma=sigma,
            half_life_seconds=half_life_seconds,
            model_type=model_type,
            dt_seconds=dt_seconds
        )

        # Add identifiers and parameters to summary
        summary["stock"] = stock
        summary["date"] = date
        summary["model_type"] = model_type
        summary["lambda_hat"] = lambda_hat
        summary["ADV"] = ADV
        summary["sigma"] = sigma
        summary["half_life_seconds"] = half_life_seconds

        summaries.append(summary)

    # Aggregate all stock-day results into a single dataframe
    return pd.DataFrame(summaries)

def add_bps_metrics(backtest_df):
    """
    Add P&L and impact cost in basis points of traded notional.
    
    Converts absolute P&L and impact costs to basis points (1 bps = 0.01%)
    of the total traded notional value for easier comparison across trades.
    
    Args:
        backtest_df: DataFrame with backtest results including impact_cost and daily_pnl
    
    Returns:
        DataFrame with added columns: impact_cost_bps and pnl_bps
    """

    backtest_df = backtest_df.copy()

    # Convert impact cost to basis points
    backtest_df["impact_cost_bps"] = (
        1e4 * backtest_df["impact_cost"] / backtest_df["traded_notional"]
    )

    # Convert P&L to basis points
    backtest_df["pnl_bps"] = (
        1e4 * backtest_df["daily_pnl"] / backtest_df["traded_notional"]
    )

    return backtest_df


def plot_one_stock_day_backtest_path(path_df, stock, date, model_type):
    """
    Plot mid price, execution price, and own price impact for one stock-day.
    
    Creates two visualizations:
    1. Mid price vs execution price over the trading day
    2. Price impact over time in cents
    
    Args:
        path_df: DataFrame with trading path including prices and impact
        stock: Stock ticker
        date: Trading date
        model_type: Type of impact model used in backtest
    """

    x = np.arange(len(path_df))
    tick_positions = np.linspace(0, len(path_df) - 1, 8, dtype=int)
    tick_labels = path_df.index[tick_positions]

    # Price plot: Compare mid price vs execution price
    plt.figure(figsize=(12, 4))
    plt.plot(x, path_df["mid_price"].values, label="Mid price")
    plt.plot(x, path_df["execution_price"].values, label="Execution price")
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.legend()
    plt.title(f"{stock} {date} - {model_type} one-stock-day backtest")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()

    # Impact plot in cents
    impact_in_cents = 100 * path_df["impact_in_price"]

    plt.figure(figsize=(12, 4))
    plt.plot(x, impact_in_cents.values)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.title(f"{stock} {date} - {model_type} price impact")
    plt.xlabel("Time")
    plt.ylabel("Price impact, cents")
    plt.tight_layout()
    plt.show()
def plot_backtest_summary(backtest_summary_df):
    """
    Plot summary diagnostics for one backtest result dataframe.
    
    Creates three distribution plots:
    1. Daily P&L distribution in basis points
    2. Impact cost distribution in basis points
    3. Maximum absolute price impact distribution in cents
    
    Args:
        backtest_summary_df: DataFrame with backtest summary statistics across all stock-days
    """

    model_type = backtest_summary_df["model_type"].iloc[0]

    # Daily P&L in bps
    plt.figure(figsize=(8, 4))
    plt.hist(backtest_summary_df["pnl_bps"], bins=30)
    plt.title(f"{model_type} - Distribution of daily P&L")
    plt.xlabel("Daily P&L / traded notional, bps")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Impact cost in bps
    plt.figure(figsize=(8, 4))
    plt.hist(backtest_summary_df["impact_cost_bps"], bins=30)
    plt.title(f"{model_type} - Distribution of impact costs")
    plt.xlabel("Impact cost / traded notional, bps")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Maximum price impact in cents
    plt.figure(figsize=(8, 4))
    plt.hist(100 * backtest_summary_df["max_abs_impact"], bins=30)
    plt.title(f"{model_type} - Distribution of maximum absolute impact")
    plt.xlabel("Maximum absolute impact, cents")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
def plot_model_comparison(backtest_all_models_df):
    """
    Compare backtest diagnostics across impact models.
    
    Creates side-by-side boxplots comparing:
    1. Impact cost distribution by model
    2. P&L distribution by model
    
    Useful for evaluating which impact model provides better trading outcomes.
    
    Args:
        backtest_all_models_df: DataFrame with backtest results for multiple model types
    """

    models = backtest_all_models_df["model_type"].unique()

    # Impact cost by model
    data = [
        backtest_all_models_df.loc[
            backtest_all_models_df["model_type"] == model,
            "impact_cost_bps"
        ]
        for model in models
    ]

    plt.figure(figsize=(8, 4))
    plt.boxplot(data, tick_labels=models)
    plt.title("Impact cost by model")
    plt.ylabel("Impact cost / traded notional, bps")
    plt.tight_layout()
    plt.show()

    # P&L by model
    data = [
        backtest_all_models_df.loc[
            backtest_all_models_df["model_type"] == model,
            "pnl_bps"
        ]
        for model in models
    ]

    plt.figure(figsize=(8, 4))
    plt.boxplot(data, tick_labels=models)
    plt.title("Daily P&L by model")
    plt.ylabel("Daily P&L / traded notional, bps")
    plt.tight_layout()
    plt.show()

    # Maximum impact by model
    data = [
        100 * backtest_all_models_df.loc[
            backtest_all_models_df["model_type"] == model,
            "max_abs_impact"
        ]
        for model in models
    ]

    plt.figure(figsize=(8, 4))
    plt.boxplot(data, tick_labels=models)
    plt.title("Maximum absolute price impact by model")
    plt.ylabel("Maximum absolute impact, cents")
    plt.tight_layout()
    plt.show()
def run_one_stock_day_diagnostic(
    stock,
    date,
    model_type,
    test_px_df,
    strategy_trades_df,
    test_traded_volume_df,
    scaling_df,
    fit_dfs,
    dt_seconds=10,
    plot=True
):
    """
    Run and optionally plot one stock-day backtest diagnostic.

    This uses the same logic as run_backtest_from_trade_df:
    first remove public market impact, then simulate the strategy trades.
    """

    fit_df = fit_dfs[model_type]

    prices = test_px_df.loc[(stock, date)]
    my_trades = strategy_trades_df.loc[(stock, date)]
    public_trades = test_traded_volume_df.loc[(stock, date)]

    lambda_hat = float(fit_df.loc[stock, "lambda_hat"])
    half_life_seconds = float(fit_df.loc[stock, "half_life_seconds"])

    ADV = float(scaling_df.loc[stock, "ADV"])
    sigma = float(scaling_df.loc[stock, "sigma"])

    prices = make_impact_adjusted_prices(
        prices=prices,
        public_trades=public_trades,
        lambda_hat=lambda_hat,
        ADV=ADV,
        sigma=sigma,
        half_life_seconds=half_life_seconds,
        model_type=model_type,
        dt_seconds=dt_seconds
    )

    path_df, summary = simulate_one_stock_day(
        prices=prices,
        my_trades=my_trades,
        lambda_hat=lambda_hat,
        ADV=ADV,
        sigma=sigma,
        half_life_seconds=half_life_seconds,
        model_type=model_type,
        dt_seconds=dt_seconds
    )

    summary["stock"] = stock
    summary["date"] = date
    summary["model_type"] = model_type
    summary["lambda_hat"] = lambda_hat
    summary["ADV"] = ADV
    summary["sigma"] = sigma
    summary["half_life_seconds"] = half_life_seconds

    if plot:
        plot_one_stock_day_backtest_path(
            path_df=path_df,
            stock=stock,
            date=date,
            model_type=model_type
        )

    return path_df, summary


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











