
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
from src.backtest_engine import make_impact_adjusted_prices
from src.optimal_trading_strategies import ow_target_impact_from_alpha

########## FIRST CASE : OVERNIGHT PnL ADD ON ONLY (NO CARRYING POSITION)##################



def add_overnight_pnl_addon(
    backtest_df,
    test_px_df,
    pnl_col="daily_pnl",
    position_col="final_position",
):
    """
    Add overnight PnL from the previous day's final position.

    This is not a full carry backtest.
    It keeps the intraday strategy daily-reset, but adds:

        overnight_pnl_{i,d}
        = Q_{i,d-1,close} * (P_{i,d,open} - P_{i,d-1,close})

    to day d's PnL.
    """

    result = backtest_df.copy()

    # Make sure dates are comparable and sortable.
    result["date"] = pd.to_datetime(result["date"])

    # Extract open and close prices from test_px_df.
    px = test_px_df.copy()
    px = px.astype(float)

    first_col = px.columns[0]
    last_col = px.columns[-1]

    price_info = pd.DataFrame({
        "open_price": px[first_col],
        "close_price": px[last_col],
    }).reset_index()

    price_info["date"] = pd.to_datetime(price_info["date"])

    # Merge today's open and close prices into the backtest result.
    result = result.merge(
        price_info,
        on=["stock", "date"],
        how="left",
    )

    # Sort by strategy, stock, date.
    group_cols = ["stock"]

    if "strategy" in result.columns:
        group_cols = ["strategy", "stock"]

    result = result.sort_values(group_cols + ["date"]).copy()

    # Previous day's final position and close price.
    result["previous_final_position"] = (
        result.groupby(group_cols)[position_col].shift(1)
    )

    result["previous_close_price"] = (
        result.groupby(group_cols)["close_price"].shift(1)
    )

    # Overnight price move from yesterday close to today open.
    result["overnight_price_move"] = (
        result["open_price"] - result["previous_close_price"]
    )

    result["overnight_return"] = (
        result["overnight_price_move"] / result["previous_close_price"]
    )

    # Overnight PnL from carrying yesterday's final position.
    result["overnight_pnl"] = (
        result["previous_final_position"] * result["overnight_price_move"]
    )

    # First day has no previous close position.
    result["overnight_pnl"] = result["overnight_pnl"].fillna(0.0)
    result["overnight_return"] = result["overnight_return"].fillna(0.0)
    result["previous_final_position"] = result["previous_final_position"].fillna(0.0)

    # Add overnight PnL to the existing daily-reset intraday PnL.
    result["intraday_pnl"] = result[pnl_col]
    result["total_pnl_with_overnight_addon"] = (
        result["intraday_pnl"] + result["overnight_pnl"]
    )

    return result


def add_overnight_addon_bps_metrics(backtest_df):
    """
    Add bps metrics for the overnight-adjusted PnL.
    """

    result = backtest_df.copy()

    result["intraday_pnl_bps"] = (
        1e4 * result["intraday_pnl"] / result["traded_notional"]
    )

    result["overnight_pnl_bps"] = (
        1e4 * result["overnight_pnl"] / result["traded_notional"]
    )

    result["total_pnl_with_overnight_addon_bps"] = (
        1e4
        * result["total_pnl_with_overnight_addon"]
        / result["traded_notional"]
    )

    return result




####################### SIMPLE ADD ON PLOTTING FUNCTIONS #################################


def plot_final_position_by_date_for_stock(
    backtest_df,
    stock,
    strategy_name="OW intraday alpha"
):
    """
    Plot final end-of-day position over time for one stock and one strategy.

    The position is scaled by ADV.
    """

    df = backtest_df[
        (backtest_df["stock"] == stock)
        & (backtest_df["strategy"] == strategy_name)
    ].copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df["final_position_over_ADV"] = (
        df["final_position"] / df["ADV"]
    )

    plt.figure(figsize=(11, 4))
    plt.plot(
        df["date"],
        100 * df["final_position_over_ADV"],
        marker="o"
    )
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"{stock} - final close position over time")
    plt.xlabel("Date")
    plt.ylabel("Final position / ADV, %")
    plt.tight_layout()
    plt.show()


def add_position_exposure_metrics(backtest_df):
    """
    Add inventory exposure metrics scaled by ADV.

    final_position_over_ADV:
        End-of-day inventory created by the current day's intraday strategy.

    previous_final_position_over_ADV:
        Inventory actually used to compute today's overnight PnL.
    """

    df = backtest_df.copy()

    df["final_position_over_ADV"] = df["final_position"] / df["ADV"]

    if "previous_final_position" in df.columns:
        df["previous_final_position_over_ADV"] = (
            df["previous_final_position"] / df["ADV"]
        )

    return df


def plot_ow_intraday_only_vs_with_addon_mean_pnl(
    backtest_df,
    strategy_name="OW intraday alpha",
):
    """
    Main comparison for this notebook.

    Compare:
        1. Intraday-only PnL
        2. Intraday PnL + overnight add-on

    Both are for the same OW intraday-alpha strategy.
    """

    df = backtest_df[
        backtest_df["strategy"] == strategy_name
    ].copy()

    means = pd.Series({
        "Intraday only": df["intraday_pnl_bps"].mean(),
        "Intraday + overnight add-on": df["total_pnl_with_overnight_addon_bps"].mean(),
    })

    plt.figure(figsize=(7, 4))
    bars = plt.bar(means.index, means.values)

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"{strategy_name}: intraday-only vs with overnight add-on")
    plt.ylabel("Mean PnL / traded notional, bps")

    for bar, value in zip(bars, means.values):
        va = "bottom" if value >= 0 else "top"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va=va,
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()

    return means


def plot_ow_mean_intraday_vs_addon_only_pnl(
    backtest_df,
    strategy_name="OW intraday alpha",
):
    """
    Secondary comparison.

    Compare:
        1. Mean intraday-only PnL
        2. Mean overnight add-on PnL alone

    This shows whether the add-on is economically meaningful.
    """

    df = backtest_df[
        backtest_df["strategy"] == strategy_name
    ].copy()

    means = pd.Series({
        "Intraday-only PnL": df["intraday_pnl_bps"].mean(),
        "Overnight add-on PnL": df["overnight_pnl_bps"].mean(),
    })

    plt.figure(figsize=(7, 4))
    bars = plt.bar(means.index, means.values)

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"{strategy_name}: mean intraday PnL vs add-on PnL")
    plt.ylabel("Mean PnL / traded notional, bps")

    for bar, value in zip(bars, means.values):
        va = "bottom" if value >= 0 else "top"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.6f}",
            ha="center",
            va=va,
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()

    return means


def plot_ow_cumulative_intraday_only_vs_with_addon(
    backtest_df,
    strategy_name="OW intraday alpha",
):
    """
    Plot cumulative daily PnL through time for the OW strategy only.

    Compare:
        1. Cumulative intraday-only PnL
        2. Cumulative intraday + overnight add-on PnL
    """

    df = backtest_df[
        backtest_df["strategy"] == strategy_name
    ].copy()

    df["date"] = pd.to_datetime(df["date"])

    daily = (
        df
        .groupby("date")[[
            "intraday_pnl",
            "overnight_pnl",
            "total_pnl_with_overnight_addon",
            "traded_notional",
        ]]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    daily["intraday_pnl_bps_daily"] = (
        1e4 * daily["intraday_pnl"] / daily["traded_notional"]
    )

    daily["total_pnl_with_overnight_addon_bps_daily"] = (
        1e4
        * daily["total_pnl_with_overnight_addon"]
        / daily["traded_notional"]
    )

    plt.figure(figsize=(11, 4))

    plt.plot(
        daily["date"],
        daily["intraday_pnl_bps_daily"].cumsum(),
        marker="o",
        label="Intraday only",
    )

    plt.plot(
        daily["date"],
        daily["total_pnl_with_overnight_addon_bps_daily"].cumsum(),
        marker="o",
        label="Intraday + overnight add-on",
    )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"{strategy_name}: cumulative PnL")
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL, bps")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return daily


def plot_mean_intraday_position_profile(
    trades_df,
    scaling_df,
    title="OW intraday alpha: mean intraday position profile",
):
    """
    Plot the mean position held in each intraday bin.

    The position is computed from the strategy trades:

        Q_j = cumulative sum of q_j

    The plot shows:
        1. signed mean position / ADV
        2. mean absolute position / ADV

    The signed mean can cancel across stock-days.
    The absolute mean shows the typical inventory size.
    """

    trades = trades_df.astype(float)
    position = trades.cumsum(axis=1)

    stocks = position.index.get_level_values("stock")
    adv = pd.Series(
        stocks.map(scaling_df["ADV"]),
        index=position.index,
    ).astype(float)

    position_over_adv = position.divide(adv, axis=0)

    mean_signed = position_over_adv.mean(axis=0)
    mean_abs = position_over_adv.abs().mean(axis=0)

    x = np.arange(len(position_over_adv.columns))
    tick_positions = np.linspace(0, len(x) - 1, 8, dtype=int)
    tick_labels = position_over_adv.columns[tick_positions]

    plt.figure(figsize=(11, 4))

    plt.plot(
        x,
        100 * mean_signed.values,
        label="Mean signed position",
    )

    plt.plot(
        x,
        100 * mean_abs.values,
        label="Mean absolute position",
    )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Position / ADV, %")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    profile_df = pd.DataFrame({
        "time": position_over_adv.columns,
        "mean_signed_position_over_ADV": mean_signed.values,
        "mean_abs_position_over_ADV": mean_abs.values,
    })

    return profile_df


def plot_final_position_through_time(
    backtest_df,
    strategy_name="OW intraday alpha",
):
    """
    Plot the mean absolute final close position through time.

    This explains why the overnight add-on is small or large.
    If final position is close to zero, overnight PnL must also be small.
    """

    df = backtest_df[
        backtest_df["strategy"] == strategy_name
    ].copy()

    df["date"] = pd.to_datetime(df["date"])
    df["abs_final_position_over_ADV"] = (
        df["final_position"].abs() / df["ADV"]
    )

    daily = (
        df
        .groupby("date")["abs_final_position_over_ADV"]
        .mean()
        .reset_index()
        .sort_values("date")
    )

    plt.figure(figsize=(11, 4))

    plt.plot(
        daily["date"],
        100 * daily["abs_final_position_over_ADV"],
        marker="o",
    )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"{strategy_name}: mean absolute final position through time")
    plt.xlabel("Date")
    plt.ylabel("Mean |final position| / ADV, %")
    plt.tight_layout()
    plt.show()

    return daily


def run_ow_overnight_addon_plots(
    overnight_addon_results_df,
    ow_alpha_trades_df,
    scaling_df,
    strategy_name="OW intraday alpha",
):
    """
    Run the useful plots for the first overnight add-on experiment.

    This excludes TWAP and strategy-comparison plots.
    """

    mean_comparison = plot_ow_intraday_only_vs_with_addon_mean_pnl(
        overnight_addon_results_df,
        strategy_name=strategy_name,
    )

    component_comparison = plot_ow_mean_intraday_vs_addon_only_pnl(
        overnight_addon_results_df,
        strategy_name=strategy_name,
    )

    cumulative_daily = plot_ow_cumulative_intraday_only_vs_with_addon(
        overnight_addon_results_df,
        strategy_name=strategy_name,
    )

    position_profile = plot_mean_intraday_position_profile(
        trades_df=ow_alpha_trades_df,
        scaling_df=scaling_df,
        title=f"{strategy_name}: mean intraday position profile",
    )

    final_position_daily = plot_final_position_through_time(
        overnight_addon_results_df,
        strategy_name=strategy_name,
    )

    return {
        "mean_comparison": mean_comparison,
        "component_comparison": component_comparison,
        "cumulative_daily": cumulative_daily,
        "position_profile": position_profile,
        "final_position_daily": final_position_daily,
    }




############################## ROLLING / CARRY BACKTEST FUNCTIONS ##############################

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

def make_combined_alpha_df(
    intraday_alpha_df,
    overnight_alpha_df,
    intraday_weight=1.0,
    overnight_weight=1.0,
):
    """
    Build the combined alpha panel:

        alpha_combined = intraday_weight * alpha_intraday
                         + overnight_weight * alpha_overnight

    Both inputs should be stock-date x intraday-time panels.
    """

    intraday_alpha = intraday_alpha_df.astype(float)

    overnight_alpha = (
        overnight_alpha_df
        .reindex(index=intraday_alpha.index, columns=intraday_alpha.columns)
        .fillna(0.0)
        .astype(float)
    )

    combined_alpha_df = (
        intraday_weight * intraday_alpha
        + overnight_weight * overnight_alpha
    )

    return combined_alpha_df

def recover_ow_trades_from_target_impact_with_initial_state(
    I_star,
    lambda_hat,
    ADV,
    sigma,
    half_life_seconds,
    initial_impact_state=0.0,
    dt_seconds=10,
    eps=1e-12,
):
    """
    Recover OW trades from a target impact path when the day may start
    with nonzero carried impact.

    OW equation:

        I_j = decay * I_{j-1} + lambda_hat * sigma * q_j / ADV

    Therefore:

        q_j = (I*_j - decay * previous_I) / (lambda_hat * sigma / ADV)

    For the first bin, previous_I is the carried initial impact.
    For later bins, previous_I is the previous target impact.
    """

    I_star = I_star.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    beta = np.log(2) / float(half_life_seconds)
    decay = np.exp(-beta * dt_seconds)

    lambda_eff = float(lambda_hat) * float(sigma) / float(ADV)

    if abs(lambda_eff) < eps:
        return pd.Series(0.0, index=I_star.index)

    trades = []
    previous_I = float(initial_impact_state)

    for target_I in I_star.values:
        q = (target_I - decay * previous_I) / lambda_eff
        trades.append(q)

        # If no constraints are applied, the intended next impact is I*_j.
        previous_I = target_I

    return pd.Series(trades, index=I_star.index)

def compute_ow_realized_impact_path(
    trades,
    lambda_hat,
    ADV,
    sigma,
    half_life_seconds,
    initial_impact_state=0.0,
    dt_seconds=10,
):
    """
    Compute the realized OW impact path from executed trades.

    This is the actual impact generated by the trades:

        I_j = decay * I_{j-1} + lambda_hat * sigma * q_j / ADV

    Unlike the target impact, this is based on the trades that are
    actually executed after any normalization.
    """

    trades = trades.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    beta = np.log(2) / float(half_life_seconds)
    decay = np.exp(-beta * dt_seconds)

    lambda_eff = float(lambda_hat) * float(sigma) / float(ADV)

    impact_values = []
    impact_state = float(initial_impact_state)

    for q in trades.values:
        impact_state = decay * impact_state + lambda_eff * q
        impact_values.append(impact_state)

    return pd.Series(impact_values, index=trades.index)

def simulate_one_stock_rolling_ow(
    stock,
    alpha_df,
    test_px_df,
    test_traded_volume_df,
    scaling_df,
    fit_df,
    dt_seconds=10,
    overnight_seconds=16 * 60 * 60,
    normalize_abs_volume=True,
    target_participation=0.20,
    adjust_public_impact=True,
):
    """
    Simulate one stock through the full test period, carrying:

        position
        cash
        own impact state

    across days.

    This is the rolling/carry version of the OW backtest.
    There is no position cap in this first version.
    """

    lambda_hat = float(fit_df.loc[stock, "lambda_hat"])
    half_life_seconds = float(fit_df.loc[stock, "half_life_seconds"])

    ADV = float(scaling_df.loc[stock, "ADV"])
    sigma = float(scaling_df.loc[stock, "sigma"])

    beta = np.log(2) / half_life_seconds
    overnight_decay = np.exp(-beta * overnight_seconds)

    # Dates available for this stock.
    stock_dates = list(test_px_df.loc[stock].index)
    stock_dates = sorted(stock_dates, key=lambda x: pd.to_datetime(x))

    # Carried state variables.
    position = 0.0
    cash = 0.0
    impact_state = 0.0
    previous_close_price = None

    path_rows = []
    summary_rows = []

    for date in stock_dates:

        prices_raw = test_px_df.loc[(stock, date)].astype(float)
        public_trades = test_traded_volume_df.loc[(stock, date)].astype(float)

        # Decay own impact from previous close to today's open.
        # First day starts with zero impact anyway.
        if previous_close_price is not None:
            impact_state = overnight_decay * impact_state

        start_position = position
        start_cash = cash
        start_impact_state = impact_state

        # Adjust public market impact day by day, as in the original backtest.
        if adjust_public_impact:
            prices = make_impact_adjusted_prices(
                prices=prices_raw,
                public_trades=public_trades,
                lambda_hat=lambda_hat,
                ADV=ADV,
                sigma=sigma,
                half_life_seconds=half_life_seconds,
                model_type="ow",
                dt_seconds=dt_seconds,
            )
        else:
            prices = prices_raw.copy()

        open_price = float(prices.iloc[0])
        close_price = float(prices.iloc[-1])

        # Overnight PnL comes from carrying yesterday's final position.
        if previous_close_price is None:
            overnight_pnl = 0.0
        else:
            overnight_pnl = start_position * (open_price - previous_close_price)

        open_portfolio_value = start_cash + start_position * open_price

        # Get alpha for this stock-day.
        alpha = (
            alpha_df
            .loc[(stock, date)]
            .reindex(prices.index)
            .fillna(0.0)
            .astype(float)
        )

        # Alpha -> target impact.
        I_star, alpha_prime = ow_target_impact_from_alpha(
            alpha=alpha,
            beta=beta,
            dt_seconds=dt_seconds,
        )

        # Target impact -> desired trades, accounting for carried impact.
        raw_trades = recover_ow_trades_from_target_impact_with_initial_state(
            I_star=I_star,
            lambda_hat=lambda_hat,
            ADV=ADV,
            sigma=sigma,
            half_life_seconds=half_life_seconds,
            initial_impact_state=start_impact_state,
            dt_seconds=dt_seconds,
        )

        raw_abs_volume = raw_trades.abs().sum()
        target_abs_volume = target_participation * ADV

        scale_factor = 1.0
        trades = raw_trades.copy()

        if normalize_abs_volume and raw_abs_volume > 0:
            scale_factor = target_abs_volume / raw_abs_volume
            trades = raw_trades * scale_factor

        actual_abs_volume = trades.abs().sum()

        # Realized impact from the executed trades.
        realized_impact = compute_ow_realized_impact_path(
            trades=trades,
            lambda_hat=lambda_hat,
            ADV=ADV,
            sigma=sigma,
            half_life_seconds=half_life_seconds,
            initial_impact_state=start_impact_state,
            dt_seconds=dt_seconds,
        )

        target_minus_realized = I_star - realized_impact

        impact_in_price = open_price * realized_impact
        execution_prices = prices + impact_in_price

        # Rolling position and cash.
        position_path = start_position + trades.cumsum()
        cash_path = start_cash - (trades * execution_prices).cumsum()

        portfolio_value = cash_path + position_path * prices

        final_position = float(position_path.iloc[-1])
        final_cash = float(cash_path.iloc[-1])
        final_impact_state = float(realized_impact.iloc[-1])
        close_portfolio_value = float(portfolio_value.iloc[-1])

        intraday_pnl = close_portfolio_value - open_portfolio_value
        total_pnl = overnight_pnl + intraday_pnl

        impact_cost = float((trades * impact_in_price).sum())
        avg_price = float(prices.mean())
        total_abs_traded = float(trades.abs().sum())
        net_traded = float(trades.sum())
        traded_notional = total_abs_traded * avg_price

        day_path = pd.DataFrame({
            "stock": stock,
            "date": date,
            "time": prices.index.astype(str),
            "mid_price": prices.values,
            "raw_mid_price": prices_raw.values,
            "alpha": alpha.values,
            "alpha_prime": alpha_prime.values,
            "target_impact": I_star.values,
            "trade": trades.values,
            "realized_impact": realized_impact.values,
            "target_minus_realized": target_minus_realized.values,
            "impact_in_price": impact_in_price.values,
            "execution_price": execution_prices.values,
            "position": position_path.values,
            "cash": cash_path.values,
            "portfolio_value": portfolio_value.values,
        })

        path_rows.append(day_path)

        summary_rows.append({
            "stock": stock,
            "date": date,
            "overnight_pnl": overnight_pnl,
            "intraday_pnl": intraday_pnl,
            "total_pnl": total_pnl,
            "impact_cost": impact_cost,
            "start_position": start_position,
            "final_position": final_position,
            "max_abs_position": float(position_path.abs().max()),
            "start_cash": start_cash,
            "final_cash": final_cash,
            "start_impact_state": start_impact_state,
            "final_impact_state": final_impact_state,
            "max_abs_impact_state": float(realized_impact.abs().max()),
            "max_abs_impact_price": float(impact_in_price.abs().max()),
            "open_price": open_price,
            "close_price": close_price,
            "previous_close_price": previous_close_price,
            "raw_abs_volume": raw_abs_volume,
            "target_abs_volume": target_abs_volume,
            "actual_abs_volume": actual_abs_volume,
            "scale_factor": scale_factor,
            "total_abs_traded": total_abs_traded,
            "net_traded": net_traded,
            "avg_price": avg_price,
            "traded_notional": traded_notional,
            "ADV": ADV,
            "sigma": sigma,
            "lambda_hat": lambda_hat,
            "half_life_seconds": half_life_seconds,
            "overnight_seconds": overnight_seconds,
            "target_minus_realized_mean_abs": float(target_minus_realized.abs().mean()),
            "target_minus_realized_max_abs": float(target_minus_realized.abs().max()),
        })

        # Carry state to next day.
        position = final_position
        cash = final_cash
        impact_state = final_impact_state
        previous_close_price = close_price

    path_df = pd.concat(path_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    return path_df, summary_df

def run_rolling_ow_carry_backtest(
    alpha_df,
    test_px_df,
    test_traded_volume_df,
    scaling_df,
    fit_df,
    strategy_name,
    dt_seconds=10,
    overnight_seconds=16 * 60 * 60,
    normalize_abs_volume=True,
    target_participation=0.20,
    adjust_public_impact=True,
):
    """
    Run the rolling OW carry backtest for all stocks.

    This replaces the daily-reset backtest for strategies that carry
    inventory across days.
    """

    path_results = []
    summary_results = []

    stocks = test_px_df.index.get_level_values("stock").unique()

    for stock in stocks:

        if stock not in fit_df.index:
            continue

        if stock not in scaling_df.index:
            continue

        if stock not in alpha_df.index.get_level_values("stock"):
            continue

        path_df, summary_df = simulate_one_stock_rolling_ow(
            stock=stock,
            alpha_df=alpha_df,
            test_px_df=test_px_df,
            test_traded_volume_df=test_traded_volume_df,
            scaling_df=scaling_df,
            fit_df=fit_df,
            dt_seconds=dt_seconds,
            overnight_seconds=overnight_seconds,
            normalize_abs_volume=normalize_abs_volume,
            target_participation=target_participation,
            adjust_public_impact=adjust_public_impact,
        )

        path_df["strategy"] = strategy_name
        summary_df["strategy"] = strategy_name

        path_results.append(path_df)
        summary_results.append(summary_df)

    rolling_path_df = pd.concat(path_results, ignore_index=True)
    rolling_summary_df = pd.concat(summary_results, ignore_index=True)

    return rolling_path_df, rolling_summary_df

def add_rolling_bps_metrics(summary_df):
    """
    Add PnL and cost metrics in bps of traded notional.
    """

    df = summary_df.copy()

    denom = df["traded_notional"].replace(0.0, np.nan)

    df["intraday_pnl_bps"] = 1e4 * df["intraday_pnl"] / denom
    df["overnight_pnl_bps"] = 1e4 * df["overnight_pnl"] / denom
    df["total_pnl_bps"] = 1e4 * df["total_pnl"] / denom
    df["impact_cost_bps"] = 1e4 * df["impact_cost"] / denom

    df["final_position_over_ADV"] = df["final_position"] / df["ADV"]
    df["start_position_over_ADV"] = df["start_position"] / df["ADV"]
    df["max_abs_position_over_ADV"] = df["max_abs_position"] / df["ADV"]

    return df


################################### ROLLING PLOT FUNCTIONS##############################################

def plot_rolling_mean_pnl_by_strategy(rolling_summary_df):
    """
    Compare mean intraday PnL, overnight PnL, and total PnL
    between rolling strategies.
    """

    plot_df = (
        rolling_summary_df
        .groupby("strategy")[[
            "intraday_pnl_bps",
            "overnight_pnl_bps",
            "total_pnl_bps"
        ]]
        .mean()
    )

    strategies = plot_df.index
    x = np.arange(len(strategies))
    width = 0.25

    plt.figure(figsize=(10, 4))

    plt.bar(
        x - width,
        plot_df["intraday_pnl_bps"].values,
        width,
        label="Intraday PnL"
    )

    plt.bar(
        x,
        plot_df["overnight_pnl_bps"].values,
        width,
        label="Overnight PnL"
    )

    plt.bar(
        x + width,
        plot_df["total_pnl_bps"].values,
        width,
        label="Total PnL"
    )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, strategies, rotation=25)
    plt.ylabel("Mean PnL / traded notional, bps")
    plt.title("Rolling carry backtest: mean PnL by strategy")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return plot_df

def plot_rolling_total_pnl_comparison(rolling_summary_df):
    """
    Headline comparison of mean total PnL by rolling strategy.
    """

    plot_df = (
        rolling_summary_df
        .groupby("strategy")["total_pnl_bps"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 4))

    bars = plt.bar(plot_df.index, plot_df.values)

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.ylabel("Mean total PnL / traded notional, bps")
    plt.title("Rolling carry backtest: total PnL comparison")
    plt.xticks(rotation=25)

    for bar, value in zip(bars, plot_df.values):
        va = "bottom" if value >= 0 else "top"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va=va,
            fontsize=9
        )

    plt.tight_layout()
    plt.show()

    return plot_df

def plot_rolling_cumulative_total_pnl(rolling_summary_df):
    """
    Plot cumulative total PnL by date for each rolling strategy.
    """

    df = rolling_summary_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    daily = (
        df
        .groupby(["strategy", "date"])[[
            "total_pnl",
            "intraday_pnl",
            "overnight_pnl",
            "traded_notional"
        ]]
        .sum()
        .reset_index()
    )

    daily["total_pnl_bps_daily"] = (
        1e4 * daily["total_pnl"] / daily["traded_notional"]
    )

    plt.figure(figsize=(11, 4))

    for strategy in daily["strategy"].unique():
        strategy_df = daily[daily["strategy"] == strategy].sort_values("date")

        plt.plot(
            strategy_df["date"],
            strategy_df["total_pnl_bps_daily"].cumsum(),
            marker="o",
            label=strategy
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Rolling carry backtest: cumulative total PnL")
    plt.xlabel("Date")
    plt.ylabel("Cumulative total PnL, bps")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return daily

def plot_rolling_position_exposure_by_strategy(rolling_summary_df):
    """
    Plot max absolute position over ADV by strategy.

    This checks whether the no-cap rolling strategy accumulates
    unrealistic inventory.
    """

    strategies = list(rolling_summary_df["strategy"].unique())

    data = [
        rolling_summary_df.loc[
            rolling_summary_df["strategy"] == strategy,
            "max_abs_position_over_ADV"
        ].dropna()
        for strategy in strategies
    ]

    plt.figure(figsize=(8, 4))

    plt.boxplot(data, tick_labels=strategies, showfliers=True)

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.ylabel("Max absolute position / ADV")
    plt.title("Rolling carry backtest: position exposure")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.show()

def plot_rolling_mean_position_profile(
    rolling_path_df,
    scaling_df,
    use_abs=True,
):
    """
    Plot mean intraday position profile for each rolling strategy.

    If use_abs=True:
        plots mean absolute position / ADV.

    If use_abs=False:
        plots mean signed position / ADV.
    """

    df = rolling_path_df.copy()

    stocks = df["stock"]
    adv_map = scaling_df["ADV"]

    df["ADV"] = stocks.map(adv_map).astype(float)
    df["position_over_ADV"] = df["position"] / df["ADV"]

    if use_abs:
        df["position_metric"] = df["position_over_ADV"].abs()
        ylabel = "Mean absolute position / ADV, %"
        title = "Rolling carry backtest: mean absolute intraday position profile"
    else:
        df["position_metric"] = df["position_over_ADV"]
        ylabel = "Mean signed position / ADV, %"
        title = "Rolling carry backtest: mean signed intraday position profile"

    profile = (
        df
        .groupby(["strategy", "time"])["position_metric"]
        .mean()
        .reset_index()
    )

    time_order = sorted(profile["time"].unique())
    x = np.arange(len(time_order))

    tick_positions = np.linspace(0, len(time_order) - 1, 8, dtype=int)
    tick_labels = [time_order[i] for i in tick_positions]

    plt.figure(figsize=(11, 4))

    for strategy in profile["strategy"].unique():
        strategy_df = profile[profile["strategy"] == strategy].copy()
        strategy_df["time"] = pd.Categorical(
            strategy_df["time"],
            categories=time_order,
            ordered=True
        )
        strategy_df = strategy_df.sort_values("time")

        plt.plot(
            x,
            100 * strategy_df["position_metric"].values,
            label=strategy
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.ylabel(ylabel)
    plt.xlabel("Time")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return profile

def plot_target_realized_impact_gap(rolling_summary_df):
    """
    Plot target-minus-realized impact gap by strategy.

    This is important because with carried impact and daily volume
    normalization, realized impact may differ from target impact.
    """

    strategies = list(rolling_summary_df["strategy"].unique())

    data = [
        rolling_summary_df.loc[
            rolling_summary_df["strategy"] == strategy,
            "target_minus_realized_mean_abs"
        ].dropna()
        for strategy in strategies
    ]

    plt.figure(figsize=(8, 4))

    plt.boxplot(data, tick_labels=strategies, showfliers=True)

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.ylabel("Mean absolute target-realized impact gap")
    plt.title("Target impact vs realized impact gap")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.show()

def plot_rolling_final_position_through_time(rolling_summary_df):
    """
    Plot mean absolute final position through time by strategy.
    """

    df = rolling_summary_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    daily = (
        df
        .groupby(["strategy", "date"])["final_position_over_ADV"]
        .apply(lambda x: x.abs().mean())
        .reset_index()
        .rename(columns={"final_position_over_ADV": "mean_abs_final_position_over_ADV"})
    )

    plt.figure(figsize=(11, 4))

    for strategy in daily["strategy"].unique():
        strategy_df = daily[daily["strategy"] == strategy].sort_values("date")

        plt.plot(
            strategy_df["date"],
            100 * strategy_df["mean_abs_final_position_over_ADV"],
            marker="o",
            label=strategy
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Rolling carry backtest: final position through time")
    plt.xlabel("Date")
    plt.ylabel("Mean |final position| / ADV, %")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return daily

def run_rolling_carry_diagnostic_plots(
    rolling_summary_df,
    rolling_path_df,
    scaling_df,
):
    """
    Run the main diagnostic plots for the rolling/carry backtest.
    """

    mean_pnl = plot_rolling_mean_pnl_by_strategy(
        rolling_summary_df
    )

    total_pnl = plot_rolling_total_pnl_comparison(
        rolling_summary_df
    )

    daily_pnl = plot_rolling_cumulative_total_pnl(
        rolling_summary_df
    )

    plot_rolling_position_exposure_by_strategy(
        rolling_summary_df
    )

    position_profile_abs = plot_rolling_mean_position_profile(
        rolling_path_df=rolling_path_df,
        scaling_df=scaling_df,
        use_abs=True,
    )

    position_profile_signed = plot_rolling_mean_position_profile(
        rolling_path_df=rolling_path_df,
        scaling_df=scaling_df,
        use_abs=False,
    )

    plot_target_realized_impact_gap(
        rolling_summary_df
    )

    final_position_daily = plot_rolling_final_position_through_time(
        rolling_summary_df
    )

    return {
        "mean_pnl": mean_pnl,
        "total_pnl": total_pnl,
        "daily_pnl": daily_pnl,
        "position_profile_abs": position_profile_abs,
        "position_profile_signed": position_profile_signed,
        "final_position_daily": final_position_daily,
    }


############################ OVERNIGHT WEIGHT SENSITIVITY ###############################################

def make_weighted_combined_alpha_df(
    intraday_alpha_df,
    overnight_alpha_df,
    overnight_weight=1.0,
):
    """
    Build the weighted combined alpha:

        alpha = alpha_intraday + overnight_weight * alpha_overnight

    Both inputs should be stock-date x intraday-time panels.
    """

    intraday_alpha = intraday_alpha_df.astype(float)

    overnight_alpha = (
        overnight_alpha_df
        .reindex(index=intraday_alpha.index, columns=intraday_alpha.columns)
        .fillna(0.0)
        .astype(float)
    )

    combined_alpha_df = intraday_alpha + overnight_weight * overnight_alpha

    return combined_alpha_df

def run_weighted_overnight_alpha_experiment(
    intraday_alpha_df,
    overnight_alpha_df,
    test_px_df,
    test_traded_volume_df,
    scaling_df,
    fit_df,
    overnight_weights,
    dt_seconds=10,
    overnight_seconds=16 * 60 * 60,
    normalize_abs_volume=True,
    target_participation=0.20,
    adjust_public_impact=True,
):
    """
    Run the rolling OW carry backtest for:

        alpha = alpha_intraday + w * alpha_overnight

    for several values of w.

    This keeps the rolling backtest engine unchanged.
    """

    path_results = []
    summary_results = []

    for w in overnight_weights:

        alpha_df = make_weighted_combined_alpha_df(
            intraday_alpha_df=intraday_alpha_df,
            overnight_alpha_df=overnight_alpha_df,
            overnight_weight=w,
        )

        strategy_name = f"Rolling alpha: intra + {w:g} x ON"

        path_df, summary_df = run_rolling_ow_carry_backtest(
            alpha_df=alpha_df,
            test_px_df=test_px_df,
            test_traded_volume_df=test_traded_volume_df,
            scaling_df=scaling_df,
            fit_df=fit_df,
            strategy_name=strategy_name,
            dt_seconds=dt_seconds,
            overnight_seconds=overnight_seconds,
            normalize_abs_volume=normalize_abs_volume,
            target_participation=target_participation,
            adjust_public_impact=adjust_public_impact,
        )

        path_df["overnight_weight"] = w
        summary_df["overnight_weight"] = w

        path_results.append(path_df)
        summary_results.append(summary_df)

    all_path_df = pd.concat(path_results, ignore_index=True)
    all_summary_df = pd.concat(summary_results, ignore_index=True)

    all_summary_df = add_rolling_bps_metrics(all_summary_df)

    return all_path_df, all_summary_df


#------------------Plot----------------------#

def plot_weighted_overnight_experiment_summary(summary_df):
    """
    Plot how rolling performance changes as the overnight-alpha weight changes.

    Shows:
        total PnL
        intraday PnL
        overnight PnL
        impact cost
        max position
    """

    plot_df = (
        summary_df
        .groupby("overnight_weight")[[
            "intraday_pnl_bps",
            "overnight_pnl_bps",
            "total_pnl_bps",
            "impact_cost_bps",
            "max_abs_position_over_ADV",
        ]]
        .mean()
        .reset_index()
        .sort_values("overnight_weight")
    )

    # PnL decomposition.
    plt.figure(figsize=(10, 4))

    plt.plot(
        plot_df["overnight_weight"],
        plot_df["intraday_pnl_bps"],
        marker="o",
        label="Intraday PnL"
    )

    plt.plot(
        plot_df["overnight_weight"],
        plot_df["overnight_pnl_bps"],
        marker="o",
        label="Overnight PnL"
    )

    plt.plot(
        plot_df["overnight_weight"],
        plot_df["total_pnl_bps"],
        marker="o",
        label="Total PnL"
    )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Rolling carry: PnL vs overnight-alpha weight")
    plt.xlabel("Overnight alpha weight")
    plt.ylabel("Mean PnL / traded notional, bps")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    # Inventory exposure.
    plt.figure(figsize=(10, 4))

    plt.plot(
        plot_df["overnight_weight"],
        100 * plot_df["max_abs_position_over_ADV"],
        marker="o"
    )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Rolling carry: inventory exposure vs overnight-alpha weight")
    plt.xlabel("Overnight alpha weight")
    plt.ylabel("Mean max |position| / ADV, %")
    plt.tight_layout()
    plt.show()

    return plot_df


#--------------------- Shuffling overnight weighted alpha -----------------------#

def make_shuffled_overnight_alpha_df(
    overnight_alpha_df,
    random_seed=1234,
):
    """
    Create a placebo overnight alpha by shuffling stock-days within each stock.

    This preserves:
        - the same stocks
        - the same dates
        - the same distribution of overnight alpha values
        - the same intraday repeated structure

    but destroys the alignment between today's overnight alpha and the
    actual next open return.
    """

    rng = np.random.default_rng(random_seed)

    overnight_alpha_df = overnight_alpha_df.copy().sort_index()

    shuffled_pieces = []

    stocks = overnight_alpha_df.index.get_level_values("stock").unique()

    for stock in stocks:
        stock_df = overnight_alpha_df.loc[
            overnight_alpha_df.index.get_level_values("stock") == stock
        ].copy()

        n_rows = stock_df.shape[0]

        if n_rows <= 1:
            shuffled_pieces.append(stock_df)
            continue

        permutation = rng.permutation(n_rows)

        shuffled_stock_df = pd.DataFrame(
            stock_df.to_numpy()[permutation],
            index=stock_df.index,
            columns=stock_df.columns,
        )

        shuffled_pieces.append(shuffled_stock_df)

    shuffled_overnight_alpha_df = pd.concat(shuffled_pieces).sort_index()

    return shuffled_overnight_alpha_df


#----- Plot-----#

def plot_true_vs_shuffled_overnight_alpha_experiment(
    comparison_summary_df,
):
    """
    Compare true overnight alpha against shuffled overnight alpha
    across overnight-alpha weights.

    Main question:
        Does the true overnight signal improve PnL more than a placebo signal?
    """

    plot_df = (
        comparison_summary_df
        .groupby(["overnight_alpha_type", "overnight_weight"])[[
            "intraday_pnl_bps",
            "overnight_pnl_bps",
            "total_pnl_bps",
            "impact_cost_bps",
            "max_abs_position_over_ADV",
        ]]
        .mean()
        .reset_index()
        .sort_values(["overnight_alpha_type", "overnight_weight"])
    )

    # ------------------------------------------------------------
    # 1. Total PnL comparison
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 4))

    for alpha_type in plot_df["overnight_alpha_type"].unique():
        df = plot_df[plot_df["overnight_alpha_type"] == alpha_type]

        plt.plot(
            df["overnight_weight"],
            df["total_pnl_bps"],
            marker="o",
            label=alpha_type,
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("True vs shuffled overnight alpha: total PnL")
    plt.xlabel("Overnight alpha weight")
    plt.ylabel("Mean total PnL / traded notional, bps")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 2. Overnight PnL comparison
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 4))

    for alpha_type in plot_df["overnight_alpha_type"].unique():
        df = plot_df[plot_df["overnight_alpha_type"] == alpha_type]

        plt.plot(
            df["overnight_weight"],
            df["overnight_pnl_bps"],
            marker="o",
            label=alpha_type,
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("True vs shuffled overnight alpha: overnight PnL")
    plt.xlabel("Overnight alpha weight")
    plt.ylabel("Mean overnight PnL / traded notional, bps")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 3. Inventory exposure comparison
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 4))

    for alpha_type in plot_df["overnight_alpha_type"].unique():
        df = plot_df[plot_df["overnight_alpha_type"] == alpha_type]

        plt.plot(
            df["overnight_weight"],
            100 * df["max_abs_position_over_ADV"],
            marker="o",
            label=alpha_type,
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("True vs shuffled overnight alpha: inventory exposure")
    plt.xlabel("Overnight alpha weight")
    plt.ylabel("Mean max |position| / ADV, %")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return plot_df

def plot_scaled_target_vs_realized_impact_one_day(
    weighted_path_df,
    weighted_summary_df,
    stock,
    date,
    overnight_weight=10.0,
):
    path = weighted_path_df.copy()
    summary = weighted_summary_df.copy()

    path["date"] = pd.to_datetime(path["date"])
    summary["date"] = pd.to_datetime(summary["date"])
    chosen_date = pd.to_datetime(date)

    scale_info = summary[
        (summary["stock"] == stock)
        & (summary["date"] == chosen_date)
        & (summary["overnight_weight"] == overnight_weight)
    ]

    if scale_info.empty:
        raise ValueError("No scale factor found for this stock/date/overnight_weight.")

    scale_factor = float(scale_info["scale_factor"].iloc[0])

    df = path[
        (path["stock"] == stock)
        & (path["date"] == chosen_date)
        & (path["overnight_weight"] == overnight_weight)
    ].copy()

    if df.empty:
        raise ValueError("No path rows found for this stock/date/overnight_weight.")

    df = df.sort_values("time")

    df["scaled_target_impact"] = scale_factor * df["target_impact"]

    x = np.arange(len(df))
    tick_positions = np.linspace(0, len(df) - 1, 8, dtype=int)
    tick_labels = df["time"].iloc[tick_positions]

    plt.figure(figsize=(11, 4))

    plt.plot(x, df["scaled_target_impact"], label="Scaled target impact")
    plt.plot(x, df["realized_impact"], label="Realized impact")

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(tick_positions, tick_labels, rotation=45)

    plt.title(
        f"{stock} {chosen_date.date()} - scaled target vs realized impact "
        f"(ON weight = {overnight_weight:g})"
    )

    plt.xlabel("Time")
    plt.ylabel("Impact state")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return df