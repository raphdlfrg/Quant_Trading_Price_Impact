import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_twap_trade_df(test_px_df, scaling_df, participation_rate=0.20):
    trades_df = pd.DataFrame(
        0.0,
        index=test_px_df.index,
        columns=test_px_df.columns
    )

    for stock, date in test_px_df.index:
        ADV = float(scaling_df.loc[stock, "ADV"])
        total_trade = participation_rate * ADV
        n_steps = test_px_df.shape[1]

        trades_df.loc[(stock, date)] = total_trade / n_steps

    return trades_df

def make_round_trip_twap_trade_df(test_px_df, scaling_df, participation_rate=0.20):
    trades_df = pd.DataFrame(
        0.0,
        index=test_px_df.index,
        columns=test_px_df.columns
    )

    n_steps = test_px_df.shape[1]
    half = n_steps // 2

    for stock, date in test_px_df.index:
        ADV = float(scaling_df.loc[stock, "ADV"])
        total_trade = participation_rate * ADV

        trades = pd.Series(0.0, index=test_px_df.columns)
        trades.iloc[:half] = total_trade / half
        trades.iloc[half:] = -total_trade / (n_steps - half)

        trades_df.loc[(stock, date)] = trades.values

    return trades_df



def get_best_model_fit_df(stock_results_df, best_df, model_type):
    best_half_life_minutes = float(
        best_df.loc[best_df["model_type"] == model_type, "half_life_minutes"].iloc[0]
    )

    best_half_life_seconds = best_half_life_minutes * 60

    fit_df = stock_results_df[
        (stock_results_df["model_type"] == model_type) &
        (stock_results_df["half_life_seconds"] == best_half_life_seconds)
    ].copy()

    fit_df = fit_df.set_index("stock").sort_index()

    return fit_df


def compute_impact_feature_one_day(
    my_trades,
    ADV,
    sigma,
    half_life_seconds,
    model_type="ow",
    dt_seconds=10,
    eps=1e-12
):
    """
    Compute one-day model-implied impact feature with lambda = 1.
    """

    my_trades = my_trades.astype(float)

    ADV = float(ADV)
    sigma = float(sigma)
    half_life_seconds = float(half_life_seconds)

    beta = np.log(2) / half_life_seconds
    decay = np.exp(-beta * dt_seconds)

    feature_values = []

    if model_type == "ow":
        I = 0.0

        for q in my_trades:
            I = decay * I + sigma * q / ADV
            feature_values.append(I)

    elif model_type == "afs":
        I = 0.0

        for q in my_trades:
            q_kernel = np.sign(q) * np.sqrt(abs(q))
            I = decay * I + sigma * q_kernel / np.sqrt(ADV)
            feature_values.append(I)

    elif model_type == "reduced_form":
        I = 0.0
        v = 0.0

        for q in my_trades:
            v = decay * v + abs(q)
            v = max(v, eps)

            input_term = sigma * q / np.sqrt(ADV * v)
            I = decay * I + input_term

            feature_values.append(I)

    else:
        raise ValueError("model_type must be 'ow', 'afs', or 'reduced_form'.")

    return pd.Series(feature_values, index=my_trades.index)


def make_impact_adjusted_prices(
    prices,
    public_trades,
    lambda_hat,
    ADV,
    sigma,
    half_life_seconds,
    model_type="ow",
    dt_seconds=10
):
    public_impact_feature = compute_impact_feature_one_day(
        my_trades=public_trades,
        ADV=ADV,
        sigma=sigma,
        half_life_seconds=half_life_seconds,
        model_type=model_type,
        dt_seconds=dt_seconds
    )

    public_impact_state = lambda_hat * public_impact_feature
    public_impact_in_price = prices.iloc[0] * public_impact_state

    impact_adjusted_prices = prices - public_impact_in_price

    return impact_adjusted_prices


def simulate_one_stock_day(
    prices,
    my_trades,
    lambda_hat,
    ADV,
    sigma,
    half_life_seconds,
    model_type="ow",
    dt_seconds=10
):
    """
    Simulate one stock-day using a fitted impact model.
    """

    prices = prices.astype(float)
    my_trades = my_trades.reindex(prices.index).fillna(0).astype(float)

    lambda_hat = float(lambda_hat)

    impact_feature = compute_impact_feature_one_day(
        my_trades=my_trades,
        ADV=ADV,
        sigma=sigma,
        half_life_seconds=half_life_seconds,
        model_type=model_type,
        dt_seconds=dt_seconds
    )

    impact_state = lambda_hat * impact_feature

    impact_in_price = prices.iloc[0] * impact_state

    execution_prices = prices + impact_in_price

    position = my_trades.cumsum()
    cash = -(my_trades * execution_prices).cumsum()
    portfolio_value = cash + position * prices

    impact_cost = (my_trades * impact_in_price).sum()

    avg_price = prices.mean()
    total_abs_traded = my_trades.abs().sum()
    traded_notional = total_abs_traded * avg_price

    path_df = pd.DataFrame({
        "mid_price": prices,
        "trade": my_trades,
        "impact_feature": impact_feature,
        "impact_state": impact_state,
        "impact_in_price": impact_in_price,
        "execution_price": execution_prices,
        "position": position,
        "cash": cash,
        "portfolio_value": portfolio_value
    })

    summary = {
        "daily_pnl": portfolio_value.iloc[-1],
        "impact_cost": impact_cost,
        "max_abs_impact": impact_in_price.abs().max(),
        "final_position": position.iloc[-1],
        "total_abs_traded": total_abs_traded,
        "net_traded": my_trades.sum(),
        "avg_price": avg_price,
        "traded_notional": traded_notional,
        "model_type": model_type
    }

    return path_df, summary


def run_backtest_from_trade_df(
    test_px_df,
    strategy_trades_df,
    test_traded_volume_df,
    scaling_df,
    fit_df,
    model_type,
    dt_seconds=10
):
    summaries = []

    for stock, date in test_px_df.index:
        prices = test_px_df.loc[(stock, date)]
        my_trades = strategy_trades_df.loc[(stock, date)]
        public_trades = test_traded_volume_df.loc[(stock, date)]

        lambda_hat = float(fit_df.loc[stock, "lambda_hat"])
        half_life_seconds = float(fit_df.loc[stock, "half_life_seconds"])

        ADV = float(scaling_df.loc[stock, "ADV"])
        sigma = float(scaling_df.loc[stock, "sigma"])

        #Adjusting prices

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

        summaries.append(summary)

    return pd.DataFrame(summaries)


def add_bps_metrics(backtest_df):
    """
    Add P&L and impact cost in basis points of traded notional.
    """

    backtest_df = backtest_df.copy()

    backtest_df["impact_cost_bps"] = (
        1e4 * backtest_df["impact_cost"] / backtest_df["traded_notional"]
    )

    backtest_df["pnl_bps"] = (
        1e4 * backtest_df["daily_pnl"] / backtest_df["traded_notional"]
    )

    return backtest_df

def plot_one_stock_day_backtest_path(path_df, stock, date, model_type):
    """
    Plot mid price, execution price, and own price impact for one stock-day.
    """

    x = np.arange(len(path_df))
    tick_positions = np.linspace(0, len(path_df) - 1, 8, dtype=int)
    tick_labels = path_df.index[tick_positions]

    # Price plot
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