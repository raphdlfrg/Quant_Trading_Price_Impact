import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_twap_trade_df(test_px_df, scaling_df, participation_rate=0.20):
    """
    Create a TWAP (Time-Weighted Average Price) trade schedule.
    
    Distributes a fixed percentage of Average Daily Volume (ADV) equally 
    across all intraday time steps for each stock-day.
    
    Args:
        test_px_df: DataFrame with price data indexed by (stock, date)
        scaling_df: DataFrame containing ADV values for each stock
        participation_rate: Fraction of ADV to trade (default 0.20 = 20%)
    
    Returns:
        DataFrame with trade volumes distributed equally across time steps
    """
    trades_df = pd.DataFrame(
        0.0,
        index=test_px_df.index,
        columns=test_px_df.columns
    )

    for stock, date in test_px_df.index:
        # Get average daily volume for this stock
        ADV = float(scaling_df.loc[stock, "ADV"])
        # Calculate total amount to trade as a fraction of ADV
        total_trade = participation_rate * ADV
        # Divide equally across all time bins
        n_steps = test_px_df.shape[1]

        trades_df.loc[(stock, date)] = total_trade / n_steps

    return trades_df

def make_round_trip_twap_trade_df(test_px_df, scaling_df, participation_rate=0.20):
    """
    Create a round-trip TWAP trade schedule (buy then sell).
    
    Buys a fixed percentage of ADV in the first half of the day,
    then sells it in the second half (flat ending position).
    
    Args:
        test_px_df: DataFrame with price data indexed by (stock, date)
        scaling_df: DataFrame containing ADV values for each stock
        participation_rate: Fraction of ADV to trade (default 0.20 = 20%)
    
    Returns:
        DataFrame with positive trades (buys) in first half, negative trades (sells) in second half
    """
    trades_df = pd.DataFrame(
        0.0,
        index=test_px_df.index,
        columns=test_px_df.columns
    )

    n_steps = test_px_df.shape[1]
    half = n_steps // 2

    for stock, date in test_px_df.index:
        # Get average daily volume and calculate total trade size
        ADV = float(scaling_df.loc[stock, "ADV"])
        total_trade = participation_rate * ADV

        trades = pd.Series(0.0, index=test_px_df.columns)
        # Buy (positive) in first half
        trades.iloc[:half] = total_trade / half
        # Sell (negative) in second half
        trades.iloc[half:] = -total_trade / (n_steps - half)

        trades_df.loc[(stock, date)] = trades.values

    return trades_df



def get_best_model_fit_df(stock_results_df, best_df, model_type):
    """
    Extract fitted model parameters for the best-fit model type.
    
    Retrieves stock-level fitted parameters (lambda_hat, etc.) for the 
    specified model type with its optimal half-life.
    
    Args:
        stock_results_df: DataFrame with stock-level fitted results
        best_df: DataFrame with optimal parameters by model type
        model_type: Model type (e.g., 'ow', 'afs', 'reduced_form')
    
    Returns:
        DataFrame indexed by stock with fitted parameters for the best model
    """
    # Get optimal half-life for this model type
    best_half_life_minutes = float(
        best_df.loc[best_df["model_type"] == model_type, "half_life_minutes"].iloc[0]
    )

    # Convert from minutes to seconds
    best_half_life_seconds = best_half_life_minutes * 60

    # Filter results for this model type and optimal half-life
    fit_df = stock_results_df[
        (stock_results_df["model_type"] == model_type) &
        (stock_results_df["half_life_seconds"] == best_half_life_seconds)
    ].copy()

    # Index by stock for easier lookup
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
    Compute impact state feature for one trading day across different models.
    
    Evaluates the price impact at each time step based on accumulated trading volume,
    accounting for mean-reversion of impact over time. Supports three different 
    impact models with varying specifications.
    
    Args:
        my_trades: Series of trade volumes (positive = buy, negative = sell)
        ADV: Average daily volume for normalization
        sigma: Impact sensitivity parameter
        half_life_seconds: Mean-reversion half-life of impact
        model_type: 'ow' (Obizhaeva-Wang), 'afs' (Almgren-Fruth-Schied), 
                   or 'reduced_form'
        dt_seconds: Time step between observations (default 10 seconds)
        eps: Small value for numerical stability (default 1e-12)
    
    Returns:
        Series of impact features (normalized by price) at each time step
        
    Notes:
        - OW model: Linear impact I = decay*I + sigma * q/ADV
        - AFS model: Square-root impact I = decay*I + sigma * sign(q)*sqrt(|q|)/sqrt(ADV)
        - Reduced-form: Nonlinear with volume weighting
    """

    my_trades = my_trades.astype(float)

    ADV = float(ADV)
    sigma = float(sigma)
    half_life_seconds = float(half_life_seconds)

    # Calculate decay factor from half-life: I(t) = decay * I(t-dt)
    beta = np.log(2) / half_life_seconds
    decay = np.exp(-beta * dt_seconds)

    feature_values = []

    if model_type == "ow":
        # Obizhaeva-Wang: Linear impact model
        I = 0.0

        for q in my_trades:
            # Impact decays exponentially, new trade q adds sigma*q/ADV
            I = decay * I + sigma * q / ADV
            feature_values.append(I)

    elif model_type == "afs":
        # Almgren-Fruth-Schied: Square-root model
        I = 0.0

        for q in my_trades:
            # Impact uses square-root of trade size to capture nonlinearity
            q_kernel = np.sign(q) * np.sqrt(abs(q))
            I = decay * I + sigma * q_kernel / np.sqrt(ADV)
            feature_values.append(I)

    elif model_type == "reduced_form":
        # Reduced-form: Volatility-weighted impact
        I = 0.0
        v = 0.0

        for q in my_trades:
            # Track recent volume activity for dynamic weighting
            v = decay * v + abs(q)
            v = max(v, eps)  # Avoid division by zero

            # Impact depends on trade size relative to recent volume
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
    """
    Adjust prices for the impact caused by public market trades.
    
    Removes the price impact from public trades to isolate the trading environment,
    allowing clean backtesting of trading strategies.
    
    Args:
        prices: Series of mid-prices over time
        public_trades: Series of public trading volumes
        lambda_hat: Fitted impact coefficient
        ADV: Average daily volume
        sigma: Volatility parameter
        half_life_seconds: Impact mean-reversion half-life in seconds
        model_type: Type of impact model ('ow', 'afs', 'reduced_form')
        dt_seconds: Time step in seconds (default 10)
    
    Returns:
        Series of impact-adjusted prices
    """
    # Compute impact state from public trades
    public_impact_feature = compute_impact_feature_one_day(
        my_trades=public_trades,
        ADV=ADV,
        sigma=sigma,
        half_life_seconds=half_life_seconds,
        model_type=model_type,
        dt_seconds=dt_seconds
    )

    # Calculate impact magnitude
    public_impact_state = lambda_hat * public_impact_feature
    public_impact_in_price = prices.iloc[0] * public_impact_state

    # Remove public impact from prices
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
    
    Computes execution prices, position tracking, cash flows, and portfolio value
    when executing trades under a specific price impact model. Returns both the 
    full trading path (for visualization) and summary statistics.
    
    Args:
        prices: Series of mid-market prices throughout the day
        my_trades: Series of trade volumes to execute (positive = buy, negative = sell)
        lambda_hat: Fitted impact coefficient (sensitivity parameter)
        ADV: Average daily volume
        sigma: Volatility parameter
        half_life_seconds: Impact mean-reversion half-life
        model_type: Type of impact model ('ow', 'afs', 'reduced_form')
        dt_seconds: Time step in seconds (default 10)
    
    Returns:
        path_df: DataFrame with full trading path including:
            - mid_price: Mid-market prices
            - trade: Executed trade volumes
            - impact_feature: Model impact state
            - impact_in_price: Price impact in currency units
            - execution_price: Mid price + impact
            - position: Cumulative position held
            - cash: Cumulative cash spent
            - portfolio_value: Total portfolio value (cash + position * mid-price)
        
        summary: Dict with key metrics:
            - daily_pnl: Profit/loss at end of day
            - impact_cost: Total cost due to price impact
            - max_abs_impact: Maximum absolute price impact during day
            - final_position: Position at end of day (should be 0 for execution, positive for TWAP)
            - total_abs_traded: Total absolute volume traded
            - traded_notional: Total notional value of trades
    """

    prices = prices.astype(float)
    # Align trades with price index, fill missing values with 0
    my_trades = my_trades.reindex(prices.index).fillna(0).astype(float)

    lambda_hat = float(lambda_hat)

    # Compute cumulative impact state from our trades
    impact_feature = compute_impact_feature_one_day(
        my_trades=my_trades,
        ADV=ADV,
        sigma=sigma,
        half_life_seconds=half_life_seconds,
        model_type=model_type,
        dt_seconds=dt_seconds
    )

    # Scale impact by fitted coefficient and initial price
    impact_state = lambda_hat * impact_feature
    impact_in_price = prices.iloc[0] * impact_state

    # Execution prices = mid prices + price impact
    execution_prices = prices + impact_in_price

    # Track cumulative position and cash flows
    position = my_trades.cumsum()
    cash = -(my_trades * execution_prices).cumsum()
    # Portfolio value = cash held + position value at mid prices
    portfolio_value = cash + position * prices

    # Total cost incurred due to price impact
    impact_cost = (my_trades * impact_in_price).sum()

    # Summary statistics
    avg_price = prices.mean()
    total_abs_traded = my_trades.abs().sum()
    traded_notional = total_abs_traded * avg_price

    # Create detailed path dataframe for analysis and visualization
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

    # Create summary dictionary for aggregation
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