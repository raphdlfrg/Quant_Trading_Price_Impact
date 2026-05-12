import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calibrate_synthetic_alpha_params(
    train_px_df,
    target_corr=0.9,
    lookahead_bins=1
):
    """
    Calibrate the stock-level constants used to generate synthetic alpha.

    The input price dataframe should have:

        index   = (stock, date)
        columns = intraday time bins
        values  = mid prices

    For each stock, this function estimates:

        Var(r)
        E[P^{-2}]
        x = rho^2
        y = rho * sqrt(1 - rho^2) * sqrt( Var(r) / (E[P^{-2}] * h) )

    where:

        r_{i,d,j} = (P_{i,d,j+h} - P_{i,d,j}) / P_{i,d,j}

    Parameters
    ----------
    train_px_df : pd.DataFrame
        Training price panel indexed by (stock, date).

    target_corr : float
        Target correlation between the synthetic alpha innovation
        and the realized forward return.

    lookahead_bins : int
        Number of intraday bins used for the forward return.

    Returns
    -------
    alpha_param_df : pd.DataFrame
        Stock-level calibration table indexed by stock.
    """

    if not isinstance(train_px_df.index, pd.MultiIndex):
        raise ValueError("train_px_df must have a MultiIndex with levels (stock, date).")

    if lookahead_bins < 1:
        raise ValueError("lookahead_bins must be at least 1.")

    if target_corr <= 0 or target_corr >= 1:
        raise ValueError("target_corr must be between 0 and 1.")

    rows = []

    # Get unique stocks from the multi-index
    stocks = train_px_df.index.get_level_values(0).unique()

    for stock in stocks:
        # Extract price data for this stock across all training dates
        stock_prices = train_px_df.loc[stock].astype(float)

        # Align current and future prices for forward return calculation
        current_prices = stock_prices.iloc[:, :-lookahead_bins]  # All but last h columns
        future_prices = stock_prices.iloc[:, lookahead_bins:]    # All but first h columns

        # Reset column indices to align for element-wise operations
        current_prices.columns = range(current_prices.shape[1])
        future_prices.columns = range(future_prices.shape[1])

        # Calculate forward returns: (P_{t+h} - P_t) / P_t
        forward_returns = (future_prices - current_prices) / current_prices

        # Flatten to 1D arrays for statistical calculations
        r_values = forward_returns.to_numpy().ravel()  # Forward returns
        p_values = current_prices.to_numpy().ravel()   # Current prices

        # Filter out invalid observations (NaN, inf, or negative prices)
        valid_mask = (
            np.isfinite(r_values)
            & np.isfinite(p_values)
            & (p_values > 0)
        )

        r_values = r_values[valid_mask]
        p_values = p_values[valid_mask]

        # Estimate statistical moments needed for alpha calibration
        var_r = np.var(r_values)  # Variance of forward returns
        mean_inv_price_sq = np.mean(1.0 / (p_values ** 2))  # E[1/P^2]

        # Calculate calibration parameters x and y
        # x = rho^2 (deterministic component strength)
        x = target_corr ** 2

        # y = rho * sqrt(1-rho^2) * sqrt(Var(r)/(E[1/P^2] * h))
        # (stochastic component strength, scaled by price and time)
        y = (
            target_corr
            * np.sqrt(1 - target_corr ** 2)
            * np.sqrt(var_r / (mean_inv_price_sq * lookahead_bins))
        )

        # Store calibration results for this stock
        rows.append({
            "stock": stock,
            "target_corr": target_corr,
            "lookahead_bins": lookahead_bins,
            "var_forward_return": var_r,
            "mean_inv_price_sq": mean_inv_price_sq,
            "x": x,
            "y": y,
            "n_obs": len(r_values)  # Number of valid observations
        })

    # Create and format the output DataFrame
    alpha_param_df = pd.DataFrame(rows)
    alpha_param_df = alpha_param_df.set_index("stock").sort_index()

    return alpha_param_df


def generate_synthetic_alpha_df(
    test_px_df,
    alpha_param_df,
    alpha_half_life_seconds=1800,
    alpha_level=1.0,
    dt_seconds=10,
    random_seed=1234
):
    """
    Generate a synthetic alpha panel using the calibrated stock-level
    parameters from calibrate_synthetic_alpha_params.

    Parameters
    ----------
    test_px_df : pd.DataFrame
        Test price panel.

        index   = (stock, date)
        columns = intraday time bins
        values  = mid prices

    alpha_param_df : pd.DataFrame
        Output of calibrate_synthetic_alpha_params.
        Indexed by stock and containing x, y, target_corr, lookahead_bins.

    alpha_half_life_seconds : float
        Half-life of the decaying alpha process, in seconds.

    alpha_level : float
        Multiplicative scaling applied to the final alpha process.

    dt_seconds : int
        Length of one intraday time bin, in seconds.

    random_seed : int
        Random seed used for reproducibility.

    Returns
    -------
    synthetic_alpha_df : pd.DataFrame
        Alpha panel with the same shape as test_px_df.

        index   = (stock, date)
        columns = intraday time bins
        values  = synthetic alpha values

    synthetic_alpha_diagnostics_df : pd.DataFrame
        Stock-level diagnostics for the synthetic alpha construction.
    """

    if not isinstance(test_px_df.index, pd.MultiIndex):
        raise ValueError("test_px_df must have a MultiIndex with levels (stock, date).")

    if alpha_half_life_seconds <= 0:
        raise ValueError("alpha_half_life_seconds must be positive.")

    if alpha_level <= 0:
        raise ValueError("alpha_level must be positive.")

    # Initialize random number generator for reproducible results
    rng = np.random.default_rng(random_seed)

    # Calculate decay factor for alpha process: alpha decays exponentially
    # phi_alpha = exp(-ln(2) * dt / half_life) = exp(-dt / (half_life / ln(2)))
    phi_alpha = np.exp(
        -np.log(2) * dt_seconds / alpha_half_life_seconds
    )

    # Storage for results
    alpha_rows = []      # Will hold alpha time series for each stock-day
    alpha_index = []     # Will hold (stock, date) tuples
    diagnostic_rows = [] # Will hold diagnostic statistics

    # Process each stock independently
    stocks = test_px_df.index.get_level_values(0).unique()

    for stock in stocks:
        # Skip stocks not in calibration parameters
        if stock not in alpha_param_df.index:
            continue

        # Extract calibrated parameters for this stock
        x = float(alpha_param_df.loc[stock, "x"])  # Deterministic component weight
        y = float(alpha_param_df.loc[stock, "y"])  # Stochastic component weight
        target_corr = float(alpha_param_df.loc[stock, "target_corr"])
        lookahead_bins = int(alpha_param_df.loc[stock, "lookahead_bins"])

        # Get price data for this stock across all test dates
        stock_prices_df = test_px_df.loc[stock].astype(float)

        # Storage for diagnostic calculations across all dates for this stock
        all_forward_returns = []
        all_alpha_innovations = []
        all_alpha_values = []

        # Generate alpha for each trading day
        for date in stock_prices_df.index:
            prices = stock_prices_df.loc[date].values.astype(float)
            n = len(prices)

            # Initialize arrays for this day
            alpha_innovation = np.zeros(n)  # Innovation shocks at each time step
            forward_return = np.full(n, np.nan)  # Forward returns for correlation calc

            # Generate innovation shocks for each time step
            for j in range(n - lookahead_bins):
                if prices[j] > 0:  # Valid price check
                    # Calculate realized forward return
                    r = (
                        prices[j + lookahead_bins] - prices[j]
                    ) / prices[j]

                    # Generate random shock with variance scaled by lookahead_bins
                    delta_w = rng.normal(
                        loc=0.0,
                        scale=np.sqrt(lookahead_bins)
                    )

                    # Innovation = deterministic component + stochastic component
                    # Deterministic: correlated with forward return
                    # Stochastic: uncorrelated noise, scaled by 1/price
                    alpha_innovation[j] = x * r + y * delta_w / prices[j]
                    forward_return[j] = r

            # Generate the alpha time series by integrating innovations
            alpha_values = np.zeros(n)
            alpha_state = 0.0  # Current alpha state

            for j in range(n):
                # Alpha follows AR(1) process: alpha_t = phi * alpha_{t-1} + innovation_t
                alpha_state = (
                    phi_alpha * alpha_state
                    + alpha_innovation[j]
                )

                # Apply scaling factor
                alpha_values[j] = alpha_level * alpha_state

            # Store results for this stock-day
            alpha_rows.append(alpha_values)
            alpha_index.append((stock, date))

            # Collect data for diagnostics (filtering out NaN forward returns)
            valid_mask = np.isfinite(forward_return)
            all_forward_returns.extend(forward_return[valid_mask])
            all_alpha_innovations.extend(alpha_innovation[valid_mask])
            all_alpha_values.extend(alpha_values[valid_mask])

        # Calculate diagnostic correlations for this stock
        all_forward_returns = np.array(all_forward_returns)
        all_alpha_innovations = np.array(all_alpha_innovations)
        all_alpha_values = np.array(all_alpha_values)

        if len(all_forward_returns) > 1:
            # Correlation between innovation shocks and forward returns
            innovation_corr = np.corrcoef(
                all_alpha_innovations,
                all_forward_returns
            )[0, 1]

            # Correlation between final alpha signal and forward returns
            alpha_corr = np.corrcoef(
                all_alpha_values,
                all_forward_returns
            )[0, 1]
        else:
            innovation_corr = np.nan
            alpha_corr = np.nan

        # Store diagnostic results
        diagnostic_rows.append({
            "stock": stock,
            "target_corr": target_corr,
            "realized_innovation_corr": innovation_corr,
            "realized_alpha_corr": alpha_corr,
            "alpha_half_life_seconds": alpha_half_life_seconds,
            "alpha_level": alpha_level,
            "lookahead_bins": lookahead_bins,
            "phi_alpha": phi_alpha,  # Decay factor
            "n_obs": len(all_forward_returns)  # Number of valid observations
        })

    # Create the main output: synthetic alpha DataFrame
    synthetic_alpha_df = pd.DataFrame(
        alpha_rows,
        index=pd.MultiIndex.from_tuples(
            alpha_index,
            names=test_px_df.index.names
        ),
        columns=test_px_df.columns
    )

    # Create diagnostics DataFrame
    synthetic_alpha_diagnostics_df = pd.DataFrame(diagnostic_rows)
    synthetic_alpha_diagnostics_df = (
        synthetic_alpha_diagnostics_df
        .set_index("stock")
        .sort_index()
    )

    return synthetic_alpha_df, synthetic_alpha_diagnostics_df

def plot_synthetic_alpha_parameter_comparison(
    train_px_df,
    test_px_df,
    stock,
    date,
    comparison_type,
    values,
    target_corr=0.9,
    alpha_half_life_seconds=1800,
    alpha_level=1.0,
    lookahead_bins=1,
    dt_seconds=10,
    random_seed=1234
):
    """
    Plot synthetic alpha paths for one stock-day while varying one parameter.

    Parameters
    ----------
    train_px_df : pd.DataFrame
        Training price panel indexed by (stock, date).

    test_px_df : pd.DataFrame
        Test price panel indexed by (stock, date).

    stock : str
        Stock ticker.

    date : str
        Date to plot.

    comparison_type : str
        Parameter to vary. Must be one of:
            "half_life"
            "target_corr"
            "alpha_level"

    values : list
        Values of the chosen parameter to compare.

    target_corr : float
        Baseline target correlation rho.

    alpha_half_life_seconds : float
        Baseline alpha half-life in seconds.

    alpha_level : float
        Baseline alpha level.

    lookahead_bins : int
        Number of bins used for the forward return.

    dt_seconds : int
        Length of each intraday bin in seconds.

    random_seed : int
        Random seed used to generate the same noise shocks across curves.

    Returns
    -------
    None
    """

    if comparison_type not in ["half_life", "target_corr", "alpha_level"]:
        raise ValueError("comparison_type must be 'half_life', 'target_corr', or 'alpha_level'.")

    if stock not in train_px_df.index.get_level_values(0):
        raise ValueError("stock is not available in train_px_df.")

    if (stock, date) not in test_px_df.index:
        raise ValueError("(stock, date) is not available in test_px_df.")

    prices = test_px_df.loc[(stock, date)].astype(float)
    n = len(prices)

    cumulative_return = prices / prices.iloc[0] - 1.0

    rng = np.random.default_rng(random_seed)
    noise_shocks = rng.normal(
        loc=0.0,
        scale=np.sqrt(lookahead_bins),
        size=n
    )

    x_axis = np.arange(n)
    tick_positions = np.linspace(0, n - 1, 8, dtype=int)
    tick_labels = prices.index[tick_positions]

    plt.figure(figsize=(12, 5))

    plt.plot(
        x_axis,
        100 * cumulative_return.values,
        linewidth=2,
        label="Cumulative intraday return"
    )

    for value in values:

        current_target_corr = target_corr
        current_half_life = alpha_half_life_seconds
        current_alpha_level = alpha_level

        if comparison_type == "half_life":
            current_half_life = value
        elif comparison_type == "target_corr":
            current_target_corr = value
        elif comparison_type == "alpha_level":
            current_alpha_level = value

        alpha_param_df = calibrate_synthetic_alpha_params(
            train_px_df=train_px_df,
            target_corr=current_target_corr,
            lookahead_bins=lookahead_bins
        )

        x = float(alpha_param_df.loc[stock, "x"])
        y = float(alpha_param_df.loc[stock, "y"])

        phi_alpha = np.exp(
            -np.log(2) * dt_seconds / current_half_life
        )

        alpha_innovation = np.zeros(n)

        for j in range(n - lookahead_bins):
            if prices.iloc[j] > 0:
                forward_return = (
                    prices.iloc[j + lookahead_bins] - prices.iloc[j]
                ) / prices.iloc[j]

                alpha_innovation[j] = (
                    x * forward_return
                    + y * noise_shocks[j] / prices.iloc[j]
                )

        alpha_values = np.zeros(n)
        alpha_state = 0.0

        for j in range(n):
            alpha_state = (
                phi_alpha * alpha_state
                + alpha_innovation[j]
            )

            alpha_values[j] = current_alpha_level * alpha_state

        if comparison_type == "half_life":
            label = f"$H_\\alpha$ = {value / 60:.0f} min"
        elif comparison_type == "target_corr":
            label = f"$\\rho$ = {value:.2f}"
        else:
            label = f"$A_\\alpha$ = {value:.2f}"

        plt.plot(
            x_axis,
            100 * alpha_values,
            label=label
        )

    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.title(f"{stock} on {date} - synthetic alpha comparison")
    plt.xlabel("Time")
    plt.ylabel("Return / alpha signal, %")
    plt.legend()
    plt.tight_layout()
    plt.show()