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
    alpha_level=1.0,
    random_seed=1234
):
    """
    Generate synthetic alpha exactly in the style of the lecture notebook.

    Lecture formula:

        alpha_t^h = [x (P_{t+h} - P_t) + y (W_{t+h} - W_t)] / P_t

    Then the synthetic alpha path is the cumulative sum of these synthetic returns.

    Parameters
    ----------
    test_px_df : pd.DataFrame
        index   = (stock, date)
        columns = intraday time bins
        values  = mid prices

    alpha_param_df : pd.DataFrame
        Indexed by stock.
        Must contain:
            x, y, target_corr, lookahead_bins

    alpha_level : float
        Multiplicative scale applied to final cumulative alpha.

    random_seed : int
        Random seed.

    Returns
    -------
    synthetic_alpha_df : pd.DataFrame
        Same shape as test_px_df.

    synthetic_alpha_diagnostics_df : pd.DataFrame
        Stock-level diagnostics.
    """

    if not isinstance(test_px_df.index, pd.MultiIndex):
        raise ValueError("test_px_df must have a MultiIndex with levels (stock, date).")

    if alpha_level <= 0:
        raise ValueError("alpha_level must be positive.")

    required_cols = ["x", "y", "target_corr", "lookahead_bins"]
    missing = [c for c in required_cols if c not in alpha_param_df.columns]
    if missing:
        raise ValueError(f"alpha_param_df is missing columns: {missing}")

    rng = np.random.default_rng(random_seed)

    alpha_rows = []
    alpha_index = []
    diagnostic_rows = []

    stocks = test_px_df.index.get_level_values(0).unique()

    for stock in stocks:
        if stock not in alpha_param_df.index:
            continue

        x = float(alpha_param_df.loc[stock, "x"])
        y = float(alpha_param_df.loc[stock, "y"])
        target_corr = float(alpha_param_df.loc[stock, "target_corr"])
        lookahead_bins = int(alpha_param_df.loc[stock, "lookahead_bins"])

        stock_prices_df = test_px_df.loc[stock].astype(float)

        all_realized_returns = []
        all_synthetic_returns = []
        all_synthetic_alphas = []

        for date in stock_prices_df.index:
            prices = stock_prices_df.loc[date].values.astype(float)
            n = len(prices)

            synthetic_returns = np.zeros(n)
            realized_returns = np.full(n, np.nan)

            # Brownian path W
            w_diffs = rng.normal(loc=0.0, scale=1.0, size=n - 1)
            W = np.concatenate([[0.0], np.cumsum(w_diffs)])

            for j in range(n - lookahead_bins):
                if prices[j] > 0 and np.isfinite(prices[j + lookahead_bins]):
                    price_change_h = prices[j + lookahead_bins] - prices[j]
                    brownian_change_h = W[j + lookahead_bins] - W[j]

                    synthetic_returns[j] = (
                        x * price_change_h
                        + y * brownian_change_h
                    ) / prices[j]

                    realized_returns[j] = price_change_h / prices[j]

            # Lecture notebook: synthetic alpha path = cumulative sum of synthetic returns
            synthetic_alpha = alpha_level * np.cumsum(synthetic_returns)

            alpha_rows.append(synthetic_alpha)
            alpha_index.append((stock, date))

            valid_mask = np.isfinite(realized_returns)

            all_realized_returns.extend(realized_returns[valid_mask])
            all_synthetic_returns.extend(synthetic_returns[valid_mask])
            all_synthetic_alphas.extend(synthetic_alpha[valid_mask])

        all_realized_returns = np.asarray(all_realized_returns)
        all_synthetic_returns = np.asarray(all_synthetic_returns)
        all_synthetic_alphas = np.asarray(all_synthetic_alphas)

        if len(all_realized_returns) > 1:
            return_corr = np.corrcoef(
                all_synthetic_returns,
                all_realized_returns
            )[0, 1]

            alpha_corr = np.corrcoef(
                all_synthetic_alphas,
                all_realized_returns
            )[0, 1]
        else:
            return_corr = np.nan
            alpha_corr = np.nan

        diagnostic_rows.append({
            "stock": stock,
            "target_corr": target_corr,
            "realized_return_corr": return_corr,
            "realized_alpha_corr": alpha_corr,
            "alpha_level": alpha_level,
            "lookahead_bins": lookahead_bins,
            "x": x,
            "y": y,
            "n_obs": len(all_realized_returns),
        })

    synthetic_alpha_df = pd.DataFrame(
        alpha_rows,
        index=pd.MultiIndex.from_tuples(
            alpha_index,
            names=test_px_df.index.names
        ),
        columns=test_px_df.columns
    )

    synthetic_alpha_diagnostics_df = (
        pd.DataFrame(diagnostic_rows)
        .set_index("stock")
        .sort_index()
    )

    return synthetic_alpha_df, synthetic_alpha_diagnostics_df

def generate_synthetic_alpha_decay_df(
    synthetic_alpha_df,
    dt_seconds=10
):
    """
    Generate synthetic lookahead alpha decay from a synthetic alpha path.

    Lecture convention:
        d alpha_t = mu_t dt + sigma_t dW_t
        decay_t = -mu_t

    Discretely:
        decay_t = -(alpha_{t+1} - alpha_t) / dt
    """

    alpha_decay_df = -(
        synthetic_alpha_df.shift(-1, axis=1)
        - synthetic_alpha_df
    ) / dt_seconds

    alpha_decay_df = alpha_decay_df.fillna(0.0)

    return alpha_decay_df


def plot_synthetic_alpha_decay(
    synthetic_alpha_df,
    synthetic_alpha_decay_df,
    stock,
    date,
    smooth_window=None
):
    """
    Plot synthetic alpha together with its synthetic decay signal.
    """

    key = (stock, date)

    if key not in synthetic_alpha_df.index:
        raise ValueError("(stock, date) not found in synthetic_alpha_df.")

    if key not in synthetic_alpha_decay_df.index:
        raise ValueError("(stock, date) not found in synthetic_alpha_decay_df.")

    alpha_series = synthetic_alpha_df.loc[key].astype(float)
    decay_series = synthetic_alpha_decay_df.loc[key].astype(float)

    if smooth_window is not None and smooth_window > 1:
        decay_series = (
            decay_series
            .rolling(smooth_window, min_periods=1)
            .mean()
        )

    n = len(alpha_series)
    x_axis = np.arange(n)

    tick_positions = np.linspace(0, n - 1, 8, dtype=int)
    tick_labels = alpha_series.index[tick_positions]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    alpha_color = "tab:blue"
    decay_color = "tab:orange"

    # Alpha level
    ax1.plot(
        x_axis,
        100 * alpha_series.values,
        color=alpha_color,
        linewidth=2,
        label="Synthetic alpha"
    )

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Alpha level (%)", color=alpha_color)
    ax1.tick_params(axis="y", labelcolor=alpha_color)

    # Alpha decay
    ax2 = ax1.twinx()

    ax2.plot(
        x_axis,
        decay_series.values,
        color=decay_color,
        linestyle="--",
        linewidth=2,
        label="Alpha decay"
    )

    ax2.set_ylabel("Alpha decay", color=decay_color)
    ax2.tick_params(axis="y", labelcolor=decay_color)

    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left"
    )

    plt.title(f"{stock} on {date} - synthetic alpha and decay")

    plt.tight_layout()
    plt.show()

    

def plot_synthetic_alpha_parameter_comparison(
    train_px_df,
    test_px_df,
    stock,
    date,
    comparison_type,
    values,
    target_corr=0.9,
    alpha_level=1.0,
    lookahead_bins=1,
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
            "target_corr"
            "alpha_level"

    values : list
        Values of the chosen parameter to compare.

    target_corr : float
        Baseline target correlation rho.

    alpha_level : float
        Baseline alpha level.

    lookahead_bins : int
        Number of bins used for the forward return.

    random_seed : int
        Random seed used to generate the same noise shocks across curves.

    Returns
    -------
    None
    """

    if comparison_type not in ["target_corr", "alpha_level"]:
        raise ValueError(
            "comparison_type must be 'target_corr' or 'alpha_level'."
        )

    if stock not in train_px_df.index.get_level_values(0):
        raise ValueError("stock is not available in train_px_df.")

    if (stock, date) not in test_px_df.index:
        raise ValueError("(stock, date) is not available in test_px_df.")

    prices = test_px_df.loc[(stock, date)].astype(float)
    n = len(prices)

    cumulative_return = prices / prices.iloc[0] - 1.0

    rng = np.random.default_rng(random_seed)

    # Brownian path
    w_diffs = rng.normal(loc=0.0, scale=1.0, size=n - 1)
    W = np.concatenate([[0.0], np.cumsum(w_diffs)])

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
        current_alpha_level = alpha_level

        if comparison_type == "target_corr":
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

        synthetic_returns = np.zeros(n)

        for j in range(n - lookahead_bins):

            if prices.iloc[j] > 0:

                price_change_h = (
                    prices.iloc[j + lookahead_bins]
                    - prices.iloc[j]
                )

                brownian_change_h = (
                    W[j + lookahead_bins]
                    - W[j]
                )

                synthetic_returns[j] = (
                    x * price_change_h
                    + y * brownian_change_h
                ) / prices.iloc[j]

        # Lecture notebook construction:
        # alpha path = cumulative sum of synthetic returns
        alpha_values = (
            current_alpha_level
            * np.cumsum(synthetic_returns)
        )

        if comparison_type == "target_corr":
            label = f"$\\rho$ = {value:.2f}"
        else:
            label = f"$A_\\alpha$ = {value:.2f}"

        plt.plot(
            x_axis,
            100 * alpha_values,
            label=label
        )

    plt.xticks(tick_positions, tick_labels, rotation=45)

    plt.title(
        f"{stock} on {date} - synthetic alpha comparison"
    )

    plt.xlabel("Time")
    plt.ylabel("Return / alpha signal, %")

    plt.legend()

    plt.tight_layout()
    plt.show()