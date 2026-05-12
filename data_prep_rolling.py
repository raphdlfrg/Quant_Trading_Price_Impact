"""
Rolling data preparation utilities for Quantitative Trading and Price Impact.

This module extends the baseline data-preparation step from one train/test
month and 20 stocks to rolling monthly train/test windows over the full
available stock universe.

Main outputs per month:
    - trade_panel_YYYYMM.csv
    - price_panel_YYYYMM.csv
    - scaling_YYYYMM.csv

Main rolling output:
    - rolling_train_test_manifest.csv
    - monthly_preprocessing_diagnostics.csv
    - rolling_data_preparation_summary.csv
"""

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Baseline helpers
# ============================================================

def load_bin_month(bin_sample_path: str, year: int, month: int) -> pd.DataFrame:
    """
    Load one monthly bin file.

    Expected filename format:
        binYYYYMM.csv

    Parameters
    ----------
    bin_sample_path:
        Folder containing monthly bin files.
    year:
        Year, e.g. 2019.
    month:
        Month, e.g. 1 for January.

    Returns
    -------
    pd.DataFrame
        Long-format monthly bin dataframe.
    """
    month_str = f"{month:02d}"
    file_path = os.path.join(bin_sample_path, f"bin{year}{month_str}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file: {file_path}")

    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(df["date"])
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S").dt.time

    return df


def make_panel(df: pd.DataFrame, value_col: str, fill_method: str) -> pd.DataFrame:
    """
    Convert long-format bin data into a stock-date x time matrix.

    Parameters
    ----------
    df:
        Long-format bin dataframe with columns stock, date, time.
    value_col:
        Column to pivot, for example 'trade' or 'midEnd'.
    fill_method:
        'zero'  -> fill missing values with 0, suitable for volumes.
        'price' -> forward/backward fill across time, suitable for prices.

    Returns
    -------
    pd.DataFrame
        Panel indexed by (stock, date), with intraday time bins as columns.
    """
    required = {"stock", "date", "time", value_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    panel = df.pivot(index=["stock", "date"], columns="time", values=value_col)
    panel = panel.sort_index().sort_index(axis="columns")

    if fill_method == "zero":
        panel = panel.fillna(0)
    elif fill_method == "price":
        panel = panel.ffill(axis="columns").bfill(axis="columns")
    else:
        raise ValueError("fill_method must be either 'zero' or 'price'.")

    return panel


def align_intraday_columns(*panels: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Keep only the intraday time columns common to all panels.

    This prevents train/test mismatches when later computing time differences.
    """
    if len(panels) == 0:
        raise ValueError("At least one panel must be supplied.")

    common_times = panels[0].columns
    for panel in panels[1:]:
        common_times = common_times.intersection(panel.columns)

    common_times = sorted(common_times)

    return [panel[common_times] for panel in panels]


# ============================================================
# Rolling extension helpers
# ============================================================

def find_available_bin_months(bin_sample_path: str) -> pd.DataFrame:
    """
    Find all available binYYYYMM.csv files in the raw data folder.

    Returns
    -------
    pd.DataFrame
        Columns: year, month, year_month, file_name.
    """
    if not os.path.isdir(bin_sample_path):
        raise NotADirectoryError(f"Not a valid folder: {bin_sample_path}")

    rows = []
    for file_name in os.listdir(bin_sample_path):
        match = re.match(r"bin(\d{4})(\d{2})\.csv$", file_name)
        if match is not None:
            year = int(match.group(1))
            month = int(match.group(2))
            rows.append({
                "year": year,
                "month": month,
                "year_month": f"{year}{month:02d}",
                "file_name": file_name,
            })

    months_df = pd.DataFrame(rows)
    if months_df.empty:
        raise FileNotFoundError(
            f"No files matching binYYYYMM.csv were found in {bin_sample_path}"
        )

    return months_df.sort_values(["year", "month"]).reset_index(drop=True)


def compute_monthly_scaling(
    traded_volume_df: pd.DataFrame,
    px_df: pd.DataFrame,
    volatility_method: str = "realized",
    min_price: float = 1e-8,
) -> pd.DataFrame:
    """
    Compute stock-level ADV and volatility for one month.

    The recommended volatility estimate for the rolling extension is
    realized intraday volatility:

        sigma_{i,d} = sqrt(sum_t r_{i,d,t}^2)

    where r is the 10-second return. The stock-level sigma is the mean of
    daily realized volatility across the month.

    Parameters
    ----------
    traded_volume_df:
        Stock-date x time matrix of signed traded volume.
    px_df:
        Stock-date x time matrix of prices.
    volatility_method:
        'realized'      -> use intraday realized volatility as sigma.
        'close_to_open' -> use std of daily close-to-open returns as sigma.
        'both'          -> compute both and use realized volatility as sigma.
    min_price:
        Small cutoff to avoid division by zero.

    Returns
    -------
    pd.DataFrame
        Stock-level scaling table with columns:
        ADV, sigma, sigma_realized, sigma_realized_median,
        sigma_close_to_open, n_days, total_abs_volume.
    """
    if not traded_volume_df.index.equals(px_df.index):
        # Keep common rows only if one panel is missing some stock-days.
        common_index = traded_volume_df.index.intersection(px_df.index)
        traded_volume_df = traded_volume_df.loc[common_index]
        px_df = px_df.loc[common_index]

    # ADV from absolute signed traded volume.
    daily_volume = traded_volume_df.abs().sum(axis=1)

    stock_adv = daily_volume.groupby(level="stock").mean().rename("ADV")
    stock_total_abs_volume = daily_volume.groupby(level="stock").sum().rename(
        "total_abs_volume"
    )
    stock_n_days = daily_volume.groupby(level="stock").count().rename("n_days")

    # Close-to-open volatility, retained as a diagnostic.
    daily_close_to_open_return = px_df.iloc[:, -1] / px_df.iloc[:, 0] - 1
    stock_sigma_close_to_open = daily_close_to_open_return.groupby(level="stock").std()
    stock_sigma_close_to_open = stock_sigma_close_to_open.rename("sigma_close_to_open")

    # Realized intraday volatility from all intraday returns.
    px_safe = px_df.where(px_df > min_price)
    intraday_returns = px_safe.pct_change(axis=1)
    intraday_returns = intraday_returns.replace([np.inf, -np.inf], np.nan)

    daily_realized_vol = np.sqrt((intraday_returns ** 2).sum(axis=1, skipna=True))

    stock_sigma_realized = daily_realized_vol.groupby(level="stock").mean().rename(
        "sigma_realized"
    )
    stock_sigma_realized_median = daily_realized_vol.groupby(level="stock").median().rename(
        "sigma_realized_median"
    )

    if volatility_method == "realized":
        stock_sigma = stock_sigma_realized.rename("sigma")
    elif volatility_method == "close_to_open":
        stock_sigma = stock_sigma_close_to_open.rename("sigma")
    elif volatility_method == "both":
        stock_sigma = stock_sigma_realized.rename("sigma")
    else:
        raise ValueError(
            "volatility_method must be 'realized', 'close_to_open', or 'both'."
        )

    scaling_df = pd.concat(
        [
            stock_adv,
            stock_sigma,
            stock_sigma_realized,
            stock_sigma_realized_median,
            stock_sigma_close_to_open,
            stock_n_days,
            stock_total_abs_volume,
        ],
        axis=1,
    )

    return scaling_df


def process_one_full_universe_month(
    bin_sample_path: str,
    output_path: str,
    year: int,
    month: int,
    volatility_method: str = "realized",
) -> Dict[str, object]:
    """
    Process one month of bin data for the full available universe.

    Saves:
        - trade_panel_YYYYMM.csv
        - price_panel_YYYYMM.csv
        - scaling_YYYYMM.csv

    Returns
    -------
    dict
        Diagnostics for the processed month.
    """
    os.makedirs(output_path, exist_ok=True)
    year_month = f"{year}{month:02d}"

    print(f"Processing {year_month}...")

    month_df = load_bin_month(bin_sample_path=bin_sample_path, year=year, month=month)
    month_df = month_df.sort_values(["stock", "date", "time"])

    traded_volume_df = make_panel(month_df, value_col="trade", fill_method="zero")
    px_df = make_panel(month_df, value_col="midEnd", fill_method="price")

    traded_volume_df, px_df = align_intraday_columns(traded_volume_df, px_df)

    scaling_df = compute_monthly_scaling(
        traded_volume_df=traded_volume_df,
        px_df=px_df,
        volatility_method=volatility_method,
    )

    trade_file = os.path.join(output_path, f"trade_panel_{year_month}.csv")
    price_file = os.path.join(output_path, f"price_panel_{year_month}.csv")
    scaling_file = os.path.join(output_path, f"scaling_{year_month}.csv")

    traded_volume_df.to_csv(trade_file)
    px_df.to_csv(price_file)
    scaling_df.to_csv(scaling_file)

    diagnostics = {
        "year_month": year_month,
        "n_rows_raw": len(month_df),
        "n_stock_days": traded_volume_df.shape[0],
        "n_time_bins": traded_volume_df.shape[1],
        "n_stocks": traded_volume_df.index.get_level_values("stock").nunique(),
        "total_abs_volume": float(traded_volume_df.abs().sum().sum()),
        "mean_ADV": float(scaling_df["ADV"].mean()),
        "mean_sigma": float(scaling_df["sigma"].mean()),
        "mean_sigma_realized": float(scaling_df["sigma_realized"].mean()),
        "mean_sigma_close_to_open": float(scaling_df["sigma_close_to_open"].mean()),
        "trade_file": trade_file,
        "price_file": price_file,
        "scaling_file": scaling_file,
    }

    print(
        f"Saved {year_month}: "
        f"{diagnostics['n_stocks']} stocks, "
        f"{diagnostics['n_stock_days']} stock-days, "
        f"{diagnostics['n_time_bins']} time bins"
    )

    return diagnostics


def process_all_available_months(
    bin_sample_path: str,
    output_path: str,
    volatility_method: str = "realized",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process all available monthly bin files.

    Returns
    -------
    months_df:
        Available months found in the raw data folder.
    monthly_diagnostics_df:
        Diagnostics for each processed month.
    """
    months_df = find_available_bin_months(bin_sample_path)

    monthly_diagnostics = []
    for _, row in months_df.iterrows():
        diagnostics = process_one_full_universe_month(
            bin_sample_path=bin_sample_path,
            output_path=output_path,
            year=int(row["year"]),
            month=int(row["month"]),
            volatility_method=volatility_method,
        )
        monthly_diagnostics.append(diagnostics)

    monthly_diagnostics_df = pd.DataFrame(monthly_diagnostics)
    monthly_diagnostics_df.to_csv(
        os.path.join(output_path, "monthly_preprocessing_diagnostics.csv"),
        index=False,
    )

    return months_df, monthly_diagnostics_df


def build_rolling_manifest(months_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a rolling monthly train/test manifest.

    Each row uses one available month for training and the next available
    month for testing.
    """
    rows = []
    for i in range(len(months_df) - 1):
        train_row = months_df.iloc[i]
        test_row = months_df.iloc[i + 1]

        rows.append({
            "window_id": i,
            "train_year": int(train_row["year"]),
            "train_month": int(train_row["month"]),
            "train_year_month": train_row["year_month"],
            "test_year": int(test_row["year"]),
            "test_month": int(test_row["month"]),
            "test_year_month": test_row["year_month"],
        })

    return pd.DataFrame(rows)


def load_monthly_panel(output_path: str, year_month: str, panel_type: str) -> pd.DataFrame:
    """
    Load one saved monthly panel.

    Parameters
    ----------
    output_path:
        Folder containing processed monthly CSV files.
    year_month:
        Month identifier, e.g. '201901'.
    panel_type:
        One of 'trade', 'price', or 'scaling'.
    """
    if panel_type == "trade":
        file_path = os.path.join(output_path, f"trade_panel_{year_month}.csv")
        df = pd.read_csv(file_path, index_col=[0, 1])
        df.index.names = ["stock", "date"]
    elif panel_type == "price":
        file_path = os.path.join(output_path, f"price_panel_{year_month}.csv")
        df = pd.read_csv(file_path, index_col=[0, 1])
        df.index.names = ["stock", "date"]
    elif panel_type == "scaling":
        file_path = os.path.join(output_path, f"scaling_{year_month}.csv")
        df = pd.read_csv(file_path, index_col=0)
    else:
        raise ValueError("panel_type must be 'trade', 'price', or 'scaling'.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find processed file: {file_path}")

    return df


def load_rolling_window(
    output_path: str,
    manifest_df: pd.DataFrame,
    window_id: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    Load train/test data for one rolling monthly window.

    Returns
    -------
    train_trade_df, test_trade_df, train_px_df, test_px_df, scaling_df, metadata
    """
    row = manifest_df.loc[manifest_df["window_id"] == window_id]
    if row.empty:
        raise ValueError(f"window_id {window_id} not found in manifest.")

    row = row.iloc[0]
    train_ym = row["train_year_month"]
    test_ym = row["test_year_month"]

    train_trade_df = load_monthly_panel(output_path, train_ym, "trade")
    test_trade_df = load_monthly_panel(output_path, test_ym, "trade")
    train_px_df = load_monthly_panel(output_path, train_ym, "price")
    test_px_df = load_monthly_panel(output_path, test_ym, "price")
    scaling_df = load_monthly_panel(output_path, train_ym, "scaling")

    train_trade_df, test_trade_df, train_px_df, test_px_df = align_intraday_columns(
        train_trade_df, test_trade_df, train_px_df, test_px_df
    )

    train_stocks = set(train_trade_df.index.get_level_values("stock"))
    test_stocks = set(test_trade_df.index.get_level_values("stock"))
    scaling_stocks = set(scaling_df.index)
    common_stocks = sorted(train_stocks.intersection(test_stocks).intersection(scaling_stocks))

    train_trade_df = train_trade_df.loc[common_stocks]
    test_trade_df = test_trade_df.loc[common_stocks]
    train_px_df = train_px_df.loc[common_stocks]
    test_px_df = test_px_df.loc[common_stocks]
    scaling_df = scaling_df.loc[common_stocks]

    metadata = {
        "window_id": int(window_id),
        "train_year_month": train_ym,
        "test_year_month": test_ym,
        "n_common_stocks": len(common_stocks),
        "n_train_stock_days": train_trade_df.shape[0],
        "n_test_stock_days": test_trade_df.shape[0],
        "n_time_bins": train_trade_df.shape[1],
    }

    return train_trade_df, test_trade_df, train_px_df, test_px_df, scaling_df, metadata


def make_rolling_summary(
    rolling_manifest_df: pd.DataFrame,
    monthly_diagnostics_df: pd.DataFrame,
    output_path: str,
) -> pd.DataFrame:
    """
    Create and save a compact rolling data-preparation summary table.
    """
    rolling_summary_df = rolling_manifest_df.merge(
        monthly_diagnostics_df[
            [
                "year_month",
                "n_stocks",
                "n_stock_days",
                "n_time_bins",
                "total_abs_volume",
                "mean_ADV",
                "mean_sigma",
            ]
        ],
        left_on="train_year_month",
        right_on="year_month",
        how="left",
    )

    rolling_summary_df = rolling_summary_df.rename(
        columns={
            "n_stocks": "train_n_stocks",
            "n_stock_days": "train_n_stock_days",
            "n_time_bins": "train_n_time_bins",
            "total_abs_volume": "train_total_abs_volume",
            "mean_ADV": "train_mean_ADV",
            "mean_sigma": "train_mean_sigma",
        }
    ).drop(columns=["year_month"])

    rolling_summary_df.to_csv(
        os.path.join(output_path, "rolling_data_preparation_summary.csv"),
        index=False,
    )

    return rolling_summary_df
