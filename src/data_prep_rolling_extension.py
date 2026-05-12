"""
Rolling full-universe data preparation using scaling.

This file extends the baseline data preparation used in the coursework.
It processes all available binYYYYMM.csv files, saves monthly trade/price
panels separately, and computes daily rolling scaling factors following
the methodology:

    px_vol = std of intraday returns within each stock-day
    volume = total absolute traded volume within each stock-day
    sigma, ADV = previous 20-day rolling averages, shifted by one day

The shift avoids look-ahead bias: the scaling used for a stock-date only
uses information available before that date.
"""

import os
import re
import numpy as np
import pandas as pd

from src.data_prep import load_bin_month, make_panel, align_intraday_columns


def find_available_bin_months(bin_sample_path: str) -> pd.DataFrame:
    """
    Find all available binYYYYMM.csv files in the raw bin data folder.

    Returns
    -------
    months_df : pd.DataFrame
        Columns: year, month, year_month, file_name.
    """
    rows = []

    for file_name in os.listdir(bin_sample_path):
        match = re.match(r"bin(\d{4})(\d{2})\.csv$", file_name)
        if match is None:
            continue

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


def compute_daily_stock_info(
    traded_volume_df: pd.DataFrame,
    px_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute daily stock-level information.

    For each stock-date:
    - px_vol = standard deviation of 10-second intraday returns;
    - volume = total absolute traded volume during the day.

    Parameters
    ----------
    traded_volume_df : pd.DataFrame
        stock-date x time matrix of signed traded volume.
    px_df : pd.DataFrame
        stock-date x time matrix of prices.

    Returns
    -------
    stock_info_df : pd.DataFrame
        Columns: stock, date, px_vol, volume.
    """
    px_returns = px_df.pct_change(1, axis="columns")
    px_returns = px_returns.replace([np.inf, -np.inf], np.nan)

    px_vol = px_returns.std(axis="columns").rename("px_vol")
    volume = traded_volume_df.abs().sum(axis="columns").rename("volume")

    stock_info_df = pd.concat([px_vol, volume], axis=1).reset_index()
    stock_info_df["date"] = pd.to_datetime(stock_info_df["date"])

    return stock_info_df.sort_values(["stock", "date"]).reset_index(drop=True)


def compute_rolling_scaling(
    stock_info_df: pd.DataFrame,
    num_days_precompute: int = 20,
) -> pd.DataFrame:
    """
    Compute rolling scaling factors.

    For each stock-date:
    - sigma = average px_vol over previous num_days_precompute trading days;
    - ADV = average volume over previous num_days_precompute trading days.

    The rolling mean is shifted by one day to avoid look-ahead bias.

    Parameters
    ----------
    stock_info_df : pd.DataFrame
        Columns: stock, date, px_vol, volume.
    num_days_precompute : int
        Number of previous trading days used for the rolling average.

    Returns
    -------
    rolling_scaling_df : pd.DataFrame
        Columns: stock, date, sigma, ADV, px_vol_rolling, volume_rolling.
    """
    stock_info_df = stock_info_df.copy()
    stock_info_df["date"] = pd.to_datetime(stock_info_df["date"])
    stock_info_df = stock_info_df.sort_values(["date", "stock"])

    stacked_info = (
        stock_info_df
        .pivot(index="date", columns="stock", values=["px_vol", "volume"])
        .sort_index()
        .rolling(num_days_precompute)
        .mean()
        .shift(1)
    )

    rolling_scaling_df = pd.DataFrame({
        "sigma": stacked_info["px_vol"].stack(),
        "ADV": stacked_info["volume"].stack(),
    }).reset_index()

    rolling_scaling_df = rolling_scaling_df.dropna()
    rolling_scaling_df = rolling_scaling_df.sort_values(["stock", "date"])
    rolling_scaling_df = rolling_scaling_df.reset_index(drop=True)

    return rolling_scaling_df


def process_one_full_universe_month(
    bin_sample_path: str,
    output_path: str,
    year: int,
    month: int,
) -> dict:
    """
    Process one month of bin data for the full available universe.

    Saves:
    - trade_panel_YYYYMM.csv
    - price_panel_YYYYMM.csv
    - daily_stock_info_YYYYMM.csv

    The monthly rolling scaling table is computed later after all months'
    daily stock info has been concatenated, because the rolling 20-day
    window may use observations from the previous month.
    """
    os.makedirs(output_path, exist_ok=True)
    year_month = f"{year}{month:02d}"

    print(f"Processing {year_month}...")

    month_df = load_bin_month(
        bin_sample_path=bin_sample_path,
        year=year,
        month=month,
    )

    month_df = month_df.sort_values(["stock", "date", "time"])

    traded_volume_df = make_panel(
        month_df,
        value_col="trade",
        fill_method="zero",
    )

    px_df = make_panel(
        month_df,
        value_col="midEnd",
        fill_method="price",
    )

    traded_volume_df, px_df = align_intraday_columns(traded_volume_df, px_df)

    daily_stock_info_df = compute_daily_stock_info(
        traded_volume_df=traded_volume_df,
        px_df=px_df,
    )

    trade_file = os.path.join(output_path, f"trade_panel_{year_month}.csv")
    price_file = os.path.join(output_path, f"price_panel_{year_month}.csv")
    info_file = os.path.join(output_path, f"daily_stock_info_{year_month}.csv")

    traded_volume_df.to_csv(trade_file)
    px_df.to_csv(price_file)
    daily_stock_info_df.to_csv(info_file, index=False)

    diagnostics = {
        "year_month": year_month,
        "n_rows_raw": len(month_df),
        "n_stock_days": traded_volume_df.shape[0],
        "n_time_bins": traded_volume_df.shape[1],
        "n_stocks": traded_volume_df.index.get_level_values("stock").nunique(),
        "total_abs_volume": float(traded_volume_df.abs().sum().sum()),
        "trade_file": trade_file,
        "price_file": price_file,
        "daily_stock_info_file": info_file,
    }

    print(
        f"Saved {year_month}: "
        f"{diagnostics['n_stocks']} stocks, "
        f"{diagnostics['n_stock_days']} stock-days, "
        f"{diagnostics['n_time_bins']} time bins"
    )

    return diagnostics


def process_all_full_universe_months(
    bin_sample_path: str,
    output_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process all available monthly bin files.

    Returns
    -------
    months_df : pd.DataFrame
        Available months.
    monthly_diagnostics_df : pd.DataFrame
        Diagnostics for each processed month.
    """
    months_df = find_available_bin_months(bin_sample_path)

    diagnostics = []
    for _, row in months_df.iterrows():
        diagnostics.append(
            process_one_full_universe_month(
                bin_sample_path=bin_sample_path,
                output_path=output_path,
                year=int(row["year"]),
                month=int(row["month"]),
            )
        )

    monthly_diagnostics_df = pd.DataFrame(diagnostics)
    monthly_diagnostics_df.to_csv(
        os.path.join(output_path, "monthly_preprocessing_diagnostics.csv"),
        index=False,
    )

    months_df.to_csv(
        os.path.join(output_path, "available_months.csv"),
        index=False,
    )

    return months_df, monthly_diagnostics_df


def load_all_daily_stock_info(output_path: str, months_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load and concatenate all monthly daily_stock_info_YYYYMM.csv files.
    """
    pieces = []

    for _, row in months_df.iterrows():
        ym = row["year_month"]
        file_path = os.path.join(output_path, f"daily_stock_info_{ym}.csv")
        info = pd.read_csv(file_path, parse_dates=["date"])
        pieces.append(info)

    stock_info_df = pd.concat(pieces, ignore_index=True)
    stock_info_df = stock_info_df.sort_values(["stock", "date"]).reset_index(drop=True)

    return stock_info_df


def save_rolling_scaling(
    output_path: str,
    months_df: pd.DataFrame,
    num_days_precompute: int = 20,
) -> pd.DataFrame:
    """
    Build and save the full daily rolling scaling table.

    Saves:
    - daily_stock_info_all_months.csv
    - rolling_scaling_20d.csv, or matching window length.
    """
    stock_info_df = load_all_daily_stock_info(output_path, months_df)

    stock_info_file = os.path.join(output_path, "daily_stock_info_all_months.csv")
    stock_info_df.to_csv(stock_info_file, index=False)

    rolling_scaling_df = compute_rolling_scaling(
        stock_info_df=stock_info_df,
        num_days_precompute=num_days_precompute,
    )

    scaling_file = os.path.join(
        output_path,
        f"rolling_scaling_{num_days_precompute}d.csv",
    )
    rolling_scaling_df.to_csv(scaling_file, index=False)

    return rolling_scaling_df


def build_rolling_manifest(months_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a rolling train/test manifest.

    Each row uses month m for training and the next available month m+1
    for testing.
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

    panel_type must be one of:
    - 'trade'
    - 'price'
    """
    if panel_type == "trade":
        file_path = os.path.join(output_path, f"trade_panel_{year_month}.csv")
    elif panel_type == "price":
        file_path = os.path.join(output_path, f"price_panel_{year_month}.csv")
    else:
        raise ValueError("panel_type must be 'trade' or 'price'.")

    df = pd.read_csv(file_path, index_col=[0, 1])
    df.index.names = ["stock", "date"]
    df.index = pd.MultiIndex.from_arrays(
        [
            df.index.get_level_values("stock"),
            pd.to_datetime(df.index.get_level_values("date")),
        ],
        names=["stock", "date"],
    )

    return df.sort_index().sort_index(axis="columns")


def load_rolling_scaling_for_dates(
    output_path: str,
    dates: pd.Index,
    num_days_precompute: int = 20,
) -> pd.DataFrame:
    """
    Load the rolling scaling table and filter to dates.

    Returns a MultiIndex dataframe indexed by (stock, date), with columns:
    - sigma
    - ADV
    """
    scaling_file = os.path.join(
        output_path,
        f"rolling_scaling_{num_days_precompute}d.csv",
    )
    scaling_df = pd.read_csv(scaling_file, parse_dates=["date"])

    date_values = pd.to_datetime(pd.Index(dates).unique())
    scaling_df = scaling_df[scaling_df["date"].isin(date_values)].copy()

    scaling_df = scaling_df.set_index(["stock", "date"]).sort_index()

    return scaling_df


def load_rolling_window(
    output_path: str,
    manifest_df: pd.DataFrame,
    window_id: int,
    num_days_precompute: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Load train and test data for one rolling window.

    Returns
    -------
    train_trade_df
    test_trade_df
    train_px_df
    test_px_df
    scaling_df
        Daily rolling scaling factors for train dates, indexed by
        (stock, date). This is the object to use when fitting on the train
        month with daily no-look-ahead scaling.
    metadata
    """
    row = manifest_df.loc[manifest_df["window_id"] == window_id].iloc[0]

    train_ym = row["train_year_month"]
    test_ym = row["test_year_month"]

    train_trade_df = load_monthly_panel(output_path, train_ym, "trade")
    test_trade_df = load_monthly_panel(output_path, test_ym, "trade")
    train_px_df = load_monthly_panel(output_path, train_ym, "price")
    test_px_df = load_monthly_panel(output_path, test_ym, "price")

    train_trade_df, test_trade_df, train_px_df, test_px_df = align_intraday_columns(
        train_trade_df,
        test_trade_df,
        train_px_df,
        test_px_df,
    )

    train_dates = train_trade_df.index.get_level_values("date")
    scaling_df = load_rolling_scaling_for_dates(
        output_path=output_path,
        dates=train_dates,
        num_days_precompute=num_days_precompute,
    )

    train_index = set(train_trade_df.index)
    test_index = set(test_trade_df.index)
    scaling_index = set(scaling_df.index)

    # We require the same stock-date rows in train and scaling. For test,
    # we keep stocks that also exist in the train universe; exact dates differ.
    common_train_index = sorted(train_index.intersection(scaling_index))
    common_stocks = sorted({idx[0] for idx in common_train_index}.intersection(
        set(test_trade_df.index.get_level_values("stock"))
    ))

    train_trade_df = train_trade_df.loc[common_stocks]
    train_px_df = train_px_df.loc[common_stocks]
    test_trade_df = test_trade_df.loc[common_stocks]
    test_px_df = test_px_df.loc[common_stocks]

    # Filter scaling to remaining train rows only.
    scaling_df = scaling_df.loc[scaling_df.index.intersection(train_trade_df.index)]

    metadata = {
        "window_id": window_id,
        "train_year_month": train_ym,
        "test_year_month": test_ym,
        "n_common_stocks": len(common_stocks),
        "n_train_stock_days": train_trade_df.shape[0],
        "n_test_stock_days": test_trade_df.shape[0],
        "n_time_bins": train_trade_df.shape[1],
        "num_days_precompute": num_days_precompute,
    }

    return (
        train_trade_df,
        test_trade_df,
        train_px_df,
        test_px_df,
        scaling_df,
        metadata,
    )
