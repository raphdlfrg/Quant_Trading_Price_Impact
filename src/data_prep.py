import pandas as pd

def load_bin_month(bin_sample_path: str, year: int, month: int) -> pd.DataFrame:
    """
    Load one monthly bin file.
    
    Expected filename format:
        binYYYYMM.csv
    """
    month_str = f"{month:02d}"
    file_path = os.path.join(bin_sample_path, f"bin{year}{month_str}.csv")
    
    df = pd.read_csv(file_path)
    
    df["date"] = pd.to_datetime(df["date"])
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S").dt.time
    
    return df


def select_top_liquid_stocks(df: pd.DataFrame, n_stocks: int = 20) -> list:
    """
    Select the top n stocks by absolute traded volume.
    
    Important:
    This should be applied only on the training month
    to avoid look-ahead bias.
    """
    stock_volume = (
        df.groupby("stock")["trade"]
        .apply(lambda x: x.abs().sum())
        .sort_values(ascending=False)
    )
    
    return list(stock_volume.head(n_stocks).index)


def make_panel(
    df: pd.DataFrame,
    value_col: str,
    fill_method: str
) -> pd.DataFrame:
    """
    Convert long bin data into a stock-date x time matrix.
    
    Parameters
    ----------
    df:
        Long-format bin dataframe.
    value_col:
        Column to pivot, e.g. 'trade' or 'midEnd'.
    fill_method:
        'zero'  -> fill missing values with 0, suitable for volumes.
        'price' -> forward/backward fill across time, suitable for prices.
    """
    panel = df.pivot(
        index=["stock", "date"],
        columns="time",
        values=value_col
    )
    
    # Important for later time differences
    panel = panel.sort_index().sort_index(axis="columns")
    
    if fill_method == "zero":
        panel = panel.fillna(0)
    elif fill_method == "price":
        panel = panel.ffill(axis="columns").bfill(axis="columns")
    else:
        raise ValueError("fill_method must be either 'zero' or 'price'")
    
    return panel


def align_intraday_columns(*panels):
    """
    Keep only the intraday time columns common to all panels.
    """
    common_times = panels[0].columns
    
    for panel in panels[1:]:
        common_times = common_times.intersection(panel.columns)
    
    common_times = sorted(common_times)
    
    return [panel[common_times] for panel in panels]