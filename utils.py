
import pandas as pd
import copy
import numpy as np
from pandas.tseries.offsets import Week, MonthBegin
import json
import os

# ----------------------------
# Calendar helpers
# ----------------------------
def easter_date(year: int) -> pd.Timestamp:
    """Gregorian computus (Anonymous algorithm) for Western Easter (UTC, no TZ)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return pd.Timestamp(year=year, month=month, day=day)

def good_friday(year: int) -> pd.Timestamp:
    """Good Friday for given year (2 days before Easter Sunday)."""
    return easter_date(year) - pd.Timedelta(days=2)

def third_friday(year: int, month: int) -> pd.Timestamp:
    """
    Third Friday of a given month (no holiday logic).
    weekday(): Mon=0 ... Sun=6, so Friday is 4.
    """
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (4 - first.weekday()) % 7  # to first Friday
    return first + pd.Timedelta(days=offset + 14)  # + two weeks = third Friday

def spx_monthly_expiration(year: int, month: int) -> pd.Timestamp:
    """
    Standard SPX monthly expiration date for the given month:
    - Third Friday normally
    - If that Friday is Good Friday (market holiday), move to the preceding Thursday.
    (Other rare exchange closures are not handled here; pass a custom list if needed.)
    """
    tf = third_friday(year, month)
    gf = good_friday(year)
    if tf.normalize() == gf.normalize():
        return tf - pd.Timedelta(days=1)  # preceding Thursday
    return tf

def vix_monthly_settlement_for_month(year: int, month: int) -> pd.Timestamp:
    """
    VIX monthly settlement (final settlement) that occurs in the given (year, month):
    Defined as 30 calendar days before the SPX monthly expiration of the FOLLOWING month.
    """
    # following month
    if month == 12:
        y2, m2 = year + 1, 1
    else:
        y2, m2 = year, month + 1
    spx_exp_next = spx_monthly_expiration(y2, m2)
    return spx_exp_next - pd.Timedelta(days=30)

def next_vix_monthly_settlement(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Next VIX monthly settlement date on/after 'ts' month.
    If the current month's VIX settlement is already past 'ts' (by date),
    roll to next month.
    """
    ts = pd.Timestamp(ts)
    cand = vix_monthly_settlement_for_month(ts.year, ts.month)
    if ts.normalize() > cand.normalize():
        # move to the following month
        y, m = (ts.year + 1, 1) if ts.month == 12 else (ts.year, ts.month + 1)
        cand = vix_monthly_settlement_for_month(y, m)
    return cand

def next_two_vix_settlements(ts: pd.Timestamp):
    ts = pd.Timestamp(ts)
    # First candidate
    first = vix_monthly_settlement_for_month(ts.year, ts.month)
    if ts.normalize() > first.normalize():
        # Move to next month if already passed
        y, m = (ts.year + 1, 1) if ts.month == 12 else (ts.year, ts.month + 1)
        first = vix_monthly_settlement_for_month(y, m)
        # For second, add one more month
        y2, m2 = (y + 1, 1) if m == 12 else (y, m + 1)
        second = vix_monthly_settlement_for_month(y2, m2)
    else:
        # If first is still upcoming, second is next month
        y, m = (ts.year + 1, 1) if ts.month == 12 else (ts.year, ts.month + 1)
        second = vix_monthly_settlement_for_month(y, m)
    return first, second

def add_vix_next_expiration(df, col_name: str):
    """
    Add a new column with the date of the next CBOE VIX future expiration date.
    """
    cols_initial = set(copy.deepcopy(df.columns))

    # Option A: calendar-day difference (floor). Same date => 0.
    df['days_to_next_vix_exp'] = (
        df.index.to_series().apply(next_vix_monthly_settlement) - df.index.normalize()
    ).dt.days

    # Option B: round UP partial days to next integer day
    delta = df.index.to_series().apply(next_vix_monthly_settlement) - df.index
    df['days_to_next_vix_exp_ceil'] = np.ceil(delta / pd.Timedelta(days=1)).astype(int)

    # (Optional) If you also want the actual settlement date as a column:
    col_name = col_name or "next_vix_expiration_date"
    cols_initial.add(col_name)
    df[col_name] = df.index.to_series().apply(next_vix_monthly_settlement)

    cols_to_drop = set(df.columns) - cols_initial
    df.drop(columns = cols_to_drop, inplace=True)

    return df

def add_days_to_next_two_expiries(df, remove_dates: bool = True):
    # Compute next two expiries and days until each
    df[['next_vix_exp', 'second_vix_exp']] = df.index.to_series().apply(lambda d: pd.Series(next_two_vix_settlements(d)))
    df["days_to_first_exp"] = (df["next_vix_exp"] - df.index).dt.days
    df["days_to_second_exp"] = (df["second_vix_exp"] - df.index).dt.days
    df["duration_second_fut"] = df["days_to_second_exp"] - df["days_to_first_exp"]

    if remove_dates:
        df.drop(columns=['next_vix_exp', 'second_vix_exp'], inplace=True)
    return df



def add_percentiles(df, col_name, percentiles: list):
    for p in percentiles:
        df[f"{col_name}_pct_{p}"] = df[col_name].expanding().quantile(p)
    return df


def load_dataset(dir_data, f_config):

    with open(f_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    files, dfs = os.listdir(dir_data), []
    for file in files:
        if cfg[str(file).split(".")[0].split("_")[1]]:
            dfs.append(pd.read_csv(os.path.join(dir_data, file)))
    
    # Extract the first column (assumed to be the same across all dataframes)
    first_column = dfs[0].iloc[:, 0]

    # Multiply the remaining columns element-wise
    result = dfs[0].iloc[:, 1:]
    for df in dfs[1:]:
        result *= df.iloc[:, 1:]

    # Combine the first column with the result
    df = pd.concat([first_column, result*cfg["scale"]], axis=1)
    df.columns = cfg["cols"]
    return df

def clean_xl_dates(df, col, set_index: bool = True):
    df[col] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df[col].astype(int), unit='D')
    if set_index:
        df.set_index(col, drop=True, inplace=True)
    return df