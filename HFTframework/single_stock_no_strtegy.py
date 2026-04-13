import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from datetime import datetime
import re
from single_trade_signal import calculate_signal

# -------------------------
# Load + clean your L2 ticks
# -------------------------
def _read_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)

def _quick_missing_check(df):
    has_missing = df.isnull().any().any()

    if has_missing:
        missing_cols = df.columns[df.isnull().any()].tolist()
        print(f"Missing values found in {len(missing_cols)} columns: {missing_cols}")
        return True
    else:
        print("No missing values found.")
        return False


def infer_missing_sides_with_lee_ready(df):

    original_order = df.index

    df['side'] = df['side'].astype(str).str.upper().str.strip()
    need_infer_mask = df['side'].isin(['', '-', 'N/A', 'NAN', 'NONE', 'UNKNOWN'])

    if not need_infer_mask.any():
        return df

    df = df.sort_values("timestamp").copy()

    df['mid_price'] = (df['bid1'] + df['ask1']) / 2
    df['price_change'] = df['last'].diff()

    inferred = pd.Series('', index=df.index)

    # Lee-Ready
    mask_up = df['price_change'] > 0
    mask_down = df['price_change'] < 0
    mask_stable_above = (df['price_change'] == 0) & (df['last'] > df['mid_price'])
    mask_stable_below = (df['price_change'] == 0) & (df['last'] < df['mid_price'])
    mask_stable_equal = (df['price_change'] == 0) & (df['last'] == df['mid_price'])

    inferred[mask_up] = 'B'
    inferred[mask_down] = 'S'
    inferred[mask_stable_above] = 'B'
    inferred[mask_stable_below] = 'S'

    # price=mid
    if mask_stable_equal.any():
        valid_sides = df['side'].where(~df['side'].isin(['', '-', 'N/A', 'NAN', 'NONE', 'UNKNOWN']))
        # ffill
        filled_sides = valid_sides.ffill()
        inferred[mask_stable_equal] = filled_sides[mask_stable_equal]


    inferred = inferred.fillna('U')
    df.loc[need_infer_mask, 'side'] = inferred[need_infer_mask]
    df = df.drop(columns=['mid_price', 'price_change'])
    df = df.loc[original_order]

    return df


def clean_price_anomalies(df, max_pct_change=0.2):
    """Clean price anomalies using percentage change threshold"""
    df_cleaned = df.copy()
    pct_change = df_cleaned['last'].pct_change().abs()
    anomaly_mask = (pct_change > max_pct_change) | (df_cleaned['last'] == 0)

    print(f"Found {anomaly_mask.sum()} anomalies (> {max_pct_change * 100}% change)")

    df_cleaned.loc[anomaly_mask, 'last'] = np.nan
    df_cleaned['last'] = df_cleaned['last'].ffill().bfill()
    df['last'] = df_cleaned['last']

    return df


def load_l2_ticks(file_path: str) -> pd.DataFrame:
    raw = _read_csv(file_path)

    # Drop repeated header rows inside file
    time_col = raw.columns[0]
    raw = raw[raw[time_col].astype(str) != time_col].copy()

    cols = list(raw.columns)

    # Base columns (per your sample)
    rename_map = {
        cols[0]: "time",
        cols[1]: "last",
        cols[2]: "side",        # may be 'B', 'S', '-' etc.
        cols[3]: "trade_vol",
        cols[4]: "trade_cnt",
        cols[5]: "trade_amt",
    }
    df = raw.rename(columns=rename_map)

    # Numeric conversion (robust to '-' / blanks)
    for c in ["last", "trade_vol", "trade_cnt", "trade_amt"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Timestamp (files show HH:MM:SS; use dummy date)
    date_match = re.search(r'\\(\d{8})\\', file_path)
    if date_match:
        date_str = date_match.group(1)
        file_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        print(file_date)

        df["timestamp"] = pd.to_datetime(file_date + " " + df["time"].astype(str),
                                         format='%Y-%m-%d %H:%M:%S',
                                         errors="coerce")

    else:
        print(f"Warning: Could not extract date from path: {file_path}")

    # Remaining columns are L2 book. Assumption: 5 bid levels then 5 ask levels, (px,sz) pairs.
    ob_cols = cols[6:]

    # Build bid1..bid5, ask1..ask5 and sizes
    for lvl in range(1, 6):
        bid_px_col = ob_cols[(lvl - 1) * 2 + 0]
        bid_sz_col = ob_cols[(lvl - 1) * 2 + 1]
        ask_px_col = ob_cols[10 + (lvl - 1) * 2 + 0]
        ask_sz_col = ob_cols[10 + (lvl - 1) * 2 + 1]

        df[f"bid{lvl}"] = pd.to_numeric(raw[bid_px_col], errors="coerce")
        df[f"bid{lvl}_sz"] = pd.to_numeric(raw[bid_sz_col], errors="coerce")
        df[f"ask{lvl}"] = pd.to_numeric(raw[ask_px_col], errors="coerce")
        df[f"ask{lvl}_sz"] = pd.to_numeric(raw[ask_sz_col], errors="coerce")

    # Keep valid rows
    df = df.dropna(subset=["timestamp", "last", "bid1", "ask1"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Missing Value
    ## Lee-Ready->side
    df = infer_missing_sides_with_lee_ready(df)

    # Clean & Agg
    sz_cols = [col for col in df.columns if '_sz' in col]
    for col in sz_cols:
        df[col] = df[col].replace(0, pd.NA)
        df[col] = df[col].ffill()
        df[col] = df[col].fillna(0)
    df = df.groupby('timestamp').last().reset_index()

    keep_cols = ["timestamp", "time", "last", "side", "trade_vol", "trade_cnt", "trade_amt"]
    for lvl in range(1, 6):
        keep_cols.extend([f"bid{lvl}", f"bid{lvl}_sz", f"ask{lvl}", f"ask{lvl}_sz"])

    df = df[keep_cols]

    print("After data cleaning...")
    _quick_missing_check(df)
    df = df.iloc[60:-60].reset_index(drop=True)

    return df


def fill_missing_features(df):
    df = df.ffill()
    df = df.bfill()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df

# -------------------------
# Feature engineering (NO strategy)
# -------------------------
def add_l2_and_orderflow_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # L1 market state
    df["mid"] = (df["bid1"] + df["ask1"]) / 2.0
    df["spread"] = df["ask1"] - df["bid1"]

    # Depth (L1-L5)
    df["bid_depth"] = sum(df[f"bid{lvl}_sz"] for lvl in range(1, 6))
    df["ask_depth"] = sum(df[f"ask{lvl}_sz"] for lvl in range(1, 6))
    df["depth_imbalance"] = (df["bid_depth"] - df["ask_depth"]) / (df["bid_depth"] + df["ask_depth"] + 1e-9)

    # Microprice (L1 size-weighted)
    df["microprice"] = (df["ask1"] * df["bid1_sz"] + df["bid1"] * df["ask1_sz"]) / (df["bid1_sz"] + df["ask1_sz"] + 1e-9)
    df["mp_minus_mid"] = df["microprice"] - df["mid"]

    # Returns / volatility
    df["mid_ret"] = df["mid"].pct_change()
    df["mid_ret_1s_proxy"] = df["mid"].diff()  # simple diff, since timestamps may be irregular
    df["mid_vol_roll"] = df["mid_ret"].rolling(200).std(ddof=0)

    # -------------------------
    # Order flow proxies (neutral)
    # -------------------------
    side = df["side"].astype(str).str.upper()
    df["trade_sign"] = np.where(side.str.contains("B"), 1,
                         np.where(side.str.contains("S"), -1, 0))

    # Signed trade volume
    df["signed_vol"] = df["trade_sign"] * df["trade_vol"].fillna(0)

    ######## Add #########
    # Cumulative signed volume
    df["cumulative_signed_vol"] = df["signed_vol"].cumsum()
    df["cumulative_signed_vol_ma"] = df["cumulative_signed_vol"].rolling(100).mean()

    # Trade flow imbalance (volume weighted)
    df["tfi"] = df["signed_vol"].rolling(20).sum() / (df["trade_vol"].rolling(20).sum() + 1e-9)

    # Trade size statistics
    df["trade_size_mean"] = df["trade_vol"].rolling(20).mean()
    df["trade_size_std"] = df["trade_vol"].rolling(20).std()
    df["trade_size_zscore"] = (df["trade_vol"] - df["trade_size_mean"]) / (df["trade_size_std"] + 1e-9)
    ######## Add #########

    # 2) OFI (Order Flow Imbalance) from L1 quote updates (Cont et al.-style simplification)
    #    Uses changes in bid/ask price and size.
    bpx, apx = df["bid1"], df["ask1"]
    bsz, asz = df["bid1_sz"].fillna(0), df["ask1_sz"].fillna(0)

    bpx_prev, apx_prev = bpx.shift(1), apx.shift(1)
    bsz_prev, asz_prev = bsz.shift(1), asz.shift(1)

    bid_contrib = np.where(bpx > bpx_prev, bsz,
                    np.where(bpx < bpx_prev, -bsz_prev, bsz - bsz_prev))
    ask_contrib = np.where(apx < apx_prev, asz,
                    np.where(apx > apx_prev, -asz_prev, asz - asz_prev))

    df["ofi_l1"] = bid_contrib - ask_contrib
    df["ofi_l1_roll"] = pd.Series(df["ofi_l1"]).rolling(200).sum()

    ######## Add #########
    # OFI for different windows
    df["ofi_10"] = pd.Series(df["ofi_l1"]).rolling(10).sum()
    df["ofi_50"] = pd.Series(df["ofi_l1"]).rolling(50).sum()
    df["ofi_ratio"] = df["ofi_10"] / (df["ofi_50"].abs() + 1e-9)

    # OFI momentum
    df["ofi_momentum"] = df["ofi_l1_roll"].diff(5)

    # Normalized OFI
    ofi_mean = df["ofi_l1"].rolling(200).mean()
    ofi_std = df["ofi_l1"].rolling(200).std()
    df["ofi_zscore"] = (df["ofi_l1"] - ofi_mean) / (ofi_std + 1e-9)

    # Relative spread (spread as percentage of mid_price)
    df["relative_spread"] = df["spread"] / df["mid"]

    # Weighted price levels (closer levels have more weight)
    weights = [0.4, 0.3, 0.2, 0.07, 0.03]  # weights for levels 1-5
    weighted_bid = sum(df[f"bid{lvl}"] * weights[lvl - 1] for lvl in range(1, 6))
    weighted_ask = sum(df[f"ask{lvl}"] * weights[lvl - 1] for lvl in range(1, 6))
    df["weighted_mid"] = (weighted_bid + weighted_ask) / 2.0
    df["weighted_spread"] = weighted_ask - weighted_bid

    # Enhanced microprice using L1-L3
    top3_bid_sz = sum(df[f"bid{lvl}_sz"] for lvl in range(1, 4))
    top3_ask_sz = sum(df[f"ask{lvl}_sz"] for lvl in range(1, 4))
    top3_bid_price = sum(df[f"bid{lvl}"] * df[f"bid{lvl}_sz"] for lvl in range(1, 4)) / (top3_bid_sz + 1e-9)
    top3_ask_price = sum(df[f"ask{lvl}"] * df[f"ask{lvl}_sz"] for lvl in range(1, 4)) / (top3_ask_sz + 1e-9)
    df["microprice_top3"] = (top3_ask_price * top3_bid_sz + top3_bid_price * top3_ask_sz) / (
                top3_bid_sz + top3_ask_sz + 1e-9)
    df["mp_top3_minus_mid"] = df["microprice_top3"] - df["mid"]

    # Price acceleration (second derivative)
    df["mid_acceleration"] = df["mid"].diff().diff()

    # Realized volatility for different windows
    df["vol_10"] = df["mid_ret"].rolling(10).std(ddof=0)
    df["vol_50"] = df["mid_ret"].rolling(50).std(ddof=0)
    df["vol_200"] = df["mid_ret"].rolling(200).std(ddof=0)
    df["vol_ratio_10_50"] = df["vol_10"] / (df["vol_50"] + 1e-9)

    # Price momentum features
    df["momentum_5"] = df["mid"].pct_change(5)
    df["momentum_10"] = df["mid"].pct_change(10)
    df["momentum_20"] = df["mid"].pct_change(20)
    df["momentum_ratio"] = df["momentum_5"] / (df["momentum_20"].abs() + 1e-9)

    # Price level features
    df["high_5"] = df["mid"].rolling(5).max()
    df["low_5"] = df["mid"].rolling(5).min()
    df["high_low_range_5"] = df["high_5"] - df["low_5"]
    df["mid_position_5"] = (df["mid"] - df["low_5"]) / (df["high_low_range_5"] + 1e-9)

    df["high_20"] = df["mid"].rolling(20).max()
    df["low_20"] = df["mid"].rolling(20).min()
    df["high_low_range_20"] = df["high_20"] - df["low_20"]
    df["mid_position_20"] = (df["mid"] - df["low_20"]) / (df["high_low_range_20"] + 1e-9)

    # Volume imbalance (difference between bid and ask sizes at L1)
    df["size_imbalance_l1"] = (df["bid1_sz"] - df["ask1_sz"]) / (df["bid1_sz"] + df["ask1_sz"] + 1e-9)
    df["size_imbalance_l1_ma"] = df["size_imbalance_l1"].rolling(20).mean()

    # Price impact proxy
    df["price_impact"] = df["mid_ret"] * df["signed_vol"].abs()
    df["price_impact_ma"] = df["price_impact"].rolling(20).mean()

    # Liquidity features
    df["liquidity_ratio"] = (df["bid1_sz"] + df["ask1_sz"]) / (df["bid_depth"] + df["ask_depth"] + 1e-9)
    df["liquidity_slope"] = (df["bid5"] - df["bid1"]) - (df["ask5"] - df["ask1"])  # Order book slope

    # Market state features
    df["is_spread_tight"] = df["spread"] < df["spread"].rolling(50).quantile(0.3)
    df["is_spread_wide"] = df["spread"] > df["spread"].rolling(50).quantile(0.7)
    df["is_vol_high"] = df["vol_10"] > df["vol_10"].rolling(200).quantile(0.7)
    df["is_vol_low"] = df["vol_10"] < df["vol_10"].rolling(200).quantile(0.3)

    # Correlation between features
    df["depth_spread_corr"] = df["depth_imbalance"].rolling(20).corr(df["spread"])
    df["ofi_mid_corr"] = df["ofi_l1_roll"].rolling(20).corr(df["mid_ret"])

    # Order book pressure gradient
    df['pressure_gradient'] = (
            (df['bid1_sz'] - df['bid5_sz']) / 4 -
            (df['ask5_sz'] - df['ask1_sz']) / 4
    )

    # Large order marking
    df['large_trade_flag'] = (df['trade_vol'] > df['trade_vol'].rolling(100).quantile(0.9)).astype(int)

    # Order processing speed
    df['ofi_acceleration'] = df['ofi_l1'].diff(3)
    df['ofi_jerk'] = df['ofi_acceleration'].diff(3)

    # Depth change rate
    df['depth_change_rate'] = df['depth_imbalance'].diff(3)

    # Volume-price divergence
    df['price_volume_divergence'] = (
            df['mid_ret'].abs() * 1000 -
            df['trade_vol'] / df['trade_vol'].rolling(50).mean()
    )

    # The momentum of imbalance in the order book
    df['depth_imbalance_momentum'] = df['depth_imbalance'].diff(5)

    # Price spread regression
    df['spread_reversion'] = df['spread'] - df['spread'].rolling(20).mean()

    # vwap
    df['vwap'] = df['trade_amt'] / (df['trade_vol'] + 1e-9)

    # Calculate the average transaction size
    df['avg_amount_per_trade'] = df['trade_amt'] / (df['trade_cnt'] + 1e-9)
    df['avg_volume_per_trade'] = df['trade_vol'] / (df['trade_cnt'] + 1e-9)

    # Analysis of Capital Flow Direction
    df['buy_amount'] = np.where(df['side'] == 'B', df['trade_amt'], 0)
    df['sell_amount'] = np.where(df['side'] == 'S', df['trade_amt'], 0)
    df['net_amount_flow'] = df['buy_amount'] - df['sell_amount']

    # Detecting abnormal large orders
    large_trade_threshold = 1000000
    df['is_large_trade'] = (df['trade_amt'] > large_trade_threshold).astype(int)

    # Calculate the cost of market impact
    df['price_impact_v2'] = (df['last'] - df['last'].shift(1)) / (df['last'].shift(1) + 1e-9)
    df['impact_per_amount'] = df['price_impact_v2'] / (df['trade_amt'] / 1e6 + 1e-9)
    ######## Add #########


    print("After OB feature engineering...")
    df = fill_missing_features(df)
    _quick_missing_check(df)

    return df


def add_low_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add low-frequency features using only L1 data
    No order book depth information, only price and volume
    """
    df = df.copy()

    # 1. BASIC PRICE AND VOLUME
    if 'mid' not in df.columns:
        df['mid'] = (df['bid1'] + df['ask1']) / 2.0

    if 'spread' not in df.columns:
        df['spread'] = df['ask1'] - df['bid1']

    if 'relative_spread' not in df.columns:
        df['relative_spread'] = df['spread'] / df['mid']

    # 2. MINUTE-LEVEL FEATURES
    # Create minute timestamps
    df['timestamp_min'] = df.index.floor('1min')
    df['timestamp_3min'] = df.index.floor('3min')
    df['timestamp_5min'] = df.index.floor('5min')

    # 2.1 1-MINUTE FEATURES
    # Rolling features within 1-minute windows
    window_1min = 60  # Approx 20 ticks for 1-min

    # Price statistics
    df['mid_1min_ma'] = df.groupby('timestamp_min')['mid'].transform(
        lambda x: x.rolling(window_1min, min_periods=5).mean()
    )
    df['mid_1min_std'] = df.groupby('timestamp_min')['mid'].transform(
        lambda x: x.rolling(window_1min, min_periods=5).std()
    )
    df['mid_1min_high'] = df.groupby('timestamp_min')['mid'].transform(
        lambda x: x.rolling(window_1min, min_periods=5).max()
    )
    df['mid_1min_low'] = df.groupby('timestamp_min')['mid'].transform(
        lambda x: x.rolling(window_1min, min_periods=5).min()
    )
    df['mid_1min_range'] = df['mid_1min_high'] - df['mid_1min_low']
    df['mid_position_1min'] = (df['mid'] - df['mid_1min_low']) / (df['mid_1min_range'] + 1e-9)

    # Volume statistics
    if 'trade_vol' in df.columns:
        df['volume_1min_ma'] = df.groupby('timestamp_min')['trade_vol'].transform(
            lambda x: x.rolling(window_1min, min_periods=5).mean()
        )
        df['volume_1min_sum'] = df.groupby('timestamp_min')['trade_vol'].transform(
            lambda x: x.rolling(window_1min, min_periods=5).sum()
        )

        # Price-Volume correlation (1min)
        df['price_volume_corr_1min'] = df.groupby('timestamp_min').apply(
            lambda x: x['mid'].rolling(window_1min, min_periods=5).corr(x['trade_vol'])
        ).reset_index(level=0, drop=True)

    # Returns
    df['ret_1min'] = df.groupby('timestamp_min')['mid'].transform(
        lambda x: x.pct_change(window_1min)
    )

    # 2.2 3-MINUTE FEATURES
    window_3min = 60  # Approx 60 ticks for 3-min

    df['mid_3min_ma'] = df.groupby('timestamp_3min')['mid'].transform(
        lambda x: x.rolling(window_3min, min_periods=15).mean()
    )
    df['mid_3min_std'] = df.groupby('timestamp_3min')['mid'].transform(
        lambda x: x.rolling(window_3min, min_periods=15).std()
    )
    df['mid_3min_high'] = df.groupby('timestamp_3min')['mid'].transform(
        lambda x: x.rolling(window_3min, min_periods=15).max()
    )
    df['mid_3min_low'] = df.groupby('timestamp_3min')['mid'].transform(
        lambda x: x.rolling(window_3min, min_periods=15).min()
    )
    df['mid_3min_range'] = df['mid_3min_high'] - df['mid_3min_low']
    df['mid_position_3min'] = (df['mid'] - df['mid_3min_low']) / (df['mid_3min_range'] + 1e-9)

    if 'trade_vol' in df.columns:
        df['price_volume_corr_3min'] = df.groupby('timestamp_3min').apply(
            lambda x: x['mid'].rolling(window_3min, min_periods=15).corr(x['trade_vol'])
        ).reset_index(level=0, drop=True)

    df['ret_3min'] = df.groupby('timestamp_3min')['mid'].transform(
        lambda x: x.pct_change(window_3min)
    )

    # 2.3 5-MINUTE FEATURES
    window_5min = 100  # Approx 100 ticks for 5-min

    df['mid_5min_ma'] = df.groupby('timestamp_5min')['mid'].transform(
        lambda x: x.rolling(window_5min, min_periods=25).mean()
    )
    df['mid_5min_std'] = df.groupby('timestamp_5min')['mid'].transform(
        lambda x: x.rolling(window_5min, min_periods=25).std()
    )
    df['mid_5min_high'] = df.groupby('timestamp_5min')['mid'].transform(
        lambda x: x.rolling(window_5min, min_periods=25).max()
    )
    df['mid_5min_low'] = df.groupby('timestamp_5min')['mid'].transform(
        lambda x: x.rolling(window_5min, min_periods=25).min()
    )
    df['mid_5min_range'] = df['mid_5min_high'] - df['mid_5min_low']
    df['mid_position_5min'] = (df['mid'] - df['mid_5min_low']) / (df['mid_5min_range'] + 1e-9)

    if 'trade_vol' in df.columns:
        df['price_volume_corr_5min'] = df.groupby('timestamp_5min').apply(
            lambda x: x['mid'].rolling(window_5min, min_periods=25).corr(x['trade_vol'])
        ).reset_index(level=0, drop=True)

    df['ret_5min'] = df.groupby('timestamp_5min')['mid'].transform(
        lambda x: x.pct_change(window_5min)
    )

    # 3. MEAN REVERSION FEATURES
    # Distance to moving averages
    df['dist_to_ma_1min'] = (df['mid'] - df['mid_1min_ma']) / df['mid_1min_ma']
    df['dist_to_ma_3min'] = (df['mid'] - df['mid_3min_ma']) / df['mid_3min_ma']
    df['dist_to_ma_5min'] = (df['mid'] - df['mid_5min_ma']) / df['mid_5min_ma']

    # Z-score (standardized distance)
    df['zscore_1min'] = (df['mid'] - df['mid_1min_ma']) / (df['mid_1min_std'] + 1e-9)
    df['zscore_3min'] = (df['mid'] - df['mid_3min_ma']) / (df['mid_3min_std'] + 1e-9)
    df['zscore_5min'] = (df['mid'] - df['mid_5min_ma']) / (df['mid_5min_std'] + 1e-9)

    # 4. CHANNEL BREAKOUT FEATURES
    # Bollinger Band style
    df['bb_upper_1min'] = df['mid_1min_ma'] + 2 * df['mid_1min_std']
    df['bb_lower_1min'] = df['mid_1min_ma'] - 2 * df['mid_1min_std']
    df['bb_position_1min'] = (df['mid'] - df['bb_lower_1min']) / (df['bb_upper_1min'] - df['bb_lower_1min'] + 1e-9)

    # Range breakout
    df['atr_1min'] = df.groupby('timestamp_min')['mid'].transform(
        lambda x: (x.rolling(window_1min, min_periods=5).max() -
                   x.rolling(window_1min, min_periods=5).min()).rolling(window_1min, min_periods=5).mean()
    )

    # 5. MOMENTUM FEATURES
    # RSI-like momentum
    df['up_move_1min'] = df.groupby('timestamp_min')['mid'].transform(
        lambda x: x.diff().apply(lambda y: y if y > 0 else 0)
    )
    df['down_move_1min'] = df.groupby('timestamp_min')['mid'].transform(
        lambda x: x.diff().apply(lambda y: -y if y < 0 else 0)
    )
    df['avg_up_1min'] = df.groupby('timestamp_min')['up_move_1min'].transform(
        lambda x: x.rolling(window_1min, min_periods=5).mean()
    )
    df['avg_down_1min'] = df.groupby('timestamp_min')['down_move_1min'].transform(
        lambda x: x.rolling(window_1min, min_periods=5).mean()
    )
    df['rs_1min'] = df['avg_up_1min'] / (df['avg_down_1min'] + 1e-9)
    df['rsi_1min'] = 100 - (100 / (1 + df['rs_1min']))

    # 6. VOLUME-BASED FEATURES
    if 'trade_vol' in df.columns:
        # Volume ratio (current vs average)
        df['volume_ratio_1min'] = df['trade_vol'] / (df['volume_1min_ma'] + 1e-9)

        # Volume spike detection
        df['volume_zscore_1min'] = (df['trade_vol'] - df['volume_1min_ma']) / (
                    df.groupby('timestamp_min')['trade_vol'].transform(
                        lambda x: x.rolling(window_1min, min_periods=5).std()) + 1e-9)

        # On-Balance Volume (simplified)
        df['obv_signal'] = np.where(df['mid'].diff() > 0, df['trade_vol'],
                                    np.where(df['mid'].diff() < 0, -df['trade_vol'], 0))
        df['obv_1min'] = df.groupby('timestamp_min')['obv_signal'].transform(
            lambda x: x.rolling(window_1min, min_periods=5).sum()
        )

    # 7. SPREAD-RELATED FEATURES
    df['spread_1min_ma'] = df.groupby('timestamp_min')['spread'].transform(
        lambda x: x.rolling(window_1min, min_periods=5).mean()
    )
    df['spread_zscore_1min'] = (df['spread'] - df['spread_1min_ma']) / (df.groupby('timestamp_min')['spread'].transform(
        lambda x: x.rolling(window_1min, min_periods=5).std()) + 1e-9)

    # 8. PRICE ACCELERATION (higher timeframe)
    df['accel_1min'] = df.groupby('timestamp_min')['mid'].transform(
        lambda x: x.diff().diff()
    )
    df['accel_3min'] = df.groupby('timestamp_3min')['mid'].transform(
        lambda x: x.diff(window_3min // 10).diff(window_3min // 10)
    )

    # 9. SUPPORT/RESISTANCE LEVELS
    # Recent highs and lows
    df['recent_high_1min'] = df.groupby('timestamp_min')['mid'].transform(
        lambda x: x.rolling(window_1min * 2, min_periods=10).max()
    )
    df['recent_low_1min'] = df.groupby('timestamp_min')['mid'].transform(
        lambda x: x.rolling(window_1min * 2, min_periods=10).min()
    )
    df['dist_to_recent_high'] = (df['recent_high_1min'] - df['mid']) / df['mid']
    df['dist_to_recent_low'] = (df['mid'] - df['recent_low_1min']) / df['mid']

    # 10. TREND STRENGTH
    df['trend_strength_1min'] = abs(df['ret_1min']) / (df['mid_1min_std'] + 1e-9)
    df['trend_strength_3min'] = abs(df['ret_3min']) / (df['mid_3min_std'] + 1e-9)

    # Remove temporary columns
    cols_to_drop = ['timestamp_min', 'timestamp_3min', 'timestamp_5min',
                    'up_move_1min', 'down_move_1min', 'avg_up_1min', 'avg_down_1min',
                    'obv_signal']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    print("After LF feature engineering...")
    df = fill_missing_features(df)
    _quick_missing_check(df)

    return df


def add_rolling_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling volatility features for different time windows
    Focus on realized volatility and volatility regimes
    """
    df = df.copy()

    # Ensure mid returns exist
    if 'mid_ret' not in df.columns:
        df['mid_ret'] = df['mid'].pct_change()

    # 1. REALIZED VOLATILITY (Standard deviation of returns)
    df['realized_vol_5s'] = df['mid_ret'].rolling(5).std(ddof=0)
    df['realized_vol_10s'] = df['mid_ret'].rolling(10).std(ddof=0)
    df['realized_vol_30s'] = df['mid_ret'].rolling(30).std(ddof=0)
    df['realized_vol_1min'] = df['mid_ret'].rolling(60).std(ddof=0)
    df['realized_vol_3min'] = df['mid_ret'].rolling(180).std(ddof=0)
    df['realized_vol_5min'] = df['mid_ret'].rolling(300).std(ddof=0)
    df['realized_vol_10min'] = df['mid_ret'].rolling(600).std(ddof=0)
    df['realized_vol_15min'] = df['mid_ret'].rolling(900).std(ddof=0)

    # 2. VOLATILITY RATIOS (Term structure)
    df['vol_ratio_5s_30s'] = df['realized_vol_5s'] / (df['realized_vol_30s'] + 1e-9)
    df['vol_ratio_10s_1min'] = df['realized_vol_10s'] / (df['realized_vol_1min'] + 1e-9)
    df['vol_ratio_30s_3min'] = df['realized_vol_30s'] / (df['realized_vol_3min'] + 1e-9)
    df['vol_ratio_1min_5min'] = df['realized_vol_1min'] / (df['realized_vol_5min'] + 1e-9)
    df['vol_ratio_3min_10min'] = df['realized_vol_3min'] / (df['realized_vol_10min'] + 1e-9)
    df['vol_ratio_5min_15min'] = df['realized_vol_5min'] / (df['realized_vol_15min'] + 1e-9)

    # 3. VOLATILITY MOMENTUM (Changes in volatility)
    df['vol_momentum_5s'] = df['realized_vol_5s'].pct_change(5)
    df['vol_momentum_30s'] = df['realized_vol_30s'].pct_change(10)
    df['vol_momentum_1min'] = df['realized_vol_1min'].pct_change(20)
    df['vol_momentum_5min'] = df['realized_vol_5min'].pct_change(50)
    df['vol_accel_30s'] = df['realized_vol_30s'].diff().diff()
    df['vol_accel_1min'] = df['realized_vol_1min'].diff().diff()

    # 4. VOLATILITY REGIME DETECTION
    df['vol_percentile_30s'] = df['realized_vol_30s'].rolling(200).rank(pct=True)
    df['vol_percentile_1min'] = df['realized_vol_1min'].rolling(200).rank(pct=True)
    df['vol_percentile_5min'] = df['realized_vol_5min'].rolling(200).rank(pct=True)

    df['is_high_vol_30s'] = (df['vol_percentile_30s'] > 0.7).astype(int)
    df['is_low_vol_30s'] = (df['vol_percentile_30s'] < 0.3).astype(int)
    df['is_high_vol_1min'] = (df['vol_percentile_1min'] > 0.7).astype(int)
    df['is_low_vol_1min'] = (df['vol_percentile_1min'] < 0.3).astype(int)

    # 5. VOLATILITY MEAN REVERSION
    df['vol_zscore_30s'] = (
            (df['realized_vol_30s'] - df['realized_vol_30s'].rolling(100).mean()) /
            (df['realized_vol_30s'].rolling(100).std() + 1e-9)
    )
    df['vol_zscore_1min'] = (
            (df['realized_vol_1min'] - df['realized_vol_1min'].rolling(100).mean()) /
            (df['realized_vol_1min'].rolling(100).std() + 1e-9)
    )
    df['vol_zscore_5min'] = (
            (df['realized_vol_5min'] - df['realized_vol_5min'].rolling(100).mean()) /
            (df['realized_vol_5min'].rolling(100).std() + 1e-9)
    )

    # 6. VOLATILITY CLUSTERING (Fixed autocorrelation calculation)
    df['vol_cluster_5'] = df['mid_ret'].pow(2).rolling(5).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 2 else 0, raw=False
    )
    df['vol_cluster_10'] = df['mid_ret'].pow(2).rolling(10).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 2 else 0, raw=False
    )
    df['vol_cluster_20'] = df['mid_ret'].pow(2).rolling(20).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 2 else 0, raw=False
    )

    # 7. VOLATILITY SMILE/SKEW
    df['up_vol_30s'] = df['mid_ret'].where(df['mid_ret'] > 0, 0).rolling(30).std(ddof=0)
    df['down_vol_30s'] = (-df['mid_ret']).where(df['mid_ret'] < 0, 0).rolling(30).std(ddof=0)
    df['up_down_vol_ratio_30s'] = df['up_vol_30s'] / (df['down_vol_30s'] + 1e-9)

    df['up_vol_1min'] = df['mid_ret'].where(df['mid_ret'] > 0, 0).rolling(60).std(ddof=0)
    df['down_vol_1min'] = (-df['mid_ret']).where(df['mid_ret'] < 0, 0).rolling(60).std(ddof=0)
    df['up_down_vol_ratio_1min'] = df['up_vol_1min'] / (df['down_vol_1min'] + 1e-9)

    # 8. VOLATILITY OF VOLATILITY (VoV)
    df['vov_30s'] = df['realized_vol_30s'].rolling(30).std(ddof=0)
    df['vov_1min'] = df['realized_vol_1min'].rolling(60).std(ddof=0)
    df['vov_5min'] = df['realized_vol_5min'].rolling(100).std(ddof=0)

    # 9. VOLATILITY BREAKOUT DETECTION
    df['vol_expansion_30s'] = (df['realized_vol_30s'] > df['realized_vol_30s'].rolling(20).max()).astype(int)
    df['vol_contraction_30s'] = (df['realized_vol_30s'] < df['realized_vol_30s'].rolling(20).min()).astype(int)
    df['vol_expansion_1min'] = (df['realized_vol_1min'] > df['realized_vol_1min'].rolling(20).max()).astype(int)
    df['vol_contraction_1min'] = (df['realized_vol_1min'] < df['realized_vol_1min'].rolling(20).min()).astype(int)

    # 10. VOLATILITY-SPREAD RELATIONSHIP
    if 'spread' in df.columns:
        df['vol_spread_corr_30s'] = df['realized_vol_30s'].rolling(30).corr(df['spread'])
        df['vol_spread_corr_1min'] = df['realized_vol_1min'].rolling(60).corr(df['spread'])

    print("After Vol feature engineering...")
    df = fill_missing_features(df)
    _quick_missing_check(df)

    return df


def filter_premarket_signals(df, target_pos, debug=True):
    """
    Filter out trading signals during 9:00-9:30 pre-market period
    """
    filtered_signals = target_pos.copy()

    # Check if timestamp column exists
    if 'timestamp' not in df.columns:
        if debug:
            print("  ⚠️ Cannot filter pre-market: Missing timestamp column")
        return filtered_signals

    try:
        # Ensure timestamp is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            timestamps = pd.to_datetime(df['timestamp'])
        else:
            timestamps = df['timestamp']

        # Create time mask: 9:00-9:30
        # Use .dt.time to access time part
        premarket_mask = (timestamps.dt.time >= pd.Timestamp('09:00:00').time()) & \
                         (timestamps.dt.time < pd.Timestamp('09:30:00').time())

        # Count filtered signals
        filtered_mask = premarket_mask & (filtered_signals != 0)
        filtered_count = filtered_mask.sum()

        if filtered_count > 0 and debug:
            print(f"  Filtered pre-market signals: {filtered_count}")
            # Show first few filtered signals
            filtered_indices = filtered_mask[filtered_mask].index
            for i, idx in enumerate(filtered_indices[:3]):  # Show first 3 only
                if idx < len(timestamps):
                    time_str = timestamps.iloc[idx]
                    signal_val = filtered_signals.iloc[idx]
                    print(f"    Time {time_str}: Signal {signal_val} → 0")
        else:
            print("    No signals in pre-market period")

        # Set non-zero signals to 0 during pre-market
        filtered_signals.loc[filtered_mask] = 0

    except Exception as e:
        if debug:
            print(f"  ⚠️ Pre-market filtering failed: {e}")

    return filtered_signals


def generate_signals(df: pd.DataFrame, signal_name,
                     min_hold_ticks=300,
                     max_hold_ticks=300,
                     signal_aggregation_window=3,  # New: signal aggregation window
                     debug=True) -> pd.Series:

    # 1. Get raw signals
    raw_signals = calculate_signal(df, signal_name)

    if debug:
        print(f"\n[generate_signals] Strategy: {signal_name}")
        print(f"  Raw signal stats: 1={sum(raw_signals == 1)}, -1={sum(raw_signals == -1)}, 0={sum(raw_signals == 0)}")

    # 2. Signal aggregation - merge adjacent signals
    aggregated_signals = pd.Series(0, index=df.index)

    if signal_aggregation_window > 1:
        for i in range(len(raw_signals)):
            if i < signal_aggregation_window - 1:
                aggregated_signals.iloc[i] = 0
                continue

            # Get signals in window
            window_signals = raw_signals.iloc[i - signal_aggregation_window + 1:i + 1]

            # Calculate weighted average
            window_mean = window_signals.mean()

            # Convert to integer signal based on threshold
            if window_mean > 0.3:  # Positive signal threshold
                aggregated_signals.iloc[i] = 1
            elif window_mean < -0.3:  # Negative signal threshold
                aggregated_signals.iloc[i] = -1
            else:
                aggregated_signals.iloc[i] = 0
    else:
        # No aggregation, directly round
        aggregated_signals = raw_signals.round().clip(-1, 1).astype(int)

    if debug:
        print(f"  Aggregation window: {signal_aggregation_window}")
        print(f"  After aggregation: 1={sum(aggregated_signals == 1)}, -1={sum(aggregated_signals == -1)}, 0={sum(aggregated_signals == 0)}")

    # 3. Signal confirmation
    int_signals = aggregated_signals.astype(int)
    smoothed_signals = pd.Series(0, index=df.index)

    # Simple signal confirmation: just need current signal ≠ 0
    # (aggregation already handles signal stability)
    for i in range(len(int_signals)):
        smoothed_signals.iloc[i] = int_signals.iloc[i]  # Directly use aggregated signal

    if debug:
        print(f"  Smooth signal stats: 1={sum(smoothed_signals == 1)}, -1={sum(smoothed_signals == -1)}, 0={sum(smoothed_signals == 0)}")

    # 4. Prevent frequent signal flipping
    stable_signals = pd.Series(0, index=df.index)
    prev_signal = 0
    signal_duration = 0

    for i in range(len(smoothed_signals)):
        current_signal = smoothed_signals.iloc[i]

        # If new signal is opposite to previous signal
        if current_signal == -prev_signal and prev_signal != 0:
            # Check if previous signal duration is too short
            if signal_duration < 5:  # If previous signal lasted very briefly
                # Ignore this flip, keep previous signal
                stable_signals.iloc[i] = prev_signal
                signal_duration += 1
            else:
                # Accept flip
                stable_signals.iloc[i] = current_signal
                prev_signal = current_signal
                signal_duration = 1
        else:
            if current_signal == prev_signal:
                signal_duration += 1
            else:
                signal_duration = 1
                prev_signal = current_signal

            stable_signals.iloc[i] = current_signal

    if debug:
        print(f"  After stabilization: 1={sum(stable_signals == 1)}, -1={sum(stable_signals == -1)}, 0={sum(stable_signals == 0)}")

    # 5. Modify position management logic
    final_positions = pd.Series(0, index=df.index)
    current_position = 0
    position_entry_tick = -1

    for i in range(len(stable_signals)):
        signal = stable_signals.iloc[i]

        if current_position == 0:
            # No position, can open
            if signal != 0:
                current_position = signal
                position_entry_tick = i
                final_positions.iloc[i] = current_position

        else:
            # Has position
            held_ticks = i - position_entry_tick

            # Check if reached max hold time
            if held_ticks >= max_hold_ticks:
                # Force close
                current_position = 0
                position_entry_tick = -1
                final_positions.iloc[i] = 0

            # Check if reached min hold time
            elif held_ticks >= min_hold_ticks:
                # Can close, but not forced
                if signal == 0 or signal == -current_position:
                    # Close position
                    current_position = 0
                    position_entry_tick = -1
                    final_positions.iloc[i] = 0
                else:
                    # Continue holding
                    final_positions.iloc[i] = current_position

            else:
                # Haven't reached min hold, forced to hold
                final_positions.iloc[i] = current_position

    # 6. Add debug information
    if debug:
        print(f"  Min hold time: {min_hold_ticks} ticks")
        print(f"  Max hold time: {max_hold_ticks} ticks")
        print(f"  Final position stats: 1={sum(final_positions == 1)}, -1={sum(final_positions == -1)}, 0={sum(final_positions == 0)}")

        # Check position continuity
        position_changes = (final_positions.diff() != 0).sum()
        print(f"  Position change count: {position_changes}")

        if len(final_positions) > 0:
            # Calculate position statistics
            positions = final_positions.values
            hold_durations = []
            exit_reasons = {'min_hold': 0, 'signal_exit': 0, 'max_hold': 0}

            i = 0
            while i < len(positions):
                if positions[i] != 0:
                    pos = positions[i]
                    start = i
                    entry_tick = i

                    # Find position end
                    while i < len(positions) and positions[i] == pos:
                        held_ticks = i - entry_tick

                        # Check exit reason
                        if held_ticks >= max_hold_ticks:
                            exit_reasons['max_hold'] += 1
                            break
                        elif i + 1 < len(positions) and (positions[i + 1] == 0 or positions[i + 1] == -pos):
                            if held_ticks >= min_hold_ticks:
                                exit_reasons['signal_exit'] += 1
                            else:
                                exit_reasons['min_hold'] += 1

                        i += 1

                    duration = i - start
                    hold_durations.append(duration)
                else:
                    i += 1

            if hold_durations:
                avg_hold = sum(hold_durations) / len(hold_durations)
                min_hold_actual = min(hold_durations)
                max_hold_actual = max(hold_durations)

                print(f"\n  Hold duration statistics:")
                print(f"    Average hold: {avg_hold:.1f} ticks")
                print(f"    Shortest hold: {min_hold_actual} ticks")
                print(f"    Longest hold: {max_hold_actual} ticks")
                print(f"    Position count: {len(hold_durations)}")

                # Hold efficiency
                if min_hold_ticks > 0:
                    efficiency = avg_hold / min_hold_ticks
                    print(f"    Hold efficiency: {efficiency:.1%} (average/min)")

                if max_hold_ticks > 0:
                    max_efficiency = avg_hold / max_hold_ticks
                    print(f"    Max hold utilization: {max_efficiency:.1%} (average/max)")

                # Exit reason statistics
                print(f"\n  Exit reason statistics:")
                total_exits = sum(exit_reasons.values())
                if total_exits > 0:
                    for reason, count in exit_reasons.items():
                        percentage = count / total_exits * 100
                        reason_name = {
                            'min_hold': 'Min hold forced close',
                            'signal_exit': 'Signal triggered close',
                            'max_hold': 'Max hold forced close'
                        }.get(reason, reason)
                        print(f"    {reason_name}: {count} ({percentage:.1f}%)")

                # Hold duration distribution
                print(f"\n  Hold duration distribution:")
                duration_bins = {
                    'Very short (<min_hold)': 0,
                    f'Short ({min_hold_ticks}-{min_hold_ticks * 2})': 0,
                    f'Medium ({min_hold_ticks * 2}-{max_hold_ticks // 2})': 0,
                    f'Long (>{max_hold_ticks // 2})': 0
                }

                for duration in hold_durations:
                    if duration < min_hold_ticks:
                        duration_bins['Very short (<min_hold)'] += 1
                    elif duration < min_hold_ticks * 2:
                        duration_bins[f'Short ({min_hold_ticks}-{min_hold_ticks * 2})'] += 1
                    elif duration < max_hold_ticks // 2:
                        duration_bins[f'Medium ({min_hold_ticks * 2}-{max_hold_ticks // 2})'] += 1
                    else:
                        duration_bins[f'Long (>{max_hold_ticks // 2})'] += 1

                for bin_name, count in duration_bins.items():
                    if count > 0:
                        percentage = count / len(hold_durations) * 100
                        print(f"    {bin_name}: {count} ({percentage:.1f}%)")

    return final_positions


# -------------------------
# Backtest engine (execution-only, neutral)
# -------------------------
class TradeLogger:
    """Trading log recorder"""

    def __init__(self, filename="trade_log.csv"):
        self.trades = []  # Store trade records
        self.current_trade = None
        self.filename = filename

    def open_trade(self, timestamp, direction, entry_price, entry_type, signal_name=""):
        """Open position record"""
        self.current_trade = {
            'entry_time': timestamp,
            'direction': direction,
            'entry_price': entry_price,
            'entry_type': entry_type,
            'signal_name': signal_name,
            'exit_time': None,
            'exit_price': None,
            'pnl': None,
            'duration_ticks': 0
        }

    def close_trade(self, timestamp, exit_price, exit_type):
        """Close position record"""
        if self.current_trade:
            self.current_trade['exit_time'] = timestamp
            self.current_trade['exit_price'] = exit_price
            self.current_trade['exit_type'] = exit_type

            # Calculate P&L
            if self.current_trade['direction'] == 'long':
                pnl = exit_price - self.current_trade['entry_price']
            else:  # short
                pnl = self.current_trade['entry_price'] - exit_price

            self.current_trade['pnl'] = pnl

            # Calculate holding time
            if isinstance(self.current_trade['entry_time'], (pd.Timestamp, datetime)):
                duration = (timestamp - self.current_trade['entry_time']).total_seconds()
            else:
                duration = 0
            self.current_trade['duration_seconds'] = duration

            # Add to trade list
            self.trades.append(self.current_trade.copy())
            self.current_trade = None

    def get_trade_log_df(self):
        """Get trade log DataFrame"""
        if not self.trades:
            return pd.DataFrame()

        df = pd.DataFrame(self.trades)

        # Add sequence number
        df.insert(0, 'trade_id', range(1, len(df) + 1))

        # Reorder columns
        columns = ['trade_id', 'signal_name', 'direction',
                   'entry_time', 'entry_price', 'entry_type',
                   'exit_time', 'exit_price', 'exit_type',
                   'pnl', 'duration_seconds']

        # Keep only existing columns
        columns = [col for col in columns if col in df.columns]

        return df[columns]

    def save_to_csv(self, filename=None):
        """Save to CSV file"""
        if filename is None:
            filename = self.filename

        df = self.get_trade_log_df()
        if len(df) > 0:
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"Trade log saved to: {filename}")
        return df


def backtest_cross_spread(df: pd.DataFrame, target_pos: pd.Series):
    """
    target_pos: desired position at each row (-1/0/+1).
    Execution: when position changes, trade at ask1 for buys and bid1 for sells.
    PnL computed per 1 unit.
    """
    df = df.copy()
    target_pos = target_pos.reindex(df.index).fillna(0).astype(int)

    pos = 0
    entry_px = np.nan
    trade_pnl = []

    for i in range(len(df)):
        desired = int(target_pos.iloc[i])

        # Only allow -1/0/+1
        desired = -1 if desired < 0 else (1 if desired > 0 else 0)

        if desired == pos:
            continue

        # Close existing position if any
        if pos == 1 and desired != 1:
            exit_px = df.loc[i, "bid1"]
            trade_pnl.append(exit_px - entry_px)
            entry_px = np.nan
        elif pos == -1 and desired != -1:
            exit_px = df.loc[i, "ask1"]
            trade_pnl.append(entry_px - exit_px)
            entry_px = np.nan

        # Open new position if needed
        if desired == 1:
            entry_px = df.loc[i, "ask1"]
        elif desired == -1:
            entry_px = df.loc[i, "bid1"]

        pos = desired

    trade_pnl = np.array(trade_pnl, dtype=float)
    stats = {
        "total_pnl": float(np.nansum(trade_pnl)),
        "num_trades": int(np.sum(~np.isnan(trade_pnl))),
        "win_rate": float(np.mean(trade_pnl > 0)) if len(trade_pnl) > 0 else 0.0
    }
    return trade_pnl, stats


def backtest_cross_spread_with_log(df: pd.DataFrame, target_pos: pd.Series, signal_name="", log_trades=False):
    """
    Backtest function with trade logging
    Fix: Get timestamps from timestamp column
    """
    df = df.copy()
    target_pos = target_pos.reindex(df.index).fillna(0).astype(int)

    # Check if timestamp column exists
    if 'timestamp' not in df.columns:
        print("⚠️ Warning: No 'timestamp' column in DataFrame, using index as time")
        use_timestamp_col = False
    else:
        use_timestamp_col = True

    pos = 0
    entry_px = np.nan
    trade_pnl = []

    # Initialize trade logger
    trade_logger = TradeLogger(f"trade_log_{signal_name}.csv")

    for i in range(len(df)):
        # Get timestamp from timestamp column
        if use_timestamp_col:
            timestamp = df.iloc[i]["timestamp"]
        else:
            # Fallback to index
            timestamp = df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i

        desired = int(target_pos.iloc[i])

        # Only allow -1/0/+1
        desired = -1 if desired < 0 else (1 if desired > 0 else 0)

        if desired == pos:
            continue

        # Close existing position if any
        if pos == 1 and desired != 1:
            exit_px = df.iloc[i]["bid1"]
            trade_pnl.append(exit_px - entry_px)

            if log_trades and trade_logger.current_trade:
                trade_logger.close_trade(timestamp, exit_px, "bid_close")

            entry_px = np.nan

        elif pos == -1 and desired != -1:
            exit_px = df.iloc[i]["ask1"]
            trade_pnl.append(entry_px - exit_px)

            if log_trades and trade_logger.current_trade:
                trade_logger.close_trade(timestamp, exit_px, "ask_close")

            entry_px = np.nan

        # Open new position if needed
        if desired == 1:
            entry_px = df.iloc[i]["ask1"]
            if log_trades:
                trade_logger.open_trade(timestamp, "long", entry_px, "ask_open", signal_name)

        elif desired == -1:
            entry_px = df.iloc[i]["bid1"]
            if log_trades:
                trade_logger.open_trade(timestamp, "short", entry_px, "bid_open", signal_name)

        pos = desired

    # Handle any open trades
    if log_trades and trade_logger.current_trade:
        last_idx = len(df) - 1

        # Get last timestamp
        if use_timestamp_col:
            last_timestamp = df.iloc[last_idx]["timestamp"]
        else:
            last_timestamp = df.index[last_idx] if isinstance(df.index, pd.DatetimeIndex) else last_idx

        last_price = df.iloc[last_idx]["mid"] if "mid" in df.columns else df.iloc[last_idx]["last"]

        if trade_logger.current_trade['direction'] == 'long':
            exit_price = last_price
            exit_type = "mid_close"
        else:
            exit_price = last_price
            exit_type = "mid_close"

        trade_logger.close_trade(last_timestamp, exit_price, exit_type)

    trade_pnl = np.array(trade_pnl, dtype=float)
    stats = {
        "total_pnl": float(np.nansum(trade_pnl)),
        "num_trades": int(np.sum(~np.isnan(trade_pnl))),
        "win_rate": float(np.mean(trade_pnl > 0)) if len(trade_pnl) > 0 else 0.0
    }

    return trade_pnl, stats, trade_logger

