import pandas as pd
import numpy as np

# ================ risk control ================
def _add_max_hold_time(signals: pd.Series, max_hold_ticks: int = 4200) -> pd.Series:
    """
    Add maximum holding time risk control
    Each signal may occur continuously for no more than: max_hold_ticks
    """
    if len(signals) == 0:
        return signals

    result = signals.copy()
    pos_start_idx = -1
    current_pos = 0

    for i in range(len(signals)):
        if signals.iloc[i] != 0:  # trading signal
            if current_pos != signals.iloc[i]:  # signal change
                current_pos = signals.iloc[i]
                pos_start_idx = i
            elif i - pos_start_idx >= max_hold_ticks:  # over time
                result.iloc[i] = 0
                current_pos = 0
        else:  # no signal
            current_pos = 0


    return result


def _add_stop_loss(signals: pd.Series, df: pd.DataFrame, stop_loss_pct: float = 0.001) -> pd.Series:
    """
    Add stop-loss logic to trading signals
    """

    def to_scalar(value):
        """Convert any value to scalar float"""
        if isinstance(value, (pd.Series, pd.DataFrame)):
            if len(value) > 0:
                return float(value.iloc[0])
            return 0.0
        elif isinstance(value, np.ndarray):
            return float(value.item())
        elif hasattr(value, 'item'):
            return float(value.item())
        else:
            return float(value)

    result = signals.copy()
    in_position = False
    entry_price = 0.0
    position_type = 0  # 1: long, -1: short

    for i in range(len(signals)):
        # Get current signal
        current_signal = signals.iloc[i]

        # Open new position
        if not in_position and current_signal != 0:
            in_position = True
            position_type = current_signal

            # Get entry price
            if position_type == 1:  # Long
                entry_price = to_scalar(df["ask1"].iloc[i])
            else:  # Short
                entry_price = to_scalar(df["bid1"].iloc[i])

            # Validate entry price
            if pd.isna(entry_price) or entry_price <= 0:
                in_position = False
                result.iloc[i] = 0
                continue

        # Check stop-loss
        elif in_position:
            # Get current price
            if position_type == 1:  # Long
                current_price = to_scalar(df["bid1"].iloc[i])
            else:  # Short
                current_price = to_scalar(df["ask1"].iloc[i])

            # Validate prices
            if (pd.isna(current_price) or current_price <= 0 or
                    pd.isna(entry_price) or entry_price <= 0):
                continue

            # Calculate return
            if position_type == 1:  # Long
                ret = (current_price - entry_price) / entry_price
            else:  # Short
                ret = (entry_price - current_price) / entry_price

            # Apply stop-loss
            if ret < -stop_loss_pct:
                result.iloc[i] = 0
                in_position = False

        # Close position
        if in_position and current_signal == 0:
            in_position = False

    return result



# ================ strategy signals ================
def signal_orderflow_imbalance(df: pd.DataFrame) -> pd.Series:
    """
    Strategy 1: Order Flow Imbalance Reversion Strategy
    Fade extreme OFI signals with low-frequency and volatility confirmation
    """
    signals = pd.Series(0, index=df.index)

    # 1. CHECK OB FEATURES (Sufficient: 2 market-state + 1 order-flow)
    # Market-state features: relative_spread, depth_imbalance ✓
    # Order-flow features: ofi_l1_roll, signed_vol ✓
    # All good, no need to add more OB features

    # 2. ADD LOW-FREQUENCY FEATURES
    # Use 1-minute price position for mean reversion context
    lf_price_position = df.get('mid_position_1min', 0.5)  # Default to middle if not available
    lf_zscore = df.get('zscore_1min', 0)  # 1-minute z-score for mean reversion

    # 3. ADD VOLATILITY FEATURES
    # Use 1-minute realized volatility for regime filtering
    vol_1min = df.get('realized_vol_1min', 0.001)
    vol_percentile = df.get('vol_percentile_1min', 0.5)  # Volatility regime
    vol_ratio = df.get('vol_ratio_30s_3min', 1.0)  # Volatility term structure

    # THRESHOLDS FOR EXTREME OFI
    ofi_extreme_threshold = 30
    depth_extreme_threshold = 0.02

    # SIMPLE FILTERS
    spread_ok = df['relative_spread'] < df['relative_spread'].rolling(50).quantile(0.7)
    volume_ok = df['trade_vol'] > df['trade_vol'].rolling(20).mean() * 0.2

    # LOW-FREQUENCY FILTERS
    # Only trade in moderate volatility regimes (avoid extremes)
    moderate_vol = (vol_percentile > 0.2) & (vol_percentile < 0.8)

    # Prefer mean reversion when price is at extremes
    oversold_lf = lf_price_position < 0.3
    overbought_lf = lf_price_position > 0.7

    # VOLATILITY FILTERS
    # Avoid trading when volatility is expanding rapidly
    vol_not_expanding = vol_ratio < 1.5
    # Prefer normal or declining volatility
    normal_vol_structure = (vol_ratio > 0.7) & (vol_ratio < 1.3)

    # REVERSED LOGIC WITH LF & VOL CONFIRMATION
    long_entry = (
            (df['ofi_l1_roll'] < -ofi_extreme_threshold) &  # Extreme negative OFI
            (df['depth_imbalance'] < -depth_extreme_threshold) &  # Extreme negative depth
            oversold_lf &  # LF: Price at lower range
            (lf_zscore < -1.0) &  # LF: Significantly below mean
            spread_ok &
            volume_ok &
            moderate_vol &  # Vol: Not extreme volatility
            vol_not_expanding &  # Vol: Volatility not expanding
            (df['mid'].pct_change(5) < 0) &  # Recent price decline
            (df['signed_vol'].rolling(3).mean() < 0)  # Recent selling
    )

    short_entry = (
            (df['ofi_l1_roll'] > ofi_extreme_threshold) &  # Extreme positive OFI
            (df['depth_imbalance'] > depth_extreme_threshold) &  # Extreme positive depth
            overbought_lf &  # LF: Price at upper range
            (lf_zscore > 1.0) &  # LF: Significantly above mean
            spread_ok &
            volume_ok &
            moderate_vol &
            vol_not_expanding &
            (df['mid'].pct_change(5) > 0) &  # Recent price rise
            (df['signed_vol'].rolling(3).mean() > 0)  # Recent buying
    )

    # MEAN REVERSION EXITS WITH VOLATILITY AWARENESS
    price_for_exit = df['microprice_top3']

    long_exit = (
            (df['ofi_l1_roll'] > -ofi_extreme_threshold * 0.3) |  # OFI normalized
            (lf_zscore > -0.5) |  # LF: Price returned near mean
            (vol_ratio > 1.8) |  # Vol: Volatility expanded too much
            (price_for_exit > price_for_exit.rolling(10).max() * 0.999) |  # Small profit
            (df['mid'].pct_change(2) > 0.0002)  # Quick bounce
    )

    short_exit = (
            (df['ofi_l1_roll'] < ofi_extreme_threshold * 0.3) |
            (lf_zscore < 0.5) |
            (vol_ratio > 1.8) |
            (price_for_exit < price_for_exit.rolling(10).min() * 1.001) |
            (df['mid'].pct_change(2) < -0.0002)
    )

    # DYNAMIC HOLDING BASED ON VOLATILITY
    # Hold longer in low volatility, shorter in high volatility
    if 'vol_percentile_1min' in df.columns:
        vol_level = df['vol_percentile_1min']
        # Map volatility percentile to hold time: low vol -> longer, high vol -> shorter
        hold_ticks = np.where(vol_level < 0.3, 16,  # Low vol: 16 ticks
                              np.where(vol_level > 0.7, 8,  # High vol: 8 ticks
                                       12))  # Normal vol: 12 ticks
    else:
        hold_ticks = 12

    # DYNAMIC STOP LOSS BASED ON VOLATILITY
    if 'realized_vol_1min' in df.columns:
        # Stop loss as multiple of volatility
        base_stop = 0.0005
        vol_multiplier = df['realized_vol_1min'] / df['realized_vol_1min'].rolling(100).mean()
        stop_pct = base_stop * np.clip(vol_multiplier, 0.7, 1.5)
    else:
        stop_pct = 0.0005

    # Apply signals
    signals[long_entry] = 1
    signals[short_entry] = -1

    signals[long_exit & (signals == 1)] = 0
    signals[short_exit & (signals == -1)] = 0

    # VERY SHORT HOLDS for reversion trades
    signals = _add_max_hold_time(signals)

    # TIGHT STOPS
    stop_value = float(stop_pct.iloc[0]) if hasattr(stop_pct, '__len__') else stop_pct
    signals = _add_stop_loss(signals, df, stop_loss_pct=stop_value)

    # PREVENT OVERTRADING
    for i in range(1, len(signals)):
        if signals.iloc[i] != 0 and signals.iloc[i - 1] != 0:
            if signals.iloc[i] == -signals.iloc[i - 1]:
                signals.iloc[i] = 0

    return signals


def signal_microprice_momentum(df: pd.DataFrame) -> pd.Series:
    """
    Strategy 2: Simple Micro-price Momentum with Basic Filters
    Simplified version for more signals
    """
    signals = pd.Series(0, index=df.index)

    # SIMPLE THRESHOLDS
    momentum_threshold = 0.00001  # Reduced threshold
    depth_threshold = 0.0005  # Minimal depth advantage

    # BASIC FILTERS
    spread_ok = df['relative_spread'] < df['relative_spread'].rolling(50).quantile(0.8)  # Relaxed
    volume_ok = df['trade_vol'] > 50  # Minimal volume

    # SIMPLE MICROPRICE MOMENTUM
    mp_change = df['microprice'].diff(2)  # 2-tick change (faster)
    mp_momentum = mp_change  # No smoothing for more signals

    # ADD BASIC LOW-FREQUENCY CONFIRMATION (if available)
    lf_available = False
    if 'trend_strength_1min' in df.columns and 'mid_position_1min' in df.columns:
        lf_available = True
        # Simple LF filters
        moderate_trend = df['trend_strength_1min'].abs() < 2.0
        not_extreme_position = (df['mid_position_1min'] > 0.2) & (df['mid_position_1min'] < 0.8)

    # ADD BASIC VOLATILITY FILTER (if available)
    vol_available = False
    if 'realized_vol_30s' in df.columns:
        vol_available = True
        # Avoid extreme volatility
        avg_vol = df['realized_vol_30s'].rolling(100).mean()
        moderate_vol = df['realized_vol_30s'] < avg_vol * 2.0

    # SIMPLE ENTRY CONDITIONS
    base_long = (
            (mp_momentum > momentum_threshold) &  # Microprice rising
            (df['mp_minus_mid'] > 0) &  # Microprice above mid
            (df['size_imbalance_l1'] > depth_threshold) &  # Some buying pressure
            spread_ok &
            volume_ok
    )

    base_short = (
            (mp_momentum < -momentum_threshold) &  # Microprice falling
            (df['mp_minus_mid'] < 0) &  # Microprice below mid
            (df['size_imbalance_l1'] < -depth_threshold) &  # Some selling pressure
            spread_ok &
            volume_ok
    )

    # ADD OPTIONAL FILTERS IF AVAILABLE
    if lf_available and vol_available:
        long_entry = base_long & moderate_trend & not_extreme_position & moderate_vol
        short_entry = base_short & moderate_trend & not_extreme_position & moderate_vol
    elif lf_available:
        long_entry = base_long & moderate_trend & not_extreme_position
        short_entry = base_short & moderate_trend & not_extreme_position
    elif vol_available:
        long_entry = base_long & moderate_vol
        short_entry = base_short & moderate_vol
    else:
        long_entry = base_long
        short_entry = base_short

    # SIMPLE EXIT CONDITIONS
    price_for_exit = df['microprice_top3']

    long_exit = (
            (mp_momentum < 0) |  # Momentum turned negative
            (price_for_exit > price_for_exit.rolling(5).max() * 0.999)  # Near recent high
    )

    short_exit = (
            (mp_momentum > 0) |  # Momentum turned positive
            (price_for_exit < price_for_exit.rolling(5).min() * 1.001)  # Near recent low
    )

    # APPLY SIGNALS
    signals[long_entry] = 1
    signals[short_entry] = -1

    signals[long_exit & (signals == 1)] = 0
    signals[short_exit & (signals == -1)] = 0

    # BASIC COOLDOWN
    for i in range(1, len(signals)):
        if signals.iloc[i] != 0 and signals.iloc[i - 1] != 0:
            if signals.iloc[i] == -signals.iloc[i - 1]:
                signals.iloc[i] = 0

    # SIMPLE RISK CONTROLS
    signals = _add_max_hold_time(signals)  # Shorter holds
    signals = _add_stop_loss(signals, df, stop_loss_pct=0.0008)  # Tighter stop

    return signals


def signal_depth_spread_arb(df: pd.DataFrame) -> pd.Series:
    """
    Strategy 3: Simple Depth-Spread Strategy with Bollinger Band and Volatility Filters
    Use Bollinger Band position instead of range position for low-frequency context
    """
    signals = pd.Series(0, index=df.index)

    # SIMPLIFY TO MINIMUM OB FEATURES
    # Market-state features: depth_imbalance, relative_spread (2 features)
    # Add 1 order-flow feature: ofi_l1 (Order Flow Imbalance)

    # RELAXED THRESHOLDS FOR MORE SIGNALS
    depth_threshold = 0.015
    spread_widen = 1.15
    spread_tighten = 0.85

    # SIMPLE VOLUME FILTER
    volume_ok = df['trade_vol'] > df['trade_vol'].rolling(20).mean() * 0.05

    # USE BOLLINGER BAND AS LOW-FREQUENCY FEATURE
    if 'bb_position_1min' in df.columns:
        # Use Bollinger Band position (0-1 scale)
        bb_position = df['bb_position_1min']
        # Trade at BB extremes for mean reversion
        at_bb_extreme = (bb_position < 0.2) | (bb_position > 0.8)
    else:
        at_bb_extreme = pd.Series(True, index=df.index)

    # ADD VOLATILITY FEATURE
    if 'realized_vol_1min' in df.columns:
        vol_1min = df['realized_vol_1min']
        avg_vol = vol_1min.rolling(100).mean()
        # Prefer moderate volatility
        moderate_vol = (vol_1min > avg_vol * 0.5) & (vol_1min < avg_vol * 1.5)
    else:
        moderate_vol = pd.Series(True, index=df.index)

    # SIMPLE ENTRY CONDITIONS WITH BOLLINGER BAND
    # Long: Buy depth + Wide spread + Price at lower BB
    long_entry = (
            (df['depth_imbalance'] > depth_threshold) &  # Buy depth
            (df['relative_spread'] > df['relative_spread'].rolling(50).mean() * spread_widen) &
            (df['ofi_l1'] > 0) &  # Positive OFI
            (bb_position < 0.3) &  # Price near lower Bollinger Band (oversold)
            moderate_vol &  # Moderate volatility
            volume_ok
    )

    # Short: Sell depth + Tight spread + Price at upper BB
    short_entry = (
            (df['depth_imbalance'] < -depth_threshold) &  # Sell depth
            (df['relative_spread'] < df['relative_spread'].rolling(50).mean() * spread_tighten) &  # Tight spread
            (df['ofi_l1'] < 0) &  # Negative OFI
            (bb_position > 0.7) &  # Price near upper Bollinger Band (overbought)
            moderate_vol &
            volume_ok
    )

    # SIMPLE EXITS
    long_exit = (
            (df['depth_imbalance'] < 0) |  # Depth turned negative
            (bb_position > 0.5) |  # Price moved to middle of BB
            (df['mid'].pct_change(2) < -0.0003)  # Stop loss
    )

    short_exit = (
            (df['depth_imbalance'] > 0) |  # Depth turned positive
            (bb_position < 0.5) |  # Price moved to middle of BB
            (df['mid'].pct_change(2) > 0.0003)  # Stop loss
    )

    # APPLY SIGNALS
    signals[long_entry] = 1
    signals[short_entry] = -1

    signals[long_exit & (signals == 1)] = 0
    signals[short_exit & (signals == -1)] = 0


    # DYNAMIC STOP LOSS
    if 'realized_vol_1min' in df.columns and len(df) > 0:
        current_vol = df['realized_vol_1min'].iloc[0]
        stop_pct = min(current_vol * 2, 0.001)
    else:
        stop_pct = 0.0006

    # RISK CONTROLS
    signals = _add_max_hold_time(signals)
    signals = _add_stop_loss(signals, df, stop_loss_pct=stop_pct)

    # BASIC COOLDOWN
    for i in range(1, len(signals)):
        if signals.iloc[i] != 0 and signals.iloc[i - 1] != 0:
            if signals.iloc[i] == -signals.iloc[i - 1]:
                signals.iloc[i] = 0

    return signals


def signal_volatility_breakout_reversed(df: pd.DataFrame) -> pd.Series:
    """
    Strategy 4: Simple Volatility Breakout with High-Low Channel
    REVERSED VERSION: Trade the opposite signals
    """
    signals = pd.Series(0, index=df.index)

    # 1. SIMPLIFY TO MINIMUM OB FEATURES
    # Market-state: relative_spread, depth_imbalance (2 features)
    # Order-flow: signed_vol (1 feature)

    # RELAXED THRESHOLDS
    vol_ratio = 0.7
    price_break = 0.0002

    # SIMPLE VOLUME FILTER
    volume_ok = df['trade_vol'] > df['trade_vol'].rolling(20).mean() * 0.1

    # LOW VOLATILITY DETECTION
    current_vol = df['mid_ret'].abs().rolling(5).mean()
    avg_vol = df['mid_ret'].abs().rolling(50).mean()
    low_vol = current_vol < avg_vol * vol_ratio

    # 2. USE HIGH-LOW CHANNEL AS LOW-FREQUENCY FEATURE
    if 'mid_position_5min' in df.columns:
        channel_position = df['mid_position_5min']
    else:
        channel_position = pd.Series(0.5, index=df.index)

    # SIMPLE PRICE MOMENTUM
    price_change = df['mid'].pct_change(3)

    # ***********************************************
    # ORIGINAL ENTRY CONDITIONS (for reference)
    # Long: Fade downside breakout in low volatility
    original_long_entry = (
            low_vol &
            (price_change < -price_break) &
            (df['depth_imbalance'] < 0) &
            (df['signed_vol'].rolling(3).mean() < 0) &
            (channel_position < 0.3) &
            (df['relative_spread'] < df['relative_spread'].rolling(50).quantile(0.8)) &
            volume_ok
    )

    # Short: Fade upside breakout in low volatility
    original_short_entry = (
            low_vol &
            (price_change > price_break) &
            (df['depth_imbalance'] > 0) &
            (df['signed_vol'].rolling(3).mean() > 0) &
            (channel_position > 0.7) &
            (df['relative_spread'] < df['relative_spread'].rolling(50).quantile(0.8)) &
            volume_ok
    )

    # ***********************************************
    # REVERSED ENTRY CONDITIONS
    # Long: When original strategy wants to short
    reversed_long_entry = original_short_entry

    # Short: When original strategy wants to long
    reversed_short_entry = original_long_entry

    # ***********************************************
    # REVERSED EXITS
    # Long exit: When original short would exit
    reversed_long_exit = (
            (channel_position < 0.5)
    )

    # Short exit: When original long would exit
    reversed_short_exit = (
            (channel_position > 0.5)
    )

    current_position = 0

    for i in range(len(signals)):
        if current_position == 0:
            if reversed_long_entry.iloc[i]:
                current_position = 1
            elif reversed_short_entry.iloc[i]:
                current_position = -1

        else:
            if current_position == 1 and reversed_long_exit.iloc[i]:
                current_position = 0
            elif current_position == -1 and reversed_short_exit.iloc[i]:
                current_position = 0

        signals.iloc[i] = current_position

    return signals


def signal_momentum_depth_convergence(df: pd.DataFrame) -> pd.Series:
    """
    Strategy 5: REVERSED Momentum with Bollinger Band and RSI
    Opposite of original signals (since original had 0% win rate)
    """
    signals = pd.Series(0, index=df.index)

    # 1. SIMPLIFY TO MINIMUM OB FEATURES
    # SIMPLE MOMENTUM
    momentum = df['momentum_5']

    # SIMPLE VOLUME FILTER
    volume_ok = df['trade_vol'] > df['trade_vol'].rolling(20).mean() * 0.2

    # 2. USE BOLLINGER BAND AND RSI AS LOW-FREQUENCY FEATURES
    if 'bb_position_1min' in df.columns:
        bb_position = df['bb_position_1min']
    else:
        bb_position = pd.Series(0.5, index=df.index)

    if 'rsi_1min' in df.columns:
        rsi = df['rsi_1min']
    else:
        rsi = pd.Series(50, index=df.index)

    # REVERSED ENTRY CONDITIONS
    # Original: long when momentum positive, RSI bullish, etc.
    # Reversed: SHORT when original would go LONG

    long_entry = (
            (momentum < -0.0002) &  # REVERSED: Negative momentum (was positive)
            (df['signed_vol'].rolling(3).mean() < 0) &  # REVERSED: Selling volume (was buying)
            (bb_position < 0.3) &  # REVERSED: Lower BB (was middle)
            (rsi < 30) &  # REVERSED: Oversold (was bullish)
            (df['relative_spread'] < df['relative_spread'].rolling(50).quantile(0.7)) &
            volume_ok
    )

    short_entry = (
            (momentum > 0.0002) &  # REVERSED: Positive momentum (was negative)
            (df['signed_vol'].rolling(3).mean() > 0) &  # REVERSED: Buying volume (was selling)
            (bb_position > 0.7) &  # REVERSED: Upper BB (was middle)
            (rsi > 70) &  # REVERSED: Overbought (was bearish)
            (df['relative_spread'] < df['relative_spread'].rolling(50).quantile(0.7)) &
            volume_ok
    )

    # REVERSED EXITS
    long_exit = (
            (momentum > 0) |  # REVERSED: Momentum turned positive
            (bb_position > 0.5) |  # Moved to middle BB
            (rsi > 50) |  # RSI normalized
            (df['mid'].pct_change(2) < -0.0004)  # Wider stop for reversal
    )

    short_exit = (
            (momentum < 0) |
            (bb_position < 0.5) |
            (rsi < 50) |
            (df['mid'].pct_change(2) > 0.0004)
    )

    # APPLY SIGNALS
    signals[long_entry] = 1
    signals[short_entry] = -1

    signals[long_exit & (signals == 1)] = 0
    signals[short_exit & (signals == -1)] = 0

    # WIDER STOPS FOR REVERSAL
    stop_pct = 0.0008

    # RISK CONTROLS
    signals = _add_max_hold_time(signals)
    signals = _add_stop_loss(signals, df, stop_loss_pct=stop_pct)

    return signals



def signal_diagnostic_test(df: pd.DataFrame) -> pd.Series:
    """
    Diagnostic Strategy: Test if ANY signal can be profitable
    """
    signals = pd.Series(0, index=df.index)

    print("=== DIAGNOSTIC TEST ===")
    print(f"Data points: {len(df)}")

    if len(df) == 0:
        return signals

    # TEST 1: Check basic statistics
    print(f"Price range: {df['mid'].min():.4f} to {df['mid'].max():.4f}")
    print(f"Average spread: {df['spread'].mean():.6f}")
    print(f"Average relative spread: {df['relative_spread'].mean():.6%}")

    # TEST 2: Check returns distribution
    returns = df['mid'].pct_change()
    pos_returns = (returns > 0).sum()
    neg_returns = (returns < 0).sum()
    zero_returns = (returns == 0).sum()

    print(f"Positive returns: {pos_returns} ({pos_returns / len(df) * 100:.1f}%)")
    print(f"Negative returns: {neg_returns} ({neg_returns / len(df) * 100:.1f}%)")
    print(f"Zero returns: {zero_returns} ({zero_returns / len(df) * 100:.1f}%)")

    # TEST 3: Simple buy-and-hold for 1 tick
    # Buy at every tick, sell next tick
    for i in range(len(df) - 1):
        signals.iloc[i] = 1

    # TEST 4: Check if spread is too wide
    avg_spread = df['spread'].mean()
    avg_price = df['mid'].mean()
    spread_pct = avg_spread / avg_price

    print(f"Average spread as % of price: {spread_pct:.6%}")

    if spread_pct > 0.001:  # Spread > 0.1%
        print(f"WARNING: Spread is very wide ({spread_pct:.4%})")
        print("Trading may be impossible due to high transaction costs")

    return signals


# ================ signal_name_dict ================
SIGNAL_REGISTRY = {
    "signal_diagnostic_test": signal_diagnostic_test,
    'orderflow_imbalance': signal_orderflow_imbalance,
    'microprice_momentum': signal_microprice_momentum,
    'depth_spread_arb': signal_depth_spread_arb,
    'volatility_breakout_reversed': signal_volatility_breakout_reversed
}


def get_signal_names() -> list:
    """Obtain all available signal names"""
    return list(SIGNAL_REGISTRY.keys())


def calculate_signal(df: pd.DataFrame, signal_name: str) -> pd.Series:
    """
    Args:
        df:
        signal_name:
    Returns:
         [-1, 0, 1]
    """
    if signal_name not in SIGNAL_REGISTRY:
        raise ValueError(f"unknown signal: {signal_name}")

    return SIGNAL_REGISTRY[signal_name](df)