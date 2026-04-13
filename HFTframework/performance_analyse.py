import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle

def plot_price_chart(df, stock_id, output_dir='plots', every_n_ticks=7200, figsize=(20, 8)):

    os.makedirs(output_dir, exist_ok=True)
    df_plot = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df_plot['timestamp']):
        df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'], errors='coerce')

    df_plot['timestamp_str'] = df_plot['timestamp'].dt.strftime('%m-%d %H:%M:%S')

    if 'trading_date' in df_plot.columns:
        trading_days = df_plot['trading_date'].nunique()
    else:
        trading_days = df_plot['timestamp'].dt.date.nunique()

    plt.figure(figsize=figsize)
    plt.plot(df_plot['timestamp_str'], df_plot['last'],
             linewidth=0.5, color='blue', alpha=0.7, label='Last Price')

    n = len(df_plot)
    if n > every_n_ticks * 2:
        indices = np.arange(0, n, every_n_ticks)
        indices = indices[indices < n]
        tick_labels = df_plot['timestamp_str'].iloc[indices].tolist()
        plt.xticks(indices, tick_labels, rotation=45, ha='right', fontsize=8)
    else:
        plt.xticks(rotation=45)

    if n > 0:
        date_changes = df_plot['timestamp'].dt.date.diff() != pd.Timedelta(0)
        date_change_indices = df_plot[date_changes].index.tolist()

        for idx in date_change_indices[1:]:
            plt.axvline(x=idx, color='red', alpha=0.3, linestyle=':',
                        linewidth=1, label='Trading Day Change' if idx == date_change_indices[1] else "")

    plt.xlabel('Time (MM-DD HH:MM:SS)', fontsize=12)
    plt.ylabel('Last Price', fontsize=12)
    plt.title(f'Last Price for {stock_id} ({trading_days} trading days)',
              fontsize=14, fontweight='bold')

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()

    filename = f"{stock_id}_price_chart.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Price chart saved: {filepath}")

    plt.show()
    return filepath


def plot_price_volume_combined(df, trade_log_path, stock_id, output_dir,
                               every_n_ticks=7200, figsize=(20, 8)):
    from matplotlib.patches import Patch

    os.makedirs(output_dir, exist_ok=True)
    df_plot = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df_plot['timestamp']):
        df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'], errors='coerce')

    df_plot['timestamp_str'] = df_plot['timestamp'].dt.strftime('%m-%d %H:%M:%S')
    print(f"Data points: {len(df_plot):,}")

    try:
        trades_df = pd.read_csv(trade_log_path)
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        print(f"Loaded {len(trades_df)} strategy trades")
    except Exception as e:
        print(f"Failed to load trade log: {e}")
        trades_df = pd.DataFrame()

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    if len(trades_df) > 0:
        print("Drawing strategy position background...")

        for idx, trade in trades_df.iterrows():
            entry_time = trade['entry_time']
            exit_time = trade['exit_time']
            direction = trade['direction']

            mask = (df_plot['timestamp'] >= entry_time) & (df_plot['timestamp'] <= exit_time)
            if mask.any():
                indices = df_plot[mask].index
                start_idx = indices[0]
                end_idx = indices[-1]

                color = 'lightgreen' if direction == 'long' else 'lightcoral'
                ax1.axvspan(start_idx, end_idx, alpha=0.3, color=color)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(trades_df):
                print(f"  Processed {idx + 1}/{len(trades_df)} trades")

    ax1.plot(df_plot['timestamp_str'], df_plot['last'],
             linewidth=2, color='blue', alpha=0.7, label='Last Price')

    if 'trade_vol' in df_plot.columns and 'side' in df_plot.columns:
        print("Separating buyer/seller volumes...")

        n = len(df_plot)
        x_indices = np.arange(n)

        buyer_mask = df_plot['side'].astype(str).str.upper() == 'B'
        seller_mask = df_plot['side'].astype(str).str.upper() == 'S'

        buyer_count = buyer_mask.sum()
        seller_count = seller_mask.sum()

        print(f"Buyer trades: {buyer_count}, Seller trades: {seller_count}")

        if buyer_count > 0:
            buyer_indices = x_indices[buyer_mask]
            buyer_volumes = df_plot.loc[buyer_mask, 'trade_vol'].values

            ax2.plot(buyer_indices, buyer_volumes,
                     color='darkgreen', linewidth=1.0, alpha=0.9,
                     label='Buyer Market (B)')

            ax2.fill_between(buyer_indices, 0, buyer_volumes,
                             color='darkgreen', alpha=0.3)

        if seller_count > 0:
            seller_indices = x_indices[seller_mask]
            seller_volumes = df_plot.loc[seller_mask, 'trade_vol'].values

            ax2.plot(seller_indices, seller_volumes,
                     color='darkred', linewidth=1.0, alpha=0.9,
                     label='Seller Market (S)')

            ax2.fill_between(seller_indices, 0, seller_volumes,
                             color='darkred', alpha=0.3)

    n = len(df_plot)
    if n > every_n_ticks * 2:
        indices = np.arange(0, n, every_n_ticks)
        indices = indices[indices < n]
        tick_labels = df_plot['timestamp_str'].iloc[indices].tolist()
        ax1.set_xticks(indices)
        ax1.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
        print(f"Set x-axis ticks: every {every_n_ticks} ticks, {len(indices)} ticks total")
    else:
        ax1.tick_params(axis='x', rotation=45)
        print(f"Few data points ({n}), showing all ticks")

    if n > 0:
        date_changes = df_plot['timestamp'].dt.date.diff() != pd.Timedelta(0)
        date_change_indices = df_plot[date_changes].index.tolist()

        for idx in date_change_indices[1:]:
            ax1.axvline(x=idx, color='red', alpha=0.3, linestyle=':',
                        linewidth=1, label='Trading Day Change' if idx == date_change_indices[1] else "")

    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlabel('Time (MM-DD HH:MM:SS)', fontsize=10)
    ax1.set_ylabel('Price', fontsize=10, color='blue')
    ax2.set_ylabel('Trade Volume', fontsize=10, color='black')

    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='black')

    if 'trading_date' in df_plot.columns:
        trading_days = df_plot['trading_date'].nunique()
    else:
        trading_days = df_plot['timestamp'].dt.date.nunique()

    try:
        if 'trades_df' in locals():
            total_trades = len(trades_df)
            long_trades = len(trades_df[trades_df['direction'] == 'long'])
            short_trades = len(trades_df[trades_df['direction'] == 'short'])
            total_pnl = trades_df['pnl'].sum() if 'pnl' in trades_df.columns else 0

            title = (f'{stock_id} - Price & Volume with Positions ({trading_days} trading days)\n'
                     f'Trades: {total_trades} (Long: {long_trades}, Short: {short_trades}) | '
                     f'Total PnL: {total_pnl:.2f}')
        else:
            title = f'{stock_id} - Price & Volume with Positions ({trading_days} trading days)'
    except:
        title = f'{stock_id} - Price & Volume with Positions ({trading_days} trading days)'

    ax1.set_title(title, fontsize=12, fontweight='bold')

    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='lightgreen', alpha=0.3, label='Long Position'),
        Patch(facecolor='lightcoral', alpha=0.3, label='Short Position'),
        Line2D([0], [0], color='blue', lw=2, label='Price'),
    ]

    if 'trade_vol' in df_plot.columns and 'side' in df_plot.columns:
        if buyer_count > 0:
            legend_elements.append(Line2D([0], [0], color='darkgreen', lw=2, label='Buyer Market (B)'))
        if seller_count > 0:
            legend_elements.append(Line2D([0], [0], color='darkred', lw=2, label='Seller Market (S)'))

    ax1.legend(handles=legend_elements, fontsize=8, loc='best')
    plt.tight_layout()

    filename = f"{stock_id}_price_volume_combined.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Price-volume combined chart saved: {filepath}")

    plt.show()
    return filepath


def plot_daily_trade_analysis(trade_log_path):

    df = pd.read_csv(trade_log_path)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['date'] = df['entry_time'].dt.date

    daily_stats = df.groupby('date').agg({
        'pnl': ['sum', 'mean', 'count'],
        'direction': lambda x: (x == 'long').sum(),
        'trade_id': lambda x: (df.loc[x.index, 'pnl'] > 0).sum()
    })

    daily_stats.columns = ['total_pnl', 'avg_pnl', 'trade_count', 'long_count', 'win_count']
    daily_stats['win_rate'] = daily_stats['win_count'] / daily_stats['trade_count'] * 100
    daily_stats['short_count'] = daily_stats['trade_count'] - daily_stats['long_count']
    daily_stats = daily_stats.reset_index()
    daily_stats['date_str'] = daily_stats['date'].astype(str)

    fig_width = 16
    fig_height = 6

    plt.figure(figsize=(fig_width, fig_height))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    bars = ax1.bar(daily_stats['date_str'], daily_stats['total_pnl'],
                   color='skyblue', alpha=0.7, width=0.6, label='Daily PnL')

    for bar, pnl in zip(bars, daily_stats['total_pnl']):
        height = bar.get_height()
        if height != 0:
            ax1.text(bar.get_x() + bar.get_width() / 2.,
                     height + (0.01 if height > 0 else -0.01) * max(abs(daily_stats['total_pnl'])),
                     f'{pnl:.2f}', ha='center', va='bottom' if pnl > 0 else 'top',
                     fontsize=8, fontweight='bold',
                     color='green' if pnl > 0 else 'red')

    line = ax2.plot(daily_stats['date_str'], daily_stats['win_rate'],
                    color='darkred', linewidth=2.5, marker='o', markersize=6,
                    label='Win Rate (%)')

    for i, (date, win_rate) in enumerate(zip(daily_stats['date_str'], daily_stats['win_rate'])):
        ax2.text(i, win_rate + 1, f'{win_rate:.1f}%',
                 ha='center', va='bottom', fontsize=9, fontweight='bold',
                 color='darkred')

    max_pnl = max(abs(daily_stats['total_pnl'].min()), abs(daily_stats['total_pnl'].max()))
    ax1.set_ylim([-max_pnl * 1.2, max_pnl * 1.3])
    ax2.set_ylim([0, 110])

    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Daily PnL', fontsize=12, fontweight='bold', color='blue')
    ax2.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold', color='darkred')

    strategy_name = df['signal_name'].iloc[0] if 'signal_name' in df.columns else 'Strategy'
    total_trades = len(df)
    total_pnl = df['pnl'].sum()
    avg_pnl = df['pnl'].mean()

    plt.title(f'{strategy_name} - Daily Performance\n'
              f'Total Trades: {total_trades} | Total PnL: {total_pnl:.2f} | Avg PnL: {avg_pnl:.2f}',
              fontsize=14, fontweight='bold', pad=20)

    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.xticks(rotation=45, ha='right')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    ax1.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(fig_width, fig_height))
    bar_width = 0.6
    dates = daily_stats['date_str']

    plt.bar(dates, daily_stats['long_count'],
            color='green', alpha=0.7, width=bar_width,
            label='Long Trades', edgecolor='darkgreen', linewidth=1)

    plt.bar(dates, daily_stats['short_count'],
            bottom=daily_stats['long_count'],
            color='red', alpha=0.7, width=bar_width,
            label='Short Trades', edgecolor='darkred', linewidth=1)

    for i, (date, long_cnt, short_cnt, total) in enumerate(zip(dates,
                                                               daily_stats['long_count'],
                                                               daily_stats['short_count'],
                                                               daily_stats['trade_count'])):
        if long_cnt > 0:
            plt.text(i, long_cnt / 2, f'{int(long_cnt)}',
                     ha='center', va='center', fontsize=9, fontweight='bold',
                     color='white')

        if short_cnt > 0:
            plt.text(i, long_cnt + short_cnt / 2, f'{int(short_cnt)}',
                     ha='center', va='center', fontsize=9, fontweight='bold',
                     color='white')

        plt.text(i, total + 0.1, f'Total: {int(total)}',
                 ha='center', va='bottom', fontsize=8, fontstyle='italic',
                 color='gray')

    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Trades', fontsize=12, fontweight='bold')

    long_total = daily_stats['long_count'].sum()
    short_total = daily_stats['short_count'].sum()
    long_ratio = long_total / (long_total + short_total) * 100 if (long_total + short_total) > 0 else 0

    plt.title(f'{strategy_name} - Daily Long/Short Distribution\n'
              f'Total Long: {long_total} ({long_ratio:.1f}%) | Total Short: {short_total} ({100 - long_ratio:.1f}%)',
              fontsize=14, fontweight='bold', pad=20)

    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc='upper left', fontsize=10)

    for i in range(len(dates) - 1):
        plt.axvline(x=i + 0.5, color='gray', alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    plt.show()

    print("=" * 60)
    print("DAILY TRADE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Analysis Period: {len(daily_stats)} trading days")
    print(f"Total Trades: {total_trades}")
    print(f"Total PnL: {total_pnl:.2f}")
    print(f"Average Daily PnL: {daily_stats['total_pnl'].mean():.2f}")
    print(f"Average Daily Trades: {daily_stats['trade_count'].mean():.1f}")
    print(f"Overall Win Rate: {(df['pnl'] > 0).sum() / len(df) * 100:.1f}%")
    print(f"Long/Short Ratio: {long_ratio:.1f}% / {100 - long_ratio:.1f}%")
    print("=" * 60)

    return daily_stats


def evaluate_strategy_performance(trade_log_df, note, performance_file="performance.csv"):

    if trade_log_df is None or len(trade_log_df) == 0:
        print(f"Invalid trade log: {note}")
        return None

    required_cols = ['pnl', 'entry_price']
    missing_cols = [col for col in required_cols if col not in trade_log_df.columns]
    if missing_cols:
        print(f"Trade log missing required columns: {missing_cols}")
        return None

    df = trade_log_df.copy()
    df['return_rate'] = df['pnl'] / df['entry_price']

    total_pnl = df['pnl'].sum()
    total_return = df['return_rate'].sum()
    num_trades = len(df)
    win_rate = (df['pnl'] > 0).mean() if num_trades > 0 else 0

    winning_trades = df[df['pnl'] > 0]
    losing_trades = df[df['pnl'] < 0]
    total_win = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
    total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
    profit_factor = total_win / total_loss if total_loss > 0 else float('inf')

    cumulative_returns = (1 + df['return_rate']).cumprod() - 1
    running_max = (1 + cumulative_returns).expanding().max()
    drawdown = (1 + cumulative_returns) / running_max - 1
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

    trade_volatility = df['return_rate'].std() if num_trades > 1 else 0
    avg_return = df['return_rate'].mean() if num_trades > 0 else 0
    sharpe_ratio = avg_return / trade_volatility if trade_volatility > 0 else 0

    evaluation = {
        'note': note,
        'total_pnl': total_pnl,
        'total_return': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'trade_volatility': trade_volatility,
        'sharpe_ratio': sharpe_ratio,
    }

    save_to_performance_csv(evaluation, performance_file)
    return evaluation


def save_to_performance_csv(evaluation, performance_file="performance.csv"):
    df_eval = pd.DataFrame([evaluation])

    if os.path.exists(performance_file):
        df_eval.to_csv(performance_file, mode='a', header=False, index=False)
    else:
        df_eval.to_csv(performance_file, index=False)

    print(f"✅ evaluation saved to {performance_file}")


def plot_combined_performance_MaxHolding(csv_path="performance.csv"):

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"CSV file is empty: {csv_path}")
            return None

        def extract_tick_number(note_str):
            match = re.search(r'max_hold_ticks_(\d+)', note_str)
            if match:
                return int(match.group(1))
            return None

        df['tick_number'] = df['note'].apply(extract_tick_number)
        df_filtered = df.dropna(subset=['tick_number']).sort_values('tick_number')

        if df_filtered.empty:
            print("No valid data with 'max_hold_ticks_{number}' format found")
            return None

        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 10
        fig, ax1 = plt.subplots(figsize=(12, 7))

        x_positions = df_filtered['tick_number']
        x_ticks = np.arange(len(x_positions))

        bars = ax1.bar(x_ticks, df_filtered['total_return'],
                       alpha=0.7, color='#1f77b4', width=0.6,
                       label='Return')
        ax1.set_xlabel('Max Hold Ticks', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Return', fontsize=12, fontweight='bold', color='#1f77b4')
        ax1.tick_params(axis='y', labelcolor='#1f77b4')

        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_positions.astype(int))

        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{height:.4f}', ha='center', va='bottom',
                     fontsize=8, color='#1f77b4')

        ax2 = ax1.twinx()
        line, = ax2.plot(x_ticks, df_filtered['win_rate'],
                         color='#ff7f0e', marker='o', linewidth=2,
                         markersize=8, label='Win Rate', zorder=5)
        ax2.set_ylabel('Win Rate', fontsize=12, fontweight='bold', color='#ff7f0e')
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')

        plt.title('Return and Win Rate vs Max Hold Ticks',
                  fontsize=14, fontweight='bold', pad=20)

        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
        lines_labels = [bars, line]
        labels = [bar.get_label() for bar in lines_labels]
        ax1.legend(lines_labels, labels, loc='upper left', fontsize=10)

        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.tight_layout()

        combined_plot_png = "combined_performance_max.png"
        plt.savefig(combined_plot_png, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        print(f"Combined chart saved: {combined_plot_png}")

        print("\n📊 Data statistics:")
        print("=" * 50)
        print(f"Data points: {len(df_filtered)}")
        print(f"Min Hold Ticks range: {x_positions.min()} to {x_positions.max()}")
        print(f"Return mean: {df_filtered['total_return'].mean():.4f}")
        print(f"Return max: {df_filtered['total_return'].max():.4f} (Min Hold Ticks={df_filtered.loc[df_filtered['total_return'].idxmax(), 'tick_number']})")
        print(f"Return min: {df_filtered['total_return'].min():.4f} (Min Hold Ticks={df_filtered.loc[df_filtered['total_return'].idxmin(), 'tick_number']})")
        print(f"Win Rate mean: {df_filtered['win_rate'].mean():.4f}")
        print(f"Win Rate max: {df_filtered['win_rate'].max():.4f} (Min Hold Ticks={df_filtered.loc[df_filtered['win_rate'].idxmax(), 'tick_number']})")

        return combined_plot_png

    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return None
    except Exception as e:
        print(f"Error reading or plotting: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_combined_performance_MinHolding(csv_path="performance.csv"):

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"CSV file is empty: {csv_path}")
            return None

        def extract_tick_number(note_str):
            match = re.search(r'min_hold_ticks_(\d+)', note_str)
            if match:
                return int(match.group(1))
            return None

        df['tick_number'] = df['note'].apply(extract_tick_number)
        df_filtered = df.dropna(subset=['tick_number']).sort_values('tick_number')

        if df_filtered.empty:
            print("No valid data with 'min_hold_ticks_{number}' format found")
            return None

        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 10
        fig, ax1 = plt.subplots(figsize=(12, 7))

        x_positions = df_filtered['tick_number']
        x_ticks = np.arange(len(x_positions))

        bars = ax1.bar(x_ticks, df_filtered['total_return'],
                       alpha=0.7, color='#1f77b4', width=0.6,
                       label='Return')
        ax1.set_xlabel('Min Hold Ticks', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Return', fontsize=12, fontweight='bold', color='#1f77b4')
        ax1.tick_params(axis='y', labelcolor='#1f77b4')

        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_positions.astype(int))

        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{height:.4f}', ha='center', va='bottom',
                     fontsize=8, color='#1f77b4')

        ax2 = ax1.twinx()
        line, = ax2.plot(x_ticks, df_filtered['win_rate'],
                         color='#ff7f0e', marker='o', linewidth=2,
                         markersize=8, label='Win Rate', zorder=5)
        ax2.set_ylabel('Win Rate', fontsize=12, fontweight='bold', color='#ff7f0e')
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')

        plt.title('Return and Win Rate vs Min Hold Ticks',
                  fontsize=14, fontweight='bold', pad=20)

        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
        lines_labels = [bars, line]
        labels = [bar.get_label() for bar in lines_labels]
        ax1.legend(lines_labels, labels, loc='upper left', fontsize=10)

        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.tight_layout()

        combined_plot_png = "combined_performance_min.png"
        plt.savefig(combined_plot_png, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        print(f"Combined chart saved: {combined_plot_png}")

        print("\n📊 Data statistics:")
        print("=" * 50)
        print(f"Data points: {len(df_filtered)}")
        print(f"Min Hold Ticks range: {x_positions.min()} to {x_positions.max()}")
        print(f"Return mean: {df_filtered['total_return'].mean():.4f}")
        print(f"Return max: {df_filtered['total_return'].max():.4f} (Min Hold Ticks={df_filtered.loc[df_filtered['total_return'].idxmax(), 'tick_number']})")
        print(f"Return min: {df_filtered['total_return'].min():.4f} (Min Hold Ticks={df_filtered.loc[df_filtered['total_return'].idxmin(), 'tick_number']})")
        print(f"Win Rate mean: {df_filtered['win_rate'].mean():.4f}")
        print(f"Win Rate max: {df_filtered['win_rate'].max():.4f} (Min Hold Ticks={df_filtered.loc[df_filtered['win_rate'].idxmax(), 'tick_number']})")

        return combined_plot_png

    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return None
    except Exception as e:
        print(f"Error reading or plotting: {e}")
        import traceback
        traceback.print_exc()
        return None

# ---------------------------------------------------------------------------
# plot_combined_performance_MaxHolding(csv_path="performance.csv")
# plot_combined_performance_MinHolding(csv_path="performance.csv")


