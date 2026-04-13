import warnings
warnings.filterwarnings('ignore')
from main_batch import *
from performance_analyse import *

def main():

    my_days = ['20250401', '20250402', '20250403', '20250407', '20250408', '20250409', '20250410']

    stock = "02015.csv"
    stock_code = stock[:-4]
    signal_name = 'volatility_breakout_reversed'
    df = load_all_days_simple(stock, my_days)
    print("=" * 50)
    print(df.columns)
    print("=" * 50)
    df = clean_price_anomalies(df)


    # 1. The target price chart and strategy signal chart
    plot_price_volume_combined(
        df=df,
        trade_log_path="LOG.csv",
        stock_id=stock_code,
        output_dir='test_plots',
        every_n_ticks=7200,
        figsize=(20, 8)
    )

    # 2. Data processing and feature engineering
    df = add_l2_and_orderflow_features(df)
    df = df.set_index('timestamp')
    df = add_low_frequency_features(df)
    # df = add_rolling_volatility_features(df)
    df = df.reset_index()

    # 3. Strategies and Signals
    target_pos = generate_signals(df, signal_name)
    target_pos = filter_premarket_signals(df, target_pos, debug=True)

    # 4. Backtesting
    # pnl, stats = backtest_cross_spread(df, target_pos)
    pnl, stats, log = backtest_cross_spread_with_log(df, target_pos, signal_name, log_trades=True)
    log.save_to_csv()

    # --------------------------- Ablation Experiment ---------------------------
    # trade_df = log.get_trade_log_df()
    # evaluation = evaluate_strategy_performance(trade_df, "no_channel_position")
    # ---------------------------------------------------------------------------

    # 5. PnL Plot
    plt.figure(figsize=(16, 6))

    cumulative_pnl = np.cumsum(pnl)
    plt.plot(cumulative_pnl, color='#1f77b4', linewidth=2.0, label='Cumulative PnL')
    plt.fill_between(range(len(cumulative_pnl)), cumulative_pnl,
                     color='#87CEEB', alpha=0.3)

    if len(cumulative_pnl) > 0:
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown_idx = np.argmax(drawdown)

        if max_drawdown_idx > 0:
            peak_idx = np.argmax(cumulative_pnl[:max_drawdown_idx + 1])

            recovery_idx = max_drawdown_idx
            for i in range(max_drawdown_idx, len(cumulative_pnl)):
                if cumulative_pnl[i] >= cumulative_pnl[peak_idx]:
                    recovery_idx = i
                    break

            plt.scatter(peak_idx, cumulative_pnl[peak_idx],
                        color='red', s=100, zorder=5, marker='o',
                        label=f'Peak (Start of Max Drawdown)')

            plt.scatter(max_drawdown_idx, cumulative_pnl[max_drawdown_idx],
                        color='darkred', s=100, zorder=5, marker='o',
                        label=f'Max Drawdown: {drawdown[max_drawdown_idx]:.2f}')

            if recovery_idx > max_drawdown_idx:
                plt.scatter(recovery_idx, cumulative_pnl[recovery_idx],
                            color='green', s=100, zorder=5, marker='o',
                            label='Recovery')

            if recovery_idx > peak_idx:
                plt.axvspan(peak_idx, recovery_idx, alpha=0.2, color='red',
                            label=f'Max Drawdown Period: {recovery_idx - peak_idx} trades')

            plt.plot([peak_idx, max_drawdown_idx],
                     [cumulative_pnl[peak_idx], cumulative_pnl[max_drawdown_idx]],
                     color='red', linestyle='--', linewidth=1.5, alpha=0.7)

            if recovery_idx > max_drawdown_idx:
                plt.plot([max_drawdown_idx, recovery_idx],
                         [cumulative_pnl[max_drawdown_idx], cumulative_pnl[recovery_idx]],
                         color='green', linestyle='--', linewidth=1.5, alpha=0.7)

    plt.grid(True, color='gray', linestyle='--', linewidth=0.6, alpha=0.9)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.6)

    plt.title(f"Cumulative PnL (per unit) [{signal_name}]", fontsize=12, fontweight='bold')
    plt.xlabel("Trade No.", fontsize=10)
    plt.ylabel("PnL", fontsize=10)
    plt.legend(fontsize=8, loc='best')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()