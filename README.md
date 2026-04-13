# High-frequency-trading-simple-framework
A lightweight HFT backtesting framework for order-book strategies. Trades at counterparty prices, ignoring slippage. Provides data preprocessing, feature engineering, strategy templates, and risk controls (rebalance limits, holding periods, stop-loss). Outputs performance metrics. Supports rapid strategy prototyping.

**main_batch.py**<br>
>Batch backtest for Hong Kong stock trading strategies, testing multiple strategies on 60+ stocks.<br>
>
>*Core Functions*
>* load_all_days_simple: Merge multi-day stock data
>* process_stock_data: Process single stock (feature engineering)
>* evaluate_stock_strategy: Evaluate single strategy performance (P&L, trade count, win rate)
>* main: Main function, execute batch backtesting to 60+ HK stocks
>
>*Data*
>* Location: `F:\HKdata`subfolders
>* Dates: 2025-04-01 to 2025-04-10 (7 trading days)
>* Format: CSV Level 2 data
>
>*Output*
>* Console: Backtest progress and results
>* summary.csv: Detailed records of profitable strategies<br>
>
>| stock | strategy | total_pnl | num_trades | win_rate |
>|-------|----------|-----------|------------|----------|
>| 00700.csv | orderflow_imbalance | 10.999999999999943 | 56 | 0.5 |
>| 00700.csv | microprice_momentum | 25.0 | 16 | 0.625 |
>| 00883.csv | volatility_breakout_reversed | 0.0199999999999995 | 3 | 0.6666666666666666 |
>| 09985.csv | weighted_microprice_reversal | 7.9600000000000115 | 315 | 0.4380952380952381 |
>| 01810.csv | trade_flow_imbalance | 1.5500000000000114 | 64 | 0.546875 |
>| 01810.csv | microprice_depth_convergence | 0.5999999999999872 | 21 | 0.5714285714285714 |
>| 00939.csv | microprice_momentum | 0.0199999999999995 | 1 | 1.0 |
>| 02382.csv | trade_flow_imbalance | 3.550000000000012 | 40 | 0.6 |


**main_single.py**<br>
>Single-stock strategy testing script for Hong Kong stocks. Analyzes and visualizes trading strategy performance for a specific stock.<br>
>Complete workflow: Data loading → processing → signals → backtesting → visualization
>
>*Core Functions*
>* load_all_days_simple(): Load multi-day stock data
>* clean_price_anomalies(): Data cleaning
>* plot_price_volume_combined(): Price-volume visualization
>* add_l2_and_orderflow_features(): Add L2/order flow features
>* add_low_frequency_features(): Add technical indicators
>* generate_signals(): Generate trading signals
>* filter_premarket_signals(): Filter pre-market trades
>* backtest_cross_spread_with_log(): Backtest with detailed logging
>
>Data
>* Data source: `F:\HKdata`folders
>* Trading days: 2025-04-01 to 2025-04-10 (7 days)
>* Sample stock: 02015.csv(Li Auto-W)
>* Strategy: volatility_breakout_reversed
>  
>*Key Parameters*
>* stock: Stock filename (default: "02015.csv")
>* signal_name: Strategy name (default: 'volatility_breakout_reversed')
>* every_n_ticks: Plot frequency (default: 7200 ticks)
>* figsize: Plot dimensions (default: 20×8 inches)
>
>*Output*
>* Console: Column info, processing logs
>* Plots: Price-volume signal chart + PnL chart
>* CSV file: LOG.csvwith trade details
>* Stats: Backtesting statistics dictionary
