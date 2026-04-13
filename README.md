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

<small>  
  
>>| stock | strategy | total_pnl | num_trades | win_rate |
>>|-------|----------|-----------|------------|----------|
>>| 00700.csv | orderflow_imbalance | 10.999999999999943 | 56 | 0.5 |
>>| 00700.csv | microprice_momentum | 25.0 | 16 | 0.625 |
>>| 00883.csv | volatility_breakout_reversed | 0.0199999999999995 | 3 | 0.6666666666666666 |
>>| 09985.csv | weighted_microprice_reversal | 7.9600000000000115 | 315 | 0.4380952380952381 |
>>| 01810.csv | trade_flow_imbalance | 1.5500000000000114 | 64 | 0.546875 |
>>| 01810.csv | microprice_depth_convergence | 0.5999999999999872 | 21 | 0.5714285714285714 |
>>| 00939.csv | microprice_momentum | 0.0199999999999995 | 1 | 1.0 |
>>| 02382.csv | trade_flow_imbalance | 3.550000000000012 | 40 | 0.6 |
</small>

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
>* CSV file: LOG.csvwith trade details

<small>  

>>| trade_id | signal_name | direction | entry_time | entry_price | entry_type | exit_time | exit_price | exit_type | pnl | duration_seconds |
>>|----------|----------------------|-----------|---------------------|-------------|------------|---------------------|------------|-----------|-------------------|-------------------|
>>| 1 | volatility_breakout_reversed | long | 2025_04_01_09:30:22 | 98.8 | ask_open | 2025_04_01_09:35:24 | 99.2 | bid_close | 0.4000000000000057 | 302.0 |
>>| 2 | volatility_breakout_reversed | long | 2025_04_02_09:33:11 | 99.7 | ask_open | 2025_04_02_09:38:20 | 100.1 | bid_close | 0.3999999999999915 | 309.0 |
>>| 3 | volatility_breakout_reversed | short | 2025_04_02_09:39:35 | 99.7 | bid_open | 2025_04_02_09:44:38 | 99.55 | ask_close | 0.15000000000000568 | 303.0 |
>>| 4 | volatility_breakout_reversed | long | 2025_04_03_09:32:02 | 98.4 | ask_open | 2025_04_03_09:37:04 | 98.8 | bid_close | 0.3999999999999915 | 302.0 |
>>| 5 | volatility_breakout_reversed | long | 2025_04_03_09:37:05 | 98.85 | ask_open | 2025_04_03_09:42:08 | 98.7 | bid_close | -0.14999999999999147 | 303.0 |
>>| 6 | volatility_breakout_reversed | long | 2025_04_03_10:01:23 | 98.0 | ask_open | 2025_04_03_10:06:24 | 97.3 | bid_close | -0.7000000000000028 | 301.0 |
>>| 7 | volatility_breakout_reversed | short | 2025_04_03_10:06:25 | 97.3 | bid_open | 2025_04_03_10:11:25 | 97.3 | ask_close | 0.0 | 300.0 |
>>| 8 | volatility_breakout_reversed | short | 2025_04_07_09:37:14 | 87.5 | bid_open | 2025_04_07_09:42:14 | 86.6 | ask_close | 0.9000000000000057 | 300.0 |
</small>

>
>* Plots: Price-volume signal chart + PnL chart
>* Stats: Backtesting statistics dictionary

**single_stock_no_strategy.py**<br>
>Core trading system module. Handles data processing, feature engineering, and backtesting without​ specific trading strategies.
>
>*Core Functions*
>* load_l2_ticks()​ - Load and parse CSV data
>* add_l2_and_orderflow_features()​ - Add 100+ OB features
>* add_low_frequency_features()​ - Add minute-level features
>* add_rolling_volatility_features()​ - Add volatility metrics
>* generate_signals()​ - Process raw signals into positions
>* backtest_cross_spread()​ - Execute trading strategy
>* backtest_cross_spread_with_log()​ - Backtest with logging
>* filter_premarket_signals()​ - Remove 9:00-9:30 signals
>
>*Trade Loggers*
>* TradeLogger​ class: Record all trade details
>* Outputs: Entry/exit time, price, P&L, duration
>* Saves to: trade_log_[strategy].csv
