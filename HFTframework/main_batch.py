import os
import warnings
warnings.filterwarnings('ignore')
from single_stock_no_strtegy import *
from single_trade_signal import *


def load_all_days_simple(csv_file, days_list=None):
    if days_list is None:
        days_list = ['20250401', '20250402', '20250403', '20250407', '20250408', '20250409', '20250410']

    all_days_data = []

    for day in days_list:
        if day == '20250410':
            data_dir = r"F:\HKdata\hk 10\_data\stock\20250410\hk"
        else:
            day_num = int(day[-2:])
            folder_name = f"hk 0{day_num}" if 1 <= day_num <= 9 else f"hk {day_num}"
            data_dir = os.path.join(r"F:\HKdata", folder_name, "_data", "stock", day, "hk")

        file_path = os.path.join(data_dir, csv_file)

        if not os.path.exists(file_path):
            print(f"⚠️ File does not exist: {file_path}")
            continue

        df = load_l2_ticks(file_path)

        all_days_data.append(df)

    combined_df = pd.concat(all_days_data, axis=0)
    combined_df.sort_values('timestamp', inplace=True)
    combined_df = combined_df.reset_index(drop=True)

    return combined_df


def process_stock_data(stock_id):
    my_days = ['20250401', '20250402', '20250403', '20250407', '20250408', '20250409', '20250410']

    try:
        df = load_all_days_simple(stock_id, my_days)

        if df is None or len(df) == 0:
            print(f"    ⚠️ The data is empty. Skip this stock.")
            return None

        df = add_l2_and_orderflow_features(df)
        df = df.set_index('timestamp')
        df = add_low_frequency_features(df)
        df = add_rolling_volatility_features(df)
        df = df.reset_index()

        return df

    except Exception as e:
        print(f"    ❌ Data processing failed: {str(e)[:50]}...")
        return None


def evaluate_stock_strategy(stock_id, signal_name, stock_df):

    target_pos = generate_signals(stock_df, signal_name)
    pnl, stats = backtest_cross_spread(stock_df, target_pos)

    print("Stats:", stats)

    if stats.get('total_pnl', 0) > 0:

        summary_data = {
            'stock': stock_id,
            'strategy': signal_name,
            'total_pnl': stats.get('total_pnl', 0),
            'num_trades': stats.get('num_trades', 0),
            'win_rate': stats.get('win_rate', 0)
        }

        for key, value in stats.items():
            if key not in summary_data:
                summary_data[key] = value

        import pandas as pd
        summary_df = pd.DataFrame([summary_data])

        try:
            existing_df = pd.read_csv('summary.csv')
            combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
        except FileNotFoundError:
            combined_df = summary_df

        combined_df.to_csv('summary.csv', index=False)
        print(f"save summary.csv")

    return stats


def main():
    stock_pool = [
        # === Popular Stocks ===
        "02513.csv",  # Zhipu (AI large model)
        "09988.csv",  # Alibaba-W (Technology/Internet leader)
        "00700.csv",  # Tencent Holdings (Technology leader/HK market bellwether) [3,10](@ref)
        "01729.csv",  # HuiJu Tech (Optical communication/AI hardware)
        "06809.csv",  # Montage Tech (Memory chips)
        "01347.csv",  # Huahong Semiconductor (Semiconductor sector) [1,3](@ref)
        "06869.csv",  # Yangtze Optical (Optical communication) [7](@ref)
        "09903.csv",  # Tianshu Zhixin (Chip concept)
        "01815.csv",  # Everest Gold (Gold hedging/2025 top performer) [5,11](@ref)
        "02099.csv",  # China Gold Intl (Resources) [1](@ref)
        "09995.csv",  # RemeGen (Innovative pharma leader/2025 double) [1,7](@ref)
        "00358.csv",  # Jiangxi Copper (Resources/2025 top performer) [1,7](@ref)
        "03690.csv",  # Meituan-W (Hang Seng Tech) [10](@ref)
        "00941.csv",  # China Mobile (High dividend, common blue-chip) [3](@ref)
        "01299.csv",  # AIA (Financial weight) [10](@ref)
        "00005.csv",  # HSBC Holdings (Financial/high dividend core) [10](@ref)
        "00883.csv",  # CNOOC (Energy high dividend) [10](@ref)
        "00388.csv",  # HKEX (Bull market flag/brokerage representative) [10,11](@ref)
        "02318.csv",  # Ping An (Comprehensive finance) [10](@ref)
        "00981.csv",  # SMIC (Foundry leader) [3,8](@ref)
        "09992.csv",  # Pop Mart (New consumer/2025 up 200%+) [2,6](@ref)
        "01211.csv",  # BYD (NEV leader) [10](@ref)
        "02899.csv",  # Zijin Mining (Non-ferrous leader/2025 up 140%+) [8,10](@ref)
        "01024.csv",  # Kuaishou-W (Short video/AI application) [3,10](@ref)
        "00998.csv",  # China CITIC Bank (Chinese bank representative)
        "02628.csv",  # China Life (Chinese insurance/2025 strong) [8](@ref)
        "06181.csv",  # Lao Pu Gold (New consumer/2025 H1 up 300%+) [2,9](@ref)
        "01810.csv",  # Xiaomi-W (Tech/auto/phones) [3,6](@ref)
        "02020.csv",  # Anta Sports (Sportswear leader)
        "02382.csv",  # Sunny Optical (Phone supply chain)
        # === Tech & Internet Leaders ===
        "09999.csv",  # NetEase-S (Tech leader)
        "09618.csv",  # JD.com-SW (E-commerce/logistics)
        "09898.csv",  # XPeng-W (NEV)
        "02015.csv",  # Li Auto-W (NEV)
        "03896.csv",  # Kingsoft Cloud (Cloud computing/AI)
        "01788.csv",  # Guotai Junan Intl (Chinese brokerage)
        "00763.csv",  # ZTE (Telecom equipment)

        # === Innovative Pharma & Biotech (18A & popular) ===
        "06160.csv",  # BeiGene (Innovative pharma global leader)
        "01801.csv",  # Innovent Bio (Innovative pharma/weight loss drugs)
        "09926.csv",  # Akeso (Bispecific antibody leader)
        "03692.csv",  # Hansoh Pharma (Innovation transformation)
        "01177.csv",  # Sino Biopharm (Integrated pharma)
        "01093.csv",  # CSPC Pharma (Integrated pharma)
        "06990.csv",  # Kelun-BTB (ADC drugs)
        "02171.csv",  # CARsgen (CAR-T therapy)
        "02197.csv",  # Clover Biopharma (Vaccines/RSV)
        "01877.csv",  # Junshi Biosciences (Innovative pharma)

        # === New Consumer & Beverage ===
        "02097.csv",  # Mixue Group (New tea chain IPO)
        "09991.csv",  # Baozun (Brand e-commerce)
        "02331.csv",  # Li Ning (Guochao sportswear)
        "02313.csv",  # Shenzhou International (Apparel manufacturing)
        "01929.csv",  # Chow Tai Fook (Jewelry retail)
        "02145.csv",  # Shanghai Jahwa (Skincare/beauty)
        "09985.csv",  # Weilong Delicious (Snack food)

        # === Non-ferrous & Resources (Cyclicals) ===
        "03993.csv",  # CMOC (Copper/cobalt resources)
        "01378.csv",  # China Hongqiao (Aluminum)
        "02600.csv",  # CHALCO (Aluminum)
        "01208.csv",  # MMG (Copper mining)
        "01258.csv",  # CNMC (Copper)
        "03330.csv",  # Lingbao Gold (Gold)
        "00340.csv",  # Tongguan Gold (Gold)

        # === Finance & High Dividend ===
        "00939.csv",  # CCB (Big bank/high dividend)
        "01398.csv",  # ICBC (Big bank/high dividend)
        "02628.csv",  # China Life (Insurance, if not duplicate)
        "02318.csv",  # Ping An (Comprehensive finance, if not duplicate)
        "01339.csv",  # PICC Group (Insurance)
        "02888.csv",  # Standard Chartered (International bank)

        # === Other Popular & Segment Leaders ===
        "01772.csv",  # Ganfeng Lithium (Lithium battery)
        "02269.csv",  # WuXi Biologics (CXO)
        "06078.csv",  # Hygeia Healthcare (Medical services)
        "01530.csv",  # 3SBio (Innovative pharma)
        "09868.csv",  # XPeng (if needed as backup)
    ]

    stock_pool = list(set(stock_pool))

    print(f"Starting batch testing")
    print(f"Stock pool: {len(stock_pool)} stocks")
    print(stock_pool)
    print(f"Strategy count: {len(SIGNAL_REGISTRY)}")
    print("=" * 50)

    profitable_count = 0

    for i, stock in enumerate(stock_pool, 1):
        print(f"\n{'=' * 50}")
        print(f"Processing stock {i}/{len(stock_pool)}: {stock}")

        s_df = process_stock_data(stock)
        if s_df is None:
            print(f"    ⏭️ Skipping this stock, moving to next")
            continue

        for signal_name in SIGNAL_REGISTRY.keys():
            print(f"\n  → Strategy: {signal_name}")

            try:
                stats = evaluate_stock_strategy(stock, signal_name, s_df)

                if stats and stats.get('total_pnl', 0) > 0:
                    profitable_count += 1
                    print(f"    ✅ Profitable: {stats['total_pnl']:.2f}")
                else:
                    pnl = stats.get('total_pnl', 0) if stats else 0
                    print(f"    ❌ Loss: {pnl:.2f}")

            except Exception as e:
                print(f"    💥 Error: {str(e)[:50]}...")
                continue

    print(f"\n{'=' * 50}")
    print(f"Batch testing completed")
    print(f"Total profitable strategies: {profitable_count}")
    print(f"Total tested combinations: {len(stock_pool) * len(SIGNAL_REGISTRY)}")

    try:
        import pandas as pd
        if os.path.exists('summary.csv'):
            summary_df = pd.read_csv('summary.csv')
            print(f"\nsummary.csv content:")
            print(f"Record count: {len(summary_df)}")
            if len(summary_df) > 0:
                print("\nFirst 5 records:")
                print(summary_df.head().to_string())
    except:
        pass




if __name__ == "__main__":
    main()