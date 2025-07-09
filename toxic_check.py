import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import glob
import os

from sklearn.linear_model import LinearRegression
import pull_data
warnings.filterwarnings('ignore')

class BinanceCounterPartyPNL:
    def __init__(self, csv_pattern: str = "SOLUSDT-aggTrades-*.csv"):
        """
        Initialize with a pattern to match multiple CSV files
        
        Args:
            csv_pattern: Pattern to match CSV files (e.g., "SOLUSDT-aggTrades-*.csv")
        """
        self.csv_pattern = os.path.join("data", csv_pattern)
        self.symbol = "SOLUSDT"
        self.combined_data = None
        
    def find_csv_files(self) -> list:
        """Find all CSV files matching the pattern"""
        csv_files = glob.glob(self.csv_pattern)
        csv_files.sort()  # Sort to ensure chronological order
        
        print(f"Found {len(csv_files)} CSV files matching pattern '{self.csv_pattern}':")
        for file in csv_files:
            print(f"  - {file}")
        
        return csv_files
    
    def load_and_combine_csv_data(self) -> pd.DataFrame:
        """Load and combine multiple CSV files"""
        csv_files = self.find_csv_files()
        
        if not csv_files:
            print(f"No CSV files found matching pattern: {self.csv_pattern}")
            return pd.DataFrame()
        
        combined_df = pd.DataFrame()
        
        for file in csv_files:
            try:
                print(f"Loading {file}...")
                df = pd.read_csv(file)
                
                # Add source file info
                df['source_file'] = os.path.basename(file)
                
                # Combine with existing data
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                
                print(f"  - Loaded {len(df)} trades from {file}")
                
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
        
        if combined_df.empty:
            print("No data loaded successfully")
            return pd.DataFrame()
        
        print(f"\nCombined dataset: {len(combined_df)} total trades")
        print(f"Columns: {list(combined_df.columns)}")
        
        # Convert data types
        combined_df['price'] = combined_df['price'].astype(float)
        combined_df['quantity'] = combined_df['quantity'].astype(float)
        combined_df['is_buyer_maker'] = combined_df['is_buyer_maker'].astype(bool)
        
        # Convert timestamp to datetime
        combined_df['transact_time'] = pd.to_datetime(combined_df['transact_time'], unit='ms')
        
        # Sort by time to ensure chronological order
        combined_df = combined_df.sort_values('transact_time').reset_index(drop=True)
        
        # Remove any potential duplicates based on timestamp and trade info
        print(f"Removing duplicates...")
        original_len = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['transact_time', 'price', 'quantity', 'is_buyer_maker'])
        final_len = len(combined_df)
        
        if original_len != final_len:
            print(f"Removed {original_len - final_len} duplicate trades")
        
        print(f"Final dataset: {len(combined_df)} trades")
        print(f"Data range: {combined_df['transact_time'].min()} to {combined_df['transact_time'].max()}")
        print(f"Duration: {combined_df['transact_time'].max() - combined_df['transact_time'].min()}")
        
        self.combined_data = combined_df
        return combined_df
    
    def calculate_counterparty_pnl(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate counterparty PnL assuming we take the maker side of every trade.
        
        Corrected Logic:
        - If is_buyer_maker=True (buyer is maker), we buy (+ quantity)
        - If is_buyer_maker=False (seller is maker), we sell (- quantity)
        """

        if df.empty:
            print("No data to process")
            return pd.DataFrame()

        result_df = df.copy()

        # Corrected maker side qty sign
        result_df['our_qty'] = np.where(result_df['is_buyer_maker'], 
                                    result_df['quantity'],    # Buyer is maker → buy
                                    -result_df['quantity'])   # Seller is maker → sell

        # Side label for clarity
        result_df['our_side'] = np.where(result_df['is_buyer_maker'], 'buy', 'sell')

        # Calculate cumulative position
        result_df['cumulative_qty'] = result_df['our_qty'].cumsum()

        # Calculate cumulative cost basis (sum of qty * price)
        result_df['cumulative_cost'] = (result_df['our_qty'] * result_df['price']).cumsum()

        # Calculate average price only when cumulative_qty != 0
        result_df['avg_price'] = np.where(
            result_df['cumulative_qty'] != 0,
            result_df['cumulative_cost'] / result_df['cumulative_qty'],
            np.nan
        )

        # Forward fill avg_price when position is flat (optional, for plotting consistency)
        result_df['avg_price'] = result_df['avg_price'].ffill()

        # Current price for unrealized PnL is the last trade price
        current_price = result_df['price'].iloc[-1]

        # Unrealized PnL = position * (current price - avg price)
        # For zero position, unrealized PnL is zero
        result_df['unrealized_pnl'] = np.where(
            result_df['cumulative_qty'] != 0,
            result_df['cumulative_qty'] * (current_price - result_df['avg_price']),
            0
        )

    
        result_df['position_value'] = result_df['cumulative_qty'] * result_df['avg_price']

       
        result_df['trade_pnl_impact'] = result_df['unrealized_pnl'].diff().fillna(0)

        # Rolling metrics
        window_size = max(100, len(result_df) // 100)
        result_df['rolling_pnl'] = result_df['unrealized_pnl'].rolling(window=window_size, min_periods=1).mean()
        result_df['rolling_qty'] = result_df['cumulative_qty'].rolling(window=window_size, min_periods=1).mean()
        result_df['pnl_volatility'] = result_df['unrealized_pnl'].rolling(window=window_size, min_periods=1).std()
        result_df['price_vs_avg'] = result_df['price'] - result_df['avg_price']

        return result_df
    
    def compute_toxicity_score(self, df):
        """
        Uses linear regression to estimate PnL slope.
        Returns a toxicity classification and slope value.
        """
        df = df.copy()
        df = df.dropna(subset=['unrealized_pnl'])
        if len(df) < 2:
            return "Insufficient data", 0.0

        # Convert timestamps to numeric
        df['time_numeric'] = pd.to_datetime(df['transact_time']).astype(np.int64) // 10**9

        X = df['time_numeric'].values.reshape(-1, 1)
        y = df['unrealized_pnl'].values

        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]

        # Classify based on slope
        if slope < -0.001:
            toxicity = "Toxic"
        elif slope > 0.001:
            toxicity = "Benign"
        else:
            toxicity = "Neutral"

        return toxicity, slope
    
    def create_pnl_plots(self, df: pd.DataFrame, save_path: str = None):
        """Create comprehensive PnL plots"""
        if df.empty:
            print("No data to plot")
            return
        
        # Create a simplified figure with 2 rows and 2 columns (only 3 plots used)
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)

        # Main title
        duration = df['transact_time'].max() - df['transact_time'].min()
        unique_files = df['source_file'].nunique() if 'source_file' in df.columns else 1
        fig.suptitle(f'Counterparty PnL Analysis - Combined Data\n'
                    f'Duration: {duration} | Total Trades: {len(df):,} | Files: {unique_files}\n'
                    f'(Assuming maker side of every aggregate trade)', 
                    fontsize=20, fontweight='bold', y=0.98)

        # Plot 1: Cumulative PnL over time (main plot - spans top row)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df['transact_time'], df['unrealized_pnl'], color='#1f77b4', linewidth=1.5, alpha=0.8)
        ax1.set_title('Cumulative Unrealized PnL Over Time', fontweight='bold', fontsize=16)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('PnL (USDT)')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        ax1.fill_between(df['transact_time'], df['unrealized_pnl'], 0, 
                        where=(df['unrealized_pnl'] >= 0), 
                        color='green', alpha=0.3, interpolate=True, label='Profit')
        ax1.fill_between(df['transact_time'], df['unrealized_pnl'], 0, 
                        where=(df['unrealized_pnl'] < 0), 
                        color='red', alpha=0.3, interpolate=True, label='Loss')
        ax1.legend()

        # Add PnL statistics
        final_pnl = df['unrealized_pnl'].iloc[-1]
        max_pnl = df['unrealized_pnl'].max()
        min_pnl = df['unrealized_pnl'].min()
        ax1.text(0.02, 0.98, f'Final PnL: ${final_pnl:,.2f}\nMax PnL: ${max_pnl:,.2f}\nMin PnL: ${min_pnl:,.2f}', 
                transform=ax1.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=12)

        # Plot 2: Rolling Average PnL
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(df['transact_time'], df['rolling_pnl'], color='#ff7f0e', linewidth=2)
        ax3.set_title('Rolling Average PnL', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('PnL (USDT)')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.7)

        # Plot 3: Daily PnL by File or PnL Volatility
        ax8 = fig.add_subplot(gs[1, 1])
        if 'source_file' in df.columns and df['source_file'].nunique() > 1:
            daily_pnl = df.groupby('source_file')['unrealized_pnl'].last().reset_index()
            daily_pnl['date'] = daily_pnl['source_file'].str.extract(r'(\d{4}-\d{2}-\d{2})')
            daily_pnl = daily_pnl.sort_values('date')

            ax8.bar(range(len(daily_pnl)), daily_pnl['unrealized_pnl'], 
                    color=['green' if x > 0 else 'red' for x in daily_pnl['unrealized_pnl']], alpha=0.7)
            ax8.set_title('PnL by File/Day', fontweight='bold', fontsize=14)
            ax8.set_xlabel('File')
            ax8.set_ylabel('PnL (USDT)')
            ax8.set_xticks(range(len(daily_pnl)))
            ax8.set_xticklabels(daily_pnl['date'], rotation=45)
            ax8.grid(True, alpha=0.3)
        else:
            ax8.plot(df['transact_time'], df['pnl_volatility'], color='darkred', linewidth=1.5)
            ax8.set_title('PnL Volatility', fontweight='bold', fontsize=14)
            ax8.set_xlabel('Time')
            ax8.set_ylabel('PnL Std Dev')
            ax8.grid(True, alpha=0.3)
            ax8.tick_params(axis='x', rotation=45)

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        plt.show()
    
    def print_detailed_summary(self, df: pd.DataFrame):
        """Print comprehensive summary statistics"""
        if df.empty:
            print("No data available")
            return
        
        print("\n" + "="*70)
        print("DETAILED COUNTERPARTY PnL ANALYSIS - COMBINED DATA")
        print("="*70)
        
        # Basic info
        print(f"Symbol: {self.symbol}")
        print(f"CSV Pattern: {self.csv_pattern}")
        unique_files = df['source_file'].nunique() if 'source_file' in df.columns else 1
        print(f"Files Combined: {unique_files}")
        if 'source_file' in df.columns:
            print("Files included:")
            for file in sorted(df['source_file'].unique()):
                file_count = (df['source_file'] == file).sum()
                print(f"  - {file}: {file_count:,} trades")
        
        print(f"Analysis Period: {df['transact_time'].min()} to {df['transact_time'].max()}")
        print(f"Duration: {df['transact_time'].max() - df['transact_time'].min()}")
        
        # Trade statistics
        total_trades = len(df)
        
        print(f"\nTRADE STATISTICS:")
        print(f"Total Aggregate Trades: {total_trades:,}")
        
        # Price information
        start_price = df['price'].iloc[0]
        end_price = df['price'].iloc[-1]
        price_change = end_price - start_price
        price_change_pct = (price_change / start_price) * 100
        
        print(f"\nPRICE INFORMATION:")
        print(f"Start Price: ${start_price:,.2f}")
        print(f"End Price: ${end_price:,.2f}")
        print(f"Price Change: ${price_change:,.2f} ({price_change_pct:+.2f}%)")
        print(f"Price Range: ${df['price'].min():,.2f} - ${df['price'].max():,.2f}")
                
        # PnL analysis
        final_pnl = df['unrealized_pnl'].iloc[-1]
        max_pnl = df['unrealized_pnl'].max()
        min_pnl = df['unrealized_pnl'].min()
        avg_pnl = df['unrealized_pnl'].mean()
        pnl_volatility = df['unrealized_pnl'].std()
        
        print(f"\nPnL ANALYSIS:")
        print(f"Final Unrealized PnL: ${final_pnl:,.2f}")
        print(f"Maximum PnL: ${max_pnl:,.2f}")
        print(f"Minimum PnL: ${min_pnl:,.2f}")
        print(f"Average PnL: ${avg_pnl:,.2f}")
        print(f"PnL Volatility: ${pnl_volatility:,.2f}")
        
        # Performance metrics
        positive_pnl_pct = (df['unrealized_pnl'] > 0).mean() * 100
        profitable_trades = (df['trade_pnl_impact'] > 0).sum()
        losing_trades = (df['trade_pnl_impact'] < 0).sum()
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"Time in Profit: {positive_pnl_pct:.1f}%")
        print(f"Profitable Trade Impacts: {profitable_trades:,}")
        print(f"Losing Trade Impacts: {losing_trades:,}")
        
        # Side analysis
        buy_trades = df[~df['is_buyer_maker']]
        sell_trades = df[df['is_buyer_maker']]
        
        print(f"\nSIDE ANALYSIS:")
        print(f"Trades where we bought: {len(buy_trades):,} ({len(buy_trades)/total_trades*100:.1f}%)")
        print(f"Trades where we sold: {len(sell_trades):,} ({len(sell_trades)/total_trades*100:.1f}%)")
        
        # Risk metrics
        max_drawdown = (df['unrealized_pnl'].cummax() - df['unrealized_pnl']).max()
        sharpe_ratio = (df['trade_pnl_impact'].mean() / df['trade_pnl_impact'].std()) if df['trade_pnl_impact'].std() > 0 else 0
        
        print(f"\nRISK METRICS:")
        print(f"Maximum Drawdown: ${max_drawdown:,.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        toxic, slope = self.compute_toxicity_score(df)
        print(f'Toxic Flow is {toxic} with slope {slope}')
        print("="*70)
    
    def save_combined_data(self, filename: str = None):
        """Save the combined data to a CSV file"""
        if self.combined_data is None:
            print("No combined data to save. Run load_and_combine_csv_data() first.")
            return
        
        if filename is None:
            filename = f"combined_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        self.combined_data.to_csv(filename, index=False)
        print(f"Combined data saved to: {filename}")
        return filename
    
    def run_analysis(self, save_plot: bool = True, save_combined_data: bool = True):
        """Run complete analysis on multiple CSV files"""
        print("Starting Counterparty PnL Analysis from Multiple CSV Files...")
        
        # Load and combine data
        df = self.load_and_combine_csv_data()
        if df.empty:
            print("Failed to load data")
            return None
        
        # Save combined data if requested
        if save_combined_data:
            self.save_combined_data()
        
        # Calculate counterparty PnL
        print("Calculating counterparty PnL...")
        analysis_df = self.calculate_counterparty_pnl(df)
        
        if analysis_df.empty:
            print("Failed to calculate PnL")
            return None
        
        # Print detailed summary
        self.print_detailed_summary(analysis_df)
        
        # Create plots
        save_path = f"counterparty_pnl_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png" if save_plot else None
        self.create_pnl_plots(analysis_df, save_path)
        
        return analysis_df

# Example usage
if __name__ == "__main__":
    # Initialize analyzer with pattern to match your CSV files
    START_DATE = "2025-07-06"
    END_DATE = "2025-07-08"
    SYMBOL = "ROSEUSDT"
    pull_data.main(START_DATE, END_DATE, SYMBOL)
    csv_pattern = f"{SYMBOL}-aggTrades-*.csv"  # This will match all your files
    
    try:
        analyzer = BinanceCounterPartyPNL(csv_pattern)
        df = analyzer.run_analysis(save_plot=True, save_combined_data=True)
        if df is not None:
            print(f"\nAnalysis complete! Combined dataset contains {len(df)} trades.")
            print(f"Data covers period: {df['transact_time'].min()} to {df['transact_time'].max()}")
            
            # You can also access the raw combined data
            if analyzer.combined_data is not None:
                print(f"Raw combined data shape: {analyzer.combined_data.shape}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")