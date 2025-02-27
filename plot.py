import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import sys
from datetime import datetime

def visualize_trading_performance(csv_path, initial_investment=10000):
    """
    Visualize trading performance with emphasis on net profit/loss display.
    
    Parameters:
    csv_path (str): Path to the CSV file with trading data
    initial_investment (float): Initial investment amount for calculating returns
    
    Returns:
    matplotlib.figure.Figure: The figure containing the visualizations
    """
    print(f"Loading CSV file: {csv_path}")
    print(f"Using initial investment amount: ${initial_investment:.2f}")
    
    # Ensure file exists
    if not os.path.exists(csv_path):
        print(f"ERROR: File {csv_path} does not exist!")
        sys.exit(1)
        
    # Close any existing plots to avoid caching issues
    plt.close('all')
    
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        print(f"CSV loaded successfully with {len(df)} rows and {len(df.columns)} columns")
        print(f"Column names: {df.columns.tolist()}")
        
        # Sample first few rows for debugging
        print("First few rows:")
        print(df.head())

        # Flexible column name handling
        # Try to identify key columns
        datetime_col = None
        action_col = None
        profit_col = None
        
        # Common column name patterns
        datetime_patterns = ['datetime', 'date', 'time', 'timestamp']
        action_patterns = ['action', 'decision', 'trade', 'signal']
        profit_patterns = ['profit', 'returns', 'pnl', 'gain']
        
        # Check actual column names
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in datetime_patterns):
                datetime_col = col
            elif any(pattern in col_lower for pattern in action_patterns):
                action_col = col
            elif any(pattern in col_lower for pattern in profit_patterns):
                profit_col = col
        
        # If we couldn't identify columns by name, try to guess based on position
        if datetime_col is None and len(df.columns) >= 2:
            datetime_col = df.columns[1]
        if action_col is None and len(df.columns) >= 3:
            action_col = df.columns[2]
        if profit_col is None and len(df.columns) >= 4:
            profit_col = df.columns[3]
            
        print(f"Using columns: datetime={datetime_col}, action={action_col}, profit={profit_col}")
        
        # Create a cleaned DataFrame with standardized columns
        cleaned_df = pd.DataFrame()
        
        # Convert datetime strings to actual datetime objects if possible
        try:
            if pd.api.types.is_object_dtype(df[datetime_col]):
                # Replace underscores with spaces if they exist
                if df[datetime_col].str.contains('_').any():
                    cleaned_df['datetime'] = pd.to_datetime(
                        df[datetime_col].str.replace('_', ' '), 
                        format='%Y-%m-%d %H-%M-%S', 
                        errors='coerce'
                    )
                else:
                    cleaned_df['datetime'] = pd.to_datetime(df[datetime_col], errors='coerce')
            else:
                cleaned_df['datetime'] = pd.to_datetime(df[datetime_col], errors='coerce')
        except Exception as e:
            print(f"Warning: Could not convert {datetime_col} to datetime format. Using original values. Error: {e}")
            cleaned_df['datetime'] = df[datetime_col]
        
        # Copy action column
        cleaned_df['action'] = df[action_col]
        
        # Copy and ensure profit column is numeric
        try:
            cleaned_df['profit'] = pd.to_numeric(df[profit_col], errors='coerce')
        except Exception as e:
            print(f"Warning: Could not convert {profit_col} to numeric format. Error: {e}")
            cleaned_df['profit'] = df[profit_col]
        
        # Add index column
        cleaned_df['index'] = range(len(cleaned_df))
        
        # Replace cleaned DataFrame
        df = cleaned_df
        
        # Sort by datetime if possible
        if pd.api.types.is_datetime64_dtype(df['datetime']):
            df = df.sort_values('datetime')
        
        # Calculate cumulative profit
        df['cumulative_profit'] = df['profit'].cumsum()
        
        # Standardize action names (case-insensitive)
        action_mapping = {
            'long': 'Long',
            'short': 'Short',
            'close': 'Close',
            'idle': 'Idle',
            'buy': 'Long',
            'sell': 'Short',
            'hold': 'Idle',
            'exit': 'Close'
        }
        
        if pd.api.types.is_object_dtype(df['action']):
            df['action'] = df['action'].str.lower().map(lambda x: action_mapping.get(x, x))
        
        # Create color mapping for different actions
        color_map = {
            'Long': 'green',
            'Short': 'red',
            'Close': 'gray',
            'Idle': 'blue'
        }
        
        # Create marker mapping for different actions
        marker_map = {
            'Long': '^',
            'Short': 'v',
            'Close': 'o',
            'Idle': '.'
        }
        
        # Calculate date column for daily aggregation
        if pd.api.types.is_datetime64_dtype(df['datetime']):
            df['date'] = df['datetime'].dt.date
        else:
            # If datetime isn't actually a datetime, try to extract date portion
            try:
                df['date'] = df['datetime'].astype(str).str.split(' ').str[0]
            except:
                print("Warning: Could not parse dates for daily aggregation.")
                df['date'] = 'Unknown'
        
        # Calculate profit by day
        try:
            daily_profit = df.groupby('date')['profit'].sum().reset_index()
            print(f"Daily profit calculated successfully for {len(daily_profit)} days")
        except Exception as e:
            print(f"Warning: Could not calculate daily profit. Error: {e}")
            daily_profit = pd.DataFrame(columns=['date', 'profit'])
        
        # Calculate key statistics
        total_trades = len(df)
        profitable_trades = len(df[df['profit'] > 0])
        win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0
        avg_profit = df['profit'].mean()
        max_profit = df['profit'].max()
        min_profit = df['profit'].min()
        total_profit = df['profit'].sum()
        
        # Calculate return on investment
        roi_percent = (total_profit / initial_investment) * 100
        final_value = initial_investment + total_profit
        
        # Create a figure with multiple subplots
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(20, 15))
        
        # Create a prominent title with net profit/loss information
        title_color = 'green' if total_profit >= 0 else 'red'
        title_text = f'Trading Bot Performance Analysis\n{os.path.basename(csv_path)}\n'
        title_text += f'NET PROFIT/LOSS: ${total_profit:.2f} ({roi_percent:.2f}%)'
        
        fig.suptitle(title_text, fontsize=20, color=title_color, weight='bold')
        
        # Define our grid: 3 rows, 2 columns
        gs = fig.add_gridspec(3, 2)
        
        # Cumulative Profit Chart with Trade Actions
        ax1 = fig.add_subplot(gs[0, :])
        line, = ax1.plot(range(len(df)), df['cumulative_profit'], 'b-', linewidth=2.5, alpha=0.8)
        
        # Color the line based on final profit
        line.set_color('green' if total_profit >= 0 else 'red')
        
        ax1.set_title('Cumulative Profit Over Time', fontsize=16)
        ax1.set_xlabel('Trade Number', fontsize=12)
        ax1.set_ylabel('Cumulative Profit ($)', fontsize=12)
        
        # Add a horizontal line at y=0
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        
        # Add a background color to easily see profit/loss regions
        ax1.axhspan(0, df['cumulative_profit'].max() * 1.1 if df['cumulative_profit'].max() > 0 else 10, 
                   facecolor='green', alpha=0.1)
        ax1.axhspan(df['cumulative_profit'].min() * 1.1 if df['cumulative_profit'].min() < 0 else -10, 
                   0, facecolor='red', alpha=0.1)
        
        # Annotate the final profit
        final_profit = df['cumulative_profit'].iloc[-1]
        color = 'green' if final_profit >= 0 else 'red'
        
        ax1.annotate(f'Final Profit: ${final_profit:.2f}', 
                    xy=(len(df)-1, final_profit),
                    xytext=(len(df)*0.8, final_profit * (1.1 if final_profit > 0 else 0.9)),
                    color=color,
                    weight='bold',
                    fontsize=14,
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))
        
        # Mark significant points (max profit, min profit)
        max_idx = df['cumulative_profit'].idxmax()
        min_idx = df['cumulative_profit'].idxmin()
        
        ax1.scatter(max_idx, df['cumulative_profit'].max(), color='green', s=100, 
                   marker='*', label=f'Max Profit: ${df["cumulative_profit"].max():.2f}')
        ax1.scatter(min_idx, df['cumulative_profit'].min(), color='red', s=100, 
                   marker='*', label=f'Max Drawdown: ${df["cumulative_profit"].min():.2f}')
        
        # Individual Trade Actions
        if pd.api.types.is_object_dtype(df['action']):
            for action in color_map.keys():
                if action in df['action'].values:
                    action_df = df[df['action'] == action]
                    ax1.scatter(action_df.index, action_df['cumulative_profit'], 
                              color=color_map.get(action, 'black'), 
                              marker=marker_map.get(action, 'o'), 
                              s=30, alpha=0.6, label=action)
            
            ax1.legend(loc='best', fontsize=12)
        
        # Daily Profit Chart
        ax2 = fig.add_subplot(gs[1, 0])
        
        if len(daily_profit) > 0:
            # Convert date to position index if needed
            if not pd.api.types.is_datetime64_dtype(daily_profit['date']):
                pos = range(len(daily_profit))
            else:
                pos = daily_profit['date']
                
            bars = ax2.bar(pos, daily_profit['profit'], alpha=0.7)
            
            # Color bars based on profit value
            for i, bar in enumerate(bars):
                bar.set_color('green' if daily_profit['profit'].iloc[i] >= 0 else 'red')
            
            ax2.set_title('Daily Profit', fontsize=16)
            ax2.set_xlabel('Day' if not pd.api.types.is_datetime64_dtype(daily_profit['date']) else 'Date', fontsize=12)
            ax2.set_ylabel('Profit ($)', fontsize=12)
            
            # Add a horizontal line at y=0
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
            
            # Label the most profitable and least profitable days
            if len(daily_profit) > 0:
                best_day_idx = daily_profit['profit'].idxmax()
                worst_day_idx = daily_profit['profit'].idxmin()
                
                best_day = daily_profit.iloc[best_day_idx]
                worst_day = daily_profit.iloc[worst_day_idx]
                
                # Add annotations
                ax2.annotate(f'Best: ${best_day["profit"]:.2f}',
                            xy=(pos[best_day_idx] if hasattr(pos, '__getitem__') else best_day_idx, best_day['profit']),
                            xytext=(0, 20), textcoords='offset points',
                            color='green', weight='bold', ha='center')
                
                ax2.annotate(f'Worst: ${worst_day["profit"]:.2f}',
                            xy=(pos[worst_day_idx] if hasattr(pos, '__getitem__') else worst_day_idx, worst_day['profit']),
                            xytext=(0, -20), textcoords='offset points',
                            color='red', weight='bold', ha='center')
        else:
            ax2.text(0.5, 0.5, 'No daily profit data available', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax2.transAxes)
        
        # Net Profit and Key Metrics - NEW PANEL SPECIFICALLY FOR NET PROFIT
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')  # Turn off axis
        
        # Create a table-like visual display of key metrics
        profit_color = 'green' if total_profit >= 0 else 'red'
        
        # Create text content with initial investment info
        metric_text = f"""
        INITIAL INVESTMENT: ${initial_investment:.2f}
        FINAL VALUE: ${final_value:.2f}
        
        NET PROFIT/LOSS: ${total_profit:.2f}
        RETURN ON INVESTMENT: {roi_percent:.2f}%
        
        Win Rate: {win_rate:.1f}%
        Profitable Trades: {profitable_trades} of {total_trades}
        
        Average Profit per Trade: ${avg_profit:.2f}
        Largest Single Profit: ${max_profit:.2f}
        Largest Single Loss: ${min_profit:.2f}
        """
        
        # Add text to the center of the panel
        ax3.text(0.5, 0.5, metric_text, 
                 horizontalalignment='center', 
                 verticalalignment='center',
                 transform=ax3.transAxes, 
                 fontsize=18,
                 color=profit_color,
                 weight='bold',
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
        
        # Trading Actions Distribution
        ax4 = fig.add_subplot(gs[2, 0])
        
        # Count actions
        if pd.api.types.is_object_dtype(df['action']):
            action_counts = df['action'].value_counts()
            action_profits = df.groupby('action')['profit'].sum()
            
            # Create a stacked bar chart showing counts and their contribution to profit
            actions = action_counts.index
            counts = action_counts.values
            profits = [action_profits.get(action, 0) for action in actions]
            
            # Define colors
            colors = [color_map.get(action, 'gray') for action in actions]
            
            # Plot bars for counts
            ax4.bar(actions, counts, color=colors, alpha=0.7)
            
            # Add labels with profit contributions
            for i, (action, count, profit) in enumerate(zip(actions, counts, profits)):
                ax4.text(i, count + 0.1, f'${profit:.2f}', 
                       ha='center', va='bottom',
                       color='green' if profit >= 0 else 'red',
                       weight='bold')
            
            ax4.set_title('Action Distribution and Profit Contribution', fontsize=16)
            ax4.set_xlabel('Action Type', fontsize=12)
            ax4.set_ylabel('Count', fontsize=12)
        else:
            ax4.text(0.5, 0.5, 'No action distribution data available', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax4.transAxes)
        
        # Profit Distribution Histogram
        ax5 = fig.add_subplot(gs[2, 1])
        
        # Ensure we have numeric profit values
        valid_profits = df['profit'].dropna()
        if len(valid_profits) > 0 and pd.api.types.is_numeric_dtype(valid_profits):
            # Create separate histograms for profits and losses
            profits = valid_profits[valid_profits >= 0]
            losses = valid_profits[valid_profits < 0]
            
            bins = min(50, len(valid_profits)//5+1)
            
            if len(profits) > 0:
                ax5.hist(profits, bins=bins, alpha=0.7, color='green', edgecolor='black', label='Profits')
            
            if len(losses) > 0:
                ax5.hist(losses, bins=bins, alpha=0.7, color='red', edgecolor='black', label='Losses')
            
            ax5.set_title('Profit/Loss Distribution', fontsize=16)
            ax5.set_xlabel('Amount ($)', fontsize=12)
            ax5.set_ylabel('Frequency', fontsize=12)
            ax5.axvline(x=0, color='k', linestyle='-', alpha=0.5)
            ax5.legend()
            
            # Add statistics to the plot
            textstr = f"""
            Mean Profit: ${valid_profits.mean():.2f}
            Median: ${valid_profits.median():.2f}
            Std Dev: ${valid_profits.std():.2f}
            """
            
            # Place a text box
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax5.text(0.05, 0.95, textstr, transform=ax5.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
        else:
            ax5.text(0.5, 0.5, 'No valid profit data for histogram', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax5.transAxes)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.90])  # Adjust to make room for title
        
        # Save the figure
        output_dir = os.path.dirname(csv_path)
        output_file = os.path.join(output_dir, f"trading_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        print(f"Analysis saved to: {output_file}")
        
        return fig
    
    except Exception as e:
        print(f"Error visualizing trading performance: {e}")
        import traceback
        traceback.print_exc()
        
        # Create a simple error figure
        fig = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error processing CSV file:\n\n{str(e)}\n\nCheck console for details.", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=12)
        plt.title("Error Visualizing Trading Performance")
        return fig

# Function to allow running the script directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize trading bot performance from CSV file.')
    parser.add_argument('csv_path', type=str, nargs='?', help='Path to the CSV file with trading data')
    parser.add_argument('--initial', type=float, default=10000.0, help='Initial investment amount (default: $10,000)')
    args = parser.parse_args()
    
    # Use provided path or ask for input if not provided
    if args.csv_path:
        csv_path = args.csv_path
    else:
        # Prompt user for input
        csv_path = input("Enter the full path to your CSV file: ")
        csv_path = csv_path.strip('"').strip("'")  # Remove quotes if they were included
    
    # Get initial investment amount
    initial_investment = args.initial
    
    print(f"Visualizing trading performance for: {csv_path}")
    print(f"Using initial investment amount: ${initial_investment:.2f}")
    
    # Visualize the trading performance
    fig = visualize_trading_performance(csv_path, initial_investment)
    
    # Show the plot if running interactively
    plt.show()