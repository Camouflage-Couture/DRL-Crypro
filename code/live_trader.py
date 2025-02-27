import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import argparse
import torch
import torch.nn as nn
from binance.client import Client
import matplotlib.pyplot as plt
from PIL import Image
import glob
from dotenv import load_dotenv
import schedule

# Load environment variables
load_dotenv()

# Actions enum
class Actions:
    IDLE = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3

def get_legal_actions(position):
    """Get legal actions based on current position"""
    if position == 0:  # No position
        legal_actions = [Actions.IDLE, Actions.LONG, Actions.SHORT]
    elif position == 1 or position == -1:  # Long or short position
        legal_actions = [Actions.IDLE, Actions.CLOSE]
    
    # Return action values, not the enum objects
    return legal_actions

# Define model matching your trained weights
class ActorPPO(nn.Module):
    def __init__(self, input_size=338, action_size=4):
        super(ActorPPO, self).__init__()
        # No CNN backbone in this model - it uses raw features
        self.actor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
    
    def forward(self, x, account=None, action_mask=None):
        # For raw data models, x is already a feature vector
        logits = self.actor(x)
        
        if action_mask is not None:
            logits[action_mask == 0] = -1e9  # Mask out illegal actions
        
        return torch.distributions.Categorical(logits=logits)

class BinanceDataDownloader:
    def __init__(self, api_key=None, api_secret=None):
        print("Initializing Binance data downloader...")
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        
        try:
            self.client = Client(self.api_key, self.api_secret)
            print("Connected to Binance API successfully")
        except Exception as e:
            print(f"Failed to connect to Binance API: {e}")
            raise
    
    def download_recent_data(self, symbol="BTCUSDT", lookback_hours=24):
        """
        Download recent data for all required timeframes
        """
        print(f"Downloading {lookback_hours} hours of recent data for {symbol}...")
        
        # Define intervals and their client constants
        intervals = {
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '30m': Client.KLINE_INTERVAL_30MINUTE,
            '2h': Client.KLINE_INTERVAL_2HOUR
        }
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_hours)
        
        print(f"Time range: {start_time} to {end_time}")
        
        dataframes = {}
        
        # Download data for each interval
        for interval_name, interval in intervals.items():
            print(f"Downloading {interval_name} data...")
            klines = self.client.get_historical_klines(
                symbol, interval,
                start_time.strftime("%d %b %Y %H:%M:%S"),
                end_time.strftime("%d %b %Y %H:%M:%S")
            )
            
            if not klines:
                print(f"No data from Binance for {interval_name} timeframe!")
                dataframes[interval_name] = pd.DataFrame()
                continue
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert string columns to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Add Moving Averages
            for period in [5, 20, 50]:
                df[f'MA_{period}'] = df['close'].rolling(window=period, min_periods=1).mean()
            
            # Add scaled close price
            df['scaled'] = df['close'] / 100000
            
            print(f"Downloaded {len(df)} {interval_name} candles for {symbol}")
            dataframes[interval_name] = df
        
        return dataframes

class LiveTrader:
    def __init__(self, model_path, device='cpu'):
        """Initialize inference trader with model path"""
        print(f"Initializing trader with model: {model_path}")
        self.model_path = model_path
        self.device = device
        
        # Load the model
        self.net = self._load_model()
        
        # Initialize position state
        self.position = 0  # 0: no position, 1: long, -1: short
        self.fund_rate = 0.0  # Default fund rate
        
        print(f"Model loaded successfully. Using device: {device}")
        
        # Trade history
        self.trades = []
        self.trade_file = "trade_history.csv"
        
        # Load previous trades if file exists
        if os.path.exists(self.trade_file):
            try:
                trade_history = pd.read_csv(self.trade_file)
                self.trades = trade_history.to_dict('records')
                # Get the latest position
                if len(self.trades) > 0:
                    self.position = self.trades[-1].get('position', 0)
                    print(f"Loaded previous position: {self.position}")
            except Exception as e:
                print(f"Error loading trade history: {e}")
    
    def _load_model(self):
        """Load the PPO model with exact architecture matching saved weights"""
        print("Loading PPO model...")
        # Create the actor network with exact input size
        actor = ActorPPO(input_size=338, action_size=4)
        
        # Load model weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle state dict mismatch
        new_state_dict = {}
        for key, value in checkpoint['net'].items():
            if key.startswith('actor'):
                new_state_dict[key] = value
        
        # Load the modified state dict
        actor.load_state_dict(new_state_dict)
        actor.to(self.device)
        actor.eval()
        
        print("Model loaded successfully")
        return actor
    
    def _extract_features(self, current_data):
        """
        Extract feature vector from current market data
        
        Args:
            current_data: Dictionary with 15m, 30m, 2h dataframes
        
        Returns:
            Feature tensor of shape [1, 338]
        """
        print("Extracting features from market data...")
        # Initialize an empty feature vector
        features = torch.zeros(338)
        
        try:
            # Get latest data
            df_15m = current_data['15m']
            df_30m = current_data['30m']
            df_2h = current_data['2h']
            
            if df_15m.empty or df_30m.empty or df_2h.empty:
                print("Error: One or more dataframes are empty")
                return features.unsqueeze(0).to(self.device)
            
            # Get the latest candle from 15m
            latest_15m = df_15m.iloc[-1]
            
            # Basic OHLCV features
            features[0] = float(latest_15m['close'] / latest_15m['open'])  # ratio
            features[1] = float(latest_15m['close'])  # close price
            features[2] = float(latest_15m['high'])   # high price
            
            # Calculate volatility (simple version: (high-low)/low)
            recent = df_15m.iloc[-5:]
            avg_volatility = float(((recent['high'] - recent['low'])/recent['low']).mean())
            features[3] = avg_volatility
            
            # Add hour of day as a feature
            current_hour = datetime.now().hour
            features[4] = current_hour / 24.0
            
            # Add moving averages
            features[5] = float(latest_15m['MA_5'])
            features[6] = float(latest_15m['MA_20'])
            features[7] = float(latest_15m['MA_50'])
            
            # Add some trend features
            # Simple: current close vs previous close
            if len(df_15m) > 1:
                prev_close = df_15m.iloc[-2]['close']
                features[8] = float(latest_15m['close'] / prev_close - 1)  # percent change
            
            # More detailed feature extraction can be added here...
            
            # Add account state features at the end
            features[336] = self.position
            features[337] = self.fund_rate
            
            print(f"Feature extraction completed. First few values: {features[:5]}")
        except Exception as e:
            print(f"Error extracting features: {e}")
        
        return features.unsqueeze(0).to(self.device)
    
    def predict_action(self, current_data):
        """
        Predict action from current market data
        
        Args:
            current_data: Dictionary with 15m, 30m, 2h dataframes
            
        Returns:
            action_name: Name of the predicted action
            action: Action index
        """
        print("Predicting action...")
        # Extract features from current data
        input_features = self._extract_features(current_data)
        
        # Get legal actions
        legal_actions = get_legal_actions(self.position)
        action_mask = torch.zeros((1, 4), device=self.device)
        for action in legal_actions:
            action_mask[0, action] = 1
        
        print(f"Current position: {self.position}, Legal actions: {legal_actions}")
        
        # Predict action
        with torch.no_grad():
            dist = self.net(input_features, action_mask=action_mask)
            action = dist.sample().cpu().numpy()[0]
        
        # Update position
        old_position = self.position
        self._update_position(action)
        
        # Convert action to name
        action_names = ['IDLE', 'LONG', 'SHORT', 'CLOSE']
        print(f"Predicted action: {action_names[action]}")
        
        # Record the trade if position changed
        if old_position != self.position:
            self._record_trade(current_data, action_names[action], action)
        
        return action_names[action], action
    
    def _update_position(self, action):
        """Update position based on action"""
        old_position = self.position
        
        if action == Actions.IDLE:
            # Position stays the same
            pass
        elif action == Actions.LONG:
            if self.position == 0:
                self.position = 1
            elif self.position == -1:
                self.position = 1  # Close short and open long
        elif action == Actions.SHORT:
            if self.position == 0:
                self.position = -1
            elif self.position == 1:
                self.position = -1  # Close long and open short
        elif action == Actions.CLOSE:
            self.position = 0
            
        if old_position != self.position:
            print(f"Position changed: {old_position} -> {self.position}")
        else:
            print(f"Position unchanged: {self.position}")
    
    def _record_trade(self, current_data, action_name, action):
        """Record trade in history"""
        try:
            # Get current timestamp and price
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            price = current_data['15m'].iloc[-1]['close'] if not current_data['15m'].empty else 0
            
            # Create trade record
            trade = {
                'timestamp': timestamp,
                'action': action,
                'action_name': action_name,
                'price': price,
                'position': self.position
            }
            
            self.trades.append(trade)
            
            # Save to CSV
            trade_df = pd.DataFrame(self.trades)
            trade_df.to_csv(self.trade_file, index=False)
            
            print(f"Trade recorded: {action_name} at {price} -> Position: {self.position}")
        except Exception as e:
            print(f"Error recording trade: {e}")

def run_inference_cycle(model_path, device='cpu'):
    """Run a complete inference cycle: download data, predict action"""
    print("\n===== STARTING INFERENCE CYCLE =====")
    start_time = time.time()
    
    try:
        # Step 1: Download latest data
        print("Step 1: Downloading latest market data...")
        downloader = BinanceDataDownloader()
        dataframes = downloader.download_recent_data(lookback_hours=24)
        
        # Check if we have enough data
        if any(df.empty for df in dataframes.values()):
            print("Not enough data for inference. Skipping cycle.")
            return None
        
        # Step 2: Run inference
        print("Step 2: Running inference with latest data...")
        trader = LiveTrader(model_path=model_path, device=device)
        action_name, action = trader.predict_action(dataframes)
        
        # Step 3: Log the action
        latest_price = dataframes['15m'].iloc[-1]['close'] if not dataframes['15m'].empty else 0
        print(f"\nINFERENCE RESULT: Action={action_name}, Position={trader.position}, Price={latest_price}")
        
        # Here you could add code to execute the trade on Binance if you want
        
        end_time = time.time()
        print(f"Cycle completed in {end_time - start_time:.2f} seconds")
        print("===== INFERENCE CYCLE COMPLETED =====\n")
        
        return action_name, trader.position
    
    except Exception as e:
        print(f"Error in inference cycle: {e}")
        print("===== INFERENCE CYCLE FAILED =====\n")
        return None

def scheduled_job(model_path, device):
    """Scheduled job that runs every 15 minutes"""
    print(f"\n===== SCHEDULED JOB - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
    result = run_inference_cycle(model_path, device)
    
    if result:
        action, position = result
        print(f"Summary - Action: {action}, Position: {position}")
    else:
        print("No result from inference cycle")

def main():
    parser = argparse.ArgumentParser(description="Simple Live Trader")
    parser.add_argument('--model', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run inference on (cpu or cuda:0)')
    parser.add_argument('--interval', type=int, default=15, help='Interval in minutes between inference cycles')
    parser.add_argument('--run_now', action='store_true', help='Run inference immediately on startup')
    
    args = parser.parse_args()
    
    print("===== STARTING SIMPLE LIVE TRADER =====")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Interval: {args.interval} minutes")
    
    # Run immediately if requested
    if args.run_now:
        print("Running immediate inference...")
        scheduled_job(args.model, args.device)
    
    # Schedule regular runs
    print(f"Scheduling job to run every {args.interval} minutes")
    schedule.every(args.interval).minutes.do(scheduled_job, model_path=args.model, device=args.device)
    
    print(f"Trader is now running. Press Ctrl+C to exit.")
    print("Waiting for next scheduled run...")
    
    # Keep running until keyboard interrupt
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
    
    print("===== TRADER STOPPED =====")

if __name__ == "__main__":
    main()