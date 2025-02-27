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
import json
import logging
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LiveTrader")

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
        logger.info("Initializing Binance data downloader...")
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            logger.error("Missing API credentials")
            raise ValueError("Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        
        try:
            self.client = Client(self.api_key, self.api_secret)
            logger.info("Connected to Binance API successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Binance API: {e}")
            raise
    
    def download_recent_data(self, symbol="BTCUSDT", lookback_hours=24):
        """
        Download recent data for all required timeframes
        """
        logger.info(f"Downloading {lookback_hours} hours of recent data for {symbol}...")
        
        # Define intervals and their client constants
        intervals = {
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '30m': Client.KLINE_INTERVAL_30MINUTE,
            '2h': Client.KLINE_INTERVAL_2HOUR
        }
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_hours)
        
        logger.info(f"Time range: {start_time} to {end_time}")
        
        dataframes = {}
        
        # Download data for each interval
        for interval_name, interval in intervals.items():
            logger.info(f"Downloading {interval_name} data...")
            klines = self.client.get_historical_klines(
                symbol, interval,
                start_time.strftime("%d %b %Y %H:%M:%S"),
                end_time.strftime("%d %b %Y %H:%M:%S")
            )
            
            if not klines:
                logger.warning(f"No data from Binance for {interval_name} timeframe!")
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
            
            logger.info(f"Downloaded {len(df)} {interval_name} candles for {symbol}")
            dataframes[interval_name] = df
        
        return dataframes

class ProfitTracker:
    def __init__(self, initial_balance=10000, commission=0.0005):
        self.initial_balance = float(initial_balance)
        self.current_balance = float(initial_balance)
        self.current_value = float(initial_balance)
        self.commission = float(commission)  # 0.05% per trade
        self.profit_log_file = "profit_log.json"
        self.profit_history = []
        self.entry_price = 0.0
        self.position = 0  # 0: no position, 1: long, -1: short
        self.position_size = 0.0
        self.trade_count = 0
        self.profitable_trades = 0
        self.loss_trades = 0
        self.largest_profit = 0.0
        self.largest_loss = 0.0
        
        # Load existing profit history if file exists
        if os.path.exists(self.profit_log_file):
            try:
                with open(self.profit_log_file, 'r') as f:
                    data = json.load(f)
                    self.profit_history = data.get('history', [])
                    self.current_balance = float(data.get('current_balance', initial_balance))
                    self.current_value = float(data.get('current_value', initial_balance))
                    self.position = int(data.get('position', 0))
                    self.entry_price = float(data.get('entry_price', 0.0))
                    self.position_size = float(data.get('position_size', 0.0))
                    self.trade_count = int(data.get('trade_count', 0))
                    self.profitable_trades = int(data.get('profitable_trades', 0))
                    self.loss_trades = int(data.get('loss_trades', 0))
                    self.largest_profit = float(data.get('largest_profit', 0.0))
                    self.largest_loss = float(data.get('largest_loss', 0.0))
                    
                logger.info(f"Loaded profit history: {len(self.profit_history)} entries")
                logger.info(f"Current balance: ${self.current_balance:.2f}")
                logger.info(f"Current portfolio value: ${self.current_value:.2f}")
                logger.info(f"Current position: {self.position}")
                logger.info(f"Entry price: ${self.entry_price:.2f}")
                logger.info(f"Position size: {self.position_size:.6f}")
            except Exception as e:
                logger.error(f"Error loading profit history: {e}")
                logger.error(traceback.format_exc())
    
    def update(self, action, price):
        """
        Update profit tracker based on action and current price
        """
        # Convert inputs to native Python types to prevent JSON serialization issues
        action = int(action)
        price = float(price)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trade_profit = 0.0
        old_position = self.position
        pnl_percentage = 0.0
        
        # Calculate portfolio value before action
        if self.position == 0:
            old_value = self.current_balance
        elif self.position == 1:  # Long
            old_value = self.current_balance + (self.position_size * price)
        else:  # Short
            old_value = self.current_balance - (self.position_size * price)
        
        logger.info(f"Starting update - Current position: {self.position}, Action: {action}, Price: ${price:.2f}")
        logger.info(f"Pre-update stats - Balance: ${self.current_balance:.2f}, Value: ${old_value:.2f}")
        
        # Process action
        if action == Actions.IDLE:
            # Update valuation for existing positions
            if self.position == 1:  # Long
                self.current_value = self.current_balance + (self.position_size * price)
            elif self.position == -1:  # Short
                self.current_value = self.current_balance - (self.position_size * price)
            else:
                self.current_value = self.current_balance
            logger.info("Action IDLE - Updated valuation")
        
        elif action == Actions.LONG:
            if self.position == 0:
                # Open new long position
                trade_size = self.current_balance * 0.95  # Use 95% of balance
                trade_commission = trade_size * self.commission
                self.position_size = (trade_size - trade_commission) / price
                self.current_balance -= trade_size
                self.entry_price = price
                self.position = 1
                self.trade_count += 1
                logger.info(f"OPENED LONG: {self.position_size} units at ${price:.2f}")
                logger.info(f"Commission paid: ${trade_commission:.2f}")
            
            elif self.position == -1:
                # Close short and open long
                short_profit = self.position_size * (self.entry_price - price)
                close_commission = (self.position_size * price) * self.commission
                self.current_balance += short_profit - close_commission
                trade_profit = short_profit - close_commission
                
                # Then open long position
                trade_size = self.current_balance * 0.95
                trade_commission = trade_size * self.commission
                self.position_size = (trade_size - trade_commission) / price
                self.current_balance -= trade_size
                self.entry_price = price
                self.position = 1
                
                self.trade_count += 1
                if trade_profit > 0:
                    self.profitable_trades += 1
                else:
                    self.loss_trades += 1
                
                self.largest_profit = max(self.largest_profit, trade_profit)
                self.largest_loss = min(self.largest_loss, trade_profit)
                
                logger.info(f"CLOSED SHORT with profit ${trade_profit:.2f} and OPENED LONG: {self.position_size} units at ${price:.2f}")
        
        elif action == Actions.SHORT:
            if self.position == 0:
                # Open new short position
                trade_size = self.current_balance * 0.95
                trade_commission = trade_size * self.commission
                self.position_size = (trade_size - trade_commission) / price
                self.current_balance -= trade_commission
                self.entry_price = price
                self.position = -1
                self.trade_count += 1
                logger.info(f"OPENED SHORT: {self.position_size} units at ${price:.2f}")
                logger.info(f"Commission paid: ${trade_commission:.2f}")
            
            elif self.position == 1:
                # Close long and open short
                long_profit = self.position_size * (price - self.entry_price)
                close_commission = (self.position_size * price) * self.commission
                self.current_balance += (self.position_size * price) - close_commission
                trade_profit = long_profit - close_commission
                
                # Then open short position
                trade_size = self.current_balance * 0.95
                trade_commission = trade_size * self.commission
                self.position_size = (trade_size - trade_commission) / price
                self.current_balance -= trade_commission
                self.entry_price = price
                self.position = -1
                
                self.trade_count += 1
                if trade_profit > 0:
                    self.profitable_trades += 1
                else:
                    self.loss_trades += 1
                
                self.largest_profit = max(self.largest_profit, trade_profit)
                self.largest_loss = min(self.largest_loss, trade_profit)
                
                logger.info(f"CLOSED LONG with profit ${trade_profit:.2f} and OPENED SHORT: {self.position_size} units at ${price:.2f}")
        
        elif action == Actions.CLOSE:
            if self.position == 1:
                # Close long position
                long_profit = self.position_size * (price - self.entry_price)
                close_commission = (self.position_size * price) * self.commission
                self.current_balance += (self.position_size * price) - close_commission
                trade_profit = long_profit - close_commission
                self.position_size = 0
                self.position = 0
                
                self.trade_count += 1
                if trade_profit > 0:
                    self.profitable_trades += 1
                else:
                    self.loss_trades += 1
                
                self.largest_profit = max(self.largest_profit, trade_profit)
                self.largest_loss = min(self.largest_loss, trade_profit)
                
                logger.info(f"CLOSED LONG with profit ${trade_profit:.2f}")
                logger.info(f"Commission paid: ${close_commission:.2f}")
            
            elif self.position == -1:
                # Close short position
                short_profit = self.position_size * (self.entry_price - price)
                close_commission = (self.position_size * price) * self.commission
                self.current_balance += short_profit - close_commission
                trade_profit = short_profit - close_commission
                self.position_size = 0
                self.position = 0
                
                self.trade_count += 1
                if trade_profit > 0:
                    self.profitable_trades += 1
                else:
                    self.loss_trades += 1
                
                self.largest_profit = max(self.largest_profit, trade_profit)
                self.largest_loss = min(self.largest_loss, trade_profit)
                
                logger.info(f"CLOSED SHORT with profit ${trade_profit:.2f}")
                logger.info(f"Commission paid: ${close_commission:.2f}")
        
        # Calculate current portfolio value
        if self.position == 0:
            self.current_value = self.current_balance
        elif self.position == 1:  # Long
            self.current_value = self.current_balance + (self.position_size * price)
        else:  # Short
            self.current_value = self.current_balance - (self.position_size * price)
        
        # Calculate PnL
        pnl = self.current_value - self.initial_balance
        pnl_percentage = (self.current_value / self.initial_balance - 1) * 100
        
        logger.info(f"Post-update stats - Balance: ${self.current_balance:.2f}, Value: ${self.current_value:.2f}")
        logger.info(f"Total P&L: ${pnl:.2f} ({pnl_percentage:.2f}%)")
        
        # Create profit info dictionary
        profit_info = {
            'timestamp': timestamp,
            'action': int(action),
            'price': float(price),
            'trade_profit': float(trade_profit),
            'current_balance': float(self.current_balance),
            'current_value': float(self.current_value),
            'position': int(self.position),
            'position_size': float(self.position_size),
            'entry_price': float(self.entry_price),
            'total_pnl': float(pnl),
            'total_pnl_percentage': float(pnl_percentage)
        }
        
        # Add to history and save
        self.profit_history.append(profit_info)
        self._save_profit_log()
        
        return profit_info
    
    def _save_profit_log(self):
        """Save profit history to file"""
        try:
            # Ensure all values are native Python types for JSON serialization
            data = {
                'history': self.profit_history,
                'current_balance': float(self.current_balance),
                'current_value': float(self.current_value),
                'position': int(self.position),
                'entry_price': float(self.entry_price),
                'position_size': float(self.position_size),
                'initial_balance': float(self.initial_balance),
                'trade_count': int(self.trade_count),
                'profitable_trades': int(self.profitable_trades),
                'loss_trades': int(self.loss_trades),
                'largest_profit': float(self.largest_profit),
                'largest_loss': float(self.largest_loss)
            }
            
            # Create a backup of the previous file
            if os.path.exists(self.profit_log_file):
                backup_file = f"{self.profit_log_file}.bak"
                try:
                    os.replace(self.profit_log_file, backup_file)
                    logger.info(f"Created backup of profit log: {backup_file}")
                except Exception as e:
                    logger.warning(f"Could not create backup: {e}")
            
            # Write the new file
            with open(self.profit_log_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Profit log saved successfully with {len(self.profit_history)} entries")
        except Exception as e:
            logger.error(f"Error saving profit log: {e}")
            logger.error(traceback.format_exc())
    
    def generate_summary(self):
        """Generate a profit summary"""
        if not self.profit_history:
            logger.warning("No trade history available for summary")
            return "No trade history available."
        
        win_rate = self.profitable_trades / self.trade_count * 100 if self.trade_count > 0 else 0
        
        total_profit = self.current_value - self.initial_balance
        profit_percentage = (self.current_value / self.initial_balance - 1) * 100
        
        logger.info(f"Generating summary - Total trades: {self.trade_count}, Win rate: {win_rate:.2f}%")
        logger.info(f"Total P&L: ${total_profit:.2f} ({profit_percentage:.2f}%)")
        
        # Prepare the summary
        summary = f"""
=== TRADING PERFORMANCE SUMMARY ===
Initial Balance: ${self.initial_balance:.2f}
Current Balance: ${self.current_balance:.2f}
Current Portfolio Value: ${self.current_value:.2f}
Total P&L: ${total_profit:.2f} ({profit_percentage:.2f}%)

Total Trades: {self.trade_count}
Profitable Trades: {self.profitable_trades} ({win_rate:.2f}%)
Loss Trades: {self.loss_trades}

Current Position: {"Long" if self.position == 1 else "Short" if self.position == -1 else "None"}
Position Size: {self.position_size:.6f}
Entry Price: ${self.entry_price:.2f}

Largest Profit: ${self.largest_profit:.2f}
Largest Loss: ${self.largest_loss:.2f}
==============================
"""
        return summary

class StateManager:
    """Manages the global state between runs to ensure consistency"""
    
    def __init__(self, state_file="trader_state.json"):
        self.state_file = state_file
        self.state = {
            'last_run_time': None,
            'model_path': None,
            'device': 'cpu',
            'initial_balance': 10000.0,
            'version': 1  # For future migrations if needed
        }
        self.load_state()
        
    def load_state(self):
        """Load state from file if exists"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    loaded_state = json.load(f)
                    self.state.update(loaded_state)
                logger.info(f"Loaded state: Last run time: {self.state['last_run_time']}")
            except Exception as e:
                logger.error(f"Error loading state file: {e}")
                logger.error(traceback.format_exc())
    
    def save_state(self, **kwargs):
        """Save current state to file with optional updates"""
        self.state.update(kwargs)
        self.state['last_run_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            logger.info(f"Saved state: Last run time: {self.state['last_run_time']}")
        except Exception as e:
            logger.error(f"Error saving state file: {e}")
            logger.error(traceback.format_exc())
    
    def get(self, key, default=None):
        """Get value from state"""
        return self.state.get(key, default)

# Global state manager
state_manager = StateManager()

class LiveTrader:
    def __init__(self, model_path, device='cpu', initial_balance=10000):
        """Initialize inference trader with model path"""
        logger.info(f"Initializing trader with model: {model_path}")
        self.model_path = model_path
        self.device = device
        
        # Load the model
        self.net = self._load_model()
        
        # Initialize position state
        self.position = 0  # 0: no position, 1: long, -1: short
        self.fund_rate = 0.0  # Default fund rate
        
        logger.info(f"Model loaded successfully. Using device: {device}")
        
        # Trade history
        self.trades = []
        self.trade_file = "trade_history.csv"
        
        # Initialize profit tracker
        self.profit_tracker = ProfitTracker(initial_balance=initial_balance)
        
        # Sync position with profit tracker state for consistency
        self.position = self.profit_tracker.position
        logger.info(f"Using position from profit tracker: {self.position}")
        
        # Load trade history for reference
        self._load_trade_history()
    
    def _load_trade_history(self):
        """Load trade history from file"""
        if os.path.exists(self.trade_file):
            try:
                trade_history = pd.read_csv(self.trade_file)
                self.trades = trade_history.to_dict('records')
                # Log trade history stats
                logger.info(f"Loaded {len(self.trades)} historical trades")
                if len(self.trades) > 0:
                    last_trade = self.trades[-1]
                    logger.info(f"Last trade: {last_trade['action_name']} at ${last_trade['price']:.2f}")
            except Exception as e:
                logger.error(f"Error loading trade history: {e}")
                logger.error(traceback.format_exc())
    
    def _load_model(self):
        """Load the PPO model with exact architecture matching saved weights"""
        logger.info("Loading PPO model...")
        # Create the actor network with exact input size
        actor = ActorPPO(input_size=338, action_size=4)
        
        try:
            # Load model weights with safety option if available in your PyTorch version
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            except TypeError:  # Older PyTorch versions don't have weights_only
                logger.warning("Using older PyTorch version without weights_only support")
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
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(traceback.format_exc())
            raise
            
        return actor
    
    def _extract_features(self, current_data):
        """
        Extract feature vector from current market data
        
        Args:
            current_data: Dictionary with 15m, 30m, 2h dataframes
        
        Returns:
            Feature tensor of shape [1, 338]
        """
        logger.info("Extracting features from market data...")
        # Initialize an empty feature vector
        features = torch.zeros(338)
        
        try:
            # Get latest data
            df_15m = current_data['15m']
            df_30m = current_data['30m']
            df_2h = current_data['2h']
            
            if df_15m.empty or df_30m.empty or df_2h.empty:
                logger.error("Error: One or more dataframes are empty")
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
            
            logger.info(f"Feature extraction completed. First few values: {features[:5]}")
            logger.info(f"Position feature: {features[336]}, Fund rate feature: {features[337]}")
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            logger.error(traceback.format_exc())
        
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
        logger.info("Predicting action...")
        # Extract features from current data
        input_features = self._extract_features(current_data)
        
        # Get legal actions
        legal_actions = get_legal_actions(self.position)
        action_mask = torch.zeros((1, 4), device=self.device)
        for action in legal_actions:
            action_mask[0, action] = 1
        
        logger.info(f"Current position: {self.position}, Legal actions: {legal_actions}")
        
        # Predict action
        with torch.no_grad():
            dist = self.net(input_features, action_mask=action_mask)
            action = dist.sample().cpu().numpy()[0]
            action_probs = torch.softmax(dist.logits, dim=-1).cpu().numpy()[0]
            
        # Log action probabilities
        action_names = ['IDLE', 'LONG', 'SHORT', 'CLOSE']
        for i, name in enumerate(action_names):
            if action_mask[0, i] > 0:  # Only log legal actions
                logger.info(f"Action {name} probability: {action_probs[i]:.4f}")
        
        # Update position
        old_position = self.position
        self._update_position(action)
        
        # Convert action to name
        logger.info(f"Predicted action: {action_names[action]}")
        
        # Record the trade if position changed
        if old_position != self.position:
            current_price = float(current_data['15m'].iloc[-1]['close'])
            self._record_trade(current_data, action_names[action], action)
        
        return action_names[action], action
    
    def _update_position(self, action):
        """Update position based on predicted action"""
        old_position = self.position  # Store the old position before updating
        self.position = action
        logger.info(f"Position updated from {old_position} to {self.position}")
    
    def _record_trade(self, current_data, action_name, action):
        """Record a trade in the trade history"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trade_info = {
            'timestamp': timestamp,
            'action_name': action_name,
            'action': action,
            'price': float(current_data['15m'].iloc[-1]['close']),
            'position': self.position
        }
        self.trades.append(trade_info)
        self._save_trade_history()
    
    def _save_trade_history(self):
        """Save trade history to file"""
        try:
            trade_history = pd.DataFrame(self.trades)
            trade_history.to_csv(self.trade_file, index=False)
            logger.info(f"Trade history saved successfully with {len(self.trades)} trades")
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")
            logger.error(traceback.format_exc())

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Live Trader with Profit Tracking")
    parser.add_argument('--model', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run inference on (cpu or cuda:0)')
    parser.add_argument('--interval', type=int, default=15, help='Interval in minutes between inference cycles')
    parser.add_argument('--run_now', action='store_true', help='Run inference immediately on startup')
    parser.add_argument('--initial_balance', type=float, default=10000, help='Initial balance for profit tracking')
    
    args = parser.parse_args()
    
    # Initialize trader
    trader = LiveTrader(
        model_path=args.model,
        device=args.device,
        initial_balance=args.initial_balance
    )
    
    # Initialize data downloader
    downloader = BinanceDataDownloader()
    
    def run_inference():
        try:
            # Download recent data
            current_data = downloader.download_recent_data(lookback_hours=24)
            
            # Check if we have valid data
            if any(df.empty for df in current_data.values()):
                logger.error("Received empty data from Binance")
                return
            
            # Get prediction
            action_name, action = trader.predict_action(current_data)
            
            # Get current price
            current_price = float(current_data['15m'].iloc[-1]['close'])
            
            # Update profit tracker
            profit_info = trader.profit_tracker.update(action, current_price)
            
            # Log summary
            logger.info("\n" + trader.profit_tracker.generate_summary())
            
        except Exception as e:
            logger.error(f"Error in inference cycle: {e}")
            logger.error(traceback.format_exc())
    
    # Schedule regular runs
    schedule.every(args.interval).minutes.do(run_inference)
    
    # Run immediately if requested
    if args.run_now:
        run_inference()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()