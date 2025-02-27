import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Configuration
MAX_PRICE = 100000  # Used for price scaling
PERIODS = [5, 20, 50]  # MA periods
COLUMNS = ['open', 'high', 'low', 'close']  # Basic OHLC columns
OUTPUT_SIZE = (224, 224)  # Output image size as per paper

class CandlestickDataDownloader:
    def __init__(self, api_key=None, api_secret=None):
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        
        self.client = Client(self.api_key, self.api_secret)
    
    def download_data(self, symbol="BTCUSDT", days=30):
        """
        Download data for all required timeframes (15m, 30m, 2h)
        """
        print(f"Downloading {days} days of data for {symbol}...")
        
        # Define intervals and their client constants
        intervals = {
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '30m': Client.KLINE_INTERVAL_30MINUTE,
            '2h': Client.KLINE_INTERVAL_2HOUR
        }
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        dataframes = {}
        
        # Download data for each interval
        for interval_name, interval in intervals.items():
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
            for col in COLUMNS:
                df[col] = df[col].astype(float)
            
            # Add Moving Averages
            for period in PERIODS:
                df[f'MA_{period}'] = df['close'].rolling(window=period, min_periods=1).mean()
            
            # Add scaled close price
            df['scaled'] = df['close'] / MAX_PRICE
            
            print(f"Downloaded {len(df)} {interval_name} candles for {symbol}")
            dataframes[interval_name] = df
        
        return dataframes


class MultiResolutionImageGenerator:
    def __init__(self, dataframes):
        """
        Initialize with dataframes containing all required timeframes
        
        Args:
            dataframes: Dict with keys '15m', '30m', '2h' containing respective dataframes
        """
        self.dataframes = dataframes
        self.dataframes_half = {k: dataframes[k] for k in ['30m', '2h']}
        self.dataframes_all = {k: dataframes[k] for k in ['15m']}
    
    def generate_image(self, index, output_dir="./inference_images"):
        """
        Generate a multi-resolution candlestick image for inference
        
        Args:
            index: Index in the 15m dataframe to use as endpoint
            output_dir: Directory to save the generated image
        
        Returns:
            Path to the generated image
        """
        # Make sure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Calculate index for all timeframes
        # We need to align the data by time
        time_15m = self.dataframes['15m'].index[index]
        
        # Find corresponding indices in other timeframes
        # Using the closest time that's not in the future
        index_30m = self.dataframes['30m'].index.get_indexer([time_15m], method='pad')[0]
        index_2h = self.dataframes['2h'].index.get_indexer([time_15m], method='pad')[0]
        
        # Define slices (24 for 15m, 12 for 30m, 12 for 2h)
        df_15m = self.dataframes['15m'].iloc[max(0, index-23):index+1].copy()
        df_30m = self.dataframes['30m'].iloc[max(0, index_30m-11):index_30m+1].copy()
        df_2h = self.dataframes['2h'].iloc[max(0, index_2h-11):index_2h+1].copy()
        
        # Make sure we have enough data
        if len(df_15m) < 24 or len(df_30m) < 12 or len(df_2h) < 12:
            # Pad with NaN if necessary
            if len(df_15m) < 24:
                df_15m = pd.concat([pd.DataFrame(index=range(24-len(df_15m))), df_15m])
            if len(df_30m) < 12:
                df_30m = pd.concat([pd.DataFrame(index=range(12-len(df_30m))), df_30m])
            if len(df_2h) < 12:
                df_2h = pd.concat([pd.DataFrame(index=range(12-len(df_2h))), df_2h])
        
        # Create the image
        return self._create_multi_resolution_image(df_15m, df_30m, df_2h, output_dir)
    
    def _create_multi_resolution_image(self, df_15m, df_30m, df_2h, output_dir):
        """
        Create a single multi-resolution candlestick image
        """
        # Create blank image (224x224 RGB)
        img_width, img_height = OUTPUT_SIZE
        image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        # Height allocations according to paper
        top_row_height = 90
        bottom_row_height = 134
        
        # Candle dimensions
        candle_width = 6
        candle_spacing = 3
        
        # Create blue channel for MAs
        ma_image = np.zeros((img_height, img_width), dtype=np.uint8)
        
        # Calculate section dimensions
        section_2h_width = img_width // 2
        section_2h_height = top_row_height
        section_2h_left = img_width // 2
        
        section_30m_width = img_width // 2
        section_30m_height = top_row_height
        section_30m_left = 0
        
        section_15m_width = img_width
        section_15m_height = bottom_row_height
        section_15m_left = 0
        section_15m_top = top_row_height
        
        # Draw ratio bar between 30-min and 2-h areas
        try:
            latest_close = df_30m['close'].iloc[-1]
            ratio_height = int((latest_close / MAX_PRICE) * top_row_height)
            ma_image[0:ratio_height, img_width//2-1:img_width//2+1] = 255
        except:
            # Handle empty dataframe or other issues
            pass
        
        # Draw 2-hour candles (top-right)
        self._draw_candles(
            df_2h, image, ma_image,
            section_left=section_2h_left,
            section_top=0,
            section_height=section_2h_height,
            section_width=section_2h_width,
            max_candles=12
        )
        
        # Draw 30-minute candles (top-left)
        self._draw_candles(
            df_30m, image, ma_image,
            section_left=section_30m_left,
            section_top=0,
            section_height=section_30m_height,
            section_width=section_30m_width,
            max_candles=12
        )
        
        # Draw 15-minute candles (bottom)
        self._draw_candles(
            df_15m, image, ma_image,
            section_left=section_15m_left,
            section_top=section_15m_top,
            section_height=section_15m_height,
            section_width=section_15m_width,
            max_candles=24
        )
        
        # Merge MA image into blue channel
        image[:, :, 2] = ma_image
        
        # No horizontal line separating top and bottom rows - removed to match original dataset
        
        # Save the image with specified format including price information
        try:
            # Get timestamp and price data
            if not df_15m.empty and isinstance(df_15m.index[-1], pd.Timestamp):
                timestamp = df_15m.index[-1].strftime("%Y-%m-%d_%H-%M-%S")
                
                # Get relevant price information [ratio, close, high, volatility]
                # ratio = close/open
                close_price = df_15m['close'].iloc[-1]
                open_price = df_15m['open'].iloc[-1]
                high_price = df_15m['high'].iloc[-1]
                ratio = round(close_price/open_price if open_price != 0 else 0, 4)
                
                # Volatility as a simplified measure (high-low)/low for the last few candles
                recent = df_15m.iloc[-5:]
                avg_volatility = round(((recent['high'] - recent['low'])/recent['low']).mean(), 4)
                
                # Create price info array
                price_info = [ratio, close_price, high_price, avg_volatility]
                
                filename = os.path.join(output_dir, f"{timestamp}_{price_info}.png")
            else:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = os.path.join(output_dir, f"{timestamp}_[0,0,0,0].png")
                
        except Exception as e:
            # Fallback if timestamp extraction fails
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(output_dir, f"{timestamp}_[0,0,0,0].png")
            print(f"Warning: Using fallback filename due to: {e}")
        
        plt.imsave(filename, image)
        print(f"Image saved to: {filename}")
        
        return filename
    
    def _draw_candles(self, df, image, ma_image, section_left, section_top, 
                     section_height, section_width, max_candles):
        """
        Draw candles for a specific timeframe section
        """
        if df.empty:
            return
        
        # Calculate price range
        min_price = df['low'].min()
        max_price = df['high'].max()
        price_range = max_price - min_price
        
        if price_range == 0:  # Prevent division by zero
            price_range = 1
        
        # Calculate width of each candle
        candle_width = max(1, int(section_width / max_candles * 0.7))
        
        # Draw each candle
        for i in range(min(len(df), max_candles)):
            try:
                candle = df.iloc[i]
                
                # Calculate x position
                candle_left = section_left + i * (section_width // max_candles)
                
                # Scale prices to fit height
                scaled_open = section_top + section_height - int(
                    (candle['open'] - min_price) / price_range * section_height)
                scaled_close = section_top + section_height - int(
                    (candle['close'] - min_price) / price_range * section_height)
                scaled_high = section_top + section_height - int(
                    (candle['high'] - min_price) / price_range * section_height)
                scaled_low = section_top + section_height - int(
                    (candle['low'] - min_price) / price_range * section_height)
                
                # Ensure values are within bounds
                scaled_open = max(section_top, min(scaled_open, section_top + section_height - 1))
                scaled_close = max(section_top, min(scaled_close, section_top + section_height - 1))
                scaled_high = max(section_top, min(scaled_high, section_top + section_height - 1))
                scaled_low = max(section_top, min(scaled_low, section_top + section_height - 1))
                
                # Draw candle body
                top = min(scaled_open, scaled_close)
                bottom = max(scaled_open, scaled_close)
                
                # Bullish (close > open) = green, Bearish (close < open) = red, Doji = yellow
                if candle['close'] > candle['open']:
                    # Bullish - green channel
                    image[top:bottom+1, candle_left:candle_left+candle_width, 1] = 255
                elif candle['close'] < candle['open']:
                    # Bearish - red channel
                    image[top:bottom+1, candle_left:candle_left+candle_width, 0] = 255
                else:
                    # Doji - yellow (red + green)
                    image[top:bottom+1, candle_left:candle_left+candle_width, 0] = 255
                    image[top:bottom+1, candle_left:candle_left+candle_width, 1] = 255
                
                # Draw wicks
                middle = candle_left + candle_width // 2
                wick_width = max(1, candle_width // 3)
                
                # Upper wick
                image[scaled_high:top+1, middle-wick_width//2:middle+wick_width//2+1, 0] = image[top, candle_left, 0]
                image[scaled_high:top+1, middle-wick_width//2:middle+wick_width//2+1, 1] = image[top, candle_left, 1]
                
                # Lower wick
                image[bottom:scaled_low+1, middle-wick_width//2:middle+wick_width//2+1, 0] = image[bottom, candle_left, 0]
                image[bottom:scaled_low+1, middle-wick_width//2:middle+wick_width//2+1, 1] = image[bottom, candle_left, 1]
                
                # Draw MA lines
                for period, intensity in zip(PERIODS, [127, 170, 212]):
                    ma_name = f'MA_{period}'
                    if ma_name in candle and not pd.isna(candle[ma_name]):
                        ma_y = section_top + section_height - int(
                            (candle[ma_name] - min_price) / price_range * section_height)
                        ma_y = max(section_top, min(ma_y, section_top + section_height - 1))
                        
                        # Draw MA point with proper intensity in blue channel
                        ma_image[ma_y-1:ma_y+2, candle_left:candle_left+candle_width] = intensity
            except Exception as e:
                print(f"Error drawing candle: {e}")


def generate_inference_images(symbol="BTCUSDT", days=30, output_dir="./inference_images"):
    """
    Main function to download data and generate inference images for one month
    """
    # Download data
    downloader = CandlestickDataDownloader()
    dataframes = downloader.download_data(symbol=symbol, days=days)
    
    # Check if we have enough data
    if any(df.empty for df in dataframes.values()):
        print("Not enough data to generate images.")
        return []
    
    # Create image generator
    generator = MultiResolutionImageGenerator(dataframes)
    
    # Generate images for the entire month
    # Need at least 24 candles for 15m
    min_required = 24
    num_images = len(dataframes['15m']) - min_required
    
    # Only process every 8 candles to avoid too many similar images
    # This gives us roughly 3-4 images per day which is reasonable
    step = 8
    
    image_paths = []
    for i in range(0, num_images, step):
        idx = i + min_required
        if idx < len(dataframes['15m']):
            print(f"Generating image {len(image_paths)+1}/{num_images//step}...")
            image_path = generator.generate_image(idx, output_dir=output_dir)
            image_paths.append(image_path)
    
    return image_paths


if __name__ == "__main__":
    symbol = "BTCUSDT"
    days = 30  # One month of data
    output_dir = "./inference_images"
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_paths = generate_inference_images(symbol=symbol, days=days, output_dir=output_dir)
    
    if image_paths:
        print("\nSUMMARY:")
        print(f"Generated {len(image_paths)} inference images in {output_dir}")
        print(f"Image format: YYYY-MM-DD_HH-MM-SS_[ratio,close,high,volatility].png")
    else:
        print("No images were generated.")