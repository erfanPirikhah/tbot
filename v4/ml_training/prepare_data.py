import pandas as pd
import numpy as np
import logging
import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Generates technical indicators and features for ML model
    """
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame
        """
        data = df.copy()
        close = data['close']
        high = data['high']
        low = data['low']
        
        # 1. Trend Indicators
        # SMAs
        data['sma_50'] = close.rolling(window=50).mean()
        data['sma_200'] = close.rolling(window=200).mean()
        data['sma_50_200_ratio'] = data['sma_50'] / data['sma_200']
        data['price_to_sma_50'] = close / data['sma_50']
        
        # EMAs
        data['ema_9'] = close.ewm(span=9, adjust=False).mean()
        data['ema_21'] = close.ewm(span=21, adjust=False).mean()
        data['ema_9_21_ratio'] = data['ema_9'] / data['ema_21']
        
        # MACD
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        data['macd'] = exp12 - exp26
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']
        
        # 2. Momentum Indicators
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        low_min = low.rolling(window=14).min()
        high_max = high.rolling(window=14).max()
        data['stoch_k'] = 100 * ((close - low_min) / (high_max - low_min))
        data['stoch_d'] = data['stoch_k'].rolling(window=3).mean()
        
        # 3. Volatility Indicators
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data['atr'] = tr.rolling(window=14).mean()
        data['atr_ratio'] = data['atr'] / close
        
        # Bollinger Bands
        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        data['bb_upper'] = sma_20 + (std_20 * 2)
        data['bb_lower'] = sma_20 - (std_20 * 2)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / sma_20
        data['bb_position'] = (close - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # 4. Price Action
        data['body_size'] = abs(close - data['open']) / close
        data['shadow_upper'] = (high - data[['open', 'close']].max(axis=1)) / close
        data['shadow_lower'] = (data[['open', 'close']].min(axis=1) - low) / close
        data['close_change_1'] = close.pct_change(1)
        data['close_change_3'] = close.pct_change(3)
        data['close_change_5'] = close.pct_change(5)
        
        # Drop NaNs created by rolling windows
        data = data.dropna()
        
        return data

class DataLabeler:
    """
    Labels data based on future profitability
    """
    
    @staticmethod
    def label_data(df: pd.DataFrame, 
                   tp_percent: float = 0.02, 
                   sl_percent: float = 0.01, 
                   horizon: int = 50) -> pd.DataFrame:
        """
        Label candles as 1 (Win) or 0 (Loss)
        Win = Hits TP before SL within horizon
        """
        data = df.copy()
        data['target'] = 0
        
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        # Vectorized labeling would be complex due to path dependency (TP before SL)
        # Using a faster iterative approach
        
        targets = np.zeros(len(data))
        
        for i in range(len(data) - horizon):
            entry_price = close[i]
            tp_price = entry_price * (1 + tp_percent)
            sl_price = entry_price * (1 - sl_percent)
            
            future_highs = high[i+1 : i+1+horizon]
            future_lows = low[i+1 : i+1+horizon]
            
            # Check if TP or SL is hit first
            hit_tp = False
            hit_sl = False
            
            for j in range(len(future_highs)):
                if future_lows[j] <= sl_price:
                    hit_sl = True
                    break # Hit SL first (or same candle)
                if future_highs[j] >= tp_price:
                    hit_tp = True
                    break # Hit TP first
            
            if hit_tp:
                targets[i] = 1
            else:
                targets[i] = 0
                
        data['target'] = targets.astype(int)
        
        # Remove last 'horizon' rows as they can't be labeled
        data = data.iloc[:-horizon]
        
        return data

def prepare_dataset(file_path: str = None, 
                   symbol: str = "BTCUSDT", 
                   interval: str = "1h", 
                   limit: int = 5000,
                   output_path: str = "ml_training/dataset.csv"):
    
    logger.info("Starting data preparation...")
    
    # 1. Load Data
    if file_path and os.path.exists(file_path):
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        # Ensure column names are lowercase
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
    else:
        logger.info(f"Fetching data from provider for {symbol} {interval}...")
        from data.data_fetcher import DataFetcher
        fetcher = DataFetcher()
        df = fetcher.fetch_market_data(symbol=symbol, interval=interval, limit=limit)
        
    if df.empty:
        logger.error("No data found or fetched!")
        return
    
    logger.info(f"Raw data shape: {df.shape}")
    
    # 2. Feature Engineering
    logger.info("Generating technical features...")
    df_features = FeatureEngineer.add_technical_indicators(df)
    logger.info(f"Features shape: {df_features.shape}")
    
    # 3. Labeling
    logger.info("Labeling data (Target: Win/Loss)...")
    df_labeled = DataLabeler.label_data(df_features)
    
    distribution = df_labeled['target'].value_counts(normalize=True)
    logger.info(f"Class distribution:\n{distribution}")
    
    # 4. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_labeled.to_csv(output_path)
    logger.info(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    # Default behavior: try to fetch BTC data
    prepare_dataset(symbol="BTCUSDT", interval="1h", limit=5000)
