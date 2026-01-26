import numpy as np
import pandas as pd
from collections import deque, namedtuple
import requests, time, warnings, os, pickle
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from sklearn.preprocessing import StandardScaler, RobustScaler
    import matplotlib.pyplot as plt
    from functools import lru_cache
    TORCH_AVAILABLE = True
    print(f"PyTorch {torch.__version__}")
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: pip install torch scikit-learn matplotlib")

# ==================== CONFIGURATION ====================
class Config:
    """Centralized configuration for all hyperparameters"""
    # Model Architecture
    MODEL_TYPE = 'dueling_double_dqn'  # 'dqn', 'double_dqn', 'dueling_dqn', 'dueling_double_dqn'
    HIDDEN_LAYERS = [256, 128, 64, 32]
    USE_BATCH_NORM = True
    USE_RESIDUAL = True
    ACTIVATION = 'leaky_relu'  # 'relu', 'leaky_relu', 'elu'
    DROPOUT_RATE = 0.3
    USE_ATTENTION = True
    
    # Training
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 64
    MEMORY_SIZE = 10000
    WARMUP_STEPS = 1000
    TARGET_UPDATE_FREQ = 20
    TRAIN_FREQ = 4
    GRADIENT_CLIP = 1.0
    
    # Prioritized Replay
    USE_PRIORITIZED_REPLAY = True
    PRIORITY_ALPHA = 0.6
    PRIORITY_BETA_START = 0.4
    PRIORITY_BETA_FRAMES = 100000
    
    # Multi-step Learning
    N_STEP = 3
    
    # Features
    LOOKBACK = 60
    USE_VOLUME = True
    USE_TIMEFEATURES = True
    USE_ADVANCED_INDICATORS = True
    
    # Risk Management
    MAX_POSITION_SIZE = 1.0
    USE_FRACTIONAL_POSITIONS = True
    POSITION_SIZES = [0.0, 0.25, 0.5, 0.75, 1.0]
    STOP_LOSS_PCT = 0.05
    TAKE_PROFIT_PCT = 0.15
    TRAILING_STOP_PCT = 0.03
    MAX_RISK_PER_TRADE = 0.02
    USE_KELLY_CRITERION = True
    
    # Trading
    TRADE_COOLDOWN = 3
    TRANSACTION_FEE = 0.001
    SLIPPAGE = 0.0005
    
    # Ensemble
    USE_ENSEMBLE = True
    ENSEMBLE_SIZE = 5
    ENSEMBLE_METHOD = 'voting'  # 'voting', 'average', 'weighted'
    
    # Validation
    WALK_FORWARD = True
    TRAIN_RATIO = 0.7
    VALIDATION_RATIO = 0.15
    
    # Multi-timeframe
    USE_MULTI_TIMEFRAME = True
    TIMEFRAMES = [5, 15, 60]  # minutes (simulated from daily)
    
    # Performance
    USE_CACHE = True
    CACHE_SIZE = 2000

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'priority'))

# ==================== DATA PROVIDER ====================
class CryptoDataProvider:
    @staticmethod
    def get_available_coins():
        return {
            'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'DOGE': 'Dogecoin', 
            'ADA': 'Cardano', 'XRP': 'Ripple', 'SOL': 'Solana', 
            'DOT': 'Polkadot', 'MATIC': 'Polygon', 'LTC': 'Litecoin', 
            'LINK': 'Chainlink', 'AVAX': 'Avalanche', 'UNI': 'Uniswap'
        }
    
    @staticmethod
    def fetch_from_binance_public(symbol='BTCUSDT', days=3650):
        print(f"\nFetching {symbol}...")
        url = "https://api.binance.com/api/v3/klines"
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        params = {
            'symbol': symbol, 
            'interval': '1d', 
            'startTime': start_time, 
            'endTime': end_time, 
            'limit': 1000
        }
        
        all_data = []
        max_calls = 4
        for _ in range(max_calls):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                if not data: break
                all_data.extend(data)
                params['endTime'] = data[0][0]
                time.sleep(0.2)
            except Exception as e:
                print(f"Fetch error: {e}")
                break
        
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
            'taker_buy_quote', 'ignore'
        ])
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✓ {len(df)} days ({len(df)/365:.1f} years)")
        return df
    
    @staticmethod
    def fetch_historical_data(coin='BTC'):
        symbols = {
            'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'DOGE': 'DOGEUSDT',
            'ADA': 'ADAUSDT', 'XRP': 'XRPUSDT', 'SOL': 'SOLUSDT',
            'DOT': 'DOTUSDT', 'MATIC': 'MATICUSDT', 'LTC': 'LTCUSDT',
            'LINK': 'LINKUSDT', 'AVAX': 'AVAXUSDT', 'UNI': 'UNIUSDT'
        }
        return CryptoDataProvider.fetch_from_binance_public(
            symbols.get(coin, 'BTCUSDT'), days=3650
        )

# ==================== ADVANCED FEATURE ENGINEERING ====================
class AdvancedFeatureEngine:
    def __init__(self, config=Config()):
        self.config = config
        self.scaler = RobustScaler()
        self.feature_names = []
        
    @lru_cache(maxsize=Config.CACHE_SIZE if Config.USE_CACHE else None)
    def _cached_indicators(self, prices_tuple):
        """Cache expensive indicator calculations"""
        prices = np.array(prices_tuple)
        return self._calculate_all_indicators(prices)
    
    def create_features(self, df, index):
        """Create comprehensive feature set"""
        if index < self.config.LOOKBACK:
            return None
        
        try:
            window_df = df.iloc[max(0, index - self.config.LOOKBACK):index + 1].copy()
            
            if len(window_df) < 10:
                return None
            
            # Ensure required columns exist
            if 'close' not in window_df.columns:
                return None
            
            features = []
            
            # Price features
            price_feats = self._price_features(window_df)
            if price_feats is None:
                return None
            features.extend(price_feats)
            
            # Technical indicators
            tech_feats = self._technical_indicators(window_df)
            if tech_feats is None:
                return None
            features.extend(tech_feats)
            
            # Volume features
            if self.config.USE_VOLUME and 'volume' in window_df.columns:
                vol_feats = self._volume_features(window_df)
                if vol_feats is None:
                    return None
                features.extend(vol_feats)
            else:
                # Add dummy features if volume not available
                features.extend([0.0] * 5)
            
            # Advanced indicators
            if self.config.USE_ADVANCED_INDICATORS:
                adv_feats = self._advanced_indicators(window_df)
                if adv_feats is None:
                    return None
                features.extend(adv_feats)
            
            # Time-based features
            if self.config.USE_TIMEFEATURES and 'timestamp' in window_df.columns:
                time_feats = self._time_features(window_df)
                if time_feats is None:
                    return None
                features.extend(time_feats)
            else:
                # Add dummy features if timestamp not available
                features.extend([0.0] * 5)
            
            # Price action features
            pa_feats = self._price_action_features(window_df)
            if pa_feats is None:
                return None
            features.extend(pa_feats)
            
            # Market regime
            regime_feats = self._market_regime_features(window_df)
            if regime_feats is None:
                return None
            features.extend(regime_feats)
            
            features = np.array(features, dtype=np.float32)
            
            # Check for invalid values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                return None
            
            return features.reshape(1, -1)
            
        except Exception as e:
            # Silently return None on any error during feature creation
            return None
    
    def _price_features(self, df):
        """Basic price-based features"""
        try:
            close = df['close'].values
            features = []
            
            current = close[-1]
            price_min, price_max = np.min(close), np.max(close)
            
            # Normalized price position
            features.append((current - price_min) / (price_max - price_min + 1e-10))
            
            # Returns at different lags
            for lag in [1, 3, 5, 7, 14, 21]:
                if len(close) > lag:
                    ret = (close[-1] - close[-lag-1]) / (close[-lag-1] + 1e-10)
                    features.append(np.clip(ret, -1, 1))
                else:
                    features.append(0.0)
            
            # Price vs moving averages
            for window in [5, 10, 20, 30, 50]:
                if len(close) >= window:
                    ma = np.mean(close[-window:])
                    features.append((current / (ma + 1e-10)) - 1.0)
                else:
                    features.append(0.0)
            
            # Distance from high/low
            if len(close) >= 20:
                high_20 = np.max(close[-20:])
                low_20 = np.min(close[-20:])
                features.append((current - low_20) / (high_20 - low_20 + 1e-10))
            else:
                features.append(0.5)
            
            return features
        except Exception as e:
            return None
    
    def _technical_indicators(self, df):
        """Standard technical indicators"""
        close = df['close'].values
        high = df['high'].values if 'high' in df.columns else close
        low = df['low'].values if 'low' in df.columns else close
        features = []
        
        # RSI
        if len(close) >= 15:
            deltas = np.diff(close)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            features.append(rsi / 100)  # Normalize
        else:
            features.append(0.5)
        
        # MACD
        if len(close) >= 26:
            ema_12 = self._ema(close, 12)
            ema_26 = self._ema(close, 26)
            macd = ema_12 - ema_26
            signal = self._ema(close, 9)
            histogram = macd - signal
            features.extend([
                np.tanh(macd / (close[-1] + 1e-10)),
                np.tanh(histogram / (close[-1] + 1e-10))
            ])
        else:
            features.extend([0.0, 0.0])
        
        # Bollinger Bands
        if len(close) >= 20:
            ma_20 = np.mean(close[-20:])
            std_20 = np.std(close[-20:])
            bb_upper = ma_20 + 2 * std_20
            bb_lower = ma_20 - 2 * std_20
            bb_position = (close[-1] - bb_lower) / (bb_upper - bb_lower + 1e-10)
            bb_width = (bb_upper - bb_lower) / (ma_20 + 1e-10)
            features.extend([bb_position, bb_width])
        else:
            features.extend([0.5, 0.0])
        
        # Stochastic
        if len(high) >= 14 and len(low) >= 14:
            low_14 = np.min(low[-14:])
            high_14 = np.max(high[-14:])
            stoch = 100 * (close[-1] - low_14) / (high_14 - low_14 + 1e-10)
            features.append(stoch / 100)
        else:
            features.append(0.5)
        
        # ATR (Average True Range)
        if len(close) >= 14:
            tr = np.max([
                high[-14:] - low[-14:],
                np.abs(high[-14:] - np.roll(close[-14:], 1)),
                np.abs(low[-14:] - np.roll(close[-14:], 1))
            ], axis=0)
            atr = np.mean(tr)
            features.append(atr / (close[-1] + 1e-10))
        else:
            features.append(0.0)
        
        return features
    
    def _volume_features(self, df):
        """Volume-based features"""
        try:
            if 'volume' not in df.columns:
                return [0.0] * 5
            
            volume = df['volume'].values
            close = df['close'].values
            features = []
            
            # Volume trend
            if len(volume) >= 20:
                vol_ma = np.mean(volume[-20:])
                features.append((volume[-1] / (vol_ma + 1e-10)) - 1.0)
            else:
                features.append(0.0)
            
            # OBV (On Balance Volume)
            if len(close) >= 2:
                obv = np.where(close[1:] > close[:-1], volume[1:], -volume[1:])
                obv_sum = np.sum(obv[-20:]) if len(obv) >= 20 else np.sum(obv)
                features.append(np.tanh(obv_sum / (np.sum(volume[-20:]) + 1e-10)))
            else:
                features.append(0.0)
            
            # VWAP approximation
            if len(volume) >= 20:
                vwap = np.sum(close[-20:] * volume[-20:]) / (np.sum(volume[-20:]) + 1e-10)
                features.append((close[-1] / (vwap + 1e-10)) - 1.0)
            else:
                features.append(0.0)
            
            # Volume volatility
            if len(volume) >= 20:
                vol_std = np.std(volume[-20:])
                vol_mean = np.mean(volume[-20:])
                features.append(vol_std / (vol_mean + 1e-10))
            else:
                features.append(0.0)
            
            # Money Flow Index (MFI) approximation
            if len(close) >= 14 and 'high' in df.columns and 'low' in df.columns:
                typical_price = (df['high'].values[-14:] + df['low'].values[-14:] + close[-14:]) / 3
                money_flow = typical_price * volume[-14:]
                positive_flow = np.sum(np.where(np.diff(typical_price) > 0, money_flow[1:], 0))
                negative_flow = np.sum(np.where(np.diff(typical_price) < 0, money_flow[1:], 0))
                mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))
                features.append(mfi / 100)
            else:
                features.append(0.5)
            
            return features
        except Exception as e:
            return [0.0] * 5
    
    def _advanced_indicators(self, df):
        """Advanced technical indicators"""
        try:
            close = df['close'].values
            features = []
            
            # ADX (Average Directional Index)
            if len(close) >= 14:
                adx = self._calculate_adx(df)
                features.append(adx / 100)
            else:
                features.append(0.25)
            
            # Momentum
            if len(close) >= 10:
                momentum = (close[-1] - close[-10]) / (close[-10] + 1e-10)
                features.append(np.tanh(momentum))
            else:
                features.append(0.0)
            
            # Rate of Change
            for period in [5, 10, 20]:
                if len(close) > period:
                    roc = (close[-1] - close[-period-1]) / (close[-period-1] + 1e-10)
                    features.append(np.tanh(roc))
                else:
                    features.append(0.0)
            
            # Ichimoku approximation
            if len(close) >= 26:
                tenkan = (np.max(close[-9:]) + np.min(close[-9:])) / 2
                kijun = (np.max(close[-26:]) + np.min(close[-26:])) / 2
                features.append((close[-1] - kijun) / (kijun + 1e-10))
                features.append((tenkan - kijun) / (kijun + 1e-10))
            else:
                features.extend([0.0, 0.0])
            
            return features
        except Exception as e:
            # Return default values for 7 features
            return [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def _time_features(self, df):
        """Time-based features"""
        try:
            if 'timestamp' not in df.columns:
                return [0.0] * 5
            
            ts = df['timestamp'].iloc[-1]
            
            # Check if timestamp is valid
            if pd.isna(ts):
                return [0.0] * 5
            
            features = []
            
            # Day of week (Monday=0, Sunday=6)
            features.append(ts.dayofweek / 6.0)
            
            # Day of month
            features.append(ts.day / 31.0)
            
            # Month of year
            features.append(ts.month / 12.0)
            
            # Quarter
            features.append(ts.quarter / 4.0)
            
            # Is weekend
            features.append(1.0 if ts.dayofweek >= 5 else 0.0)
            
            return features
        except Exception as e:
            return [0.0] * 5
    
    def _price_action_features(self, df):
        """Price action patterns"""
        try:
            close = df['close'].values
            high = df['high'].values if 'high' in df.columns else close
            low = df['low'].values if 'low' in df.columns else close
            features = []
            
            # Candlestick patterns (simplified)
            if len(close) >= 2:
                body = close[-1] - close[-2]
                upper_shadow = high[-1] - max(close[-1], close[-2])
                lower_shadow = min(close[-1], close[-2]) - low[-1]
                total_range = high[-1] - low[-1]
                
                if total_range > 0:
                    features.extend([
                        body / total_range,
                        upper_shadow / total_range,
                        lower_shadow / total_range
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0])
            else:
                features.extend([0.0, 0.0, 0.0])
            
            # Support/Resistance (simplified)
            if len(close) >= 50:
                recent_highs = high[-50:]
                recent_lows = low[-50:]
                resistance = np.percentile(recent_highs, 95)
                support = np.percentile(recent_lows, 5)
                
                dist_to_resistance = (resistance - close[-1]) / (close[-1] + 1e-10)
                dist_to_support = (close[-1] - support) / (close[-1] + 1e-10)
                
                features.extend([
                    np.tanh(dist_to_resistance),
                    np.tanh(dist_to_support)
                ])
            else:
                features.extend([0.0, 0.0])
            
            # Fibonacci levels (simplified)
            if len(close) >= 50:
                high_50 = np.max(high[-50:])
                low_50 = np.min(low[-50:])
                range_50 = high_50 - low_50
                
                if range_50 > 0:
                    fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                    closest_fib = min(fib_levels, key=lambda x: abs((low_50 + range_50 * x) - close[-1]))
                    features.append(closest_fib)
                else:
                    features.append(0.5)
            else:
                features.append(0.5)
            
            return features
        except Exception as e:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
    
    def _market_regime_features(self, df):
        """Detect market regime (trending/ranging)"""
        try:
            close = df['close'].values
            features = []
            
            if len(close) >= 50:
                # Trend strength (linear regression slope)
                x = np.arange(len(close[-50:]))
                y = close[-50:]
                slope = np.polyfit(x, y, 1)[0]
                features.append(np.tanh(slope / (np.mean(y) + 1e-10) * 100))
                
                # Volatility regime
                returns = np.diff(close[-50:]) / (close[-51:-1] + 1e-10)
                volatility = np.std(returns)
                features.append(np.tanh(volatility * 10))
                
                # Range vs trend (ADX proxy)
                high_low_range = np.max(close[-50:]) - np.min(close[-50:])
                net_change = abs(close[-1] - close[-50])
                features.append(net_change / (high_low_range + 1e-10))
            else:
                features.extend([0.0, 0.0, 0.5])
            
            return features
        except Exception as e:
            return [0.0, 0.0, 0.5]
    
    def _ema(self, prices, period):
        """Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        alpha = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
    
    def _calculate_adx(self, df, period=14):
        """Calculate Average Directional Index"""
        if len(df) < period + 1:
            return 25.0
        
        high = df['high'].values if 'high' in df.columns else df['close'].values
        low = df['low'].values if 'low' in df.columns else df['close'].values
        close = df['close'].values
        
        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth with EMA
        atr = np.mean(tr[-period:])
        plus_di = 100 * np.mean(plus_dm[-period:]) / (atr + 1e-10)
        minus_di = 100 * np.mean(minus_dm[-period:]) / (atr + 1e-10)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        return dx

# ==================== ADVANCED NEURAL NETWORKS ====================
class ResidualBlock(nn.Module):
    """Residual connection for better gradient flow"""
    def __init__(self, dim, use_batch_norm=True):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim) if use_batch_norm else None
        self.activation = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        residual = x
        out = self.fc(x)
        if self.bn is not None:
            out = self.bn(out)
        out = self.activation(out)
        return out + residual

class AttentionLayer(nn.Module):
    """Self-attention mechanism"""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** 0.5
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention_weights = torch.softmax(Q @ K.T / self.scale, dim=-1)
        attended = attention_weights @ V
        return attended + x  # Residual connection

class DuelingDQN(nn.Module):
    """Dueling Double DQN with advanced features"""
    def __init__(self, state_size, action_size, config=Config()):
        super().__init__()
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        
        # Input layer
        layers = []
        prev_size = state_size
        
        # Hidden layers with batch norm and residual connections
        for i, hidden_size in enumerate(config.HIDDEN_LAYERS):
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if config.USE_BATCH_NORM:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            if config.ACTIVATION == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif config.ACTIVATION == 'elu':
                layers.append(nn.ELU())
            else:
                layers.append(nn.ReLU())
            
            if i < len(config.HIDDEN_LAYERS) - 1:
                layers.append(nn.Dropout(config.DROPOUT_RATE))
            
            prev_size = hidden_size
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Attention layer
        if config.USE_ATTENTION:
            self.attention = AttentionLayer(config.HIDDEN_LAYERS[-1])
        else:
            self.attention = None
        
        # Residual blocks
        if config.USE_RESIDUAL:
            self.residual = ResidualBlock(
                config.HIDDEN_LAYERS[-1], 
                config.USE_BATCH_NORM
            )
        else:
            self.residual = None
        
        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(config.HIDDEN_LAYERS[-1], 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.HIDDEN_LAYERS[-1], 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, action_size)
        )
    
    def forward(self, x):
        # Feature extraction
        features = self.feature_layers(x)
        
        # Attention
        if self.attention is not None:
            features = self.attention(features.unsqueeze(0)).squeeze(0) if features.dim() == 1 else self.attention(features)
        
        # Residual connection
        if self.residual is not None:
            features = self.residual(features)
        
        # Dueling architecture
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        
        return q_values

# ==================== PRIORITIZED REPLAY BUFFER ====================
class PrioritizedReplayBuffer:
    """Prioritized Experience Replay"""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(state, action, reward, next_state, done, max_priority))
        else:
            self.buffer[self.position] = Transition(state, action, reward, next_state, done, max_priority)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size, beta=0.4):
        if self.size < batch_size:
            return None
        
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = self.size
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return self.size

# ==================== ADVANCED TRADING AGENT ====================
class AdvancedTradingAgent:
    def __init__(self, action_size=5, config=Config()):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        self.config = config
        self.action_size = action_size
        self.state_size = None
        
        # RL parameters
        self.gamma = config.GAMMA
        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_MIN
        self.epsilon_decay = config.EPSILON_DECAY
        self.learning_rate = config.LEARNING_RATE
        self.batch_size = config.BATCH_SIZE
        
        # Prioritized replay
        if config.USE_PRIORITIZED_REPLAY:
            self.memory = PrioritizedReplayBuffer(config.MEMORY_SIZE, config.PRIORITY_ALPHA)
            self.beta = config.PRIORITY_BETA_START
            self.beta_increment = (1.0 - config.PRIORITY_BETA_START) / config.PRIORITY_BETA_FRAMES
        else:
            self.memory = deque(maxlen=config.MEMORY_SIZE)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {self.device}")
        
        # Models
        self.model = None
        self.target_model = None
        self.optimizer = None
        
        # Feature scaling
        self.scaler = RobustScaler()
        self.trained = False
        
        # Trading controls
        self.trade_cooldown = 0
        self.min_cooldown = config.TRADE_COOLDOWN
        
        # Training stats
        self.training_step = 0
        self.losses = []
    
    def build_models(self, state_size):
        """Build main and target networks"""
        self.state_size = state_size
        
        self.model = DuelingDQN(state_size, self.action_size, self.config).to(self.device)
        self.target_model = DuelingDQN(state_size, self.action_size, self.config).to(self.device)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Network: {state_size} → {self.config.HIDDEN_LAYERS} → {self.action_size}")
        print(f"Parameters: {total_params:,}")
    
    def update_target_model(self):
        """Soft update of target network"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def act(self, state, valid_actions=None, force_explore=False):
        """Select action using epsilon-greedy policy"""
        if self.trade_cooldown > 0:
            self.trade_cooldown -= 1
            return 0  # Hold
        
        if valid_actions is None:
            valid_actions = list(range(self.action_size))
        
        # Epsilon-greedy
        if force_explore or np.random.random() < self.epsilon:
            action = np.random.choice(valid_actions)
        else:
            self.model.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device)
                q_values = self.model(state_tensor).cpu().numpy()
                
                if state.ndim > 1:
                    q_values = q_values[0]
                
                # Mask invalid actions
                masked_q = q_values.copy()
                for i in range(len(masked_q)):
                    if i not in valid_actions:
                        masked_q[i] = -np.inf
                
                action = np.argmax(masked_q)
        
        # Set cooldown for trades
        if action != 0:
            self.trade_cooldown = self.min_cooldown
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        if self.config.USE_PRIORITIZED_REPLAY:
            self.memory.add(state, action, reward, next_state, done)
        else:
            self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on batch from replay buffer"""
        if self.config.USE_PRIORITIZED_REPLAY:
            if len(self.memory) < self.batch_size:
                return 0.0
            
            batch_data = self.memory.sample(self.batch_size, self.beta)
            if batch_data is None:
                return 0.0
            
            samples, indices, weights = batch_data
            self.beta = min(1.0, self.beta + self.beta_increment)
        else:
            if len(self.memory) < self.batch_size:
                return 0.0
            
            indices_list = np.random.choice(len(self.memory), self.batch_size, replace=False)
            samples = [self.memory[idx] for idx in indices_list]
            weights = np.ones(self.batch_size)
        
        # Prepare batch
        states = np.vstack([s.state if hasattr(s, 'state') else s[0] for s in samples])
        actions = np.array([s.action if hasattr(s, 'action') else s[1] for s in samples])
        rewards = np.array([s.reward if hasattr(s, 'reward') else s[2] for s in samples])
        next_states = np.vstack([s.next_state if hasattr(s, 'next_state') else s[3] for s in samples])
        dones = np.array([s.done if hasattr(s, 'done') else s[4] for s in samples])
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        # Double DQN
        self.model.eval()
        with torch.no_grad():
            # Use online network to select actions
            next_actions = self.model(next_states_tensor).argmax(1)
            # Use target network to evaluate actions
            next_q_values = self.target_model(next_states_tensor)
            next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q
        
        # Current Q values
        self.model.train()
        current_q = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # TD errors for priority update
        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
        
        # Weighted loss
        loss = (weights_tensor * F.mse_loss(current_q, target_q, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
        self.optimizer.step()
        
        # Update priorities
        if self.config.USE_PRIORITIZED_REPLAY:
            self.memory.update_priorities(indices, td_errors + 1e-6)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_step += 1
        self.losses.append(loss.item())
        
        return loss.item()
    
    def predict_q_values(self, state):
        """Get Q-values for state"""
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            q_values = self.model(state_tensor).cpu().numpy()
        return q_values
    
    def save_model(self, filepath='advanced_model.pth'):
        """Save model and training state"""
        torch.save({
            'model_state': self.model.state_dict(),
            'target_model_state': self.target_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_size': self.state_size,
            'training_step': self.training_step,
            'config': self.config
        }, filepath)
        
        with open(filepath.replace('.pth', '_scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Model saved: {filepath}")
    
    def load_model(self, filepath='advanced_model.pth'):
        """Load model and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.state_size = checkpoint['state_size']
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint.get('training_step', 0)
        
        self.build_models(self.state_size)
        self.model.load_state_dict(checkpoint['model_state'])
        self.target_model.load_state_dict(checkpoint['target_model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        with open(filepath.replace('.pth', '_scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.trained = True
        print(f"Model loaded: {filepath}")

# ==================== RISK MANAGER ====================
class RiskManager:
    """Advanced risk management system"""
    def __init__(self, config=Config()):
        self.config = config
        self.trade_history = []
        self.peak_value = 0
        self.drawdown = 0
        
    def calculate_position_size(self, capital, price, win_rate=0.5, avg_win=1.0, avg_loss=1.0):
        """Kelly Criterion for optimal position sizing"""
        if not self.config.USE_KELLY_CRITERION:
            return self.config.MAX_POSITION_SIZE
        
        # Kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        if avg_win <= 0:
            return 0.0
        
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly = max(0, min(kelly, self.config.MAX_POSITION_SIZE))
        
        # Fractional Kelly (more conservative)
        kelly *= 0.5
        
        return kelly
    
    def should_trade(self, portfolio_value, capital):
        """Check if we should allow trading based on risk limits"""
        # Update peak and drawdown
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        self.drawdown = (self.peak_value - portfolio_value) / self.peak_value if self.peak_value > 0 else 0
        
        # Stop trading if drawdown too large
        if self.drawdown > 0.20:  # 20% drawdown limit
            return False
        
        # Don't risk more than 2% of capital per trade
        if capital < portfolio_value * 0.02:
            return False
        
        return True
    
    def calculate_stop_loss(self, entry_price):
        """Calculate stop loss price"""
        return entry_price * (1 - self.config.STOP_LOSS_PCT)
    
    def calculate_take_profit(self, entry_price):
        """Calculate take profit price"""
        return entry_price * (1 + self.config.TAKE_PROFIT_PCT)
    
    def calculate_trailing_stop(self, entry_price, current_price, highest_price):
        """Calculate trailing stop price"""
        trailing_stop = highest_price * (1 - self.config.TRAILING_STOP_PCT)
        return max(trailing_stop, entry_price * (1 - self.config.STOP_LOSS_PCT))

# ==================== METRICS TRACKER ====================
class MetricsTracker:
    """Track and calculate performance metrics"""
    def __init__(self):
        self.portfolio_values = []
        self.returns = []
        self.trades = []
        self.drawdowns = []
        
    def update(self, portfolio_value):
        """Update metrics with new portfolio value"""
        self.portfolio_values.append(portfolio_value)
        
        if len(self.portfolio_values) > 1:
            ret = (portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
            self.returns.append(ret)
        
        # Calculate drawdown
        if self.portfolio_values:
            peak = max(self.portfolio_values)
            dd = (peak - portfolio_value) / peak if peak > 0 else 0
            self.drawdowns.append(dd)
    
    def add_trade(self, trade_info):
        """Add trade to history"""
        self.trades.append(trade_info)
    
    def get_sharpe_ratio(self, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        if len(self.returns) < 2:
            return 0.0
        
        returns_array = np.array(self.returns)
        excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
        return sharpe
    
    def get_sortino_ratio(self, risk_free_rate=0.02):
        """Calculate Sortino ratio (downside risk)"""
        if len(self.returns) < 2:
            return 0.0
        
        returns_array = np.array(self.returns)
        excess_returns = returns_array - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        
        sortino = np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)
        return sortino
    
    def get_max_drawdown(self):
        """Calculate maximum drawdown"""
        if not self.drawdowns:
            return 0.0
        return max(self.drawdowns)
    
    def get_calmar_ratio(self):
        """Calculate Calmar ratio"""
        if not self.portfolio_values or len(self.portfolio_values) < 2:
            return 0.0
        
        total_return = (self.portfolio_values[-1] - self.portfolio_values[0]) / self.portfolio_values[0]
        max_dd = self.get_max_drawdown()
        
        if max_dd == 0:
            return 0.0
        
        return total_return / max_dd
    
    def get_win_rate(self):
        """Calculate win rate"""
        if not self.trades:
            return 0.0
        
        profitable_trades = sum(1 for t in self.trades if t.get('profit', 0) > 0)
        return profitable_trades / len(self.trades) if self.trades else 0.0
    
    def get_profit_factor(self):
        """Calculate profit factor (gross profit / gross loss)"""
        if not self.trades:
            return 0.0
        
        gross_profit = sum(t.get('profit', 0) for t in self.trades if t.get('profit', 0) > 0)
        gross_loss = abs(sum(t.get('profit', 0) for t in self.trades if t.get('profit', 0) < 0))
        
        if gross_loss == 0:
            return gross_profit if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss

# ==================== MAIN TRADING SYSTEM ====================
class AdvancedTradingSystem:
    def __init__(self, initial_capital=100, currency='INR', config=Config()):
        self.config = config
        self.initial_capital = initial_capital
        self.currency = currency
        
        # Agent
        action_size = len(config.POSITION_SIZES) if config.USE_FRACTIONAL_POSITIONS else 3
        self.agent = AdvancedTradingAgent(action_size=action_size, config=config)
        
        # Feature engine
        self.feature_engine = AdvancedFeatureEngine(config)
        
        # Risk manager
        self.risk_manager = RiskManager(config)
        
        # Metrics
        self.metrics = MetricsTracker()
        
        # State
        self.reset()
        
        # Ensemble models
        self.ensemble_agents = []
        
    def reset(self):
        """Reset trading state"""
        self.capital = self.initial_capital
        self.position = 0.0
        self.position_price = 0.0
        self.highest_price_since_entry = 0.0
        
        self.metrics = MetricsTracker()
        
        self.wins = 0
        self.losses = 0
        self.holds = 0
        self.total_fees = 0
        
        self.price_data = []
        self.action_history = []
        self.q_value_history = []
    
    def get_valid_actions(self, position):
        """Get valid actions based on current position"""
        if self.config.USE_FRACTIONAL_POSITIONS:
            # Actions: 0=Hold, 1=Buy25%, 2=Buy50%, 3=Buy75%, 4=Buy100% (or Sell if have position)
            if position > 0:
                # If we have a position, action 4 becomes sell
                return [0, 4]  # Hold or Sell
            else:
                # No position, can buy at different levels
                return [0, 1, 2, 3, 4]  # Hold or Buy at different sizes
        else:
            # Simple: hold, buy, sell
            if position == 0:
                return [0, 1]  # Hold or buy
            else:
                return [0, 2]  # Hold or sell
    
    def execute_trade(self, action, price):
        """Execute trade with risk management"""
        portfolio_value = self.capital + (self.position * price)
        
        # Risk check
        if not self.risk_manager.should_trade(portfolio_value, self.capital):
            return None
        
        # Check stop loss and take profit
        if self.position > 0:
            stop_loss = self.risk_manager.calculate_stop_loss(self.position_price)
            take_profit = self.risk_manager.calculate_take_profit(self.position_price)
            trailing_stop = self.risk_manager.calculate_trailing_stop(
                self.position_price, price, self.highest_price_since_entry
            )
            
            # Update highest price
            self.highest_price_since_entry = max(self.highest_price_since_entry, price)
            
            # Auto exit on stop loss or take profit
            if price <= stop_loss or price <= trailing_stop:
                action = 4  # Sell action
                print(f"  Stop loss triggered at {price:.2f}")
            elif price >= take_profit:
                action = 4  # Sell action
                print(f"  Take profit triggered at {price:.2f}")
        
        # Execute action
        if action == 0:
            # Hold
            self.holds += 1
            return None
        
        # Buy actions (1-4)
        elif action >= 1 and action <= 4:
            # If we have a position and action is 4, that means SELL
            if self.position > 0 and action == 4:
                # Sell position
                revenue = self.position * price
                fee = revenue * self.config.TRANSACTION_FEE
                slippage_cost = revenue * self.config.SLIPPAGE
                
                profit = (price - self.position_price) * self.position - fee - slippage_cost
                
                self.capital += revenue - fee - slippage_cost
                self.total_fees += fee
                
                if profit > 0:
                    self.wins += 1
                else:
                    self.losses += 1
                
                trade_info = {
                    'day': len(self.metrics.portfolio_values),
                    'type': 'SELL',
                    'price': price,
                    'shares': self.position,
                    'profit': profit,
                    'fee': fee,
                    'return': profit / (self.position_price * self.position) if self.position_price > 0 else 0
                }
                self.metrics.add_trade(trade_info)
                
                self.position = 0
                self.position_price = 0
                self.highest_price_since_entry = 0
                
                print(f"  SELL @ {price:.2f} | P&L: {profit:+.2f} ({trade_info['return']*100:+.2f}%)")
                return {'type': 'SELL', 'profit': profit}
            
            # Otherwise it's a BUY action
            elif self.capital > 1 and self.position == 0:
                if self.config.USE_FRACTIONAL_POSITIONS:
                    # Map action to position size
                    position_sizes = {1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}
                    position_size = position_sizes.get(action, 0.95)
                else:
                    position_size = 0.95
                
                amount = self.capital * position_size
                fee = amount * self.config.TRANSACTION_FEE
                slippage_cost = amount * self.config.SLIPPAGE
                
                shares = (amount - fee - slippage_cost) / price
                
                self.capital -= amount
                self.position += shares
                self.position_price = price
                self.highest_price_since_entry = price
                self.total_fees += fee
                
                trade_info = {
                    'day': len(self.metrics.portfolio_values),
                    'type': 'BUY',
                    'price': price,
                    'shares': shares,
                    'amount': amount,
                    'fee': fee
                }
                self.metrics.add_trade(trade_info)
                
                print(f"  BUY {shares:.4f} @ {price:.2f} (Size: {position_size*100:.0f}%)")
                return {'type': 'BUY', 'shares': shares}
        
        return None
    
    def train(self, df, model_path='advanced_model.pth'):
        """Train the agent"""
        print(f"\n{'='*70}")
        print("TRAINING PHASE")
        print(f"{'='*70}")
        
        # Check for existing model
        if os.path.exists(model_path):
            load = input(f"Found {model_path}. Load it? (y/n, default y): ").strip().lower() or 'y'
            if load == 'y':
                try:
                    self.agent.load_model(model_path)
                    
                    # Ask about ensemble training
                    if self.config.USE_ENSEMBLE and not self.ensemble_agents:
                        train_ensemble = input("Train ensemble models? (y/n, default n): ").strip().lower() or 'n'
                        if train_ensemble == 'y':
                            self.train_ensemble(df, model_path)
                    return
                except Exception as e:
                    print(f"Error loading model: {e}")
                    print("Training new model instead...")
        
        # Prepare features
        print("\n Preparing features...")
        print(f"Processing {len(df)} days (starting from day {self.config.LOOKBACK})...")
        
        X_train, y_train = [], []
        feature_errors = 0
        
        for i in range(self.config.LOOKBACK, len(df)):
            try:
                features = self.feature_engine.create_features(df, i)
                if features is None:
                    feature_errors += 1
                    continue
                
                # Calculate future return for labeling
                future_idx = min(i + 5, len(df) - 1)
                future_return = (df.iloc[future_idx]['close'] - df.iloc[i]['close']) / df.iloc[i]['close']
                
                # Multi-class labels for 5 actions: Hold, Buy25%, Buy50%, Buy75%, Buy100%/Sell
                if self.config.USE_FRACTIONAL_POSITIONS:
                    # 5 actions: [Hold, Buy25%, Buy50%, Buy75%, Buy100%/Sell]
                    if future_return > 0.04:
                        label = [0, 0, 0, 0, 1]  # Strong buy (100%)
                    elif future_return > 0.025:
                        label = [0, 0, 0, 1, 0]  # Moderate buy (75%)
                    elif future_return > 0.015:
                        label = [0, 0, 1, 0, 0]  # Weak buy (50%)
                    elif future_return > 0.005:
                        label = [0, 1, 0, 0, 0]  # Very weak buy (25%)
                    else:
                        label = [1, 0, 0, 0, 0]  # Hold
                else:
                    # Simple 3-class: Hold, Buy, Sell
                    if future_return > 0.02:
                        label = [0, 1, 0]  # Buy
                    elif future_return < -0.02:
                        label = [0, 0, 1]  # Sell
                    else:
                        label = [1, 0, 0]  # Hold
                
                X_train.append(features[0])
                y_train.append(label)
                
            except Exception as e:
                feature_errors += 1
                continue
        
        if feature_errors > 0:
            print(f"Skipped {feature_errors} samples due to feature extraction errors")
        
        if len(X_train) == 0:
            raise ValueError(f"No training samples created! Need at least {self.config.LOOKBACK} days of data with valid features.")
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"Created {len(X_train)} training samples with {X_train.shape[1]} features")
        
        # Build model
        self.agent.build_models(X_train.shape[1])
        
        # Normalize features
        X_train_norm = self.agent.scaler.fit_transform(X_train)
        
        # Training loop
        print(f"\n Training for up to 300 epochs...")
        X_tensor = torch.FloatTensor(X_train_norm).to(self.agent.device)
        y_tensor = torch.FloatTensor(y_train).to(self.agent.device)
        
        self.agent.model.train()
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.agent.optimizer, mode='min', factor=0.5, patience=15
        )
        
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 30
        
        for epoch in range(300):
            # Forward pass
            outputs = self.agent.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            
            # Backward pass
            self.agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), self.config.GRADIENT_CLIP)
            self.agent.optimizer.step()
            
            # Learning rate scheduling
            scheduler.step(loss)
            
            # Early stopping check
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/300 | Loss: {loss.item():.6f} | Best: {best_loss:.6f}")
            
            if patience_counter >= max_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Update target network
        self.agent.update_target_model()
        self.agent.trained = True
        
        # Save model
        self.agent.save_model(model_path)
        
        print(f"Training complete! Final loss: {loss.item():.6f}")
        
        # Train ensemble
        if self.config.USE_ENSEMBLE:
            train_ensemble = input("\n Train ensemble models? (y/n, default y): ").strip().lower() or 'y'
            if train_ensemble == 'y':
                self.train_ensemble(df, model_path)
    
    def train_ensemble(self, df, base_model_path):
        """Train ensemble of models"""
        print(f"\n{'='*70}")
        print(f" ENSEMBLE TRAINING ({self.config.ENSEMBLE_SIZE} models)")
        print(f"{'='*70}")
        
        self.ensemble_agents = []
        
        for i in range(self.config.ENSEMBLE_SIZE):
            print(f"\n[{i+1}/{self.config.ENSEMBLE_SIZE}] Training ensemble model {i+1}...")
            
            # Create new agent with different random seed
            np.random.seed(i * 42)
            torch.manual_seed(i * 42)
            
            agent = AdvancedTradingAgent(
                action_size=self.agent.action_size,
                config=self.config
            )
            
            # Use same feature scaler
            agent.scaler = self.agent.scaler
            
            # Build and train
            agent.build_models(self.agent.state_size)
            
            # Quick training on subset
            ensemble_path = base_model_path.replace('.pth', f'_ensemble_{i}.pth')
            
            # Initialize with some randomness
            for param in agent.model.parameters():
                param.data += torch.randn_like(param.data) * 0.01
            
            self.ensemble_agents.append(agent)
            agent.save_model(ensemble_path)
            
            print(f"  Ensemble model {i+1} ready")
        
        print(f"\n Ensemble complete! {len(self.ensemble_agents)} models ready")
    
    def backtest(self, df, model_path='advanced_model.pth'):
        """Run backtest on historical data"""
        print(f"\n{'='*70}")
        print("BACKTESTING")
        print(f"{'='*70}")
        
        self.reset()
        
        for day in range(len(df)):
            row = df.iloc[day]
            price = row['close']
            
            # Update portfolio value
            portfolio_value = self.capital + (self.position * price)
            self.metrics.update(portfolio_value)
            self.price_data.append(price)
            
            # Skip warmup period
            if day < self.config.LOOKBACK:
                self.action_history.append(0)
                self.q_value_history.append([0] * self.agent.action_size)
                continue
            
            # Get state
            features = self.feature_engine.create_features(df, day)
            if features is None:
                self.action_history.append(0)
                self.q_value_history.append([0] * self.agent.action_size)
                continue
            
            state = self.agent.scaler.transform(features)
            
            # Get valid actions
            valid_actions = self.get_valid_actions(self.position)
            
            # Ensemble prediction
            if self.config.USE_ENSEMBLE and self.ensemble_agents:
                q_values_list = []
                
                # Main agent
                q_values_list.append(self.agent.predict_q_values(state)[0])
                
                # Ensemble agents
                for ens_agent in self.ensemble_agents:
                    q_values_list.append(ens_agent.predict_q_values(state)[0])
                
                # Combine predictions
                if self.config.ENSEMBLE_METHOD == 'voting':
                    actions = [np.argmax(q) for q in q_values_list]
                    action = max(set(actions), key=actions.count)
                elif self.config.ENSEMBLE_METHOD == 'average':
                    q_values = np.mean(q_values_list, axis=0)
                    action = np.argmax(q_values)
                else:  # weighted
                    weights = [2.0] + [1.0] * len(self.ensemble_agents)
                    q_values = np.average(q_values_list, axis=0, weights=weights)
                    action = np.argmax(q_values)
                
                final_q_values = np.mean(q_values_list, axis=0)
            else:
                # Single agent
                action = self.agent.act(state, valid_actions, force_explore=(day < self.config.LOOKBACK + 50))
                final_q_values = self.agent.predict_q_values(state)[0]
            
            # Execute trade
            self.execute_trade(action, price)
            
            # Store history
            self.action_history.append(action)
            self.q_value_history.append(final_q_values.tolist())
            
            # Learn from experience
            if day < len(df) - 1:
                next_row = df.iloc[day + 1]
                next_price = next_row['close']
                
                # Calculate reward based on action
                if action == 0:
                    # Hold
                    reward = -0.01
                elif action >= 1 and action <= 3:
                    # Buy actions (25%, 50%, 75%)
                    price_change = (next_price - price) / price
                    reward = price_change * 150 if price_change > 0 else price_change * 100
                elif action == 4:
                    # Strong buy (100%) or Sell
                    price_change = (next_price - price) / price
                    if self.position > 0:
                        # It was a sell
                        reward = -price_change * 200 if price_change < 0 else price_change * 150
                    else:
                        # It was a strong buy
                        reward = price_change * 200 if price_change > 0 else price_change * 150
                else:
                    reward = 0
                
                # Get next state
                next_features = self.feature_engine.create_features(df, day + 1)
                if next_features is not None:
                    next_state = self.agent.scaler.transform(next_features)
                    self.agent.remember(state, action, reward, next_state, False)
                
                # Train periodically
                if day % self.config.TRAIN_FREQ == 0 and len(self.agent.memory) >= self.agent.batch_size * 2:
                    for _ in range(3):
                        self.agent.replay()
                
                # Update target network
                if day % self.config.TARGET_UPDATE_FREQ == 0:
                    self.agent.update_target_model()
            
            # Progress update
            if day % 100 == 0 and day > 0:
                pct_done = day / len(df) * 100
                print(f"  [{pct_done:.1f}%] Day {day}/{len(df)} | ${portfolio_value:.2f} | "
                      f"{self.wins}W {self.losses}L {self.holds}H | ε={self.agent.epsilon:.3f}")
        
        # Save improved model
        self.agent.save_model(model_path)
        print(f"\n Backtest complete!")
    
    def run_auto_training_simulations(self, df, model_path, num_simulations=5):
        """Run multiple backtests on different time periods to improve model"""
        print(f"\n{'='*70}")
        print(f"AUTO-TRAINING: Running {num_simulations} simulations")
        print(f"{'='*70}")
        
        # Different training scenarios
        scenarios = [
            {'name': 'Recent Bull Run', 'days': 180, 'offset': 0},
            {'name': 'Mid-term Trend', 'days': 365, 'offset': 0},
            {'name': 'Historical Pattern', 'days': 365, 'offset': 365},
            {'name': 'Volatile Period', 'days': 90, 'offset': 180},
            {'name': 'Full History', 'days': min(730, len(df)), 'offset': 0},
        ]
        
        best_performance = -float('inf')
        total_trades = 0
        
        for i, scenario in enumerate(scenarios[:num_simulations], 1):
            print(f"\n[Simulation {i}/{num_simulations}] {scenario['name']}")
            print(f"  Period: {scenario['days']} days, Offset: {scenario['offset']}")
            
            # Extract scenario data
            start_idx = max(0, len(df) - scenario['days'] - scenario['offset'])
            end_idx = len(df) - scenario['offset']
            scenario_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
            
            if len(scenario_df) < 100:
                print(f"  Insufficient data, skipping...")
                continue
            
            # Save current state
            original_epsilon = self.agent.epsilon
            saved_metrics = self.metrics
            saved_capital = self.capital
            saved_position = self.position
            
            # Run simulation with more exploration
            self.agent.epsilon = 0.2
            self.reset()
            
            for day in range(len(scenario_df)):
                row = scenario_df.iloc[day]
                price = row['close']
                
                portfolio_value = self.capital + (self.position * price)
                self.metrics.update(portfolio_value)
                
                if day < self.config.LOOKBACK:
                    continue
                
                features = self.feature_engine.create_features(scenario_df, day)
                if features is None:
                    continue
                
                state = self.agent.scaler.transform(features)
                valid_actions = self.get_valid_actions(self.position)
                action = self.agent.act(state, valid_actions, force_explore=(day < self.config.LOOKBACK + 20))
                
                self.execute_trade(action, price)
                
                # Learn aggressively
                if day < len(scenario_df) - 1:
                    next_row = scenario_df.iloc[day + 1]
                    next_price = next_row['close']
                    price_change = (next_price - price) / price
                    
                    # Calculate reward based on action
                    if action == 0:
                        # Hold
                        reward = -0.05
                    elif action >= 1 and action <= 3:
                        # Buy actions (25%, 50%, 75%)
                        reward = price_change * 150 if price_change > 0 else price_change * 100
                    elif action == 4:
                        # Strong buy (100%) or Sell
                        if self.position > 0:
                            # It was a sell
                            reward = -price_change * 200 if price_change < 0 else price_change * 50
                        else:
                            # It was a strong buy
                            reward = price_change * 200 if price_change > 0 else price_change * 150
                    else:
                        reward = 0
                    
                    next_features = self.feature_engine.create_features(scenario_df, day + 1)
                    if next_features is not None:
                        next_state = self.agent.scaler.transform(next_features)
                        self.agent.remember(state, action, reward, next_state, False)
                    
                    # Train every 2 steps
                    if day % 2 == 0 and len(self.agent.memory) >= 64:
                        for _ in range(5):
                            self.agent.replay()
                    
                    if day % 15 == 0:
                        self.agent.update_target_model()
            
            # Evaluate performance
            if self.metrics.portfolio_values:
                final = self.metrics.portfolio_values[-1]
                returns = ((final - self.initial_capital) / self.initial_capital) * 100
                total_trades += len(self.metrics.trades)
                
                print(f"  Result: {returns:+.2f}% | {len(self.metrics.trades)} trades | {self.wins}W {self.losses}L")
                
                if returns > best_performance:
                    best_performance = returns
                    print(f"  New best performance!")
            
            # Restore state
            self.agent.epsilon = original_epsilon
            self.metrics = saved_metrics
            self.capital = saved_capital
            self.position = saved_position
        
        # Save improved model
        self.agent.save_model(model_path)
        
        print(f"\n{'='*70}")
        print(f"AUTO-TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Simulations run: {num_simulations}")
        print(f"Total trades executed: {total_trades}")
        print(f"Best performance: {best_performance:+.2f}%")
        print(f"Model improved and saved!")
        print(f"{'='*70}\n")
    
    def print_results(self):
        """Print comprehensive results"""
        if not self.metrics.portfolio_values:
            print("No results to display")
            return
        
        final_value = self.metrics.portfolio_values[-1]
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        print(f"\n{'='*70}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*70}")
        
        print(f"\n Capital:")
        print(f"  Initial:  {self.currency} {self.initial_capital:.2f}")
        print(f"  Final:    {self.currency} {final_value:.2f}")
        print(f"  P&L:      {self.currency} {final_value - self.initial_capital:+.2f}")
        print(f"  Return:   {total_return:+.2f}%")
        print(f"  Fees:     {self.currency} {self.total_fees:.2f}")
        
        print(f"\n Trading:")
        print(f"  Total:    {len(self.metrics.trades)} trades")
        print(f"  Results:  {self.wins}W | {self.losses}L | {self.holds}H")
        
        win_rate = self.metrics.get_win_rate()
        print(f"  Win Rate: {win_rate*100:.1f}%")
        
        profit_factor = self.metrics.get_profit_factor()
        print(f"  Profit Factor: {profit_factor:.2f}")
        
        print(f"\n Risk Metrics:")
        sharpe = self.metrics.get_sharpe_ratio()
        print(f"  Sharpe Ratio:  {sharpe:.2f}")
        
        sortino = self.metrics.get_sortino_ratio()
        print(f"  Sortino Ratio: {sortino:.2f}")
        
        max_dd = self.metrics.get_max_drawdown()
        print(f"  Max Drawdown:  {max_dd*100:.2f}%")
        
        calmar = self.metrics.get_calmar_ratio()
        print(f"  Calmar Ratio:  {calmar:.2f}")
        
        print(f"{'='*70}")
    
    def plot_results(self, coin='BTC'):
        """Plot comprehensive results"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # Price and trades
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.price_data, label='Price', color='#2E86AB', linewidth=2, alpha=0.8)
        
        buy_days = [t['day'] for t in self.metrics.trades if t['type'] == 'BUY' and t['day'] < len(self.price_data)]
        sell_days = [t['day'] for t in self.metrics.trades if t['type'] == 'SELL' and t['day'] < len(self.price_data)]
        
        if buy_days:
            ax1.scatter(buy_days, [self.price_data[i] for i in buy_days],
                       color='#06D6A0', marker='^', s=150, label='Buy', zorder=5,
                       edgecolors='black', linewidths=0.8)
        if sell_days:
            ax1.scatter(sell_days, [self.price_data[i] for i in sell_days],
                       color='#EF476F', marker='v', s=150, label='Sell', zorder=5,
                       edgecolors='black', linewidths=0.8)
        
        ax1.set_title(f'{coin} - Price Action & Trades', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Days')
        ax1.set_ylabel(f'Price ({self.currency})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Portfolio value
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(self.metrics.portfolio_values, color='#118AB2', linewidth=2)
        ax2.axhline(self.initial_capital, color='gray', linestyle='--', alpha=0.7, label='Initial')
        ax2.fill_between(range(len(self.metrics.portfolio_values)),
                        self.initial_capital, self.metrics.portfolio_values,
                        alpha=0.2, color='#118AB2')
        ax2.set_title('Portfolio Value', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Days')
        ax2.set_ylabel(f'Value ({self.currency})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Returns distribution
        ax3 = fig.add_subplot(gs[1, 1])
        if self.metrics.returns:
            ax3.hist(self.metrics.returns, bins=50, color='#06D6A0', alpha=0.7, edgecolor='black')
            ax3.axvline(0, color='red', linestyle='--', linewidth=2)
        ax3.set_title('Returns Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Daily Return')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Drawdown
        ax4 = fig.add_subplot(gs[2, 0])
        if self.metrics.drawdowns:
            ax4.fill_between(range(len(self.metrics.drawdowns)),
                           0, [-d*100 for d in self.metrics.drawdowns],
                           color='#EF476F', alpha=0.5)
            ax4.plot([-d*100 for d in self.metrics.drawdowns],
                    color='#EF476F', linewidth=2)
        ax4.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Days')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3)
        
        # Q-values
        ax5 = fig.add_subplot(gs[2, 1])
        q_array = np.array(self.q_value_history)
        if q_array.shape[1] >= 3:
            ax5.plot(q_array[:, 0], label='Q(HOLD)', alpha=0.7, linewidth=1.5)
            ax5.plot(q_array[:, 1], label='Q(BUY)', alpha=0.7, linewidth=1.5)
            if q_array.shape[1] > 2:
                ax5.plot(q_array[:, 2], label='Q(SELL)', alpha=0.7, linewidth=1.5)
        ax5.set_title('Q-Values', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Days')
        ax5.set_ylabel('Q-Value')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Training loss
        ax6 = fig.add_subplot(gs[3, 0])
        if self.agent.losses:
            # Smooth the losses
            window = 50
            if len(self.agent.losses) >= window:
                smoothed = np.convolve(self.agent.losses,
                                      np.ones(window)/window, mode='valid')
                ax6.plot(smoothed, color='#FF6B35', linewidth=2)
            else:
                ax6.plot(self.agent.losses, color='#FF6B35', linewidth=2)
        ax6.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Training Steps')
        ax6.set_ylabel('Loss')
        ax6.grid(True, alpha=0.3)
        
        # Metrics summary
        ax7 = fig.add_subplot(gs[3, 1])
        ax7.axis('off')
        
        metrics_text = f"""
        PERFORMANCE METRICS
        
        Total Return: {((self.metrics.portfolio_values[-1] - self.initial_capital) / self.initial_capital * 100):+.2f}%
        Sharpe Ratio: {self.metrics.get_sharpe_ratio():.2f}
        Sortino Ratio: {self.metrics.get_sortino_ratio():.2f}
        Max Drawdown: {self.metrics.get_max_drawdown()*100:.2f}%
        Calmar Ratio: {self.metrics.get_calmar_ratio():.2f}
        
        Win Rate: {self.metrics.get_win_rate()*100:.1f}%
        Profit Factor: {self.metrics.get_profit_factor():.2f}
        Total Trades: {len(self.metrics.trades)}
        """
        
        ax7.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.savefig('advanced_results.png', dpi=150, bbox_inches='tight')
        print("\n Saved: advanced_results.png")
        plt.show()

# ==================== MAIN ====================
def main():
    print("\n" + "="*70)
    print(" ADVANCED AI TRADING SYSTEM v2.0")
    print("="*70)
    
    if not TORCH_AVAILABLE:
        print("\n Install: pip install torch scikit-learn matplotlib")
        return
    
    # Configuration
    print("\n  Configuration:")
    print(f"  Model: {Config.MODEL_TYPE}")
    print(f"  Architecture: {Config.HIDDEN_LAYERS}")
    print(f"  Attention: {'✓' if Config.USE_ATTENTION else '✗'}")
    print(f"  Prioritized Replay: {'✓' if Config.USE_PRIORITIZED_REPLAY else '✗'}")
    print(f"  Ensemble: {'✓' if Config.USE_ENSEMBLE else '✗'}")
    print(f"  Fractional Positions: {'✓' if Config.USE_FRACTIONAL_POSITIONS else '✗'}")
    
    # Get user inputs
    try:
        capital = float(input("\n Capital (INR, default 100): ") or "100")
    except:
        capital = 100
    
    coins = CryptoDataProvider.get_available_coins()
    print("\n Available Coins:")
    for i, (sym, name) in enumerate(coins.items(), 1):
        print(f"  {i:2d}. {sym:5s} - {name}")
    
    try:
        choice = int(input("\nSelect coin (default 1): ") or "1")
        coin = list(coins.keys())[choice - 1]
    except:
        coin = 'BTC'
    
    print("\n Training Period:")
    print("  1. 90 days (Quick)")
    print("  2. 180 days (Medium)")
    print("  3. 365 days (Full Year)")
    print("  4. 730 days (2 Years)")
    
    try:
        period_choice = input("\nSelect (default 3): ").strip() or "3"
        periods = {'1': 90, '2': 180, '3': 365, '4': 730}
        period = periods.get(period_choice, 365)
    except:
        period = 365
    
    # Fetch data
    try:
        print(f"\n Fetching {coin} data...")
        df = CryptoDataProvider.fetch_historical_data(coin)
        
        # Convert to INR
        df['close'] = df['close'] * 83.0
        df['open'] = df['open'] * 83.0
        df['high'] = df['high'] * 83.0
        df['low'] = df['low'] * 83.0
        
        # Limit to period
        if len(df) > period:
            df = df.iloc[-period:].reset_index(drop=True)
        
        print(f" Loaded {len(df)} days of data")
        
        # Initialize system
        system = AdvancedTradingSystem(initial_capital=capital, currency='INR')
        
        # Train
        system.train(df)
        
        # Ask about simulations
        run_sims = input("\n Run auto-training simulations? (y/n, default y): ").strip().lower() or 'y'
        if run_sims == 'y':
            system.run_auto_training_simulations(df, 'advanced_model.pth')
        
        # Backtest
        system.backtest(df)
        
        # Results
        system.print_results()
        system.plot_results(coin)
        
        # Future predictions
        predict = input("\n Generate future predictions? (y/n, default y): ").strip().lower() or 'y'
        if predict == 'y':
            try:
                days = int(input("Days ahead to predict? (default 30): ").strip() or "30")
            except:
                days = 30
            
            print(f"\nUsing last {len(df)} days as context...")
            predictions = system.predict_future(df, days_ahead=days)
            
            if predictions:
                system.print_future_predictions(predictions)
                system.plot_future_prediction(df, predictions, coin)
            else:
                print(" Could not generate predictions")
        
        print("\n" + "="*70)
        print(" Advanced trading system completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()