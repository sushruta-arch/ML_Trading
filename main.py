import numpy as np
import pandas as pd
from collections import deque
import requests, time, warnings, os, pickle
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    TORCH_AVAILABLE = True
    print(f"PyTorch {torch.__version__}")
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: pip install torch scikit-learn matplotlib")

# ==================== DATA PROVIDER ====================
class CryptoDataProvider:
    @staticmethod
    def get_available_coins():
        return {
            'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'DOGE': 'Dogecoin',
            'ADA': 'Cardano', 'XRP': 'Ripple', 'SOL': 'Solana',
            'DOT': 'Polkadot', 'MATIC': 'Polygon', 'LTC': 'Litecoin',
            'LINK': 'Chainlink'
        }
    
    @staticmethod
    def fetch_from_binance_public(symbol='BTCUSDT', days=3650):
        print(f"\nFetching {symbol}...")
        url = "https://api.binance.com/api/v3/klines"
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        params = {'symbol': symbol, 'interval': '1d', 'startTime': start_time, 'endTime': end_time, 'limit': 1000}
        
        all_data = []
        for _ in range(4):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                if not data: break
                all_data.extend(data)
                params['endTime'] = data[0][0]
                time.sleep(0.2)
            except: break
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                             'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                             'taker_buy_quote', 'ignore'])
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"{len(df)} days ({len(df)/365:.1f} years)")
        return df
    
    @staticmethod
    def fetch_historical_data(coin='BTC'):
        symbols = {
            'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'DOGE': 'DOGEUSDT',
            'ADA': 'ADAUSDT', 'XRP': 'XRPUSDT', 'SOL': 'SOLUSDT',
            'DOT': 'DOTUSDT', 'MATIC': 'MATICUSDT', 'LTC': 'LTCUSDT',
            'LINK': 'LINKUSDT'
        }
        return CryptoDataProvider.fetch_from_binance_public(symbols.get(coin, 'BTCUSDT'), days=3650)

# ==================== SIMPLE FEATURES ====================
class SimpleFeatures:
    @staticmethod
    def create_features(prices, lookback=30):
        if len(prices) < lookback:
            return None
        
        try:
            recent = np.array(prices[-lookback:], dtype=float)
            current = prices[-1]
            features = []
            
            # Price position
            price_min, price_max = np.min(recent), np.max(recent)
            features.append((current - price_min) / (price_max - price_min + 1e-10))
            
            # Returns
            for lag in [1, 3, 7]:
                if len(prices) > lag:
                    features.append(np.clip((prices[-1] - prices[-lag]) / (prices[-lag] + 1e-10), -1, 1))
                else:
                    features.append(0.0)
            
            # Moving averages
            for window in [5, 10, 20]:
                if len(recent) >= window:
                    features.append(current / (np.mean(recent[-window:]) + 1e-10) - 1.0)
                else:
                    features.append(0.0)
            
            # Volatility
            if len(recent) >= 10:
                returns = np.diff(recent[-10:]) / (recent[-10:-1] + 1e-10)
                features.append(np.std(returns))
            else:
                features.append(0.0)
            
            features = np.array(features, dtype=float)
            
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                return None
            
            return features.reshape(1, -1)
        except:
            return None

# ==================== NEURAL NETWORK ====================
class TradingNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(TradingNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
    
    def forward(self, x):
        return self.net(x)

# ==================== TRADING AGENT ====================
class TradingAgent:
    def __init__(self, action_size=3):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        self.action_size = action_size
        self.state_size = None
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.memory = deque(maxlen=5000)
        self.batch_size = 32
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.scaler = StandardScaler()
        self.trained = False
    
    def build_models(self, state_size):
        self.state_size = state_size
        self.model = TradingNetwork(state_size, self.action_size).to(self.device)
        self.target_model = TradingNetwork(state_size, self.action_size).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        print(f"Network: {state_size}→128→64→32→{self.action_size}")
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def act(self, state, position=0.0):
        # Epsilon-greedy with valid actions
        if position == 0:
            valid_actions = [0, 1]  # Hold or Buy
        else:
            valid_actions = [0, 2]  # Hold or Sell
        
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(torch.FloatTensor(state).to(self.device)).cpu().numpy()[0]
            # Mask invalid actions
            for i in range(len(q_values)):
                if i not in valid_actions:
                    q_values[i] = -999999
            return np.argmax(q_values)
    
    def predict_q_values(self, state):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.FloatTensor(state).to(self.device)).cpu().numpy()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, targets = [], []
        
        for idx in batch:
            state, action, reward, next_state, done = self.memory[idx]
            target = reward
            if not done and next_state is not None:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(
                        self.target_model(torch.FloatTensor(next_state).to(self.device))
                    ).item()
            
            current_q = self.model(torch.FloatTensor(state).to(self.device)).cpu().detach().numpy()[0]
            target_q = current_q.copy()
            target_q[action] = target
            states.append(state[0])
            targets.append(target_q)
        
        self.model.train()
        outputs = self.model(torch.FloatTensor(np.array(states)).to(self.device))
        loss = nn.MSELoss()(outputs, torch.FloatTensor(np.array(targets)).to(self.device))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save_model(self, filepath='trading_model.pth'):
        torch.save({
            'model_state': self.model.state_dict(),
            'target_model_state': self.target_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_size': self.state_size
        }, filepath)
        
        with open(filepath.replace('.pth', '_scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Saved: {filepath}")
    
    def load_model(self, filepath='trading_model.pth'):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.state_size = checkpoint['state_size']
        self.epsilon = checkpoint.get('epsilon', 0.1)
        
        self.build_models(self.state_size)
        self.model.load_state_dict(checkpoint['model_state'])
        self.target_model.load_state_dict(checkpoint['target_model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        with open(filepath.replace('.pth', '_scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.trained = True
        print(f"Loaded: {filepath}, epsilon={self.epsilon:.3f}")

# ==================== PREDICTION ENGINE ====================
class PredictionEngine:
    def __init__(self):
        self.lstm_model = None
        self.prices_mean = 0
        self.prices_std = 1
    
    def calculate_technical_indicators(self, prices):
        prices_arr = np.array(prices)
        indicators = {}
        
        # RSI
        if len(prices) >= 15:
            deltas = np.diff(prices_arr)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            rs = avg_gain / (avg_loss + 1e-10)
            indicators['rsi'] = 100 - (100 / (1 + rs))
        else:
            indicators['rsi'] = 50
        
        # MACD
        if len(prices) >= 26:
            ema_12 = self._ema(prices_arr, 12)
            ema_26 = self._ema(prices_arr, 26)
            macd = ema_12 - ema_26
            # Normalize MACD as percentage of price
            macd_pct = (macd / prices_arr[-1]) * 100
            signal = self._ema(prices_arr[-9:], 9) if len(prices) > 26 else 0
            signal_pct = (signal / prices_arr[-1]) * 100 if prices_arr[-1] > 0 else 0
            indicators['macd'] = macd_pct
            indicators['macd_signal'] = signal_pct
            indicators['macd_histogram'] = macd_pct - signal_pct
        else:
            indicators['macd'] = 0
            indicators['macd_signal'] = 0
            indicators['macd_histogram'] = 0
        
        # Bollinger Bands
        if len(prices) >= 20:
            ma_20 = np.mean(prices_arr[-20:])
            std_20 = np.std(prices_arr[-20:])
            indicators['bb_upper'] = ma_20 + 2 * std_20
            indicators['bb_lower'] = ma_20 - 2 * std_20
            indicators['bb_position'] = (prices_arr[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'] + 1e-10)
        else:
            indicators['bb_upper'] = prices_arr[-1] * 1.1
            indicators['bb_lower'] = prices_arr[-1] * 0.9
            indicators['bb_position'] = 0.5
        
        # Stochastic
        if len(prices) >= 14:
            low_14 = np.min(prices_arr[-14:])
            high_14 = np.max(prices_arr[-14:])
            indicators['stochastic'] = 100 * (prices_arr[-1] - low_14) / (high_14 - low_14 + 1e-10)
        else:
            indicators['stochastic'] = 50
        
        indicators['adx'] = 25.0
        
        return indicators
    
    def _ema(self, prices, period):
        if len(prices) < period:
            return np.mean(prices)
        alpha = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
    
    def garch_volatility_forecast(self, returns, days_ahead=30):
        omega, alpha, beta = 0.000001, 0.1, 0.85
        recent_returns = returns[-30:] if len(returns) > 30 else returns
        var_t = np.var(recent_returns)
        forecast_var = []
        for _ in range(days_ahead):
            var_t = omega + alpha * (recent_returns[-1] ** 2) + beta * var_t
            forecast_var.append(np.sqrt(var_t))
        return np.array(forecast_var)
    
    def train_lstm_predictor(self, prices, epochs=30):
        if not TORCH_AVAILABLE:
            return None
        
        class LSTMPredictor(nn.Module):
            def __init__(self):
                super(LSTMPredictor, self).__init__()
                self.lstm = nn.LSTM(1, 64, 2, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(64, 1)
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.fc(lstm_out[:, -1, :])
        
        sequence_length = 30
        sequences, targets = [], []
        for i in range(sequence_length, len(prices)):
            sequences.append(prices[i-sequence_length:i])
            targets.append(prices[i])
        
        if len(sequences) < 50:
            return None
        
        self.prices_mean = np.mean(prices)
        self.prices_std = np.std(prices)
        sequences_norm = [(np.array(s) - self.prices_mean) / self.prices_std for s in sequences]
        targets_norm = [(t - self.prices_mean) / self.prices_std for t in targets]
        
        X = torch.FloatTensor(sequences_norm).unsqueeze(-1)
        y = torch.FloatTensor(targets_norm).unsqueeze(-1)
        
        self.lstm_model = LSTMPredictor()
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        self.lstm_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.lstm_model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        return self.lstm_model
    
    def lstm_predict(self, recent_prices, days_ahead=30):
        if self.lstm_model is None:
            return None
        
        self.lstm_model.eval()
        predictions = []
        current_seq = list(recent_prices[-30:])
        
        with torch.no_grad():
            for _ in range(days_ahead):
                seq_norm = (np.array(current_seq[-30:]) - self.prices_mean) / self.prices_std
                seq_tensor = torch.FloatTensor(seq_norm).unsqueeze(0).unsqueeze(-1)
                pred_norm = self.lstm_model(seq_tensor).item()
                pred_price = pred_norm * self.prices_std + self.prices_mean
                predictions.append(pred_price)
                current_seq.append(pred_price)
        
        return np.array(predictions)
    
    def statistical_forecast(self, prices, days_ahead=30):
        prices = np.array(prices)
        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        ar_coef = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0
        
        predictions = []
        last_price = prices[-1]
        last_return = returns[-1] if len(returns) > 0 else 0
        
        for i in range(days_ahead):
            ar_pred = ar_coef * last_return
            pred_price = last_price * (1 + ar_pred)
            predictions.append(pred_price)
            last_price = pred_price
            last_return = ar_pred
        
        return np.array(predictions)
    
    def monte_carlo_simulation(self, prices, days_ahead=30, n_simulations=1000):
        prices = np.array(prices)
        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        simulations = []
        for _ in range(n_simulations):
            prices_sim = [prices[-1]]
            for _ in range(days_ahead):
                shock = np.random.normal(mu, sigma)
                prices_sim.append(prices_sim[-1] * (1 + shock))
            simulations.append(prices_sim[1:])
        
        simulations = np.array(simulations)
        
        return {
            'median': np.median(simulations, axis=0),
            'lower_5': np.percentile(simulations, 5, axis=0),
            'upper_95': np.percentile(simulations, 95, axis=0),
            'lower_25': np.percentile(simulations, 25, axis=0),
            'upper_75': np.percentile(simulations, 75, axis=0)
        }
    
    def ensemble_predict(self, prices, days_ahead=30):
        prices = np.array(prices)
        
        print(f"\n{'='*70}")
        print("ENSEMBLE PREDICTION")
        print(f"{'='*70}")
        
        predictions = {}
        weights = {}
        
        # Technical Indicators
        print("[1/6] Technical indicators...")
        indicators = self.calculate_technical_indicators(prices)
        signal_strength = 0
        if indicators['rsi'] < 30:
            signal_strength += 2
        elif indicators['rsi'] > 70:
            signal_strength -= 2
        if indicators['macd_histogram'] > 0:
            signal_strength += 1
        elif indicators['macd_histogram'] < 0:
            signal_strength -= 1
        print(f"  RSI: {indicators['rsi']:.1f} | MACD: {indicators['macd']:.4f}")
        
        # LSTM
        print("\n[2/6] LSTM...")
        lstm_model = self.train_lstm_predictor(prices[-500:], epochs=30)
        if lstm_model:
            predictions['lstm'] = self.lstm_predict(prices, days_ahead)
            weights['lstm'] = 0.35
            print(" Complete")
        else:
            print(" Skipped")
        
        # Statistical
        print("\n[3/6] Statistical...")
        predictions['statistical'] = self.statistical_forecast(prices, days_ahead)
        weights['statistical'] = 0.25
        print("  Complete")
        
        # GARCH
        print("\n[4/6] GARCH volatility...")
        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        vol_forecast = self.garch_volatility_forecast(returns, days_ahead)
        print(f"  Volatility: {np.mean(vol_forecast)*100:.2f}%")
        
        # Monte Carlo
        print("\n[5/6] Monte Carlo...")
        mc_results = self.monte_carlo_simulation(prices, days_ahead, 1000)
        predictions['monte_carlo'] = mc_results['median']
        weights['monte_carlo'] = 0.20
        print(" Complete")
        
        # Trend
        print("\n[6/6] Trend...")
        ma_7 = np.mean(prices[-7:])
        ma_30 = np.mean(prices[-30:])
        base_trend = (ma_7 - ma_30) / ma_30
        
        trend_preds = []
        current = prices[-1]
        for i in range(days_ahead):
            decay = (1 - i / days_ahead)
            noise = np.random.normal(0, vol_forecast[i])
            current = current * (1 + base_trend * decay + noise)
            trend_preds.append(current)
        
        predictions['trend'] = np.array(trend_preds)
        weights['trend'] = 0.20
        print(" Complete")
        
        # Ensemble
        ensemble_pred = np.zeros(days_ahead)
        for name, pred in predictions.items():
            if name in weights and pred is not None:
                ensemble_pred += pred * weights[name]
        ensemble_pred /= sum(weights.values())
        
        # Adjust for signals
        adjustment = signal_strength * 0.005
        for i in range(days_ahead):
            ensemble_pred[i] *= (1 + adjustment * (1 - i / days_ahead))
        
        signal = 'BULLISH' if signal_strength > 0 else 'BEARISH' if signal_strength < 0 else 'NEUTRAL'
        
        print(f"\n{'='*70}")
        
        return {
            'predictions': ensemble_pred,
            'confidence_95': (mc_results['lower_5'], mc_results['upper_95']),
            'confidence_50': (mc_results['lower_25'], mc_results['upper_75']),
            'volatility': vol_forecast,
            'indicators': indicators,
            'signal': signal,
            'individual_predictions': predictions
        }

# ==================== TRADING SYSTEM ====================
class TradingSystem:
    def __init__(self, initial_capital=100, currency='INR'):
        self.initial_capital = initial_capital
        self.currency = currency
        self.fee = 0.001
        
        self.reset()
        self.agent = TradingAgent(action_size=3)
        self.features = SimpleFeatures()
    
    def reset(self):
        self.capital = self.initial_capital
        self.position = 0.0
        self.position_price = 0
        
        self.portfolio_values = []
        self.wins = 0
        self.losses = 0
        self.holds = 0
        self.total_fees = 0
        self.trade_log = []
        
        self.price_data = []
        self.action_history = []
        self.q_value_history = []
    
    def prepare_training_data(self, df):
        print("\nPreparing training data...")
        X_train, y_labels = [], []
        lookback = 30
        
        prices = df['close'].values.tolist()
        
        for i in range(lookback, len(prices) - 1):
            features = self.features.create_features(prices[:i+1], lookback)
            if features is None:
                continue
            
            future_ret = (prices[i+1] - prices[i]) / prices[i]
            
            if future_ret > 0.02:
                label = [0, 1, 0]  # Buy
            elif future_ret < -0.02:
                label = [0, 0, 1]  # Sell
            else:
                label = [1, 0, 0]  # Hold
            
            X_train.append(features[0])
            y_labels.append(label)
        
        print(f"{len(X_train)} samples")
        return np.array(X_train), np.array(y_labels)
    
    def train(self, df, model_path='trading_model.pth'):
        print(f"\n{'='*70}")
        print("TRAINING")
        print(f"{'='*70}")
        
        if os.path.exists(model_path):
            load = input(f"Load existing model? (y/n, default y): ").strip().lower() or 'y'
            if load == 'y':
                self.agent.load_model(model_path)
                return
        
        X_train, y_train = self.prepare_training_data(df)
        
        if len(X_train) == 0:
            raise ValueError("No training data!")
        
        self.agent.build_models(X_train.shape[1])
        
        X_norm = self.agent.scaler.fit_transform(X_train)
        X_tensor = torch.FloatTensor(X_norm).to(self.agent.device)
        y_tensor = torch.FloatTensor(y_train).to(self.agent.device)
        
        self.agent.model.train()
        criterion = nn.MSELoss()
        
        print("\n Training 200 epochs...")
        best_loss = float('inf')
        
        for epoch in range(200):
            outputs = self.agent.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            
            self.agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), 1.0)
            self.agent.optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: Loss={loss.item():.6f}")
        
        self.agent.update_target_model()
        self.agent.trained = True
        self.agent.save_model(model_path)
        
        print(f" Training complete! Loss={loss.item():.6f}")
    
    def execute_trade(self, action, price):
        if action == 0:
            self.holds += 1
            return None
        
        elif action == 1 and self.capital > 1:  # Buy
            amount = self.capital * 0.95
            fee = amount * self.fee
            shares = (amount - fee) / price
            
            self.capital -= amount
            self.position = shares
            self.position_price = price
            self.total_fees += fee
            
            self.trade_log.append({'type': 'BUY', 'price': price, 'shares': shares})
            print(f"  BUY {shares:.4f} @ ${price:.2f}")
            return {'type': 'BUY'}
        
        elif action == 2 and self.position > 0:  # Sell
            revenue = self.position * price
            fee = revenue * self.fee
            profit = (price - self.position_price) * self.position - fee
            
            self.capital += revenue - fee
            self.total_fees += fee
            
            if profit > 0:
                self.wins += 1
            else:
                self.losses += 1
            
            self.trade_log.append({'type': 'SELL', 'price': price, 'profit': profit})
            print(f"  SELL @ ${price:.2f} | P&L: {profit:+.2f}")
            
            self.position = 0
            self.position_price = 0
            return {'type': 'SELL', 'profit': profit}
        
        return None
    
    def backtest(self, df, model_path='trading_model.pth'):
        print(f"\n{'='*70}")
        print(" BACKTESTING")
        print(f"{'='*70}")
        
        self.reset()
        prices = df['close'].values.tolist()
        lookback = 30
        trades = 0
        
        # More aggressive epsilon
        self.agent.epsilon = 0.3
        
        for day in range(len(prices)):
            price = prices[day]
            portfolio_value = self.capital + (self.position * price)
            self.portfolio_values.append(portfolio_value)
            self.price_data.append(price)
            
            if day < lookback:
                self.action_history.append(0)
                self.q_value_history.append([0, 0, 0])
                continue
            
            state = self.features.create_features(prices[:day+1], lookback)
            if state is None:
                self.action_history.append(0)
                self.q_value_history.append([0, 0, 0])
                continue
            
            state = self.agent.scaler.transform(state)
            action = self.agent.act(state, self.position)
            q_values = self.agent.predict_q_values(state)[0]
            
            result = self.execute_trade(action, price)
            if result:
                trades += 1
            
            self.action_history.append(action)
            self.q_value_history.append(q_values.tolist())
            
            # Learn
            if day < len(prices) - 1:
                next_price = prices[day + 1]
                price_change = (next_price - price) / price
                
                if action == 0:
                    reward = 0
                elif action == 1:
                    reward = price_change * 100
                else:
                    reward = -price_change * 100
                
                next_state = self.features.create_features(prices[:day+2], lookback)
                if next_state is not None:
                    next_state = self.agent.scaler.transform(next_state)
                    self.agent.remember(state, action, reward, next_state, False)
                
                if day % 5 == 0 and len(self.agent.memory) >= 32:
                    self.agent.replay()
                
                if day % 20 == 0:
                    self.agent.update_target_model()
            
            if day % 50 == 0 and day > 0:
                print(f"  Day {day}/{len(prices)}: ${portfolio_value:.2f} | {trades} trades | {self.wins}W {self.losses}L")
        
        self.agent.save_model(model_path)
        print(f"\n Complete! {trades} trades executed")
    
    def print_results(self):
        if not self.portfolio_values:
            return
        
        final = self.portfolio_values[-1]
        returns = ((final - self.initial_capital) / self.initial_capital) * 100
        
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"Initial:  {self.currency} {self.initial_capital:.2f}")
        print(f"Final:    {self.currency} {final:.2f}")
        print(f"Return:   {returns:+.2f}%")
        print(f"Trades:   {len(self.trade_log)}")
        print(f"Results:  {self.wins}W {self.losses}L {self.holds}H")
        if self.wins + self.losses > 0:
            print(f"Win Rate: {self.wins/(self.wins+self.losses)*100:.1f}%")
        print(f"{'='*70}")
    
    def plot_results(self, coin='BTC'):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Price and trades
        axes[0].plot(self.price_data, label='Price', color='#2E86AB', linewidth=2)
        buy_days = [i for i, t in enumerate(self.trade_log) if t['type'] == 'BUY']
        sell_days = [i for i, t in enumerate(self.trade_log) if t['type'] == 'SELL']
        
        if buy_days:
            axes[0].scatter([buy_days], [self.price_data[i] for i in buy_days],
                          color='#06D6A0', marker='^', s=100, label='Buy', zorder=5)
        if sell_days:
            axes[0].scatter([sell_days], [self.price_data[i] for i in sell_days],
                          color='#EF476F', marker='v', s=100, label='Sell', zorder=5)
        
        axes[0].set_title(f'{coin} - Trading Activity')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Portfolio
        axes[1].plot(self.portfolio_values, color='#118AB2', linewidth=2)
        axes[1].axhline(self.initial_capital, color='gray', linestyle='--', alpha=0.7)
        axes[1].set_title('Portfolio Value')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('trading_results.png', dpi=150)
        print("Saved: trading_results.png")
        plt.show()
    
    def predict_future(self, df, days_ahead=30):
        print(f"\n{'='*70}")
        print(f"PREDICTING {days_ahead} DAYS")
        print(f"{'='*70}")
        
        engine = PredictionEngine()
        prices = df['close'].values.tolist()
        
        results = engine.ensemble_predict(prices, days_ahead)
        
        predictions = []
        for i, price in enumerate(results['predictions']):
            change = 0 if i == 0 else (results['predictions'][i] - results['predictions'][i-1]) / results['predictions'][i-1]
            
            if change > 0.02:
                action = 'STRONG BUY'
            elif change > 0.01:
                action = 'BUY'
            elif change < -0.02:
                action = 'STRONG SELL'
            elif change < -0.01:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            predictions.append({
                'day': i + 1,
                'price': price,
                'action': action,
                'lower_95': results['confidence_95'][0][i],
                'upper_95': results['confidence_95'][1][i]
            })
        
        self.prediction_results = results
        return predictions
    
    def print_predictions(self, predictions):
        if not predictions:
            return
        
        print(f"\n{'='*70}")
        print("FORECAST")
        print(f"{'='*70}")
        
        print(f"\n{'Day':<6} {'Price':<14} {'95% CI':<26} {'Signal':<15}")
        print(f"{'-'*70}")
        
        for p in predictions[:10]:
            print(f"{p['day']:<6} {self.currency} {p['price']:<10.2f} "
                  f"{p['lower_95']:.2f} - {p['upper_95']:.2f}{' '*8} {p['action']:<15}")
        
        if len(predictions) > 10:
            print(f"... +{len(predictions)-10} more days")
        
        current = self.price_data[-1] if self.price_data else predictions[0]['price']
        final = predictions[-1]['price']
        expected = ((final - current) / current) * 100
        
        print(f"\n{'='*70}")
        print(f"Current: {self.currency} {current:.2f}")
        print(f"Expected: {self.currency} {final:.2f} ({expected:+.2f}%)")
        print(f"{'='*70}\n")
    
    def plot_prediction(self, df, predictions, coin='BTC'):
        hist = df['close'].values.tolist()
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        hist_days = list(range(len(hist)))
        fut_days = list(range(len(hist), len(hist) + len(predictions)))
        fut_prices = [p['price'] for p in predictions]
        
        # Main chart
        axes[0].plot(hist_days, hist, label='Historical', color='#2E86AB', linewidth=2)
        axes[0].plot(fut_days, fut_prices, label='Forecast', color='#FF6B35', linewidth=2)
        
        lower = [p['lower_95'] for p in predictions]
        upper = [p['upper_95'] for p in predictions]
        axes[0].fill_between(fut_days, lower, upper, alpha=0.2, color='purple', label='95% CI')
        
        axes[0].axvline(x=len(hist)-1, color='red', linestyle=':', linewidth=2)
        axes[0].set_title(f'{coin} - Forecast')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Volatility
        if hasattr(self, 'prediction_results'):
            vol = self.prediction_results['volatility'] * 100
            axes[1].plot(fut_days, vol, color='#EF476F', linewidth=2)
            axes[1].fill_between(fut_days, 0, vol, alpha=0.3, color='#EF476F')
            axes[1].set_title('Volatility Forecast')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('prediction.png', dpi=150)
        print("Saved: prediction.png")
        plt.show()

# ==================== MAIN ====================
def main():
    print("\n" + "="*70)
    print("🚀 CRYPTO TRADING AI")
    print("="*70)
    
    if not TORCH_AVAILABLE:
        print("\n❌ Install: pip install torch scikit-learn matplotlib")
        return
    
    try:
        capital = float(input("\nCapital (INR, default 1000): ") or "1000")
    except:
        capital = 1000
    
    coins = CryptoDataProvider.get_available_coins()
    print("\n Coins:")
    for i, (sym, name) in enumerate(coins.items(), 1):
        print(f"  {i}. {sym} - {name}")
    
    try:
        choice = int(input("\nSelect (default 1): ") or "1")
        coin = list(coins.keys())[choice - 1]
    except:
        coin = 'BTC'
    
    print("\nPeriod:")
    print("  1. 90 days")
    print("  2. 180 days")
    print("  3. 365 days")
    
    try:
        period = {
            '1': 90, '2': 180, '3': 365
        }[input("\nSelect (default 3): ") or "3"]
    except:
        period = 365
    
    try:
        df = CryptoDataProvider.fetch_historical_data(coin)
        
        # Convert to INR
        df['close'] = df['close'] * 83.0
        df['open'] = df['open'] * 83.0
        df['high'] = df['high'] * 83.0
        df['low'] = df['low'] * 83.0
        
        if len(df) > period:
            df = df.iloc[-period:].reset_index(drop=True)
        
        print(f"✓ {len(df)} days loaded")
        
        system = TradingSystem(initial_capital=capital, currency='INR')
        
        # Train
        system.train(df)
        
        # Backtest
        system.backtest(df)
        system.print_results()
        system.plot_results(coin)
        
        # Predict
        predict = input("\nPredict future? (y/n, default y): ").strip().lower() or 'y'
        if predict == 'y':
            try:
                days = int(input("Days ahead (default 30): ") or "30")
            except:
                days = 30
            
            predictions = system.predict_future(df, days_ahead=days)
            if predictions:
                system.print_predictions(predictions)
                system.plot_prediction(df, predictions, coin)
        
        print("\n" + "="*70)
        print("Complete!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
