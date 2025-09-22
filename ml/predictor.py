import asyncio
import aiohttp
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import xgboost as xgb
import lightgbm as lgb
import torch
from datetime import datetime

from config import logger
from ml.models import SimpleLSTM

class AdvancedCryptoPredictor:
    """Продвинутый криптопредиктор"""
    
    def __init__(self, symbol='BTC-USDT', interval='1H', sequence_length=60):
        self.symbol = symbol
        self.interval = interval
        self.sequence_length = sequence_length
        self.base_url = 'https://www.okx.com/api/v5'
        
        # Проверка GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Инициализация скалеров
        self.scaler = RobustScaler()
        self.price_scaler = StandardScaler()
        
        # Ансамбль моделей
        self.models = {}
        self.weights = {}
        self.is_trained = False
        
        # Метрики
        self.validation_results = {}
        
    async def fetch_historical_data_async(self, limit=2000):
        """Асинхронное получение данных"""
        url = f"{self.base_url}/market/candles"
        params = {
            'instId': self.symbol,
            'bar': self.interval,
            'limit': str(limit)
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    data = await response.json()
            
            if data['code'] == '0':
                candles = data['data']
                candles.reverse()
                
                df = pd.DataFrame(candles, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                    'volume_ccy', 'vol_ccy_quote', 'confirm'
                ])
                
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_cols] = df[numeric_cols].astype(float)
                df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit="ms")
                
                return df[['timestamp'] + numeric_cols]
            else:
                logger.error(f"Ошибка API: {data['msg']}")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка получения данных: {e}")
            return None
    
    def calculate_comprehensive_indicators(self, df):
        """Расчет технических индикаторов"""
        try:
            # Базовые цены
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['hl_ratio'] = (df['high'] - df['low']) / df['close']
            df['oc_ratio'] = (df['open'] - df['close']) / df['close']
            
            # Скользящие средние
            periods = [7, 14, 21, 50]
            for period in periods:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            
            # MACD
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            sma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            df['bb_upper'] = sma20 + 2 * std20
            df['bb_lower'] = sma20 - 2 * std20
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Volatility
            df['volatility'] = df['log_returns'].rolling(20).std()
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {e}")
            return df
    
    def prepare_sequences(self, df):
        """Подготовка последовательностей для обучения"""
        # Выбор основных признаков
        feature_cols = ['close', 'returns', 'rsi', 'macd', 'bb_position', 'volume_ratio']
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Нормализация
        features = df[feature_cols].fillna(0)
        features_scaled = self.scaler.fit_transform(features)
        
        # Нормализация цен
        prices = df['close'].values.reshape(-1, 1)
        prices_scaled = self.price_scaler.fit_transform(prices)
        
        # Создание последовательностей
        X, y = [], []
        seq_len = min(self.sequence_length, 30)
        
        for i in range(seq_len, len(features_scaled) - 1):
            X.append(features_scaled[i-seq_len:i])
            y.append(prices_scaled[i+1, 0])
        
        # Создаем имена признаков для плоских данных
        flat_feature_names = []
        for i in range(seq_len):
            for col in feature_cols:
                flat_feature_names.append(f"{col}_t-{seq_len-i-1}")
        
        return np.array(X), np.array(y), flat_feature_names
    
    async def train_simple(self):
        """Упрощенное обучение модели"""
        try:
            # Загрузка данных
            df = await self.fetch_historical_data_async(limit=1000)
            if df is None or len(df) < 100:
                raise Exception("Недостаточно данных для обучения")
            
            # Расчет индикаторов
            df = self.calculate_comprehensive_indicators(df)
            
            # Подготовка данных
            X, y, feature_names = self.prepare_sequences(df)
            if len(X) < 50:
                raise Exception("Недостаточно данных после обработки")
            
            # Разделение данных
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Подготовка для XGBoost
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            
            # Обучение XGBoost с именами признаков
            self.models['xgb'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=1
            )
            self.models['xgb'].fit(X_train_flat, y_train)
            
            # Обучение LightGBM с именами признаков
            self.models['lgb'] = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=1,
                verbose=-1
            )
            self.models['lgb'].fit(X_train_flat, y_train, feature_name=feature_names)
            
            # Простые веса
            self.weights = {'xgb': 0.5, 'lgb': 0.5}
            
            self.is_trained = True
            logger.info(f"Модель для {self.symbol} обучена успешно")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обучения модели: {e}")
            return False

    
    async def predict_simple(self):
        """Упрощенное предсказание"""
        try:
            if not self.is_trained:
                success = await self.train_simple()
                if not success:
                    return None
            
            # Загрузка новых данных
            df = await self.fetch_historical_data_async(limit=200)
            if df is None:
                return None
            
            # Расчет индикаторов
            df = self.calculate_comprehensive_indicators(df)
            
            # Подготовка данных
            X, y, _ = self.prepare_sequences(df)
            if len(X) == 0:
                return None
            
            # Текущая цена
            current_price = df['close'].iloc[-1]
            
            # Предсказания
            X_flat = X[-1:].reshape(1, -1)
            
            predictions = {}
            for name, model in self.models.items():
                if hasattr(model, 'predict'):
                    pred_scaled = model.predict(X_flat)[0]
                    pred_price = self.price_scaler.inverse_transform([[pred_scaled]])[0, 0]
                    predictions[name] = pred_price
            
            # Итоговое предсказание
            if predictions:
                final_pred = sum(pred * self.weights.get(name, 0) for name, pred in predictions.items())
                change_percent = (final_pred - current_price) / current_price * 100
                
                # RSI для анализа
                current_rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
                
                return {
                    'current_price': current_price,
                    'predicted_price': final_pred,
                    'change_percent': change_percent,
                    'rsi': current_rsi,
                    'predictions': predictions,
                    'symbol': self.symbol,
                    'interval': self.interval,
                    'timestamp': datetime.now()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            return None