import asyncio
import aiohttp
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import xgboost as xgb
import lightgbm as lgb
import torch
from datetime import datetime
from sklearn.metrics import mean_absolute_error
# --- Исправлено: Импорты перенесены в начало ---
import joblib
import os
# --- Конец исправления ---

from config import logger
from ml.models import SimpleLSTM

class AdvancedCryptoPredictor:
    """Продвинутый криптопредиктор"""

    def __init__(self, symbol='BTC-USDT', interval='1H', sequence_length=60):
        self.symbol = symbol
        self.interval = interval
        self.sequence_length = sequence_length
        # ИСПРАВЛЕНО: Убран лишний пробел в URL
        self.base_url = 'https://www.okx.com/api/v5' # Исправлено

        # Проверка GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Инициализация скалеров
        self.scaler = RobustScaler()
        self.price_scaler = StandardScaler()

        # Ансамбль моделей
        self.models = {}
        self.weights = {}
        self.is_trained = False
        self.feature_names_flat = None

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
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
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
                # ИСПРАВЛЕНО: Явное преобразование timestamp в int64 перед to_datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit="ms")

                return df[['timestamp'] + numeric_cols]
            else:
                logger.error(f"Ошибка API: {data.get('msg', 'Неизвестная ошибка')}")
                return None

        except asyncio.TimeoutError:
            logger.error("Тайм-аут при получении данных от API")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"Ошибка клиента aiohttp: {e}")
            return None
        except KeyError as e:
            logger.error(f"Непредвиденная структура данных API: отсутствует ключ {e}")
            return None
        except Exception as e:
            logger.error(f"Ошибка получения данных: {e}")
            return None

    def calculate_comprehensive_indicators(self, df):
        """Расчет технических индикаторов"""
        try:
            df = df.copy()
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['hl_ratio'] = (df['high'] - df['low']) / df['close']
            df['oc_ratio'] = (df['open'] - df['close']) / df['close']

            periods = [7, 14, 21, 50]
            for period in periods:
                df[f'sma_{period}'] = df['close'].rolling(period, min_periods=1).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False, min_periods=1).mean()

            ema12 = df['close'].ewm(span=12, adjust=False, min_periods=12).mean()
            ema26 = df['close'].ewm(span=26, adjust=False, min_periods=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

            delta = df['close'].diff()
            up = delta.where(delta > 0, 0)
            down = -delta.where(delta < 0, 0)
            avg_gain = up.rolling(window=14, min_periods=1).mean()
            avg_loss = down.rolling(window=14, min_periods=1).mean()
            avg_gain = avg_gain.fillna(0)
            avg_loss = avg_loss.fillna(0)
            for i in range(1, len(avg_gain)):
                 avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * 13 + up.iloc[i]) / 14
                 avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * 13 + down.iloc[i]) / 14

            rs = avg_gain / avg_loss.replace(0, np.nan)
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi'] = df['rsi'].fillna(50)

            sma20 = df['close'].rolling(20, min_periods=1).mean()
            std20 = df['close'].rolling(20, min_periods=1).std()
            std20 = std20.fillna(0)
            df['bb_upper'] = sma20 + 2 * std20
            df['bb_lower'] = sma20 - 2 * std20
            bb_width = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = np.where(bb_width != 0, (df['close'] - df['bb_lower']) / bb_width, 0.5)

            df['volume_sma'] = df['volume'].rolling(20, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, np.nan)
            df['volume_ratio'] = df['volume_ratio'].fillna(1)

            df['volatility'] = df['log_returns'].rolling(20, min_periods=1).std()
            df['volatility'] = df['volatility'].fillna(0)

            low_14 = df['low'].rolling(14, min_periods=1).min()
            high_14 = df['high'].rolling(14, min_periods=1).max()
            df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14)).fillna(50)
            df['stoch_d'] = df['stoch_k'].rolling(3, min_periods=1).mean()

            df['tr1'] = abs(df['high'] - df['low'])
            df['tr2'] = abs(df['high'] - df['close'].shift(1))
            df['tr3'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr_14'] = df['tr'].rolling(14, min_periods=1).mean()
            df['atr_14'] = df['atr_14'].fillna(df['tr'])

            return df.dropna()

        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {e}")
            return df

    def prepare_sequences(self, df):
        """Подготовка последовательностей для обучения"""
        feature_cols = [
            'close', 'returns', 'rsi', 'macd', 'bb_position', 'volume_ratio',
            'volatility', 'stoch_k', 'stoch_d', 'atr_14'
        ]
        feature_cols = [col for col in feature_cols if col in df.columns]

        features = df[feature_cols].fillna(0)
        features_scaled = self.scaler.fit_transform(features)

        prices = df['close'].values.reshape(-1, 1)
        prices_scaled = self.price_scaler.fit_transform(prices)

        X, y = [], []
        seq_len = min(self.sequence_length, len(features_scaled) - 1)

        if seq_len <= 0:
             logger.error("Недостаточно данных для создания последовательностей.")
             return np.array([]), np.array([])

        for i in range(seq_len, len(features_scaled) - 1):
            X.append(features_scaled[i-seq_len:i])
            y.append(prices_scaled[i+1, 0])

        flat_feature_names = []
        for i in range(seq_len):
            for col in feature_cols:
                flat_feature_names.append(f"{col}_t-{seq_len-i-1}")

        self.feature_names_flat = flat_feature_names
        return np.array(X), np.array(y)

    async def train_simple(self):
        """Упрощенное обучение модели"""
        try:
            df = await self.fetch_historical_data_async(limit=1000)
            if df is None or len(df) < 100:
                raise Exception("Недостаточно данных для обучения")

            df = self.calculate_comprehensive_indicators(df)
            if df.empty:
                 raise Exception("Ошибка при расчете индикаторов, получена пустая таблица")

            X, y = self.prepare_sequences(df)
            if len(X) < 50:
                raise Exception("Недостаточно данных после обработки")

            split_idx = int(len(X) * 0.8)
            if split_idx == 0: split_idx = 1
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)

            X_train_flat_df = pd.DataFrame(X_train_flat, columns=self.feature_names_flat)
            X_val_flat_df = pd.DataFrame(X_val_flat, columns=self.feature_names_flat)

            self.models['xgb'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=1
            )
            self.models['xgb'].fit(X_train_flat_df, y_train)

            self.models['lgb'] = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=1,
                verbose=-1
            )
            self.models['lgb'].fit(X_train_flat_df, y_train)

            try:
                logger.info("Начало обучения LSTM...")
                # Создание модели
                self.models['lstm'] = SimpleLSTM(
                    input_size=X_train.shape[2], # кол-во признаков
                    hidden_size=50,
                    num_layers=2,
                    dropout=0.2 # Можно добавить dropout, если нужно
                ).to(self.device) # Переносим модель на устройство (CPU/GPU)

                # --- Добавлено: Обучение LSTM ---
                # Конвертация данных в тензоры PyTorch
                import torch.optim as optim
                import torch.nn as nn   
                import torch.nn.functional as F

                # Переносим данные на устройство
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)

                # Определение функции потерь и оптимизатора
                criterion = nn.MSELoss() # или nn.L1Loss() для MAE
                optimizer = optim.Adam(self.models['lstm'].parameters(), lr=0.001)

                # Цикл обучения
                epochs = 30
                best_val_loss = float('inf')
                patience = 5 # Для ранней остановки
                patience_counter = 0

                self.models['lstm'].train() # Переводим модель в режим обучения
                for epoch in range(epochs):
                    optimizer.zero_grad() # Обнуляем градиенты
                    outputs = self.models['lstm'](X_train_tensor) # Прямой проход
                    loss = criterion(outputs, y_train_tensor) # Вычисляем loss
                    loss.backward() # Обратный проход
                    optimizer.step() # Обновляем веса

                    # Валидация
                    self.models['lstm'].eval() # Переводим в режим оценки
                    with torch.no_grad():
                        val_outputs = self.models['lstm'](X_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor)
                    
                    if epoch % 10 == 0: # Логируем каждые 10 эпох
                         logger.info(f'Эпоха [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

                    # Ранняя остановка
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Можно сохранить лучшую модель здесь
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(f'Ранняя остановка на эпохе {epoch+1}')
                            break
                    
                    self.models['lstm'].train() # Возвращаем в режим обучения

                logger.info("LSTM обучена успешно.")
                # --- Конец обучения LSTM ---
                
            except Exception as e:
                 logger.error(f"Ошибка обучения LSTM: {e}")
                 self.models.pop('lstm', None)

            # --- Интеллектуальное взвешивание ---
            logger.info("Расчет весов моделей...")
            val_predictions = {}
            mae_scores = {}

            val_predictions['xgb'] = self.models['xgb'].predict(X_val_flat_df)
            mae_scores['xgb'] = mean_absolute_error(y_val, val_predictions['xgb'])

            val_predictions['lgb'] = self.models['lgb'].predict(X_val_flat_df)
            mae_scores['lgb'] = mean_absolute_error(y_val, val_predictions['lgb'])

            if 'lstm' in self.models:
                 try:
                     val_preds_lstm = self.models['lstm'].predict(X_val)
                     if val_preds_lstm.ndim == 2 and val_preds_lstm.shape[1] == 1:
                         val_preds_lstm = val_preds_lstm.flatten()
                     elif val_preds_lstm.ndim > 1:
                         logger.warning(f"Неожиданная форма предсказаний LSTM: {val_preds_lstm.shape}")
                         val_preds_lstm = val_preds_lstm.flatten()[:len(y_val)]

                     val_predictions['lstm'] = val_preds_lstm
                     mae_scores['lstm'] = mean_absolute_error(y_val, val_preds_lstm)
                 except Exception as e:
                     logger.error(f"Ошибка предсказания LSTM на валидации: {e}")
                     mae_scores['lstm'] = np.inf
                     self.models.pop('lstm', None)

            total_inv_mae = sum(1.0 / (mae + 1e-8) for mae in mae_scores.values())
            if total_inv_mae == 0:
                 self.weights = {k: 1.0 / len(mae_scores) for k in mae_scores}
            else:
                 self.weights = {k: (1.0 / (mae_scores[k] + 1e-8)) / total_inv_mae for k in mae_scores}

            logger.info(f"Веса моделей рассчитаны: {self.weights}")
            # --- Конец интеллектуального взвешивания ---

            self.is_trained = True
            logger.info(f"Модель для {self.symbol} обучена успешно")
            return True

        except Exception as e:
            logger.error(f"Ошибка обучения модели: {e}")
            self.is_trained = False
            return False

    def is_model_ready(self):
        """Проверяет, готова ли модель к предсказанию."""
        return self.is_trained and bool(self.models) and hasattr(self.scaler, 'scale_') and hasattr(self.price_scaler, 'scale_')

    async def predict_simple(self):
        """Упрощенное предсказание"""
        try:
            if not self.is_model_ready():
                logger.info("Модель не готова. Запуск обучения...")
                success = await self.train_simple()
                if not success:
                    logger.error("Не удалось обучить модель для предсказания.")
                    return None
                if not self.is_model_ready():
                     logger.error("Модель не готова после попытки обучения.")
                     return None

            df = await self.fetch_historical_data_async(limit=200)
            if df is None or len(df) == 0:
                logger.error("Не удалось получить данные для предсказания.")
                return None

            df = self.calculate_comprehensive_indicators(df)
            if df.empty:
                 logger.error("Ошибка при расчете индикаторов для предсказания.")
                 return None

            X, y = self.prepare_sequences(df)
            if len(X) == 0:
                logger.error("Недостаточно данных после обработки для предсказания.")
                return None

            current_price = df['close'].iloc[-1]
            last_sequence_3d = X[-1:]
            last_sequence_flat = last_sequence_3d.reshape(1, -1)
            last_sequence_flat_df = pd.DataFrame(last_sequence_flat, columns=self.feature_names_flat)

            predictions = {}
            for name in ['xgb', 'lgb']:
                 if name in self.models:
                     try:
                         pred_scaled = self.models[name].predict(last_sequence_flat_df)[0]
                         pred_price = self.price_scaler.inverse_transform([[pred_scaled]])[0, 0]
                         predictions[name] = pred_price
                     except Exception as e:
                         logger.error(f"Ошибка предсказания модели {name}: {e}")

            if 'lstm' in self.models:
                 try:
                     pred_scaled_lstm = self.models['lstm'].predict(last_sequence_3d)[0]
                     if hasattr(pred_scaled_lstm, 'item'):
                         pred_scaled_lstm = pred_scaled_lstm.item()
                     pred_price_lstm = self.price_scaler.inverse_transform([[pred_scaled_lstm]])[0, 0]
                     predictions['lstm'] = pred_price_lstm
                 except Exception as e:
                     logger.error(f"Ошибка предсказания LSTM: {e}")

            if predictions:
                total_weight = sum(self.weights.get(name, 0) for name in predictions.keys())
                if total_weight == 0:
                    logger.warning("Сумма весов для доступных предсказаний равна 0.")
                    final_pred = current_price
                else:
                    normalized_weights = {name: self.weights.get(name, 0) / total_weight for name in predictions.keys()}
                    final_pred = sum(pred * normalized_weights.get(name, 0) for name, pred in predictions.items())

                change_percent = (final_pred - current_price) / current_price * 100
                current_rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50

                return {
                    'current_price': current_price,
                    'predicted_price': final_pred,
                    'change_percent': change_percent,
                    'rsi': current_rsi,
                    'predictions': predictions,
                    'weights': normalized_weights if 'normalized_weights' in locals() else self.weights,
                    'symbol': self.symbol,
                    'interval': self.interval,
                    'timestamp': datetime.now()
                }

            logger.warning("Ни одна модель не дала предсказания.")
            return None

        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            return None

    # --- Управление версиями моделей ---
    # Импорты joblib и os перенесены в начало файла

    def save_model(self, path='saved_models'):
        """Сохраняет обученные модели, скалеры и веса."""
        try:
            os.makedirs(path, exist_ok=True)
            model_path = os.path.join(path, f"{self.symbol}_{self.interval}_model.pkl")
            scaler_path = os.path.join(path, f"{self.symbol}_{self.interval}_scaler.pkl")
            price_scaler_path = os.path.join(path, f"{self.symbol}_{self.interval}_price_scaler.pkl")
            weights_path = os.path.join(path, f"{self.symbol}_{self.interval}_weights.pkl")
            feature_names_path = os.path.join(path, f"{self.symbol}_{self.interval}_feature_names.pkl")

            joblib.dump(self.models, model_path)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.price_scaler, price_scaler_path)
            joblib.dump(self.weights, weights_path)
            joblib.dump(self.feature_names_flat, feature_names_path)

            logger.info(f"Модель для {self.symbol} сохранена в {path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")

    def load_model(self, path='saved_models'):
        """Загружает обученные модели, скалеры и веса."""
        try:
            model_path = os.path.join(path, f"{self.symbol}_{self.interval}_model.pkl")
            scaler_path = os.path.join(path, f"{self.symbol}_{self.interval}_scaler.pkl")
            price_scaler_path = os.path.join(path, f"{self.symbol}_{self.interval}_price_scaler.pkl")
            weights_path = os.path.join(path, f"{self.symbol}_{self.interval}_weights.pkl")
            feature_names_path = os.path.join(path, f"{self.symbol}_{self.interval}_feature_names.pkl")

            if not all(os.path.exists(p) for p in [model_path, scaler_path, price_scaler_path, weights_path, feature_names_path]):
                logger.warning(f"Не найдены сохраненные файлы модели в {path} для {self.symbol}")
                return False

            self.models = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.price_scaler = joblib.load(price_scaler_path)
            self.weights = joblib.load(weights_path)
            self.feature_names_flat = joblib.load(feature_names_path)

            self.is_trained = True
            logger.info(f"Модель для {self.symbol} загружена из {path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            self.is_trained = False
            return False
    # --- Конец управления версиями моделей ---
