import torch
import torch.nn as nn  # Импорт модуля нейронных сетей
import torch.optim as optim  # Импорт оптимизаторов
import torch.nn.functional as F # Часто используется, добавим на будущее
import numpy as np
from config import logger

class SimpleLSTM(nn.Module):
    """Упрощенная LSTM модель для прогнозирования временных рядов."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Инициализирует архитектуру LSTM модели.

        Args:
            input_size (int): Количество признаков на каждый временной шаг.
            hidden_size (int, optional): Количество признаков в скрытом состоянии. Defaults to 64.
            num_layers (int, optional): Количество слоев LSTM. Defaults to 2.
            dropout (float, optional): Вероятность Dropout между слоями. Defaults to 0.2.
        """
        print(f"Type of nn: {type(nn)}")  # Отладочная строка
        print(f"nn module: {nn}")  
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0 # Dropout применяется между слоями
        )
        # Dropout перед полносвязным слоем
        self.dropout = nn.Dropout(dropout) 
        # Полносвязный слой для получения одного выходного значения
        self.fc = nn.Linear(hidden_size, 1) 
        
    def forward(self, x):
        """
        Определяет прямой проход (forward pass) модели.
        """
        # lstm_out: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        # Берем выход последнего временного шага: (batch_size, hidden_size)
        out = lstm_out[:, -1, :] 
        # Применяем dropout
        out = self.dropout(out)
        # Пропускаем через полносвязный слой
        output = self.fc(out).squeeze()
        
        # Убедимся, что возвращаем правильную форму
        if output.dim() == 0:  # Если скаляр
            output = output.unsqueeze(0)  # Преобразуем в массив формы (1,)
        
        return output

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=50, lr=0.001, device=None, patience=5, batch_size=32):
        """
        Обучает модель на предоставленных данных.

        Args:
            X_train (np.ndarray): Обучающие данные, форма (N, seq_len, input_size).
            y_train (np.ndarray): Целевые значения для обучения, форма (N,).
            X_val (np.ndarray, optional): Валидационные данные. Defaults to None.
            y_val (np.ndarray, optional): Целевые значения для валидации. Defaults to None.
            epochs (int, optional): Количество эпох обучения. Defaults to 50.
            lr (float, optional): Скорость обучения. Defaults to 0.001.
            device (torch.device, optional): Устройство для обучения ('cpu' или 'cuda'). 
                                              Если None, используется устройство модели. Defaults to None.
            patience (int, optional): Количество эпох для ранней остановки. Defaults to 5.
            batch_size (int, optional): Размер батча. Defaults to 32.
        """
        if device is None:
            device = next(self.parameters()).device # Получаем устройство модели

        self.to(device) # Убедиться, что модель на нужном устройстве
        self.train() # Переводим модель в режим обучения

        # Конвертация данных в тензоры PyTorch и перенос на устройство
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
            val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Определение функции потерь и оптимизатора
        criterion = nn.MSELoss() # Можно использовать nn.L1Loss() для MAE
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Для ранней остановки
        best_val_loss = float('inf')
        patience_counter = 0

        # Создание DataLoader для обучения
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        logger.info(f"Начало обучения LSTM: эпохи={epochs}, LR={lr}, Batch Size={batch_size}")
        for epoch in range(epochs):
            # --- Обучение ---
            self.train()
            total_train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * batch_X.size(0) # Умножаем на размер батча для среднего

            avg_train_loss = total_train_loss / len(train_loader.dataset)

            # --- Валидация ---
            val_loss_str = "N/A"
            if val_loader is not None:
                self.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self(batch_X)
                        loss = criterion(outputs, batch_y)
                        total_val_loss += loss.item() * batch_X.size(0)
                
                avg_val_loss = total_val_loss / len(val_loader.dataset)
                val_loss_str = f"{avg_val_loss:.6f}"

                # --- Логика ранней остановки ---
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # logger.debug(f"Новая лучшая валидационная ошибка: {best_val_loss:.6f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f'Ранняя остановка на эпохе {epoch+1}. Лучшая Val Loss: {best_val_loss:.6f}')
                        break
                self.train() # Вернуться в режим обучения для следующей эпохи

            # Логирование прогресса
            if epoch % 10 == 0 or epoch == epochs - 1 or (val_loader and patience_counter >= patience):
                logger.info(f'Эпоха [{epoch+1:2d}/{epochs:2d}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss_str}')

        logger.info("Обучение LSTM завершено.")

    def predict(self, X_numpy, device=None):
        """
        Делает предсказание на основе numpy массива.

        Args:
            X_numpy (np.ndarray): Входные данные формы (N, seq_len, input_size).
            device (torch.device, optional): Устройство для вычислений. 
                                             Если None, используется устройство модели. Defaults to None.

        Returns:
            np.ndarray: Предсказания формы (N,).
        """
        if device is None:
            device = next(self.parameters()).device # Получаем устройство модели
            
        self.eval() # Переводим модель в режим оценки
        with torch.no_grad():
            X_tensor = torch.tensor(X_numpy, dtype=torch.float32).to(device)
            outputs = self(X_tensor)
            # Убедимся, что возвращаем numpy массив на CPU
            return outputs.cpu().numpy()