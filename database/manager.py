import os
import json
from datetime import datetime
import aiofiles

from config import logger

class DatabaseManager:
    """Менеджер данных пользователей"""
    
    def __init__(self, filename="users.json"):
        self.filename = filename
        self.users_data = {}
        
    async def load_data(self):
        """Загрузка данных пользователей"""
        try:
            if os.path.exists(self.filename):
                async with aiofiles.open(self.filename, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    self.users_data = json.loads(content) if content else {}
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            self.users_data = {}
    
    async def save_data(self):
        """Сохранение данных пользователей"""
        try:
            async with aiofiles.open(self.filename, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self.users_data, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.error(f"Ошибка сохранения данных: {e}")
    
    async def get_user(self, user_id: int) -> dict:
        """Получение данных пользователя"""
        user_id_str = str(user_id)
        if user_id_str not in self.users_data:
            self.users_data[user_id_str] = {
                'id': user_id,
                'username': '',
                'first_name': '',
                'joined_at': datetime.now().isoformat(),
                'predictions_count': 0,
                'favorite_symbols': ['BTC-USDT'],
                'settings': {
                    'interval': '1H',
                    'notifications': True
                }
            }
            await self.save_data()
        
        return self.users_data[user_id_str]
    
    async def update_user(self, user_id: int, data: dict):
        """Обновление данных пользователя"""
        user_id_str = str(user_id)
        if user_id_str in self.users_data:
            self.users_data[user_id_str].update(data)
            await self.save_data()
    
    async def increment_predictions(self, user_id: int):
        """Увеличение счетчика предсказаний"""
        user = await self.get_user(user_id)
        user['predictions_count'] += 1
        await self.update_user(user_id, user)