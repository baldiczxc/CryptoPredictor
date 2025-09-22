import asyncio
from config import logger

class NotificationManager:
    """Менеджер уведомлений"""
    
    def __init__(self, bot, db):
        self.bot = bot
        self.db = db
        self.running = False
    
    async def start_notifications(self):
        """Запуск системы уведомлений"""
        self.running = True
        logger.info("Система уведомлений запущена")
        
        while self.running:
            try:
                await self.check_price_alerts()
                await asyncio.sleep(300)
            except Exception as e:
                logger.error(f"Ошибка в системе уведомлений: {e}")
                await asyncio.sleep(60)
    
    async def check_price_alerts(self):
        """Проверка ценовых алертов"""
        pass
    
    def stop_notifications(self):
        """Остановка уведомлений"""
        self.running = False
        logger.info("Система уведомлений остановлена")

class AnalyticsManager:
    """Менеджер аналитики и статистики"""
    
    def __init__(self, db):
        self.db = db
    
    async def get_bot_statistics(self) -> dict:
        """Получение статистики бота"""
        users_data = self.db.users_data
        
        total_users = len(users_data)
        total_predictions = sum(user.get('predictions_count', 0) for user in users_data.values())
        
        # Популярные символы
        symbol_counts = {}
        for user in users_data.values():
            for symbol in user.get('favorite_symbols', []):
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        popular_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_users': total_users,
            'total_predictions': total_predictions,
            'popular_symbols': popular_symbols
        }