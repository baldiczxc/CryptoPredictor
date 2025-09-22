from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand, BotCommandScopeDefault
from aiogram.fsm.storage.memory import MemoryStorage

from config import logger
from database.manager import DatabaseManager
from bot.handlers import Handlers

class CryptoBot:
    """Основной класс бота"""
    
    def __init__(self, token: str):
        self.bot = Bot(token=token)
        self.dp = Dispatcher(storage=MemoryStorage())
        self.db = DatabaseManager()
        self.predictors = {}
        
        # Инициализация обработчиков
        self.handlers = Handlers(self)
        self.dp.include_router(self.handlers.router)
    
    async def setup_bot_commands(self):
        """Настройка команд бота"""
        commands = [
            BotCommand(command="start", description="🚀 Запуск бота"),
            BotCommand(command="predict", description="📈 Получить прогноз"),
            BotCommand(command="symbols", description="📋 Популярные символы"),
            BotCommand(command="stats", description="📊 Статистика"),
            BotCommand(command="settings", description="⚙️ Настройки"),
            BotCommand(command="help", description="❓ Справка")
        ]
        
        await self.bot.set_my_commands(commands, BotCommandScopeDefault())
    
    async def startup(self):
        """Запуск бота"""
        await self.db.load_data()
        await self.setup_bot_commands()
        logger.info("Бот запущен успешно!")
    
    async def shutdown(self):
        """Остановка бота"""
        await self.db.save_data()
        await self.bot.session.close()
        logger.info("Бот остановлен")
    
    async def run(self):
        """Основной метод запуска"""
        try:
            await self.startup()
            await self.dp.start_polling(self.bot)
        except Exception as e:
            logger.error(f"Ошибка при запуске бота: {e}")
        finally:
            await self.shutdown()
    
    async def get_user_stats(self, user_id: int) -> dict:
        """Получение статистики пользователя"""
        user_data = await self.db.get_user_data(user_id)
        return {
            'requests_count': user_data.get('requests_count', 0),
            'last_prediction': user_data.get('last_prediction'),
            'favorite_symbols': user_data.get('favorite_symbols', []),
            'registration_date': user_data.get('registration_date')
        }
    
    async def update_user_stats(self, user_id: int, **kwargs):
        """Обновление статистики пользователя"""
        await self.db.update_user_data(user_id, **kwargs)
    
    async def get_popular_symbols(self, limit: int = 10) -> list:
        """Получение списка популярных символов"""
        return await self.db.get_popular_symbols(limit)
    
    async def add_prediction_request(self, user_id: int, symbol: str):
        """Добавление запроса прогноза"""
        # Обновляем статистику пользователя
        user_stats = await self.get_user_stats(user_id)
        requests_count = user_stats['requests_count'] + 1
        
        # Обновляем любимые символы
        favorite_symbols = user_stats['favorite_symbols']
        if symbol in favorite_symbols:
            favorite_symbols.remove(symbol)
        favorite_symbols.insert(0, symbol)
        if len(favorite_symbols) > 5:
            favorite_symbols = favorite_symbols[:5]
        
        await self.update_user_stats(
            user_id,
            requests_count=requests_count,
            last_prediction=symbol,
            favorite_symbols=favorite_symbols
        )
        
        # Обновляем статистику по символам
        await self.db.increment_symbol_usage(symbol)
    
    async def get_bot_statistics(self) -> dict:
        """Получение общей статистики бота"""
        total_users = await self.db.get_total_users_count()
        total_predictions = await self.db.get_total_predictions_count()
        popular_symbols = await self.get_popular_symbols(5)
        
        return {
            'total_users': total_users,
            'total_predictions': total_predictions,
            'popular_symbols': popular_symbols
        }
    
    async def send_notification_to_user(self, user_id: int, message: str):
        """Отправка уведомления пользователю"""
        try:
            await self.bot.send_message(user_id, message)
        except Exception as e:
            logger.error(f"Ошибка отправки уведомления пользователю {user_id}: {e}")
    
    async def broadcast_message(self, message: str, user_ids: list = None):
        """Массовая рассылка сообщений"""
        if user_ids is None:
            user_ids = await self.db.get_all_user_ids()
        
        success_count = 0
        for user_id in user_ids:
            try:
                await self.send_notification_to_user(user_id, message)
                success_count += 1
            except Exception:
                continue
        
        logger.info(f"Массовая рассылка завершена. Успешно: {success_count}/{len(user_ids)}")
        return success_count