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