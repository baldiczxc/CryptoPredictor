import asyncio
from config import BOT_TOKEN, ADMIN_IDS, logger
from bot.bot_core import CryptoBot
from utils.managers import NotificationManager, AnalyticsManager

async def send_admin_stats(bot: CryptoBot, admin_id: int, analytics: AnalyticsManager):
    """Отправка статистики администратору"""
    try:
        stats = await analytics.get_bot_statistics()
        
        stats_text = (
            f"📊 <b>Статистика бота</b>\n\n"
            f"👥 Всего пользователей: {stats['total_users']}\n"
            f"🔮 Всего прогнозов: {stats['total_predictions']}\n\n"
            f"⭐ <b>Популярные символы:</b>\n"
        )
        
        for symbol, count in stats['popular_symbols']:
            stats_text += f"• {symbol}: {count} пользователей\n"
        
        await bot.bot.send_message(admin_id, stats_text, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Ошибка отправки статистики: {e}")

async def main():
    """Основная функция"""
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        logger.error("Необходимо указать токен бота в переменной BOT_TOKEN")
        return
    
    crypto_bot = CryptoBot(BOT_TOKEN)
    notification_manager = NotificationManager(crypto_bot.bot, crypto_bot.db)
    analytics_manager = AnalyticsManager(crypto_bot.db)
    
    tasks = [
        asyncio.create_task(crypto_bot.run()),
    ]
    
    async def admin_stats_task():
        while True:
            try:
                await asyncio.sleep(86400)
                for admin_id in ADMIN_IDS:
                    await send_admin_stats(crypto_bot, admin_id, analytics_manager)
            except Exception as e:
                logger.error(f"Ошибка в admin_stats_task: {e}")
    
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Получен сигнал остановки")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
    finally:
        notification_manager.stop_notifications()
        for task in tasks:
            task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"Ошибка запуска: {e}")