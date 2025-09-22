import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Конфигурация
BOT_TOKEN = "8306161575:AAGJCkXDudwiLNGwu64l0xwYxBN5XbhUGiE"
ADMIN_IDS = [769783124]

# Популярные символы
POPULAR_SYMBOLS = [
    'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'ADA-USDT',
    'SOL-USDT', 'XRP-USDT', 'DOT-USDT', 'AVAX-USDT'
]

# Интервалы
INTERVALS = {
    '15m': '15 минут',
    '1H': '1 час', 
    '4H': '4 часа',
    '1D': '1 день'
}