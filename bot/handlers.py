from aiogram import Router, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext

from config import POPULAR_SYMBOLS, INTERVALS, logger
from bot.states import UserStates
from utils.formatters import format_prediction_result

class Handlers:
    def __init__(self, crypto_bot):
        self.crypto_bot = crypto_bot
        self.router = Router()
        self.setup_handlers()
    
    def setup_handlers(self):
        """Настройка обработчиков"""
        self.router.message(CommandStart())(self.cmd_start)
        self.router.message(Command("help"))(self.cmd_help)
        self.router.message(Command("predict"))(self.cmd_predict)
        self.router.message(Command("symbols"))(self.cmd_symbols)
        self.router.message(Command("stats"))(self.cmd_stats)
        self.router.message(Command("settings"))(self.cmd_settings)
        
        self.router.callback_query(F.data.startswith("predict_"))(self.process_predict_callback)
        self.router.callback_query(F.data.startswith("symbol_"))(self.process_symbol_callback)
        self.router.callback_query(F.data.startswith("interval_"))(self.process_interval_callback)
        self.router.callback_query(F.data == "back_main")(self.back_to_main)
        
        self.router.message(UserStates.waiting_symbol)(self.process_custom_symbol)
    
    async def cmd_start(self, message: Message, state: FSMContext):
        """Команда /start"""
        user = await self.crypto_bot.db.get_user(message.from_user.id)
        
        await self.crypto_bot.db.update_user(message.from_user.id, {
            'username': message.from_user.username or '',
            'first_name': message.from_user.first_name or ''
        })
        
        welcome_text = (
            f"🚀 <b>Добро пожаловать в CryptoPredictionBot!</b>\n\n"
            f"Привет, {message.from_user.first_name}! 👋\n\n"
            f"Я использую продвинутые алгоритмы машинного обучения для "
            f"прогнозирования цен криптовалют.\n\n"
            f"<b>Что я умею:</b>\n"
            f"📈 Прогнозировать цены криптовалют\n"
            f"📊 Анализировать технические индикаторы\n"
            f"💡 Давать торговые сигналы\n"
            f"📱 Отправлять уведомления\n\n"
            f"Используйте /help для получения списка команд!"
        )
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="📈 Получить прогноз", callback_data="predict_main")],
            [InlineKeyboardButton(text="📋 Популярные символы", callback_data="symbols_popular")],
            [InlineKeyboardButton(text="⚙️ Настройки", callback_data="settings_main")]
        ])
        
        await message.answer(welcome_text, reply_markup=keyboard, parse_mode='HTML')
    
    async def cmd_help(self, message: Message):
        """Команда /help"""
        help_text = (
            "<b>🤖 Список команд CryptoPredictionBot</b>\n\n"
            "<b>Основные команды:</b>\n"
            "/start - Запуск бота\n"
            "/predict - Получить прогноз\n"
            "/symbols - Список популярных символов\n"
            "/stats - Ваша статистика\n"
            "/settings - Настройки\n"
            "/help - Эта справка\n\n"
            "<b>🎯 Как пользоваться:</b>\n"
            "1️⃣ Выберите криптовалютную пару\n"
            "2️⃣ Укажите временной интервал\n"
            "3️⃣ Получите детальный прогноз\n\n"
            "<b>📊 Поддерживаемые интервалы:</b>\n"
            "• 15m - 15 минут\n"
            "• 1H - 1 час\n"
            "• 4H - 4 часа\n"
            "• 1D - 1 день\n\n"
            "<b>💡 Торговые сигналы:</b>\n"
            "🟢 Покупка - прогнозируется рост\n"
            "🔴 Продажа - прогнозируется падение\n"
            "🟡 Ожидание - неопределенный сигнал"
        )
        
        await message.answer(help_text, parse_mode='HTML')
    
    async def cmd_predict(self, message: Message):
        """Команда /predict"""
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="⭐ Популярные", callback_data="predict_popular"),
             InlineKeyboardButton(text="🔍 Поиск символа", callback_data="predict_search")],
            [InlineKeyboardButton(text="📊 BTC-USDT", callback_data="predict_BTC-USDT"),
             InlineKeyboardButton(text="💎 ETH-USDT", callback_data="predict_ETH-USDT")]
        ])
        
        await message.answer(
            "📈 <b>Выберите криптовалютную пару для прогноза:</b>",
            reply_markup=keyboard,
            parse_mode='HTML'
        )
    
    async def cmd_symbols(self, message: Message):
        """Команда /symbols"""
        text = "📋 <b>Популярные криптовалютные пары:</b>\n\n"
        
        keyboard_buttons = []
        for i, symbol in enumerate(POPULAR_SYMBOLS):
            if i % 2 == 0:
                keyboard_buttons.append([])
            keyboard_buttons[-1].append(
                InlineKeyboardButton(text=symbol, callback_data=f"predict_{symbol}")
            )
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
        
        for symbol in POPULAR_SYMBOLS:
            text += f"• {symbol}\n"
        
        text += "\nНажмите на символ для получения прогноза!"
        
        await message.answer(text, reply_markup=keyboard, parse_mode='HTML')
    
    async def cmd_stats(self, message: Message):
        """Команда /stats"""
        user = await self.crypto_bot.db.get_user(message.from_user.id)
        
        from datetime import datetime
        joined_date = datetime.fromisoformat(user['joined_at']).strftime('%d.%m.%Y')
        
        stats_text = (
            f"📊 <b>Ваша статистика</b>\n\n"
            f"👤 <b>Пользователь:</b> {user['first_name']}\n"
            f"📅 <b>Дата регистрации:</b> {joined_date}\n"
            f"🔮 <b>Прогнозов получено:</b> {user['predictions_count']}\n"
            f"⭐ <b>Избранные символы:</b> {', '.join(user['favorite_symbols'])}\n"
            f"⏰ <b>Интервал по умолчанию:</b> {INTERVALS.get(user['settings']['interval'], user['settings']['interval'])}\n"
            f"🔔 <b>Уведомления:</b> {'Включены' if user['settings']['notifications'] else 'Выключены'}"
        )
        
        await message.answer(stats_text, parse_mode='HTML')
    
    async def cmd_settings(self, message: Message):
        """Команда /settings"""
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="⏰ Интервал по умолчанию", callback_data="settings_interval")],
            [InlineKeyboardButton(text="⭐ Избранные символы", callback_data="settings_favorites")],
            [InlineKeyboardButton(text="🔔 Уведомления", callback_data="settings_notifications")],
            [InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_main")]
        ])
        
        await message.answer(
            "⚙️ <b>Настройки</b>\n\nВыберите что хотите настроить:",
            reply_markup=keyboard,
            parse_mode='HTML'
        )
    
    async def process_predict_callback(self, callback: CallbackQuery, state: FSMContext):
        """Обработка callback для прогнозов"""
        await callback.answer()
        
        if callback.data == "predict_main":
            await self.cmd_predict(callback.message)
        elif callback.data == "predict_popular":
            await self.show_popular_symbols(callback.message)
        elif callback.data == "predict_search":
            await callback.message.answer(
                "🔍 Введите символ криптовалютной пары (например: BTC-USDT):",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="❌ Отмена", callback_data="back_main")
                ]])
            )
            await state.set_state(UserStates.waiting_symbol)
        elif callback.data.startswith("predict_") and callback.data != "predict_main":
            symbol = callback.data.replace("predict_", "")
            await self.show_interval_selection(callback.message, symbol)
    
    async def show_popular_symbols(self, message: Message):
        """Показать популярные символы"""
        keyboard_buttons = []
        for i, symbol in enumerate(POPULAR_SYMBOLS):
            if i % 2 == 0:
                keyboard_buttons.append([])
            keyboard_buttons[-1].append(
                InlineKeyboardButton(text=symbol, callback_data=f"symbol_{symbol}")
            )
        
        keyboard_buttons.append([
            InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_main")
        ])
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
        
        await message.edit_text(
            "⭐ <b>Популярные криптовалютные пары:</b>\n\n"
            "Выберите пару для прогноза:",
            reply_markup=keyboard,
            parse_mode='HTML'
        )
    
    async def process_symbol_callback(self, callback: CallbackQuery):
        """Обработка выбора символа"""
        await callback.answer()
        symbol = callback.data.replace("symbol_", "")
        await self.show_interval_selection(callback.message, symbol)
    
    async def show_interval_selection(self, message: Message, symbol: str):
        """Показать выбор интервала"""
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="15m", callback_data=f"interval_{symbol}_15m"),
             InlineKeyboardButton(text="1H", callback_data=f"interval_{symbol}_1H")],
            [InlineKeyboardButton(text="4H", callback_data=f"interval_{symbol}_4H"),
             InlineKeyboardButton(text="1D", callback_data=f"interval_{symbol}_1D")],
            [InlineKeyboardButton(text="🔙 Назад", callback_data="predict_popular")]
        ])
        
        await message.edit_text(
            f"⏰ <b>Выберите временной интервал для {symbol}:</b>",
            reply_markup=keyboard,
            parse_mode='HTML'
        )
    
    async def process_interval_callback(self, callback: CallbackQuery):
        """Обработка выбора интервала и запуск прогноза"""
        await callback.answer()
        
        parts = callback.data.replace("interval_", "").split("_")
        symbol = parts[0]
        interval = parts[1]
        
        loading_message = await callback.message.edit_text(
            f"🔄 <b>Анализирую {symbol} на интервале {interval}...</b>\n\n"
            f"⏳ Загружаю данные и обучаю модель...\n"
            f"📊 Рассчитываю технические индикаторы...\n"
            f"🤖 Генерирую прогноз...\n\n"
            f"<i>Это может занять до 30 секунд</i>",
            parse_mode='HTML'
        )
        
        try:
            predictor_key = f"{symbol}_{interval}"
            if predictor_key not in self.crypto_bot.predictors:
                from ml.predictor import AdvancedCryptoPredictor
                self.crypto_bot.predictors[predictor_key] = AdvancedCryptoPredictor(
                    symbol=symbol,
                    interval=interval,
                    sequence_length=30
                )
            
            predictor = self.crypto_bot.predictors[predictor_key]
            result = await predictor.predict_simple()
            
            if result:
                await self.crypto_bot.db.increment_predictions(callback.from_user.id)
                prediction_text = format_prediction_result(result)
                
                keyboard = InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="🔄 Обновить прогноз", callback_data=f"interval_{symbol}_{interval}")],
                    [InlineKeyboardButton(text="📊 Другой символ", callback_data="predict_popular"),
                     InlineKeyboardButton(text="⏰ Другой интервал", callback_data=f"symbol_{symbol}")],
                    [InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_main")]
                ])
                
                await loading_message.edit_text(
                    prediction_text,
                    reply_markup=keyboard,
                    parse_mode='HTML'
                )
            else:
                await loading_message.edit_text(
                    f"❌ <b>Ошибка получения прогноза для {symbol}</b>\n\n"
                    f"Возможные причины:\n"
                    f"• Недостаточно данных для анализа\n"
                    f"• Проблемы с API биржи\n"
                    f"• Технические неполадки\n\n"
                    f"Попробуйте еще раз через несколько минут.",
                    reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                        [InlineKeyboardButton(text="🔄 Попробовать снова", callback_data=f"interval_{symbol}_{interval}")],
                        [InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_main")]
                    ]),
                    parse_mode='HTML'
                )
                
        except Exception as e:
            logger.error(f"Ошибка в process_interval_callback: {e}")
            await loading_message.edit_text(
                f"❌ <b>Произошла ошибка при анализе</b>\n\n"
                f"Попробуйте выбрать другой символ или интервал.",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_main")]
                ]),
                parse_mode='HTML'
            )
    
    async def back_to_main(self, callback: CallbackQuery):
        """Возврат в главное меню"""
        await callback.answer()
        
        welcome_text = (
            f"🚀 <b>CryptoPredictionBot</b>\n\n"
            f"Выберите действие:"
        )
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="📈 Получить прогноз", callback_data="predict_main")],
            [InlineKeyboardButton(text="📋 Популярные символы", callback_data="symbols_popular")],
            [InlineKeyboardButton(text="⚙️ Настройки", callback_data="settings_main")]
        ])
        
        await callback.message.edit_text(
            welcome_text,
            reply_markup=keyboard,
            parse_mode='HTML'
        )
    
    async def process_custom_symbol(self, message: Message, state: FSMContext):
        """Обработка пользовательского символа"""
        symbol = message.text.upper().strip()
        
        if not symbol or '-' not in symbol:
            await message.answer(
                "❌ Неверный формат символа!\n\n"
                "Пример правильного формата: BTC-USDT, ETH-USDT\n"
                "Попробуйте еще раз:",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="❌ Отмена", callback_data="back_main")
                ]])
            )
            return
        
        await state.clear()
        await self.show_interval_selection(message, symbol)