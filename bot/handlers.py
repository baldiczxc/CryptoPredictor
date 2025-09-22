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
        
        self.router.callback_query()(self.process_all_callbacks)
        
        self.router.message(UserStates.waiting_symbol)(self.process_custom_symbol)
        self.router.message(UserStates.waiting_support_request)(self.process_support_request)
        self.router.message(UserStates.waiting_feedback)(self.process_feedback)
    
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
             InlineKeyboardButton(text="💎 ETH-USDT", callback_data="predict_ETH-USDT")],
            [InlineKeyboardButton(text="⭐ Избранные", callback_data="predict_favorites")]
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
        
        keyboard_buttons.append([
            InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_main")
        ])
        
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
            f"⭐ <b>Избранные символы:</b> {', '.join(user['favorite_symbols'][:3])}\n"
            f"⏰ <b>Интервал по умолчанию:</b> {INTERVALS.get(user['settings']['interval'], user['settings']['interval'])}\n"
            f"🔔 <b>Уведомления:</b> {'Включены' if user['settings']['notifications'] else 'Выключены'}"
        )
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="📈 Подробная аналитика", callback_data="stats_detailed")],
            [InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_main")]
        ])
        
        await message.answer(stats_text, reply_markup=keyboard, parse_mode='HTML')
    
    async def cmd_settings(self, message: Message):
        """Команда /settings"""
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="⏰ Интервал по умолчанию", callback_data="settings_interval")],
            [InlineKeyboardButton(text="⭐ Избранные символы", callback_data="settings_favorites")],
            [InlineKeyboardButton(text="🔔 Уведомления", callback_data="settings_notifications")],
            [InlineKeyboardButton(text="👤 Профиль", callback_data="settings_profile")],
            [InlineKeyboardButton(text="🆘 Поддержка", callback_data="settings_support")],
            [InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_main")]
        ])
        
        await message.answer(
            "⚙️ <b>Настройки</b>\n\nВыберите что хотите настроить:",
            reply_markup=keyboard,
            parse_mode='HTML'
        )
    
    async def process_all_callbacks(self, callback: CallbackQuery, state: FSMContext):
        """Универсальный обработчик всех callback-запросов"""
        try:
            # Обработка прогнозов
            if callback.data.startswith("predict_"):
                await self.process_predict_callback(callback, state)
            # Обработка символов
            elif callback.data.startswith("symbol_"):
                await self.process_symbol_callback(callback)
            # Обработка интервалов
            elif callback.data.startswith("interval_"):
                await self.process_interval_callback(callback)
            # Обработка настроек
            elif callback.data.startswith("settings_"):
                await self.process_settings_callback(callback, state)
            # Обработка статистики
            elif callback.data.startswith("stats_"):
                await self.process_stats_callback(callback)
            # Обработка избранного
            elif callback.data.startswith("favorite_"):
                await self.process_favorite_callback(callback)
            # Обработка уведомлений
            elif callback.data.startswith("notification_"):
                await self.process_notification_callback(callback)
            # Навигация
            elif callback.data == "back_main":
                await self.back_to_main(callback)
            elif callback.data == "back_predict":
                await self.back_to_predict(callback)
            elif callback.data == "back_settings":
                await self.back_to_settings(callback)
            elif callback.data == "back_stats":
                await self.back_to_stats(callback)
            else:
                # Обработка неизвестных callback-ов
                await callback.answer("❌ Неизвестная команда", show_alert=True)
                logger.warning(f"Неизвестный callback: {callback.data}")
                
        except Exception as e:
            logger.error(f"Ошибка в process_all_callbacks: {e}")
            if callback.message:
                await callback.message.edit_text(
                    "❌ Произошла ошибка. Попробуйте еще раз.",
                    reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                        InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_main")
                    ]])
                )
    
    async def process_predict_callback(self, callback: CallbackQuery, state: FSMContext):
        """Обработка callback для прогнозов"""
        await callback.answer()
        
        if callback.data == "predict_main":
            await self.show_predict_menu(callback)
        elif callback.data == "predict_popular":
            await self.show_popular_symbols(callback)
        elif callback.data == "predict_search":
            await callback.message.edit_text(
                "🔍 Введите символ криптовалютной пары (например: BTC-USDT):",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="❌ Отмена", callback_data="back_predict")
                ]])
            )
            await state.set_state(UserStates.waiting_symbol)
        elif callback.data == "predict_favorites":
            await self.show_user_favorites(callback, callback.from_user.id)
        elif callback.data.startswith("predict_") and callback.data != "predict_main":
            symbol = callback.data.replace("predict_", "")
            await self.show_interval_selection(callback, symbol)
    
    async def show_predict_menu(self, callback: CallbackQuery):
        """Показать меню прогнозов"""
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="⭐ Популярные", callback_data="predict_popular"),
             InlineKeyboardButton(text="🔍 Поиск символа", callback_data="predict_search")],
            [InlineKeyboardButton(text="📊 BTC-USDT", callback_data="predict_BTC-USDT"),
             InlineKeyboardButton(text="💎 ETH-USDT", callback_data="predict_ETH-USDT")],
            [InlineKeyboardButton(text="⭐ Избранные", callback_data="predict_favorites")],
            [InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_main")]
        ])
        
        await callback.message.edit_text(
            "📈 <b>Выберите криптовалютную пару для прогноза:</b>",
            reply_markup=keyboard,
            parse_mode='HTML'
        )
    
    async def show_popular_symbols(self, callback: CallbackQuery):
        """Показать популярные символы"""
        keyboard_buttons = []
        for i, symbol in enumerate(POPULAR_SYMBOLS):
            if i % 2 == 0:
                keyboard_buttons.append([])
            keyboard_buttons[-1].append(
                InlineKeyboardButton(text=symbol, callback_data=f"symbol_{symbol}")
            )
        
        keyboard_buttons.append([
            InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_predict")
        ])
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
        
        await callback.message.edit_text(
            "⭐ <b>Популярные криптовалютные пары:</b>\n\n"
            "Выберите пару для прогноза:",
            reply_markup=keyboard,
            parse_mode='HTML'
        )
    
    async def process_symbol_callback(self, callback: CallbackQuery):
        """Обработка выбора символа"""
        await callback.answer()
        symbol = callback.data.replace("symbol_", "")
        await self.show_interval_selection(callback, symbol)
    
    async def show_interval_selection(self, callback: CallbackQuery, symbol: str):
        """Показать выбор интервала"""
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="15m", callback_data=f"interval_{symbol}_15m"),
             InlineKeyboardButton(text="1H", callback_data=f"interval_{symbol}_1H")],
            [InlineKeyboardButton(text="4H", callback_data=f"interval_{symbol}_4H"),
             InlineKeyboardButton(text="1D", callback_data=f"interval_{symbol}_1D")],
            [InlineKeyboardButton(text="⭐ В избранное", callback_data=f"favorite_add_{symbol}")],
            [InlineKeyboardButton(text="🔙 Назад", callback_data="predict_popular")]
        ])
        
        await callback.message.edit_text(
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
        
        await callback.message.edit_text(
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
                    [InlineKeyboardButton(text="⭐ В избранное", callback_data=f"favorite_add_{symbol}")],
                    [InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_main")]
                ])
                
                await callback.message.edit_text(
                    prediction_text,
                    reply_markup=keyboard,
                    parse_mode='HTML'
                )
            else:
                await callback.message.edit_text(
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
            await callback.message.edit_text(
                f"❌ <b>Произошла ошибка при анализе</b>\n\n"
                f"Попробуйте выбрать другой символ или интервал.",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_main")]
                ]),
                parse_mode='HTML'
            )
    
    async def process_settings_callback(self, callback: CallbackQuery, state: FSMContext):
        """Обработка callback для настроек"""
        await callback.answer()
        
        if callback.data == "settings_main":
            await self.show_settings_menu(callback)
        elif callback.data == "settings_interval":
            await self.show_interval_settings(callback, callback.from_user.id)
        elif callback.data.startswith("settings_set_interval_"):
            interval = callback.data.replace("settings_set_interval_", "")
            await self.update_user_interval(callback.from_user.id, interval)
            await callback.answer("✅ Интервал обновлен!")
            await self.show_settings_menu(callback)
        elif callback.data == "settings_notifications":
            await self.show_notification_settings(callback, callback.from_user.id)
        elif callback.data == "settings_notifications_toggle":
            new_state = await self.toggle_user_notifications(callback.from_user.id)
            status = "включены" if new_state else "выключены"
            await callback.answer(f"✅ Уведомления {status}!")
            await self.show_notification_settings(callback, callback.from_user.id)
        elif callback.data == "settings_favorites":
            await self.show_user_favorites(callback, callback.from_user.id)
        elif callback.data == "settings_profile":
            await self.show_profile_settings(callback, callback.from_user.id)
        elif callback.data == "settings_support":
            await self.show_support_menu(callback)
        elif callback.data == "settings_support_request":
            await callback.message.edit_text(
                "📝 Опишите вашу проблему или предложение:",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="❌ Отмена", callback_data="settings_main")
                ]])
            )
            await state.set_state(UserStates.waiting_support_request)
        elif callback.data == "settings_feedback":
            await callback.message.edit_text(
                "⭐ Оставьте ваш отзыв о работе бота:",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="❌ Отмена", callback_data="settings_main")
                ]])
            )
            await state.set_state(UserStates.waiting_feedback)
    
    async def process_stats_callback(self, callback: CallbackQuery):
        """Обработка callback для статистики"""
        await callback.answer()
        
        if callback.data == "stats_detailed":
            await self.show_detailed_stats(callback, callback.from_user.id)
        elif callback.data == "stats_main":
            await self.show_stats_menu(callback)
    
    async def process_favorite_callback(self, callback: CallbackQuery):
        """Обработка callback для избранного"""
        await callback.answer()
        
        if callback.data.startswith("favorite_add_"):
            symbol = callback.data.replace("favorite_add_", "")
            await self.add_to_favorites(callback.from_user.id, symbol)
            await callback.answer(f"✅ {symbol} добавлен в избранное!", show_alert=True)
        elif callback.data.startswith("favorite_remove_"):
            symbol = callback.data.replace("favorite_remove_", "")
            await self.remove_from_favorites(callback.from_user.id, symbol)
            await callback.answer(f"✅ {symbol} удален из избранного!", show_alert=True)
            # Обновляем список избранных
            await self.show_user_favorites(callback, callback.from_user.id)
    
    async def process_notification_callback(self, callback: CallbackQuery):
        """Обработка callback для уведомлений"""
        await callback.answer()
        # Пока пусто, можно расширить позже
    
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
            [InlineKeyboardButton(text="📊 Статистика", callback_data="stats_main")],
            [InlineKeyboardButton(text="⚙️ Настройки", callback_data="settings_main")]
        ])
        
        await callback.message.edit_text(
            welcome_text,
            reply_markup=keyboard,
            parse_mode='HTML'
        )
    
    async def back_to_predict(self, callback: CallbackQuery):
        """Возврат в меню прогнозов"""
        await callback.answer()
        await self.show_predict_menu(callback)
    
    async def back_to_settings(self, callback: CallbackQuery):
        """Возврат в меню настроек"""
        await callback.answer()
        await self.show_settings_menu(callback)
    
    async def back_to_stats(self, callback: CallbackQuery):
        """Возврат в меню статистики"""
        await callback.answer()
        await self.show_stats_menu(callback)
    
    async def show_settings_menu(self, callback: CallbackQuery):
        """Показать меню настроек"""
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="⏰ Интервал по умолчанию", callback_data="settings_interval")],
            [InlineKeyboardButton(text="⭐ Избранные символы", callback_data="settings_favorites")],
            [InlineKeyboardButton(text="🔔 Уведомления", callback_data="settings_notifications")],
            [InlineKeyboardButton(text="👤 Профиль", callback_data="settings_profile")],
            [InlineKeyboardButton(text="🆘 Поддержка", callback_data="settings_support")],
            [InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_main")]
        ])
        
        await callback.message.edit_text(
            "⚙️ <b>Настройки</b>\n\nВыберите что хотите настроить:",
            reply_markup=keyboard,
            parse_mode='HTML'
        )
    
    async def show_stats_menu(self, callback: CallbackQuery):
        """Показать меню статистики"""
        user = await self.crypto_bot.db.get_user(callback.from_user.id)
        
        from datetime import datetime
        joined_date = datetime.fromisoformat(user['joined_at']).strftime('%d.%m.%Y')
        
        stats_text = (
            f"📊 <b>Ваша статистика</b>\n\n"
            f"👤 <b>Пользователь:</b> {user['first_name']}\n"
            f"📅 <b>Дата регистрации:</b> {joined_date}\n"
            f"🔮 <b>Прогнозов получено:</b> {user['predictions_count']}\n"
            f"⭐ <b>Избранные символы:</b> {', '.join(user['favorite_symbols'][:3])}\n"
            f"⏰ <b>Интервал по умолчанию:</b> {INTERVALS.get(user['settings']['interval'], user['settings']['interval'])}\n"
            f"🔔 <b>Уведомления:</b> {'Включены' if user['settings']['notifications'] else 'Выключены'}"
        )
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="📈 Подробная аналитика", callback_data="stats_detailed")],
            [InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_main")]
        ])
        
        await callback.message.edit_text(stats_text, reply_markup=keyboard, parse_mode='HTML')
    
    async def process_custom_symbol(self, message: Message, state: FSMContext):
        """Обработка пользовательского символа"""
        symbol = message.text.upper().strip()
        
        if not symbol or '-' not in symbol:
            await message.reply(
                "❌ Неверный формат символа!\n\n"
                "Пример правильного формата: BTC-USDT, ETH-USDT\n"
                "Попробуйте еще раз:",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="❌ Отмена", callback_data="back_predict")
                ]])
            )
            return
        
        await state.clear()
        
        # Создаем callback-подобный объект для передачи в метод
        class MockCallback:
            def __init__(self, message):
                self.message = message
                self.from_user = message.from_user
                self.answer = lambda: None
        
        mock_callback = MockCallback(message)
        await self.show_interval_selection(mock_callback, symbol)
    
    async def process_support_request(self, message: Message, state: FSMContext):
        """Обработка запроса в поддержку"""
        await state.clear()
        # Здесь можно добавить логику отправки запроса администратору
        
        # Создаем callback-подобный объект для возврата в меню
        class MockCallback:
            def __init__(self, message):
                self.message = message
                self.from_user = message.from_user
                self.answer = lambda: None
        
        mock_callback = MockCallback(message)
        await message.reply("✅ Ваш запрос отправлен в службу поддержки. Мы свяжемся с вами в ближайшее время!")
        await self.back_to_main(mock_callback)
    
    async def process_feedback(self, message: Message, state: FSMContext):
        """Обработка отзыва"""
        await state.clear()
        # Здесь можно добавить логику сохранения отзыва
        
        # Создаем callback-подобный объект для возврата в меню
        class MockCallback:
            def __init__(self, message):
                self.message = message
                self.from_user = message.from_user
                self.answer = lambda: None
        
        mock_callback = MockCallback(message)
        await message.reply("⭐ Спасибо за ваш отзыв! Он поможет нам улучшить бота!")
        await self.back_to_main(mock_callback)
    
    # Дополнительные методы
    async def get_user_favorites(self, user_id: int) -> list:
        """Получение избранных символов пользователя"""
        user = await self.crypto_bot.db.get_user(user_id)
        return user.get('favorite_symbols', [])
    
    async def add_to_favorites(self, user_id: int, symbol: str):
        """Добавление символа в избранное"""
        favorites = await self.get_user_favorites(user_id)
        if symbol not in favorites:
            favorites.append(symbol)
            if len(favorites) > 10:  # Ограничиваем 10 избранными
                favorites = favorites[-10:]
            await self.crypto_bot.db.update_user_settings(user_id, {'favorite_symbols': favorites})
    
    async def remove_from_favorites(self, user_id: int, symbol: str):
        """Удаление символа из избранного"""
        favorites = await self.get_user_favorites(user_id)
        if symbol in favorites:
            favorites.remove(symbol)
            await self.crypto_bot.db.update_user_settings(user_id, {'favorite_symbols': favorites})
    
    async def show_user_favorites(self, callback: CallbackQuery, user_id: int):
        """Показать избранные символы пользователя"""
        await callback.answer()
        favorites = await self.get_user_favorites(user_id)
        
        if not favorites:
            await callback.message.edit_text(
                "⭐ У вас пока нет избранных символов.\n\n"
                "Добавляйте символы в избранное, чтобы быстро к ним обращаться!",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_predict")
                ]]),
                parse_mode='HTML'
            )
            return
        
        keyboard_buttons = []
        for i, symbol in enumerate(favorites):
            if i % 2 == 0:
                keyboard_buttons.append([])
            keyboard_buttons[-1].append(
                InlineKeyboardButton(
                    text=f"❌ {symbol}",
                    callback_data=f"favorite_remove_{symbol}"
                )
            )
        
        keyboard_buttons.append([
            InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_predict")
        ])
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
        
        await callback.message.edit_text(
            "⭐ <b>Ваши избранные символы:</b>\n\n"
            "Нажмите на символ с ❌ для удаления:",
            reply_markup=keyboard,
            parse_mode='HTML'
        )
    
    async def update_user_interval(self, user_id: int, interval: str):
        """Обновление интервала по умолчанию для пользователя"""
        await self.crypto_bot.db.update_user_settings(user_id, {'interval': interval})
    
    async def toggle_user_notifications(self, user_id: int):
        """Переключение уведомлений для пользователя"""
        user = await self.crypto_bot.db.get_user(user_id)
        current_state = user['settings'].get('notifications', True)
        new_state = not current_state
        await self.crypto_bot.db.update_user_settings(user_id, {'notifications': new_state})
        return new_state
    
    async def show_interval_settings(self, callback: CallbackQuery, user_id: int):
        """Показать настройки интервала"""
        await callback.answer()
        user = await self.crypto_bot.db.get_user(user_id)
        current_interval = user['settings'].get('interval', '1H')
        
        intervals_list = ['15m', '1H', '4H', '1D']
        keyboard_buttons = []
        
        for i, interval in enumerate(intervals_list):
            if i % 2 == 0:
                keyboard_buttons.append([])
            status = "✅ " if interval == current_interval else ""
            keyboard_buttons[-1].append(
                InlineKeyboardButton(
                    text=f"{status}{INTERVALS.get(interval, interval)}",
                    callback_data=f"settings_set_interval_{interval}"
                )
            )
        
        keyboard_buttons.append([
            InlineKeyboardButton(text="🔙 Назад", callback_data="settings_main")
        ])
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
        
        await callback.message.edit_text(
            "⏰ <b>Выберите интервал по умолчанию:</b>\n\n"
            "Этот интервал будет использоваться при быстром прогнозе",
            reply_markup=keyboard,
            parse_mode='HTML'
        )
    
    async def show_notification_settings(self, callback: CallbackQuery, user_id: int):
        """Показать настройки уведомлений"""
        await callback.answer()
        user = await self.crypto_bot.db.get_user(user_id)
        notifications_enabled = user['settings'].get('notifications', True)
        
        status_text = "ВКЛ ✅" if notifications_enabled else "ВЫКЛ ❌"
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(
                text=f"Переключить уведомления ({status_text})",
                callback_data="settings_notifications_toggle"
            )],
            [InlineKeyboardButton(text="🔙 Назад", callback_data="settings_main")]
        ])
        
        await callback.message.edit_text(
            "🔔 <b>Настройки уведомлений</b>\n\n"
            "Уведомления помогают получать важные сигналы вовремя",
            reply_markup=keyboard,
            parse_mode='HTML'
        )
    
    async def show_profile_settings(self, callback: CallbackQuery, user_id: int):
        """Показать настройки профиля"""
        await callback.answer()
        user = await self.crypto_bot.db.get_user(user_id)
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="📊 Статистика использования", callback_data="stats_detailed")],
            [InlineKeyboardButton(text="🔙 Назад", callback_data="settings_main")]
        ])
        
        profile_text = (
            f"👤 <b>Ваш профиль</b>\n\n"
            f"Имя: {user['first_name']}\n"
            f"Username: @{user['username']}\n"
            f"ID: {user_id}\n"
            f"Дата регистрации: {user['joined_at'][:10]}"
        )
        
        await callback.message.edit_text(
            profile_text,
            reply_markup=keyboard,
            parse_mode='HTML'
        )
    
    async def show_support_menu(self, callback: CallbackQuery):
        """Показать меню поддержки"""
        await callback.answer()
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="📝 Техподдержка", callback_data="settings_support_request")],
            [InlineKeyboardButton(text="⭐ Оставить отзыв", callback_data="settings_feedback")],
            [InlineKeyboardButton(text="🔙 Назад", callback_data="settings_main")]
        ])
        
        await callback.message.edit_text(
            "🆘 <b>Поддержка и обратная связь</b>\n\n"
            "Выберите действие:",
            reply_markup=keyboard,
            parse_mode='HTML'
        )
    
    async def show_detailed_stats(self, callback: CallbackQuery, user_id: int):
        """Показать подробную статистику"""
        await callback.answer()
        user = await self.crypto_bot.db.get_user(user_id)
        favorites = await self.get_user_favorites(user_id)
        
        stats_text = (
            f"📊 <b>Подробная статистика</b>\n\n"
            f"👤 <b>Пользователь:</b> {user['first_name']}\n"
            f"📅 <b>Дата регистрации:</b> {user['joined_at'][:10]}\n"
            f"🔮 <b>Всего прогнозов:</b> {user['predictions_count']}\n"
            f"⭐ <b>Избранные символы:</b> {len(favorites)}\n"
            f"⏰ <b>Интервал по умолчанию:</b> {INTERVALS.get(user['settings']['interval'], user['settings']['interval'])}\n"
            f"🔔 <b>Уведомления:</b> {'Включены' if user['settings']['notifications'] else 'Выключены'}\n\n"
            f"<b>📈 Активность:</b>\n"
            f"• Сегодня: {user.get('daily_predictions', 0)} прогнозов\n"
            f"• Эта неделя: {user.get('weekly_predictions', 0)} прогнозов\n"
            f"• Этот месяц: {user.get('monthly_predictions', 0)} прогнозов"
        )
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Обновить", callback_data="stats_detailed")],
            [InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_main")]
        ])
        
        await callback.message.edit_text(
            stats_text,
            reply_markup=keyboard,
            parse_mode='HTML'
        )