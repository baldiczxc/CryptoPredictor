from aiogram.fsm.state import State, StatesGroup

class UserStates(StatesGroup):
    waiting_symbol = State()
    waiting_interval = State()
    waiting_horizon = State()
    
    # Добавленные состояния
    waiting_settings_choice = State()
    waiting_favorite_action = State()
    waiting_notification_toggle = State()
    waiting_interval_selection = State()
    waiting_custom_symbol = State()
    waiting_prediction_confirmation = State()
    waiting_settings_update = State()
    waiting_favorites_management = State()
    
    # Состояния для расширенных функций
    waiting_advanced_settings = State()
    waiting_timeframe_selection = State()
    waiting_model_selection = State()
    waiting_signal_preferences = State()
    
    # Состояния для настроек уведомлений
    waiting_notification_settings = State()
    waiting_price_threshold = State()
    waiting_volume_threshold = State()
    waiting_technical_indicators = State()
    
    # Состояния для управления избранным
    waiting_favorite_add = State()
    waiting_favorite_remove = State()
    waiting_favorite_reorder = State()
    
    # Состояния для статистики и аналитики
    waiting_detailed_stats = State()
    waiting_performance_review = State()
    waiting_trading_journal = State()
    
    # Состояния для помощи и поддержки
    waiting_support_request = State()
    waiting_feedback = State()
    waiting_bug_report = State()
    
    # Состояния для массовых операций
    waiting_bulk_prediction = State()
    waiting_bulk_symbol_selection = State()
    waiting_bulk_interval_selection = State()
    
    # Состояния для профиля пользователя
    waiting_profile_edit = State()
    waiting_username_change = State()
    waiting_timezone_selection = State()
    
    # Состояния для торговых сигналов
    waiting_signal_confirmation = State()
    waiting_signal_preferences_update = State()
    waiting_risk_management = State()
    
    # Состояния для исторических данных
    waiting_historical_analysis = State()
    waiting_comparison_symbol = State()
    waiting_date_range = State()