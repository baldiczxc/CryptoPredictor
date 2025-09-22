def format_prediction_result(result: dict) -> str:
    """Форматирование результата прогноза"""
    current_price = result['current_price']
    predicted_price = result['predicted_price']
    change_percent = result['change_percent']
    rsi = result['rsi']
    symbol = result['symbol']
    interval = result['interval']
    
    # Определяем тренд
    if change_percent > 1:
        trend_emoji = "🚀"
        trend_text = "СИЛЬНЫЙ РОСТ"
        trend_color = "🟢"
    elif change_percent > 0.3:
        trend_emoji = "📈"
        trend_text = "УМЕРЕННЫЙ РОСТ"
        trend_color = "🟢"
    elif change_percent < -1:
        trend_emoji = "📉"
        trend_text = "СИЛЬНОЕ ПАДЕНИЕ"
        trend_color = "🔴"
    elif change_percent < -0.3:
        trend_emoji = "📉"
        trend_text = "УМЕРЕННОЕ ПАДЕНИЕ"
        trend_color = "🔴"
    else:
        trend_emoji = "💤"
        trend_text = "КОНСОЛИДАЦИЯ"
        trend_color = "🟡"
    
    # Торговый сигнал
    if rsi < 30 and change_percent > 0.5:
        signal = "🟢 СИЛЬНАЯ ПОКУПКА"
        signal_desc = "RSI показывает перепроданность + прогноз роста"
    elif rsi > 70 and change_percent < -0.5:
        signal = "🔴 СИЛЬНАЯ ПРОДАЖА"
        signal_desc = "RSI показывает перекупленность + прогноз падения"
    elif change_percent > 0.3:
        signal = "🟢 УМЕРЕННАЯ ПОКУПКА"
        signal_desc = "Прогнозируется рост цены"
    elif change_percent < -0.3:
        signal = "🔴 УМЕРЕННАЯ ПРОДАЖА"
        signal_desc = "Прогнозируется падение цены"
    else:
        signal = "🟡 ОЖИДАНИЕ"
        signal_desc = "Неопределенный сигнал, лучше подождать"
    
    # RSI анализ
    if rsi < 30:
        rsi_status = "Перепроданность"
    elif rsi > 70:
        rsi_status = "Перекупленность"
    else:
        rsi_status = "Нормальная зона"
    
    prediction_text = (
        f"🎯 <b>ПРОГНОЗ ДЛЯ {symbol}</b>\n"
        f"⏰ <b>Интервал:</b> {interval}\n"
        f"📅 <b>Время анализа:</b> {result['timestamp'].strftime('%H:%M:%S')}\n\n"
        
        f"💰 <b>ТЕКУЩАЯ ЦЕНА:</b> ${current_price:,.2f}\n"
        f"🔮 <b>ПРОГНОЗ:</b> ${predicted_price:,.2f}\n"
        f"📊 <b>ИЗМЕНЕНИЕ:</b> {change_percent:+.2f}%\n\n"
        
        f"{trend_color} <b>ТРЕНД:</b> {trend_emoji} {trend_text}\n\n"
        
        f"📈 <b>ТЕХНИЧЕСКИЙ АНАЛИЗ:</b>\n"
        f"• RSI: {rsi:.1f} ({rsi_status})\n"
        f"• Сила тренда: {abs(change_percent):.2f}%\n\n"
        
        f"💡 <b>ТОРГОВЫЙ СИГНАЛ:</b>\n"
        f"{signal}\n"
        f"<i>{signal_desc}</i>\n\n"
        
        f"⚠️ <b>Важно:</b> Это прогноз на основе ИИ. "
        f"Всегда проводите собственный анализ перед торговлей!"
    )
    
    return prediction_text