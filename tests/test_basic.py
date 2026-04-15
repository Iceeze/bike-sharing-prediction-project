import sys
import os
# Добавляем путь к папке src, чтобы импортировать predict_demand.
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import pytest
from app import predict_demand

# Тест 1: Проверяет, что функция работает на корректном примере.
def test_predict_demand_works():
    """
    Берём типичный летний рабочий день, 17:00, хорошую погоду.
    Ожидаем, что функция вернёт не None.
    """
    # Признаки в том же порядке, что и в app.py.
    result = predict_demand(
        season=2,   # лето.
        yr=1,   # 2012 год.
        mnth=6, # июнь.
        hr=17,  # 5 вечера (час пик).
        holiday=0,  # не праздник.
        weekday=3,  # среда.
        workingday=1,   # рабочий день.
        weathersit=1,   # ясно.
        temp=0.6,   # 60% от максимальной температуры.
        atemp=0.6, # нормализованная ощущаемая температура.
        hum=0.5, # нормализованная влажность (50% от максимума).
        windspeed=0.2 # нормализованная скорость ветра.
    )
    # Проверки.
    assert result is not None, "Функция вернула None, хотя должна вернуть строку"
    assert isinstance(result, str), "Результат должен быть строкой"

# Тест 2: Проверяет формат возвращаемого значения.
def test_predict_demand_returns_correct_format():
    """
    Проверяем, что возвращается именно строка, содержащая слово "велосипедов".
    """
    result = predict_demand(
        season=1,   # Весна.
        yr=0,   # 2011 год.
        mnth=1, # Январь.
        hr=8,   # 8 утра.
        holiday=0,  # Не праздник.
        weekday=1,  # Понедельник.
        workingday=1,   # Рабочий день.
        weathersit=2,   # Облачно.
        temp=0.2,   # 20% от максимальной температуры.
        atemp=0.2,  # Нормализованная ощущаемая температура.
        hum=0.8,    # Высокая влажность (80% от максимума).
        windspeed=0.4   # Нормализованная скорость ветра.
    )
    assert isinstance(result, str), "Результат не строка"
    assert "велосипедов" in result, "В ответе нет ключевого слова 'велосипедов'"

# Тест 3: Проверяет, что веб-приложение (интерфейс) может запуститься без ошибок.
def test_gradio_interface_can_launch():
    """
    Запускаем веб-интерфейс в отдельном потоке на свободном порту,
    проверяем, что запуск не вызывает исключений и сервер принимает соединения.
    """
    import threading
    import time
    import requests
    from app import interface

    # Запускаем интерфейс на порту 7861.
    def run():
        interface.launch(server_port=7861, quiet=True, share=False)
    
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    
    # Даём серверу время подняться.
    time.sleep(3)
    
    # Проверяем, что сервер отвечает.
    try:
        response = requests.get("http://127.0.0.1:7861")
        assert response.status_code == 200, "Сервер не отвечает"
    except requests.ConnectionError:
        assert False, "Не удалось подключиться к запущенному серверу"