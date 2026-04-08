import gradio as gr
import pandas as pd
import joblib

model = joblib.load("model/best_bike_model.pkl")
scaler = joblib.load("model/scaler.pkl")


def predict_demand(
    season,
    yr,
    mnth,
    hr,
    holiday,
    weekday,
    workingday,
    weathersit,
    temp,
    atemp,
    hum,
    windspeed,
):
    input_df = pd.DataFrame(
        [
            [
                season,
                yr,
                mnth,
                hr,
                holiday,
                weekday,
                workingday,
                weathersit,
                temp,
                atemp,
                hum,
                windspeed,
            ]
        ],
        columns=[
            "season",
            "yr",
            "mnth",
            "hr",
            "holiday",
            "weekday",
            "workingday",
            "weathersit",
            "temp",
            "atemp",
            "hum",
            "windspeed",
        ],
    )

    # Масштабируем числовые признаки
    numeric_features = ["temp", "atemp", "hum", "windspeed"]
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])

    # Делаем предсказание
    prediction = model.predict(input_df)[0]

    # Округляем до целого числа и защищаемся от отрицательных прогнозов
    final_result = max(0, int(round(prediction)))

    return f"🚲 Ожидаемый спрос: {final_result} велосипедов"


# Интерфейс Gradio
inputs = [
    gr.Dropdown(
        choices=[1, 2, 3, 4], label="Сезон (1-Весна, 2-Лето, 3-Осень, 4-Зима)", value=2
    ),
    gr.Radio(choices=[0, 1], label="Год (0: 2011, 1: 2012)", value=1),
    gr.Slider(minimum=1, maximum=12, step=1, label="Месяц (1-12)", value=6),
    gr.Slider(
        minimum=0, maximum=23, step=1, label="Час дня (0-23)", value=17
    ),  # По умолчанию ставим час пик
    gr.Radio(choices=[0, 1], label="Праздничный день (0-Нет, 1-Да)", value=0),
    gr.Slider(
        minimum=0,
        maximum=6,
        step=1,
        label="День недели (0-Вс, 1-Пн ... 6-Сб)",
        value=3,
    ),
    gr.Radio(choices=[0, 1], label="Рабочий день (0-Нет, 1-Да)", value=1),
    gr.Dropdown(
        choices=[1, 2, 3, 4],
        label="Погода (1-Ясно, 2-Облачно, 3-Легкий дождь, 4-Ливень/Снег)",
        value=1,
    ),
    gr.Slider(
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        label="Температура (нормализованная 0-1)",
        value=0.6,
    ),
    gr.Slider(
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        label="Ощущаемая температура (0-1)",
        value=0.6,
    ),
    gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Влажность (0-1)", value=0.5),
    gr.Slider(
        minimum=0.0, maximum=1.0, step=0.01, label="Скорость ветра (0-1)", value=0.2
    ),
]

# Создаем само веб-приложение
interface = gr.Interface(
    fn=predict_demand,
    inputs=inputs,
    outputs=gr.Text(label="Результат"),
    title="🚴 Прогноз спроса на каршеринг велосипедов",
    description=(
        "Установите погодные и календарные параметры, чтобы модель предсказала, "
        "сколько велосипедов потребуется арендовать в этот час."
    ),
    flagging_mode="never",
)


if __name__ == "__main__":
    print("Запуск веб-интерфейса.")
    interface.launch()
