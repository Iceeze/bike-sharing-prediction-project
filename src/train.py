import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# =========================
# --- Подготовка данных ---
# =========================

df = pd.read_csv("data/hour.csv")

# X - все признаки, кроме целевой переменной 'cnt' и связанных с ней 'casual' и 'registered'
# 'instant' и 'dteday' удаляем, так как время года и месяц уже есть в других колонках
X = df.drop(columns=["cnt", "casual", "registered", "instant", "dteday"])

# y - целевая переменная (общее количество арендованных велосипедов)
y = df["cnt"]

# В данном датасете категориальные признаки уже переведены в числа создателями датасета

# Масштабирование числовых признаков
numeric_features = ["temp", "atemp", "hum", "windspeed"]
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Разделение данных:
# 85% - для обучения и кросс-валидации
# 15% - для теста
# Такое разделение позволяет дать модели большую базу для обучения
# и при этом оставить достаточно данных (~2.5к строк) для тестового прогона
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)


# ==============================
# --- Обучение и диагностика ---
# ==============================

# Выбираем два принципиально разных алгоритма:
# 1. Линейная регрессия
# 2. Случайный лес (с небольшим количеством деревьев (50) для быстрого обучения и анализа)
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
}

# Кросс-валидация
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_model_name = ""
best_model = None
best_score = float("inf")

for name, model in models.items():
    # Считаем Negative MAE (в sklearn ошибки отрицательные для кросс-валидации)
    scores = cross_val_score(
        model, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error"
    )
    mae_score = -scores.mean()
    print(f"Модель: {name} | Средняя MAE на кросс-валидации: {mae_score:.2f}")

    model.fit(X_train, y_train)

    # Диагностика как именно ошибается модель
    # Сделаем прогноз на тренировочной выборке и посмотрим, в какие часы ошибка максимальна
    train_preds = model.predict(X_train)
    errors = np.abs(train_preds - y_train)

    error_df = pd.DataFrame({"Hour": X_train["hr"], "Error": errors})
    mean_error_by_hour = error_df.groupby("Hour")["Error"].mean()

    # Находим час с максимальной средней ошибкой
    worst_hour = mean_error_by_hour.idxmax()

    # Визуализация главной ошибки
    plt.figure(figsize=(10, 5))
    mean_error_by_hour.plot(kind="bar", color="salmon")
    plt.title(f"Средняя ошибка (MAE) предсказаний {name} в зависимости от часа дня")
    plt.xlabel("Час дня (0-23)")
    plt.ylabel("Средняя ошибка (кол-во велосипедов)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    name_error_file = f"data/error_analysis_{name}.png"
    plt.savefig(name_error_file)
    print(f"График ошибок сохранен в файл '{name_error_file}'.\n")

    # Запоминаем лучшую модель
    if mae_score < best_score:
        best_score = mae_score
        best_model_name = name
        best_model = model


# ====================================
# --- Финальный отбор и сохранение ---
# ====================================

final_preds = best_model.predict(X_test)
final_mae = mean_absolute_error(y_test, final_preds)

# Сохраняем модель и scaler
joblib.dump(best_model, "model/best_bike_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("--- ИТОГОВЫЙ ОТЧЕТ ---")
print(f"Лучшая модель — {best_model_name}.")
print(f"Её ключевая метрика (MAE) на новых данных — {final_mae:.2f}.")
print(
    f"Чаще всего она ошибается в {worst_hour}:00, "
    f"так как это время максимального, но нестабильного спроса."
)
print("----------------------")
