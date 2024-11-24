import uvicorn
import json
import requests
import pickle
import nest_asyncio
import threading
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
from io import StringIO
from fastapi.responses import StreamingResponse
nest_asyncio.apply()

# Загрузка модели при старте приложения
with open("model_data.pickle", "rb") as file:
    model_data = pickle.load(file)

# Модель, стандартизатор и список признаков
model = model_data["model"]
scaler = model_data.get("scaler", None)
features_name = model_data["features_name"]

# Инициализация FastAPI-приложения
app = FastAPI()


# Описание структуры данных для запроса на основе Pydantic
class Item(BaseModel):
    name: str
    year: int
    selling_price: int = 0
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


# Список объектов Item для обработки нескольких записей
class Items(BaseModel):
    RootModel: List[Item]


# Проверка состояния сервера
@app.get("/")
def read_root():
    return {"message": "Мы работаем"}


# Функция для преобразования строковых данных в числовые
def to_numeric_(value: str) -> float:
    if value is None:
        return np.nan
    else:
        return float(''.join(filter(lambda x: x.isdigit() or x == '.', value)))


# Функция предсказания цены автомобиля
def predict_price(item: Item) -> float:
    # Подготовка данных для модели
    features_dict = {}
    for feature in features_name:
        value = getattr(item, feature)  # Получаем значение атрибута по имени
        if isinstance(value, str):    # Если это строка, извлекаем числовую часть
            value = to_numeric_(value)
        features_dict[feature] = value
    features_df = pd.DataFrame([features_dict])  # Добавялем в DataFrame признаков
    # Масштабирование данных
    features_scaled = scaler.transform(features_df)

    # Предсказание цены
    predicted_price = model.predict(features_scaled)[0]
    return max(predicted_price, 0)  # цена не может быть отрицательной


# Маршрут для предсказания цены одного автомобиля
@app.post("/predict_item")
def predict_item(item: Item) -> float:
    price = predict_price(item)
    return price


# Маршрут для предсказания цен нескольких автомобилей
@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    prices = []
    for item in items:
        price = predict_price(item)
        prices.append(price)
    return prices


# Маршрут для обработки файла с данными автомобилей
@app.post("/predict_items_file")
async def predict_items_file(file: UploadFile = File):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))

    # Обработка каждой строки в объект Item для предсказания
    predictions = []
    for _, row in df.iterrows():
        item = Item(
            name=row['name'],
            year=int(row['year']),
            selling_price=int(row.get('selling_price', 0)),
            km_driven=int(row['km_driven']),
            fuel=row['fuel'],
            seller_type=row['seller_type'],
            transmission=row['transmission'],
            owner=row['owner'],
            mileage=row['mileage'],
            engine=row['engine'],
            max_power=row['max_power'],
            torque=row['torque'],
            seats=float(row['seats']),
        )
        price = predict_price(item)
        predictions.append(price)
    df['predicted_price'] = predictions  # Добавляем предсказания в DataFrame

    # Сохранение DataFrame в CSV
    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment;filename=predictions.csv"}
    )

# Если запускать через интерпретатор кода, то можно сразу проверить примеры работы API
# Если необходимо запустить сервер через консоль, то необходимо заккоментировать всё, что ниже


# Запуск сервера в фоновом потоке
def run():
    uvicorn.run(app, host='127.0.0.1', port=8000)


threading.Thread(target=run, daemon=True).start()


# Предсказание для одной машины
url = 'http://127.0.0.1:8000/predict_item'
with open('one_item.json') as file:
    data = json.load(file)
response = requests.post(url, json=data[0])
print(f"Предсказанная цена: {response.json()}")

# Предсказание для трёх машин
url = 'http://127.0.0.1:8000/predict_items'
with open('three_items.json') as file:
    data = json.load(file)
response = requests.post(url, json=data)
print(f"Предсказанные цены: {response.json()}")

# Предсказание для файла csv
url = 'http://127.0.0.1:8000/predict_items_file'
files = {'file': open('test.csv', 'rb')}

response = requests.post(url, files=files)

# Сохранение файла с прогнозом
with open('predictions.csv', 'wb') as f:
    f.write(response.content)

print("Файл с прогнозом сохранен как predictions.csv")

pred = pd.read_csv('predictions.csv')
print(pred)
