import xgboost as xgb
from flask import Flask, request, jsonify
import numpy as np

model = xgb.Booster()
model.load_model("best_xgb_model.bin")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получаем данные из POST-запроса
        data = request.get_json()

        # Преобразуем данные в формат numpy
        features = np.array([list(data.values())], dtype=float)

        # Создаём DMatrix для XGBoost
        dmatrix = xgb.DMatrix(features)

        # Получаем предсказание вероятностей
        probabilities = model.predict(dmatrix)

        # Определяем финальный класс
        predicted_class = int(np.argmax(probabilities, axis=1)[0])

        # Формируем результат
        result = {
            "predicted_class": predicted_class,
            "probabilities": probabilities.tolist()
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8989, debug=True)
