import catboost as cb
import pandas as pd
from pydantic import BaseModel

from flask import Flask, jsonify, request

# меняем эти строки на нашу модель
model = cb.CatBoostClassifier()
model.load_model("best_catboost_model.bin")

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict():
    X = request.get_json()
    preds = model.predict_proba(pd.DataFrame(X, index=[0]))[0, 1] # меняем эту строчку
    result = {'default_proba': preds}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8989)

