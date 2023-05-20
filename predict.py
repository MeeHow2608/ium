import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

#import mlflow

# Wskazujemy ścieżkę do folderu, gdzie zostaną zapisane wyniki MLflow
#mlflow.set_tracking_uri("file:/mlflow")

# Ustawiamy nazwę eksperymentu
#mlflow.set_experiment("nazwa eksperymentu")

feature_cols = ['year', 'mileage', 'vol_engine']


feature_cols = ['year', 'mileage', 'vol_engine']

model = load_model('model.h5')
test_data = pd.read_csv('test.csv')

predictions = model.predict(test_data[feature_cols])
predicted_prices = [p[0] for p in predictions]


results = pd.DataFrame({'id': test_data['id'], 'year': test_data['year'], 'mileage': test_data['mileage'], 'vol_engine': test_data['vol_engine'], 'predicted_price': predicted_prices})
results.to_csv('predictions.csv', index=False)

y_true = test_data['price']
y_pred = [round(p[0]) for p in predictions]

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

with open('metrics.txt', 'w') as f:
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
