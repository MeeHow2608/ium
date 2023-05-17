import tensorflow as tf
from sacred import Experiment
from sacred.observers import FileStorageObserver
import pandas as pd
import sklearn
import sklearn.model_selection
import numpy as np

ex = Experiment('452662')
ex.observers.append(FileStorageObserver.create('my_runs'))
#ex.observers.append(MongoObserver(url='mongodb://admin:IUM_2021@172.17.0.1:27017', db_name='sacred'))

def normalize(df,feature_name):
    result = df.copy()
    max_value = df[feature_name].max()
    min_value = df[feature_name].min()
    result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


@ex.automain
def run_experiment():
    cars = pd.read_csv('zbior_ium/Car_Prices_Poland_Kaggle.csv')

    cars = cars.drop(73436) #wiersz z błednymi danymi
    
    cars_normalized = normalize(cars,'vol_engine')
    
    cars_train, cars_test = sklearn.model_selection.train_test_split(cars_normalized, test_size=23586, random_state=1)
    cars_dev, cars_test = sklearn.model_selection.train_test_split(cars_test, test_size=11793, random_state=1)
    cars_train.rename(columns = {list(cars_train)[0]: 'id'}, inplace = True)
    cars_test.rename(columns = {list(cars_test)[0]: 'id'}, inplace = True)
    cars_train.to_csv('train.csv')
    cars_test.to_csv('test.csv')
    
    feature_cols = ['year', 'mileage', 'vol_engine']
    inputs = tf.keras.Input(shape=(len(feature_cols),))

    x = tf.keras.layers.Dense(10, activation='relu')(inputs)
    x = tf.keras.layers.Dense(10, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse', metrics=['mae'])

    model.fit(cars_train[feature_cols], cars_train['price'], epochs=100)

    ex.add_resource('train_data.csv')
    ex.add_resource('test_data.csv')

    ex.add_artifact(__file__)

    model.save('model.h5')
    ex.add_artifact('model.h5')

    metrics = model.evaluate(cars_train[feature_cols], cars_train['price'])
    ex.log_scalar('mse', metrics[0])
    ex.log_scalar('mae', metrics[1])