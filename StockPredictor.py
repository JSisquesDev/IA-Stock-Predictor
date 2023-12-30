import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.dates as mdates
from datetime import datetime
import tensorflow as tf
import os


'''
Autor: Javier Plaza Sisqués
Metodo: download_data()
Parametros: 
    - stock: La empresa
    - start_date: Fecha inicio
    - end_date: Fecha fin 
Funcionalidad: Descargar los datos financieros de Yahoo
'''
def download_data(stock):
    START_DATE = '1000-01-01'
    
    # Obtenemos la fecha actual
    current_date = get_current_date()

    return yf.download(stock, start=START_DATE, end=current_date)

'''
Autor: Javier Plaza Sisqués
Metodo: get_first_date()
Parametros: 
    - dataset: La empresa
Funcionalidad: Obtiene la primera fecha del dataset
'''
def get_first_date(dataset):
    return dataset.index[0].strftime('%Y-%m-%d')
    
'''
Autor: Javier Plaza Sisqués
Metodo: get_training_data()
Parametros: 
    - data: Los datos con los que trabajar
    - start_date: Fecha inicio
    - end_date: Fecha fin 
Funcionalidad: Obtiene del conjunto de datos globales un dataset procesado comprendido entre dos fechas
'''
def get_processed_dataset(data, start_date, end_date):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = data['Adj Close'][start_date:end_date]
    reshaped_dataset = dataset.values.reshape(-1, 1)
    scaled_dataset = scaler.fit_transform(reshaped_dataset)

    x_dataset, y_dataset = create_sequences(scaled_dataset)
    x_dataset = np.reshape(x_dataset, (x_dataset.shape[0], x_dataset.shape[1], 1))

    return x_dataset, y_dataset

'''
Autor: Javier Plaza Sisqués
Metodo: create_sequences()
Parametros: 
    - data: Los datos con los que trabajar
    - seq_length: Longitud de la secuencia
Funcionalidad: Crea la secuencia de datos en función de los datos pasados por parametro
'''
def create_sequences(data, seq_length=60):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

'''
Autor: Javier Plaza Sisqués
Metodo: get_current_date()
Funcionalidad: Obtiene la fecha actual en formato YYYY-MM-DD
'''
def get_current_date():
    return datetime.today().strftime('%Y-%m-%d')

'''
Autor: Javier Plaza Sisqués
Metodo: get_model()
Parametros: 
    - train_dataset: Los datos con los que se entrenará el modelo
Funcionalidad: Define y obtiene el modelo de IA en base a los datos de entrenamiento
'''
def get_model(train_dataset):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(train_dataset.shape[1], 1)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(units=50, return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(units=50, return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(units=50))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units=1))
    return model


if __name__ == '__main__':

    # Obtenemos los stocks del fichero de stocks
    stock_file = open('stocks.txt', 'r')
    stocks = stock_file.readlines()

    # Recorremos todos los stocks
    for stock in stocks:

        # Descargamos los datos financieros para el stock seleccionado
        data = download_data(stock)

        # Obtenemos la primera fecha registrada del dataset
        start_date = get_first_date(data)

        # Creamos los dataset de entrenamiento y validación
        x_training_dataset, y_training_dataset = get_processed_dataset(data, start_date, '2021-12-31')
        x_validation_dataset, y_validation_dataset = get_processed_dataset(data, '2022-01-01', '2022-12-31')

        # Creamos el modelo
        model = get_model(x_training_dataset)

        # Compilamos el modelo
        model.compile(optimizer='adam', metrics = ['accuracy'], loss='mean_squared_error')

        # Creamos la ruta donde se guardará el modelo
        MODELS_PATH = '.' + os.sep + 'model'
        model_folder_name = str(stock).replace('\n', '')
        model_name = (str(stock) + '.h5').replace('\n', '')
        model_path = os.path.join(MODELS_PATH, model_folder_name, model_name)

        # Establecemos los checkpoints
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        # Entrenamos el modelo usando la GPU
        with tf.device('/gpu:0'):
            history = model.fit(x_training_dataset, y_training_dataset, epochs=100, batch_size=32, callbacks=[early_stopping, checkpoint], validation_data=(x_validation_dataset, y_validation_dataset), verbose=1)

        # Realizamos el test sobre el modelo
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Visualizamos los resultados