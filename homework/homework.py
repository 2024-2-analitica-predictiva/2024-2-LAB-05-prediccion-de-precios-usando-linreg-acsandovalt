#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import gzip
import pickle
import json
import os

# Constants
CURRENT_YEAR = 2021
TRAIN_PATH = 'files/input/train_data.csv.zip'
TEST_PATH = 'files/input/test_data.csv.zip'
MODEL_PATH = 'files/models/model.pkl.gz'
METRICS_PATH = 'files/output/metrics.json'

# Step 1: Preprocess the data
def preprocess_data(data):
    data['Age'] = CURRENT_YEAR - data['Year']
    data.drop(['Year', 'Car_Name'], axis=1, inplace=True)
    return data

# Load datasets
train_data = pd.read_csv(TRAIN_PATH, compression='zip')
test_data = pd.read_csv(TEST_PATH, compression='zip')

# Preprocess datasets
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Step 2: Split datasets
X_train = train_data.drop('Selling_Price', axis=1)
y_train = train_data['Selling_Price']
X_test = test_data.drop('Selling_Price', axis=1)
y_test = test_data['Selling_Price']

# Verificar columnas después de preprocesamiento
print("Columnas disponibles en X_train después del preprocesamiento:")
print(X_train.columns)

# Asegurar columnas categóricas y numéricas
categorical_features = [col for col in ['Fuel_Type', 'Selling_type', 'Transmission'] if col in X_train.columns]
numerical_features = [col for col in X_train.columns if col not in categorical_features]

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

# Step 3: Create pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', MinMaxScaler(), numerical_features)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selector', SelectKBest(score_func=f_regression, k='all')),
    ('regressor', LinearRegression())
])

# Step 4: Hyperparameter optimization with cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Step 5: Save the model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)  # Ensure the directory exists
with gzip.open(MODEL_PATH, 'wb') as f:
    pickle.dump(pipeline, f)

# Step 6: Calculate metrics
def calculate_metrics(model, X, y, dataset_type):
    predictions = model.predict(X)
    r2 = r2_score(y, predictions)
    mse = mean_squared_error(y, predictions)
    mad = mean_absolute_error(y, predictions)
    return {
        'type': 'metrics',
        'dataset': dataset_type,
        'r2': r2,
        'mse': mse,
        'mad': mad
    }

train_metrics = calculate_metrics(pipeline, X_train, y_train, 'train')
test_metrics = calculate_metrics(pipeline, X_test, y_test, 'test')

# Save metrics to file
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)  # Ensure the directory exists
with open(METRICS_PATH, 'w') as f:
    for metrics in [train_metrics, test_metrics]:
        f.write(json.dumps(metrics) + '\n')

print("Pipeline training complete. Model and metrics saved.")
