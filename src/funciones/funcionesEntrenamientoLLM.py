import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)


def cargar_datos_entrenamiento():
    """
    Descripción: Carga el conjunto de datos de entrenamiento desde un archivo CSV, separando todos los atributos (X) de las etiquetas (y).
    Entrada: Ninguna.
    Salida - X (DataFrame): Atributos del conjunto de entrenamiento.
           - y (Series): Etiquetas del conjunto de entrenamiento.
    """
    df = pd.read_csv("../../data/data_train.csv")
    # Separar características y etiquetas
    X = df["tweet_text"]
    y = df["class"].map({'control': 0, 'anorexia': 1})
    return X, y


def imprimir_forma(df):
    """
    Descripción: Retorna las dimensiones del DataFrame y las primeras 5 filas para una inspección rápida.
    Entrada - df (DataFrame): Conjunto de datos a inspeccionar.
    Salida - shape (tuple): Dimensiones del DataFrame.
           - head (DataFrame): Primeras 5 filas del DataFrame.
    """
    return df.shape, df.head(5)