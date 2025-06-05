# --------------------------------------------------
#
# Autor: Pablo Spínola López
# Description: Funciones centralizadas de la fase de entrenamiento de LLM.
# 
# --------------------------------------------------

# Librerías para manejo de información e integración
import pandas as pd
from tqdm import tqdm

# SciKit_Learn para reporte de clasificación 
from sklearn.metrics import classification_report

# Librería de langchain para crear un objeto de tipo Template
from langchain_core.prompts import PromptTemplate


# Crear el prompt con few-shot, con información del dataset de entrenamiento
prompt = PromptTemplate.from_template("""
Eres un asistente experto en salud mental. Tu tarea es analizar tweets e indicar si muestran señales de un posible trastorno alimenticio (como anorexia, bulimia, o similares).

Responde únicamente con "Sí" si el tweet indica un posible trastorno alimenticio, y con "No" si no lo indica.

Ejemplos:
Tweet: "no comí en todo el día y me siento bien" → Sí
Tweet: "me encanta comer con mis amigos los fines de semana" → No
Tweet: "me siento gorda incluso si no como nada" → Sí
Tweet: "hoy hice ejercicio y luego almorcé con mi familia" → No
Tweet: "ser como ellas ♡♡
  #hastaLoshuesos" → Sí
Tweet: "hoy #domingo os esperamos desde las 10:00 de la mañana para empezar el día con un buen desayuno.

también podéis reservar mesa para comer o cenar y disfrutar de nuestra carta o del menú fin de semana.
 980 56 99 34

#felizdomingo #restaurantealabros #zamora  #yum" → No

Ahora, clasifica el siguiente tweet, recuerda responder únicamente con dos letras en cada caso 'Sí' o 'No', sin explicaciones, como se muestra en los ejemplos. También recuerda que lo que recibes como input es únicamente el tweet y nada más que el tweet al que tienes que clasificar:
Tweet: "{tweet}" → 
""")


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
    Salida  - shape (tuple): Dimensiones del DataFrame.
            - head (DataFrame): Primeras 5 filas del DataFrame.
    """
    return df.shape, df.head(5)

def obtener_predicciones(chain, X):
    """
    Descripción: Obtiene las predicciones de nuestro LLM usado para clasificación, recibiendo el pipeline.
    Entrada - chain (Chain): Pipeline del modelo recibido.
            - X (DataFrame): Conjunto de datos a clasificar.
    Salida  - predicciones (Series): Lista de predicciones.
    """
    predicciones = []
    for tweet in tqdm(X, desc="Clasificando tweets con LLaMA3"):
        result = chain.invoke({"tweet": tweet})
        answer = result['text'].strip().lower()
        if "Sí" in answer or "sí" in answer:
            predicciones.append(1)
        elif "No" in answer or "no" in answer:
            predicciones.append(0)
        else:
            print("Error en tweet:", result['text'])
            break  # debug en caso de error
    return predicciones

def reporte_clasificacion_llm(predicts, y_real):
    """
    Descripción: Generar predicciones, calcular probabilidades de predicción o funciones de decisión, y crea un reporte de clasificación.
    Entrada - predicts (array): Lista de predicciones.
            - y_real (array): Conjunto de las etiquetas verdaderas.
    Salida - y_pred (array): Etiquetas predichas.
           - reporte (str): Métricas de desempeño del modelo.
    """
    y_preds = pd.DataFrame(predicts, columns=['predicciones'])
    reporte = classification_report(y_real, y_preds)
    return y_preds, reporte