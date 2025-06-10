# --------------------------------------------------
#
# Autor: Pablo Spínola López
# Description: Funciones centralizadas de la fase de entrenamiento y evaluzación del modelo BERT utilizado.
# 
# --------------------------------------------------

# Librerías utilizadas para el desarrollo del modelo BERT
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch

# Manejo de datos e información
import pandas as pd
import numpy as np
import os

# Variables globales, propias de nuestra configuración
MODEL_NAME = "PlanTL-GOB-ES/roberta-base-bne"
BATCH_SIZE = 16
RANDOM_STATE = 22
NUM_EPOCHS = 50
KFOLDS = 5
PATIENCE = 5
TARGET_AUC = 0.93
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenización
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Modelo para evaluar, usar para embedding
model2embed = AutoModel.from_pretrained(MODEL_NAME)
model2embed.eval()

##############################################################################################################################
####                   Clases utilizadas                   ####
###############################################################
class TweetDataset(Dataset):
    """
    Descripción: Clase personalizada para crear un dataset compatible con PyTorch a partir de textos (tweets) y sus etiquetas.
                 Se utiliza para el entrenamiento o evaluación de nuestro modelo basado en Transformers (RoBERTa).
    Hereda: torch.utils.data.Dataset
    Métodos: - __init__: Tokeniza los textos y almacena las etiquetas.
             - __getitem__: Devuelve una muestra del dataset en formato tensor, lista para ser usada por un DataLoader.
             - __len__: Devuelve la cantidad de muestras en el dataset.
    """

    def __init__(self, texts, labels):
        """
        Descripción: Inicializa el dataset, tokenizando los textos y almacenando las etiquetas.
        Entrada: - texts (Series): Lista de textos (tweets) a tokenizar.
                 - labels (Series): Lista de etiquetas correspondientes a cada texto.
        """
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        """
        Descripción: Devuelve una muestra individual del dataset con sus tokens y etiqueta, en formato tensor.
        Entrada: - idx (int): Índice de la muestra a devolver.
        Salida: - item (dict): Diccionario que contiene los tensores de entrada para el modelo y su etiqueta asociada.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        """
        Descripción: Devuelve la cantidad total de muestras en el dataset.
        Salida: - (int): Número de muestras.
        """
        return len(self.labels)
    

class RobertaClassifier(nn.Module):
    """
    Descripción: Clasificador basado en una arquitectura Roberta preentrenada, adaptado para nuestro objetivo de clasificación binaria.
                 Utiliza un modelo Transformer, RoBERTa) como extractor de características, finalizado con transfer learning de la red neuronal
                 para realizar la clasificación.
    Hereda: torch.nn.Module
    Métodos: - __init__: Inicializa el modelo cargando el backbone preentrenado, congelando capas, y definiendo las capas adicionales para la
                         clasificación.
             - forward: Define el paso del modelo; cómo se transforma la entrada en una predicción.
    """
    def __init__(self, unfreeze_layers=None):
        """
        Descripción: Inicializa la arquitectura del clasificador RoBERTa.
        Entrada: - unfreeze_layers (list o None): Lista de nombres o fragmentos de nombres de capas del modelo preentrenado 
                                                  que se desea descongelar para entrenamiento (fine-tuning).
        """
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        # Congelamos todos los parámetros por defecto
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
        # Si se especifican capas a descongelar, las activamos para entrenamiento
        if unfreeze_layers:
            for name, param in self.bert.named_parameters():
                if any(layer in name for layer in unfreeze_layers):
                    param.requires_grad = True
        # Capas adicionales para clasificación
        self.dropout = nn.Dropout(0.5)
        self.intermediate = nn.Linear(self.bert.config.hidden_size, 128)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(128, 1)
    
    def forward(self, input_ids, attention_mask):
        """
        Descripción: Realiza el paso hacia adelante del modelo. Pasa las entradas por el modelo preentrenado y
                     las capas adicionales para obtener una predicción final.
        Entrada: - input_ids (Tensor): IDs de tokens de entrada.
                 - attention_mask (Tensor): Máscara de atención para ignorar tokens de padding.
        Salida: - x (Tensor): Logits de salida (sin activación sigmoide).
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Tomamos la media de todos los tokens de la última capa oculta como representación del texto
        x = torch.mean(outputs.last_hidden_state, dim=1)
        x = self.dropout(x)
        x = self.relu(self.intermediate(x))
        x = self.classifier(x)
        # Salida en forma (batch_size,)
        return x.squeeze(-1)
    
##############################################################################################################################
####                   Funciones utilizadas                   ####
##################################################################
    
def generar_loader(X, y, shuffle=None):
    """
    Descripción: Crea un DataLoader a partir de los textos y etiquetas proporcionados. 
                 Esto permite agrupar los datos en batches para entrenamiento o evaluación.
    Entrada: - X (list o Series): Lista de textos de entrada.
             - y (list o Series): Lista de etiquetas correspondientes a los textos.
             - shuffle (bool o None): Indica si se deben mezclar los datos.
    Salida: - loader (DataLoader): DataLoader de PyTorch que itera sobre el conjunto de datos.
    """
    ds = TweetDataset(X, y)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)
    return loader

def obtener_probs(X, y, modelo, y_verdaderas=[], y_probs=[]):
    """
    Descripción: Calcula las probabilidades de predicción del modelo sobre un conjunto de datos.
                 Utiliza el modelo entrenado y el DataLoader generado para obtener las salidas del modelo
                 y las etiquetas verdaderas.
    Entrada: - X (list o Series): Lista de textos de entrada.
             - y (list o Series): Etiquetas verdaderas correspondientes.
             - modelo (nn.Module): Modelo entrenado que realizará las predicciones.
             - y_verdaderas (list): Lista acumulativa para almacenar las etiquetas verdaderas.
             - y_probs (list): Lista acumulativa para almacenar las probabilidades predichas.
    Salida: - y_true (ndarray): Arreglo con las etiquetas verdaderas.
            - y_probs (ndarray): Arreglo con las probabilidades predichas por el modelo.
            - y_pred (ndarray): Arreglo con las predicciones binarias generadas (umbral de 0.5).
    """
    loader = generar_loader(X, y)

    with torch.no_grad():  # Desactiva el cálculo de gradientes para acelerar y ahorrar memoria
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].cpu().numpy()

            outputs = modelo(input_ids, attention_mask)  # Salidas sin activar (logits)
            probs = torch.sigmoid(outputs).cpu().numpy()  # Se convierten a probabilidades con sigmoide

            y_probs.extend(probs)
            y_verdaderas.extend(labels)

    y_true = np.array(y_verdaderas)
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= 0.5).astype(int)  # Clasificación binaria usando umbral de 0.5

    return y_true, y_probs, y_pred

def cargar_estado_modelo():
    """
    Descripción: Carga el estado guardado del modelo RoBERTa entrenado desde un archivo `.pt`.
                 Esto permite reutilizar un modelo ya entrenado para predicción o evaluación.
    Entrada: Ninguna.
    Salida: - modelo_RoBERTa (RobertaClassifier): Instancia del modelo con los pesos cargados.
    """
    modelo_RoBERTa = RobertaClassifier()
    modelo_RoBERTa.load_state_dict(
        torch.load(
            os.path.abspath("../../models/final_best_model.pt"),
            map_location=torch.device('cpu')  # Carga el modelo en CPU, útil si no se dispone de GPU
        )
    )
    return modelo_RoBERTa

##############################################################################################################################
####                   Funciones utilizadas (Módulo final)                   ####
#################################################################################

def data4embed(test=False):
    """
    Descripción: Carga los datos desde archivos CSV, según si se requiere el conjunto de prueba o no.
                 Extrae y retorna los textos procesados para BERT y sus etiquetas correspondientes.
    Entrada: - test (bool): Si es True, carga el conjunto de prueba. Si es False, carga el conjunto de entrenamiento.
    Salida: - texts (list): Lista de textos preprocesados para BERT.
            - labels (list): Lista de etiquetas asociadas a cada texto.
    """
    # Carga de datos
    if test:
        df = pd.read_csv("../../data/ds_BETO_TEST_FINAL.csv")
    else:
        df = pd.read_csv("../../data/ds_BETO.csv")

    # División de datos
    texts = list(df["texto_bert"])
    labels = list(df["class"])

    return texts, labels

def get_embeddings(text_list):
    """
    Descripción: Genera los embeddings de una lista de textos utilizando nuestro modelo tipo BERT.
                 Usa mean pooling sobre el último estado oculto del modelo para obtener un vector por texto.
    Entrada: - text_list (list): Lista de cadenas de texto para las cuales se desea obtener embeddings.
    Salida: - embeddings (np.ndarray): Arreglo NumPy con los vectores de embeddings (uno por texto).
    """
    embeddings = []

    with torch.no_grad():
        for text in text_list:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            outputs = model2embed(**inputs)
            
            # Usamos el mean pooling del último estado oculto
            last_hidden_state = outputs.last_hidden_state
            mean_embedding = last_hidden_state.mean(dim=1)
            embeddings.append(mean_embedding.squeeze().numpy())

    return np.array(embeddings)

