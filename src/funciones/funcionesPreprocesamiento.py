# --------------------------------------------------
#
# Autor: Pablo Spínola López
# Description: Funciones centralizadas de la fase de preprocesamiento.
# 
# --------------------------------------------------

import pandas as pd
import unicodedata
import spacy
import re
from deep_translator import GoogleTranslator
from textblob import TextBlob


#######################################################################
# Variables de entorno #
########################

# Cargar modelo de spaCy para español
try:
    nlp = spacy.load('es_core_news_sm')
except:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'es_core_news_sm'])
    nlp = spacy.load('es_core_news_sm')

# Diccionario para abreviaturas comunes en español
ABREVIATURAS = {
    'q': 'que', 'x': 'por', 'd': 'de', 'k': 'que', 'tb': 'también',
    'tmb': 'también', 'pq': 'porque', 'xq': 'porque', 'dnd': 'donde',
    'kien': 'quien', 'salu2': 'saludos', 'aki': 'aquí', 'tqm': 'te quiero mucho',
    'when': 'cuando', 'wtf': 'qué carajos', 'lmao': 'risa', 'lmfao': 'risa',
    'lol': 'risa'
}

# Palabras clave relacionadas con trastornos alimenticios
PALABRAS_CLAVE = {
    'comida': ['comer', 'comida', 'alimento', 'dieta', 'caloría', 'calorías', 'peso', 'adelgazar', 'delgado', 'delgada'],
    'restriccion': ['no comer', 'ayuno', 'saltar comidas', 'evitar comer', 'prohibir', 'prohibido'],
    'purga': ['vomitar', 'vómito', 'laxante', 'diurético', 'purgar', 'purgante'],
    'imagen_corporal': ['gordo', 'gorda', 'feo', 'fea', 'grasa', 'obeso', 'obesa', 'cuerpo', 'figura'],
    'ejercicio': ['ejercicio', 'gimnasio', 'entrenar', 'quemar calorías', 'sudar']
}
# Lematización las palabras clave del diccionario
PALABRAS_CLAVE_LEMATIZADAS = {
    categoria: [token.lemma_ for token in nlp(" ".join(palabras))]
    for categoria, palabras in PALABRAS_CLAVE.items()
}
#######################################################################



#######################################################################
# Funciones para el preprocesamiento de datos #
###############################################

def limpiar_texto(texto):
    """
    Descripción: Limpia y normaliza una cadena de texto, eliminando acentos y convirtiendo todo en minúsculas.
    Entrada - texto (str): Cadena de texto que puede contener caracteres especiales o estar en codificación no UTF-8.
    Salida - texto limpio (str): Texto en minúsculas, sin acentos ni caracteres especiales.
    """
    if pd.isnull(texto):
        return ""
    try:
        texto = texto.encode("latin1").decode("utf-8")
    except:
        pass
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')
    return texto.lower()

def expandir_abreviaturas(texto):
    """
    Descripción: Reemplaza abreviaturas presentes en el texto por sus equivalentes completos contenidos en la variable ABREVIATURAS.
    Entrada - texto (str): Cadena de texto con posibles abreviaturas.
    Salida - texto expandido (str): Texto con las abreviaturas reemplazadas por su forma completa.
    """
    palabras = texto.split()
    palabras_expandidas = [ABREVIATURAS.get(p, p) for p in palabras]
    return ' '.join(palabras_expandidas)

def procesar_hashtags(texto):
    """
    Descripción: Extrae los hashtags del texto y devuelve el texto separado de ellos.
    Entrada - texto (str): Cadena de texto que posiblemente contiene hashtags (palabras precedidas de '#').
    Salida:
        - texto_sin_hashtags (str): Texto sin los hashtags.
        - hashtags (list): Lista de hashtags encontrados sin el símbolo '#'.
    """
    hashtags = re.findall(r"#(\w+)", texto)
    texto_sin_hashtags = re.sub(r"#\w+", "", texto)
    return texto_sin_hashtags.strip(), hashtags

def limpieza_final(texto):
    """
    Descripción: Elimina URLs, menciones, símbolos especiales y caracteres no alfanuméricos del texto.
    Entrada - texto (str): Cadena de texto posiblemente con menciones, enlaces y símbolos innecesarios.
    Salida - texto limpio (str): Texto limpio y legible, sin símbolos ni caracteres no deseados.
    """
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"@\w+", "", texto)
    texto = re.sub(r"&", "y", texto)
    texto = re.sub(r"[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def tokenizar_y_lematizar(texto):
    """
    Descripción: Tokeniza y lematiza el texto en español usando spaCy, eliminando stopwords, puntuación y espacios.
    Entrada - texto (str): Cadena de texto en español coherente, con puntuación y stopwords, seguramente con espacios innecesarios.
    Salida - texto procesado (str): Texto con lemas en minúsculas, sin palabras vacías ni signos de puntuación.
    """
    doc = nlp(texto)
    # Filtrar tokens que no son stopwords, puntuación o espacios
    tokens = [token.lemma_.lower() for token in doc 
              if not token.is_stop and not token.is_punct and not token.is_space]
    return " ".join(tokens)

def calcular_metricas_estilisticas(texto):
    """
    Descripción: Calcula métricas básicas de estilo sobre el texto, como longitud y número de palabras.
    Entrada - texto (str): Cadena de texto procesado.
    Salida - dict: Diccionario con dos entradas; longitud del texto en caracteres y número de palabras.
    """
    palabras = texto.split()
    caracteres = len(texto)
    
    return {
        'longitud_texto': caracteres,
        'num_palabras': len(palabras)
    }

def analizar_palabras_clave(texto):
    """
    Descripción: Cuenta cuántas palabras clave lematizadas de cada categoría aparecen en el texto.
    Entrada - texto (str): Texto lematizado, incluyendo palabras de los hashtags.
    Salida - dict: Diccionario con la frecuencia de aparición por categoría de palabras clave.
    """
    frecuencias = {}
    for categoria, palabras in PALABRAS_CLAVE_LEMATIZADAS.items():
        frecuencias[categoria] = sum(1 for palabra in palabras if palabra in texto)
    return frecuencias

def traducir_si_necesario(texto, target_lang='en'):
    """
    Descripción: Traduce el texto al idioma objetivo, por defecto inglés, usando un traductor automático.
    Entrada:
        - texto (str): Texto original.
        - target_lang (str): Idioma de destino para la traducción (por defecto "en" para inglés).
    Salida: texto traducido (str) - Texto traducido al idioma especificado o el texto original si ocurre un error.
    """
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(texto)
    except Exception as e:
        print(f"Error en traducción: {e}")
        return texto

def analizar_sentimiento(texto):
    """
    Descripción: Analiza el sentimiento del texto traducido utilizando TextBlob.
    Entrada - texto (str): Texto en cualquier idioma (será traducido automáticamente).
    Salida - dict: Diccionario con métricas de sentimiento (polaridad y subjetividad) para la cadena de texto dada.
    """
    try:
        # Traducir solo para análisis de sentimiento
        texto_traducido = traducir_si_necesario(texto)
        blob = TextBlob(texto_traducido)
        return {
            'polaridad': blob.sentiment.polarity,
            'subjetividad': blob.sentiment.subjectivity
        }
    except Exception as e:
        print(f"Error en análisis de sentimiento: {e}")
        return {
            'polaridad': 0,
            'subjetividad': 0
        }

def obtener_hashtags_frecuentes_individuales(hashtags_fila, hashtags_validos):
    """
    Descripción: Filtra hashtags válidos presentes en una fila de entrada y los une en una cadena de texto.
    Entrada:
        - hashtags_fila (list): Lista de hashtags extraídos de un tweet.
        - hashtags_validos (set o list): Conjunto de hashtags considerados válidos.
    Salida: str - Cadena con los hashtags válidos separados por espacios.
    """
    return " ".join([tag for tag in hashtags_fila if tag in hashtags_validos])

#######################################################################



#######################################################################
# Función para la extracción de características #
#################################################

def extraer_caracteristicas(tweet):
    """
    Descripción: Extrae múltiples características de un tweet aplicando las funciones descritas previamente: limpieza, expansión de abreviaturas,
                    lematización, sentimientos, métricas de estilo, hashtags y palabras clave.
    Entrada - tweet (str): Texto original del tweet.
    Salida - dict: Diccionario con características procesadas del tweet.
    """
    # Separación de hashtags
    texto_original, hashtags = procesar_hashtags(tweet)
    
    # Preprocesamiento inicial
    texto_limpio = limpiar_texto(texto_original)
    texto_expandido = expandir_abreviaturas(texto_limpio)
    texto_limpio_final = limpieza_final(texto_expandido)
    
    # Análisis de sentimiento sobre texto limpio expandido
    sentimiento = analizar_sentimiento(texto_limpio_final)

    # Lematización
    texto_lematizado = tokenizar_y_lematizar(texto_limpio_final)
    
    # Métricas estilísticas
    estilisticas = calcular_metricas_estilisticas(texto_lematizado)

    # Preparar texto completo con hashtags sin #
    texto_completo = texto_lematizado + " " + " ".join(hashtags)
    
    # Análisis de palabras clave sobre texto completo
    palabras_clave = analizar_palabras_clave(texto_completo)
    
    return {
        "tweet_text": texto_lematizado,
        "hashtags": hashtags,
        "texto_completo": texto_completo,
        "texto_bert": texto_limpio_final + ". Etiquetas clave: " + " ".join(hashtags),
        "longitud_texto": estilisticas["longitud_texto"],
        "num_palabras": estilisticas["num_palabras"],
        **palabras_clave,
        **sentimiento
    }

#######################################################################