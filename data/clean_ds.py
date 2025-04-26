# =============================================================================
# Importaciones
# =============================================================================
import pandas as pd
import re
import numpy as np
import unicodedata
import spacy
import swifter
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from deep_translator import GoogleTranslator

# =============================================================================
# Configuración inicial
# =============================================================================
# Cargar modelo de spaCy para español
try:
    nlp = spacy.load('es_core_news_sm')
except:
    print("Instalando modelo de spaCy para español...")
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'es_core_news_sm'])
    nlp = spacy.load('es_core_news_sm')

# Diccionario de abreviaturas comunes en español
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
# Lematizar las palabras clave del diccionario
def lematizar_palabras(palabras):
    return [token.lemma_ for token in nlp(" ".join(palabras))]

PALABRAS_CLAVE_LEMATIZADAS = {
    categoria: lematizar_palabras(palabras)
    for categoria, palabras in PALABRAS_CLAVE.items()
}


# Cargar stopwords en español como lista
stop_words_es = list(stopwords.words('spanish'))
# Agregar stopwords adicionales específicas del dominio
stop_words_adicionales = [
    'rt', 'https', 'http', 'tco', 'twitter', 'com',  # URLs y términos de Twitter
    't', 'co',  # Partes de 't.co'
    'q', 'k', 'd', 'tb', 'tmb', 'pq', 'xq', 'dnd', 'kien', 'salu2', 'aki', 'tqm'  # Abreviaturas comunes
]
stop_words_es.extend(stop_words_adicionales)
# Eliminar duplicados y ordenar
stop_words_es = sorted(list(set(stop_words_es)))

# =============================================================================
# Funciones de preprocesamiento de texto
# =============================================================================
def limpiar_texto(texto):
    """Limpia y normaliza el texto."""
    if pd.isnull(texto):
        return ""
    try:
        texto = texto.encode("latin1").decode("utf-8")
    except:
        pass
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')
    return texto.lower()

# Expandir abreviaturas
def expandir_abreviaturas(texto):
    palabras = texto.split()
    palabras_expandidas = [ABREVIATURAS.get(p, p) for p in palabras]
    return ' '.join(palabras_expandidas)

# Extraer hashtags y quitar del texto
def procesar_hashtags(texto):
    """Extrae hashtags y los elimina del texto."""
    hashtags = re.findall(r"#(\w+)", texto)
    texto_sin_hashtags = re.sub(r"#\w+", "", texto)
    return texto_sin_hashtags.strip(), hashtags

# Limpiar texto de urls, menciones y símbolos
def limpieza_final(texto):
    """Limpia el texto de URLs, menciones y símbolos innecesarios."""
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"@\w+", "", texto)
    texto = re.sub(r"&", "y", texto)
    #texto = re.sub(r"[^a-zA-Z0-9\s]", "", texto)
    texto = re.sub(r"[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

# Tokenización y lematización en español usando spaCy
def tokenizar_y_lematizar(texto):
    """Tokeniza y lematiza el texto usando spaCy."""
    doc = nlp(texto)
    # Filtrar tokens que no son stopwords, puntuación o espacios
    tokens = [token.lemma_.lower() for token in doc 
              if not token.is_stop and not token.is_punct and not token.is_space]
    return " ".join(tokens)

# Función de preprocesamiento completo
def preprocesar_texto_completo(texto):
    """Aplica todo el pipeline de preprocesamiento de texto."""
    texto = limpiar_texto(texto)
    texto = expandir_abreviaturas(texto)
    texto = limpieza_final(texto)
    texto = tokenizar_y_lematizar(texto)
    return texto

# Métricas estilísticas
def calcular_metricas_estilisticas(texto):
    """Calcula métricas estilísticas del texto."""
    palabras = texto.split()
    caracteres = len(texto)
    
    return {
        'longitud_texto': caracteres,
        'num_palabras': len(palabras)
    }

# Análisis de frecuencia de palabras clave
def analizar_palabras_clave(texto):
    """Analiza frecuencia de palabras clave en texto lematizado completo (texto + hashtags)."""
    frecuencias = {}
    for categoria, palabras in PALABRAS_CLAVE_LEMATIZADAS.items():
        frecuencias[categoria] = sum(1 for palabra in palabras if palabra in texto)
    return frecuencias


# Función para traducir solo cuando sea necesario (para análisis de sentimiento)
def traducir_si_necesario(texto, target_lang='en'):
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(texto)
    except Exception as e:
        print(f"Error en traducción: {e}")
        return texto

# Análisis de sentimiento mejorado
def analizar_sentimiento(texto):
    """Analiza el sentimiento del texto."""
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

# Filtrar los hashtags por fila
def obtener_hashtags_frecuentes_individuales(hashtags_fila, hashtags_validos):
    return " ".join([tag for tag in hashtags_fila if tag in hashtags_validos])

# =============================================================================
# Funciones de extracción de características
# =============================================================================
def extraer_caracteristicas(tweet):
    """Extrae todas las características de un tweet."""
    texto_original, hashtags = procesar_hashtags(tweet)
    
    # Preprocesamiento inicial sin lematizar
    texto_limpio = limpiar_texto(texto_original)
    texto_expandido = expandir_abreviaturas(texto_limpio)
    texto_limpio_final = limpieza_final(texto_expandido)
    
    # Análisis de sentimiento sobre texto limpio expandido
    sentimiento = analizar_sentimiento(texto_limpio_final)

    # Lematización
    texto_lematizado = tokenizar_y_lematizar(texto_limpio_final)
    
    # Preparar texto completo con hashtags sin #
    texto_completo = texto_lematizado + " " + " ".join(hashtags)
    
    # Análisis de palabras clave sobre texto completo
    palabras_clave = analizar_palabras_clave(texto_completo)
    
    return {
        "tweet_text": texto_lematizado,
        "hashtags": hashtags,
        "texto_completo": texto_completo,
        "texto_bert": texto_limpio_final + " " + " ".join(hashtags),
        "longitud_texto": len(texto_lematizado),
        "num_palabras": len(texto_lematizado.split()),
        **palabras_clave,
        **sentimiento
    }

# =============================================================================
# Pipeline principal
# =============================================================================
def main():
    # 1. Cargar datos
    print("Cargando datos...")
    df = pd.read_csv("data_train.csv", encoding="latin1", header=0)
    
    # 2. Preprocesamiento inicial
    print("Preprocesando texto...")
    df["tweet_text"] = df["tweet_text"].fillna("")
    
    # 3. Extraer características iniciales
    print("Extrayendo características iniciales...")
    features_df = df["tweet_text"].swifter.apply(extraer_caracteristicas).apply(pd.Series)
    
    # Reemplazar columnas duplicadas
    df["tweet_text"] = features_df["tweet_text"]
    df["hashtags"] = features_df["hashtags"]
    
    # Agregar columnas nuevas
    columnas_nuevas = features_df.drop(columns=["tweet_text", "hashtags"])
    df = pd.concat([df, columnas_nuevas], axis=1)
    
    # 4. Procesar hashtags
    print("Procesando hashtags...")
    mlb = MultiLabelBinarizer()
    hashtags_df = pd.DataFrame(mlb.fit_transform(df["hashtags"]), 
                              columns=[f"tag_{tag}" for tag in mlb.classes_])
    
    # Filtrar hashtags frecuentes
    umbral = 10
    hashtags_frecuentes = hashtags_df.columns[hashtags_df.sum() >= umbral]
    hashtags_frecuentes_df = hashtags_df[hashtags_frecuentes]
    
    # 5. Preparar texto completo
    umbral_bajo = 5
    hashtags_vectorizacion = hashtags_df.columns[hashtags_df.sum() >= umbral_bajo]
    hashtags_validos = {col.replace('tag_', '') for col in hashtags_vectorizacion}
    df["hashtags_frecuentes_bajos"] = df["hashtags"].apply(lambda h: obtener_hashtags_frecuentes_individuales(h, hashtags_validos))
    df["texto_completo"] = df["tweet_text"] + " " + df["hashtags_frecuentes_bajos"]
    
    # 6. Vectorización TF-IDF
    print("Aplicando vectorización TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1,3),
        stop_words=stop_words_es,
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
        norm='l2'
    )
    X_tfidf = vectorizer.fit_transform(df["texto_completo"])
    y = df["class"]
    
    # Imprimir información sobre las características
    print("\nInformación sobre las características TF-IDF:")
    print(f"Número total de características: {X_tfidf.shape[1]}")
    print(f"Número de muestras: {X_tfidf.shape[0]}")
    print("\nTop 10 términos más importantes:")
    feature_names = vectorizer.get_feature_names_out()
    idf_values = vectorizer.idf_
    top_terms = sorted(zip(feature_names, idf_values), key=lambda x: x[1], reverse=True)[:10]
    for term, idf in top_terms:
        print(f"{term}: {idf:.2f}")
    
    # 7. Crear DataFrame final
    print("Creando DataFrame final...")
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=[f"tfidf_{i}" for i in range(X_tfidf.shape[1])])

    # Seleccionar columnas para el DataFrame final
    columnas_base = ["tweet_id", "tweet_text", "texto_completo", "texto_bert"]
    columnas_metricas = ["longitud_texto", "num_palabras"]
    columnas_palabras_clave = ["comida", "restriccion", "purga", "imagen_corporal", "ejercicio"]
    columnas_sentimiento = ["polaridad", "subjetividad"]
    
    # Crear el DataFrame final sin la columna class
    df_final = pd.concat([
        df[columnas_base],
        df[columnas_metricas],
        df[columnas_palabras_clave],
        df[columnas_sentimiento],
        hashtags_frecuentes_df,
        tfidf_df
    ], axis=1)

    #8. Normalizar las características
    columnas_a_escalar = ["longitud_texto", "num_palabras", "polaridad", "subjetividad"]
    scaler = StandardScaler()
    df_final[columnas_a_escalar] = scaler.fit_transform(df_final[columnas_a_escalar])
    
    # Convertir etiquetas a valores numéricos y agregar al final
    df_final['class'] = df['class'].map({'control': 0, 'anorexia': 1})
    
    # 9. Guardar resultados
    print("Guardando resultados...")
    df_final.to_csv("NO_USAR_tweets_procesados.csv", index=False, encoding="utf-8")

    # 10. Guardar resultados para los modelos tradicionales
    print("Guardando resultados para modelos tradicionales...")
    ds_tradicional = df_final.drop(columns=["tweet_id", "tweet_text", "texto_completo", "texto_bert"])
    ds_tradicional.to_csv("ds_tradicional.csv", index=False, encoding="utf-8")
    
    # Imprimir información final
    print("\nDataset final guardado con las siguientes columnas:")
    print("\nColumnas base:", columnas_base)
    print("\nMétricas estilísticas:", columnas_metricas)
    print("\nPalabras clave:", columnas_palabras_clave)
    print("\nAnálisis de sentimiento:", columnas_sentimiento)
    print("\nHashtags frecuentes:", list(hashtags_frecuentes))
    print("\nTotal de características TF-IDF:", X_tfidf.shape[1])
    print("\nDistribución de clases:")
    print(df_final['class'].value_counts())

if __name__ == "__main__":
    main()