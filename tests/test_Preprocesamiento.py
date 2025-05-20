# --------------------------------------------------
#
# Autor: Pablo Spínola López
# Description: Archivo de pruebas de la fase de preprocesamiento con cobertura del 90%.
# 
# --------------------------------------------------


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from funciones.funcionesPreprocesamiento import *

""" Pruebas unitarias para las funciones de preprocesamiento """

###################################################################
# Pruebas unitarias para la primera función: limpiar_texto(string)#
###################################################################

def test_limpiar_texto_normal():
    texto = "¡Hola! ¿Cómo estás? Bien, ¿y tú?"
    resultado = limpiar_texto(texto)
    assert resultado == "hola! como estas? bien, y tu?", f"Resultado inesperado: {resultado}"

def test_limpiar_texto_con_nulo():
    resultado = limpiar_texto(None)
    assert resultado == "", f"Esperado '', pero se obtuvo: {resultado}"

def test_limpiar_texto_con_caracteres_especiales():
    texto = "niño con café golpea piñata"
    resultado = limpiar_texto(texto)
    assert resultado == "nino con cafe golpea pinata"

###########################################################################
# Pruebas unitarias para la segunda función: expandir_abreviaturas(string)#
###########################################################################

def test_expandir_abreviaturas_conocidas():
    texto = "q haces xq no vienes tmb"
    resultado = expandir_abreviaturas(texto)
    assert resultado == "que haces porque no vienes también"

def test_expandir_abreviaturas_mixto():
    texto = "hola q tal"
    resultado = expandir_abreviaturas(texto)
    assert resultado == "hola que tal"

def test_expandir_abreviaturas_sin_abreviaturas():
    texto = "hola como estas"
    resultado = expandir_abreviaturas(texto)
    assert resultado == "hola como estas"

#######################################################################
# Pruebas unitarias para la tercera función: procesar_hashtags(string)#
#######################################################################

def test_procesar_hashtags_multiples():
    texto = "Amo el #cafe y la #vida"
    texto_limpio, hashtags = procesar_hashtags(texto)
    assert texto_limpio == "Amo el  y la"
    assert hashtags == ["cafe", "vida"]

def test_procesar_hashtags_sin_hashtags():
    texto = "Buenos días profe, ¿Cómo está?"
    texto_limpio, hashtags = procesar_hashtags(texto)
    assert texto_limpio == "Buenos días profe, ¿Cómo está?"
    assert hashtags == []

def test_procesar_hashtags_unico():
    texto = "#yosoypablo"
    texto_limpio, hashtags = procesar_hashtags(texto)
    assert texto_limpio == ""
    assert hashtags == ["yosoypablo"]

###################################################################
# Pruebas unitarias para la cuarta función: limpieza_final(string)#
###################################################################

def test_limpieza_final_urls_y_menciones():
    texto = "Visita http://ejemplo.com y menciona a @usuario"
    resultado = limpieza_final(texto)
    assert resultado == "Visita y menciona a", f"Resultado inesperado: {resultado}"

def test_limpieza_final_simbolos():
    texto = "Hola & bienvenid@! #fiesta :)"
    resultado = limpieza_final(texto)
    assert resultado == "Hola y bienvenid fiesta", f"Resultado inesperado: {resultado}"

def test_limpieza_final_espacios_extra():
    texto = "  Esto    tiene   muchos   espacios   "
    resultado = limpieza_final(texto)
    assert resultado == "Esto tiene muchos espacios"

##########################################################################
# Pruebas unitarias para la quinta función: tokenizar_y_lematizar(string)#
##########################################################################

def test_tokenizar_y_lematizar_basico():
    texto = "Los gatos están durmiendo en la casa"
    resultado = tokenizar_y_lematizar(texto)
    assert "gato" in resultado and "dormir" in resultado, f"Lematización incorrecta: {resultado}"

def test_tokenizar_y_lematizar_con_stopwords():
    texto = "Este es un texto de ejemplo para probar"
    resultado = tokenizar_y_lematizar(texto)
    assert "este" not in resultado and "es" not in resultado, f"Stopwords no filtradas: {resultado}"

def test_tokenizar_y_lematizar_vacio():
    texto = ""
    resultado = tokenizar_y_lematizar(texto)
    assert resultado == "", f"Esperado '', obtenido: {resultado}"

####################################################################################
# Pruebas unitarias para la sexta función: calcular_metricas_estilisticas(string)#
####################################################################################

def test_metricas_texto_normal():
    texto = "Este es un ejemplo de texto."
    resultado = calcular_metricas_estilisticas(texto)
    assert resultado['num_palabras'] == 6, f"Número de palabras incorrecto: {resultado['num_palabras']}"
    assert resultado['longitud_texto'] == len(texto), f"Longitud incorrecta: {resultado['longitud_texto']}"

def test_metricas_texto_vacio():
    texto = ""
    resultado = calcular_metricas_estilisticas(texto)
    assert resultado['num_palabras'] == 0
    assert resultado['longitud_texto'] == 0

def test_metricas_texto_con_espacios():
    texto = "   Palabra    otra   más   "
    resultado = calcular_metricas_estilisticas(texto)
    assert resultado['num_palabras'] == 3

############################################################################
# Pruebas unitarias para la séptima función: analizar_palabras_clave(string)#
############################################################################

def test_analizar_palabras_clave_presencia():
    texto = "me voy a saltar la comida y provocar el vómito"
    resultado = analizar_palabras_clave(texto)
    assert resultado['comida'] == 1
    assert resultado['restriccion'] == 2
    assert resultado['purga'] == 1

def test_analizar_palabras_clave_ausencia():
    texto = "no hay coincidencias"
    resultado = analizar_palabras_clave(texto)
    assert resultado['comida'] == 0
    assert resultado['restriccion'] == 1
    assert resultado['purga'] == 0
    assert resultado['imagen_corporal'] == 0
    assert resultado['ejercicio'] == 0

##########################################################################
# Pruebas unitarias para la octava función: traducir_si_necesario(string)#
##########################################################################

def test_traducir_texto_espanol_a_ingles():
    texto = "Estoy feliz"
    resultado = traducir_si_necesario(texto)
    assert isinstance(resultado, str)
    assert "happy" in resultado.lower()

def test_traducir_texto_ingles_a_ingles():
    texto = "I am happy"
    resultado = traducir_si_necesario(texto)
    assert isinstance(resultado, str)
    assert "happy" in resultado.lower()

#########################################################################
# Pruebas unitarias para la novena función: analizar_sentimiento(string)#
#########################################################################

def test_analizar_sentimiento_positivo():
    texto = "Estoy muy feliz con mi cuerpo"
    resultado = analizar_sentimiento(texto)
    assert resultado['polaridad'] > 0, f"Se esperaba sentimiento positivo: {resultado}"

def test_analizar_sentimiento_negativo():
    texto = "Esto fue una pérdida de tiempo, muy mal"
    resultado = analizar_sentimiento(texto)
    assert resultado['polaridad'] < 0, f"Se esperaba sentimiento negativo: {resultado}"

##########################################################################################################
# Pruebas unitarias para la décima función: obtener_hashtags_frecuentes_individuales([string], [string])#
##########################################################################################################

def test_obtener_hashtags_frecuentes_individuales():
    hashtags_fila = ["felicidad", "amor", "vida", "random"]
    hashtags_validos = {"felicidad", "vida", "salud"}
    
    resultado = obtener_hashtags_frecuentes_individuales(hashtags_fila, hashtags_validos)
    assert resultado == "felicidad vida", f"Resultado inesperado: {resultado}"

def test_obtener_hashtags_sin_validos():
    hashtags_fila = ["invalido1", "invalido2"]
    hashtags_validos = {"felicidad", "vida"}
    
    resultado = obtener_hashtags_frecuentes_individuales(hashtags_fila, hashtags_validos)
    assert resultado == "", f"Esperado '', pero se obtuvo: {resultado}"

def test_obtener_hashtags_todos_validos():
    hashtags_fila = ["vida", "felicidad"]
    hashtags_validos = {"vida", "felicidad"}
    
    resultado = obtener_hashtags_frecuentes_individuales(hashtags_fila, hashtags_validos)
    assert resultado == "vida felicidad"

def test_obtener_hashtags_lista_vacia():
    hashtags_fila = []
    hashtags_validos = {"vida", "salud"}
    
    resultado = obtener_hashtags_frecuentes_individuales(hashtags_fila, hashtags_validos)
    assert resultado == ""

############################################################################
# Pruebas unitarias para la décima función: extraer_caracteristicas(string)#
############################################################################

def test_extraer_caracteristicas_basico():
    tweet = "Estoy en el hospital y me siento bien. #salud #bienestar"
    resultado = extraer_caracteristicas(tweet)
    
    assert "tweet_text" in resultado
    assert "hashtags" in resultado
    assert "texto_completo" in resultado
    assert "texto_bert" in resultado
    assert isinstance(resultado["hashtags"], list)
    assert "salud" in resultado["texto_completo"]
    assert isinstance(resultado["polaridad"], float)

def test_extraer_caracteristicas_sin_hashtags():
    tweet = "Nada que ver aquí, solo texto normal."
    resultado = extraer_caracteristicas(tweet)
    
    assert resultado["hashtags"] == []
    assert "texto_completo" in resultado
    assert isinstance(resultado["num_palabras"], int)

def test_extraer_caracteristicas_vacio():
    tweet = ""
    resultado = extraer_caracteristicas(tweet)
    assert resultado["tweet_text"] == ""
    assert resultado["hashtags"] == []
    assert resultado["num_palabras"] == 0
    assert resultado["polaridad"] == 0
