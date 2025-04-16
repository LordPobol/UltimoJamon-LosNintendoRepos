@echo off

echo Instalando librerías desde requirements.txt...
pip install -r requirements.txt

echo Descargando recursos de nltk...
python -m nltk.downloader all

echo Descargando modelo de spaCy...
python -m spacy download es_core_news_sm

echo Listo :D
pause
