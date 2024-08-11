import re
from itertools import combinations
import nltk

def limpieza_de_textos(limpieza, stop_words):
    textos_limpios = []
    for texto in limpieza:
        # minimizar el texto
        texto = texto.lower()
        # eliminar numeros
        texto = re.sub(r'\b\w*\d\w*\b', '', texto)
        # eliminar caracteres especiales
        texto = re.sub(r'[^\w\s]', '', texto)
        # eliminar tildes
        texto = re.sub(r'[áéíóúÁÉÍÓÚ]', lambda x: 'aeiouAEIOU'['áéíóúÁÉÍÓÚ'.index(x.group(0))], texto)
        # eliminar stopwords   
        texto = ' '.join([word for word in texto.split() if word not in stop_words])
        # eliminar espacios dobles
        texto = re.sub(r'\s+', ' ', texto).strip()
        textos_limpios.append(texto)
    return textos_limpios

def Generacion_de_skipgramas(text, n_palabras, k_saltos):
    '''Genera skipgrams de un texto dado con un tamaño de ventana n y un número de skips k.'''
    # Tokenize words
    words = nltk.word_tokenize(text)
    
    # Initialize list to store skipgrams
    skipgrams_list = []
    
    # Generate skipgrams
    for i in range(len(words)):
        # Create combinations of words with skips
        skipgrams = combinations(words[i:i+n_palabras+k_saltos], n_palabras)
        skipgrams_list.extend(skipgrams)
    
    return skipgrams_list
