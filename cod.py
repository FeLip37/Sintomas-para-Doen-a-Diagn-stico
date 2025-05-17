# Instale as bibliotecas necessárias, se ainda não tiver:
# pip install pandas numpy scikit-learn nltk

import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox

# Baixando recursos necessários
nltk.download('stopwords')

# Inicializando tokenizador
tokenizer = ToktokTokenizer()

# Função para limpar e tratar os dados de texto
def tratamento_dados(texto):
    texto = str(texto).lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    tokens = tokenizer.tokenize(texto)
    stop_words = set(stopwords.words('portuguese'))
    tokens = [palavra for palavra in tokens if palavra not in stop_words]
    return ' '.join(tokens)

# Lendo o arquivo CSV (ajuste o nome se necessário)
arquivo = 'Symptom2Disease_translated5.csv'
df = pd.read_csv(arquivo, on_bad_lines='skip')

# Extraindo colunas a partir de texto no campo 'Unnamed: 0'
df_split = df['Unnamed: 0'].str.extract(r'(\d+),\s*([^,]+),\s*"(.*)"')
df_split.columns = ['ID', 'Doenca', 'Sintomas']
df_split.dropna(inplace=True)

# Aplicando pré-processamento nos sintomas
df_split['sintomas_tratados'] = df_split['Sintomas'].apply(tratamento_dados)

# Função de recomendação com base na descrição do usuário
def recomenda_doenca(descricao_usuario, df_base):
    consulta_tratada = tratamento_dados(descricao_usuario)
    corpus = df_base['sintomas_tratados'].tolist() + [consulta_tratada]
    tfidf = TfidfVectorizer()
    tfidf_matriz = tfidf.fit_transform(corpus)
    cosine_sim = cosine_similarity(tfidf_matriz[-1], tfidf_matriz[:-1])
    indice_mais_proximo = np.argmax(cosine_sim)
    doenca_recomendada = df_base['Doenca'].iloc[indice_mais_proximo]
    similaridade = cosine_sim[0, indice_mais_proximo] * 100  # porcentagem
    return doenca_recomendada, similaridade


# Interface gráfica (GUI) com Tkinter
def botao_recomenda():
    descricao = entry.get()
    if not descricao.strip():
        messagebox.showwarning("Erro", "Por favor, insira os sintomas.")
        return
    doenca, similaridade = recomenda_doenca(descricao, df_split)
    messagebox.showinfo(
        "Diagnóstico Possível",
        f"Doença recomendada: {doenca}\nSimilaridade: {similaridade:.2f}%"
    )


# Construção da janela Tkinter
root = tk.Tk()
root.title("Sistema de Diagnóstico por Sintomas")
root.geometry('500x220')

label = tk.Label(root, text="Descreva seus sintomas (em português):")
label.pack(pady=10)

entry = tk.Entry(root, width=70)
entry.pack(pady=10)

botao = tk.Button(root, text="Diagnosticar", command=botao_recomenda)
botao.pack(pady=10)

root.mainloop()
