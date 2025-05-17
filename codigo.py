
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

nltk.download('stopwords')

tokenizer = ToktokTokenizer()

def tratamento_dados(texto):
    texto = str(texto).lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    tokens = tokenizer.tokenize(texto)
    stop_words = set(stopwords.words('portuguese'))
    tokens = [palavra for palavra in tokens if palavra not in stop_words]
    return ' '.join(tokens)

arquivo = 'Symptom2Disease_translated.csv'
df = pd.read_csv(arquivo, on_bad_lines='skip')

df_split = df['Unnamed: 0'].str.extract(r'(\d+),\s*([^,]+),\s*"(.*)"')
df_split.columns = ['ID', 'Doenca', 'Sintomas']
df_split.dropna(inplace=True)

df_split['sintomas_tratados'] = df_split['Sintomas'].apply(tratamento_dados)

def recomenda_top_doencas(descricao_usuario, df_base, top_n=3):
    consulta_tratada = tratamento_dados(descricao_usuario)
    corpus = df_base['sintomas_tratados'].tolist() + [consulta_tratada]
    tfidf = TfidfVectorizer()
    tfidf_matriz = tfidf.fit_transform(corpus)
    cosine_sim = cosine_similarity(tfidf_matriz[-1], tfidf_matriz[:-1]).flatten()

    resultados_temp = []
    for idx, score in enumerate(cosine_sim):
        doenca = df_base.iloc[idx]['Doenca']
        resultados_temp.append((doenca, score))

    resultados_dict = {}
    for doenca, score in resultados_temp:
        if doenca not in resultados_dict or score > resultados_dict[doenca]:
            resultados_dict[doenca] = score

    resultados_ordenados = sorted(resultados_dict.items(), key=lambda x: x[1], reverse=True)

    resultados_finais = [(doenca, sim * 100) for doenca, sim in resultados_ordenados[:top_n]]
    return resultados_finais




def botao_recomenda():
    descricao = entry.get()
    if not descricao.strip():
        messagebox.showwarning("Erro", "Por favor, insira os sintomas.")
        return
    resultados = recomenda_top_doencas(descricao, df_split, top_n=3)
    texto_resultado = "\n".join([f"{doenca}: {sim:.2f}%" for doenca, sim in resultados])
    messagebox.showinfo("Diagnóstico Possível", f"Doenças mais prováveis:\n\n{texto_resultado}")


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
