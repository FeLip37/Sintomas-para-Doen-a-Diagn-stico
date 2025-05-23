import os
from deep_translator import GoogleTranslator
import time
from langdetect import detect

def translate_file():
    input_file = "Symptom2Disease.csv"
    output_file = "Symptom2Disease_translated.csv"

    if not os.path.exists(input_file):
        print(f"Erro: {input_file} não encontrado!")
        return

    with open(input_file, encoding='utf-8') as infile:
        lines = infile.readlines()

    print(f"Traduzindo {len(lines)} linhas...")

    with open(output_file, 'w', encoding='utf-8-sig') as outfile:
        for idx, line in enumerate(lines):
            try:
                text = line.strip()
                if text == "":
                    translated_text = ""
                else:
                    detected_lang = detect(text)
                    if detected_lang == "en":
                        translated_text = GoogleTranslator(source='en', target='pt').translate(text)
                    else:
                        translated_text = text  # Já está em português ou outro idioma
                outfile.write(translated_text + '\n')
                if idx % 10 == 0:
                    print(f"{idx + 1}/{len(lines)} linhas processadas...")
            except Exception as e:
                print(f"Erro na linha {idx}: {e}")
                outfile.write(line)

            time.sleep(1)  # evitar sobrecarga no tradutor

    print(f"Tradução finalizada! Arquivo salvo como {output_file}")

translate_file()

