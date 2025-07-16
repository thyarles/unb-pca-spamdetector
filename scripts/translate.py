import csv
import re
import ollama
from tqdm import tqdm


OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL_NAME = 'lauchacarro/qwen2.5-translator'
SIZE_THRESHOLD = 1.5 # Se o texto corrigido for 1.5x maior, marcar como 'toCheck=yes'


# Limpa o texto
def clean_csv_text(text):
    text = re.sub(r'["|]', '', text)
    text = re.sub(r'[\r\n]', ' ', text)
    return text


# Corrige erros gramaticais em inglês
def translate(text):
    try:
        text = clean_csv_text(text)
        message = f'Translate from ENGLISH to PORTUGUESE: {text}'
        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': message}])
        corrected_text = response.get('message', {}).get('content', '').strip()
        corrected_text = clean_csv_text(corrected_text).strip()
        if not corrected_text:
            print('[ERROR] Não foi possível obter o texto corrigido.')
            return '[ERROR] Resposta inválida da API'
        return corrected_text
    except Exception as e:
        print(f'[ERROR] Erro ao corrigir o texto: {str(e)}')
        return f'[ERROR] {str(e)}'


# Verifica se o texto corrigido é muito maior que o original
def process_row(row):
    label = row['Label']
    original = row['EmailText']
    portuguese = translate(original)
    # to_check = True if (len(corrected) > len(original) * SIZE_THRESHOLD or len(corrected) < 9) else False
    # corrected_str = str(corrected)
    # if not to_check and 'gramma' in corrected_str.lower():
    #     to_check = True
    return {'Label': label, 'EmailText': original, 'EmailTextBR': portuguese}


# Processamento
def translation(input_path, output_path):

    # Definições
    fields = ['Label', 'EmailText', 'EmailTextBR']

    # Conta o número de linhas (exceto o cabeçalho)
    with open(input_path, newline='', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f) - 1

    # Cria cabeçalho da saída
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fields)
        writer.writeheader()
    
    # Processa linha por linha da entrada
    with open(input_path, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)        
        for row in tqdm(reader, total=total_lines, desc="Processando..."):
            processed_row = process_row(row)
            with open(output_path, 'a', newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fields)
                writer.writerow(processed_row)

    print(f'Arquivo salvo em: {output_path}')