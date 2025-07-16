import csv
import re
import ollama
from multiprocessing import Pool, cpu_count

OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL_NAME = 'en2en:latest'
SIZE_THRESHOLD = 1.5 # Se o texto corrigido for 1.5x maior, marcar como 'toCheck=yes'

# Limpa o texto
def clean_csv_text(text):
    text = re.sub(r'["\'|]', '', text)
    text = re.sub(r'[\r\n]', ' ', text)
    text = re.sub(r'^The correct .* is:', '', text)
    return text

# Corrige erros gramaticais em inglês
def fix_grammar(text):
    try:
        text = clean_csv_text(text)
        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': f'Fix the grammar: {text}'}])
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
    corrected = fix_grammar(original)
    to_check = 'yes' if len(corrected) > len(original) * SIZE_THRESHOLD else 'no'

    return {
        'Label': label,
        'EmailText': corrected,
        'toCheck': to_check
    }

# Processamento
def process_grammar(input_path, output_path, debug=False, num_workers=None):

    with open(input_path, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)

        if debug:
            print(f'Processando no modo debug...')
            with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
                fieldnames = ['Label', 'EmailText', 'toCheck']
                writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='|')
                writer.writeheader()
                
                for row in reader:
                    processed_row = process_row(row)
                    writer.writerow(processed_row)

        else:
            if num_workers is None:
                num_workers = max(cpu_count() - 1, 1)

            print(f'Processando com {num_workers} processos...')

            with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
                fieldnames = ['Label', 'EmailText', 'toCheck']
                writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='|')
                writer.writeheader()

                with Pool(processes=num_workers) as pool:
                    for processed_row in pool.imap_unordered(process_row, reader):
                        writer.writerow(processed_row)

    print(f'Arquivo salvo em: {output_path}')



