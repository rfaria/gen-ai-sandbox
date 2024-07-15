import psycopg2
import pandas as pd
import os
import json
import openai
import sys
import time
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()


def prepare_finetuning_data():
    conn_string = f"dbname='{os.getenv('dbname')}' user='{os.getenv('user')}' password='{os.getenv('password')}' host='{os.getenv('host')}' port='{os.getenv('port')}'"
    conn = psycopg2.connect(conn_string)
    query = '''
        SELECT 
            summary_column 
        FROM 
            public.sample 
        LIMIT 200
        '''
    df = pd.read_sql(query, conn)
    conn.close()

    # Prepare data in the chat format required for fine-tuning
    data = []
    for _, row in df.iterrows():
        message = f"Considere os dados de funis de branding por amostras estratificadas:\n{row['summary_column']}\n"
        data.append({
            "messages": [
                {"role": "system", "content": "Você é um assistente útil."},
                {"role": "user", "content": message},
                {"role": "assistant", "content": row['summary_column']}
            ]
        })

    # Save to JSONL
    output_path = os.path.join(os.path.dirname(__file__), 'fine_tuning_data.jsonl')
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def train():
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Criação do arquivo de treino no OpenAI
    response = openai.File.create(
        file=open("fine_tuning_data.jsonl", "rb"),
        purpose='fine-tune'
    )

    training_file_id = response['id']

    # Criação do job de fine-tuning
    fine_tune_response = openai.FineTuningJob.create(
        training_file=training_file_id,
        model="gpt-3.5-turbo",  # Use o modelo desejado para fine-tuning
        hyperparameters={
            "n_epochs": 10,
            "batch_size": 3,
            "learning_rate_multiplier": 0.3
        }
    )

    print(f'training_file_id:{training_file_id}')
    print(f'fine_tune_response:{fine_tune_response}')
    return fine_tune_response


def check_fine_tuning_status(fine_tune_id):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.FineTuningJob.retrieve(fine_tune_id)
    return response


def wait_for_fine_tuning(fine_tune_id, interval=60):
    while True:
        status = check_fine_tuning_status(fine_tune_id)
        print(f'Status: {status["status"]}')
        if status['status'] in ['succeeded', 'failed']:
            return status
        time.sleep(interval)


def executeFineTunedGPT(question, model_id):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.ChatCompletion.create(
        model=model_id,
        messages=[
            {"role": "user", "content": question},
            # {"role": "assistant", "content": ""}
        ],
        top_p=0.1,
    )

    return response['choices'][0]['message']['content']



if __name__ == '__main__':
    prepare_finetuning_data()
    train()
