"""Nesse exemplo o usuário faz a chamada ao modelo que decide se busca informações de acordo com a query,
porém as informações serão buscadas em uma API (blackbox), com o resultado da busca das informações o modelo
é chamado novamente para dar uma resposta personalizada"""

import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(base_url="https://api.groq.com/openai/v1")

# API de busca externa, precisa estar rodando para funcionar
# Nessa abordagem é uma caixa preta, não sabemos qual tipo de busca ela realiza
def search_kb(query: str):
    response = requests.post(
        "http://localhost:8000/search", json={"query": query, "limit": 3}
    )
    return response.json()

# Definição da Tool para obter informações de ações
tools = [
    {
        "type": "function",
        "name": "search_kb",
        "description": "Busca informações na base de conhecimento para responder perguntas",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A pergunta do usuário"},
            },
            "required": ["query"],
        },
    },
]

# Lista de mensagens de entrada para o modelo, incluindo a pergunta do usuário
input_list = [{"role": "user", "content": "what are AAPL main financial risks?"}]

# Primeira chamada ao modelo para obter informações da busca
response = client.responses.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    tools=tools,
    input=input_list,
)

# Guarda a resposta na memória
input_list += response.output

# Verifica se a Tool foi chamada e faz append na input_list
for item in response.output:
    if item.type == "function_call":
        args = json.loads(item.arguments)
        result = search_kb(**args)

        texts = [r["text"] for r in result["results"]]

        input_list.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps({"results": texts}, ensure_ascii=False),
            }
        )

# Segunda chamada ao modelo para obter uma análise baseada nos dados retornados pela função
final_response = client.responses.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    instructions="Responda à pergunta do usuário usando as informações retornadas pela busca.",
    tools=tools,
    input=input_list,
)

print(final_response.output_text)