"""Nesse exemplo fazemos duas chamadas ao modelo, sendo a primeira para obter informações de uma ação
e a segunda para obter uma análise baseada nessas informações. É importante notar que o modelo vai 
decidir se chama a Tool ou não baseado na pergunta do usuário. 
Permite consultar dados externos atuais com a chamada da Tool
"""

import json

import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(base_url="https://api.groq.com/openai/v1")

# Função para obter informações de uma ação usando yfinance
def get_stock(ticker: str):
    stock = yf.Ticker(ticker)
    info = stock.info
    output = {
        "ticker": ticker,
        "company_name": info.get("shortName", ticker),
        "current_price": info.get("currentPrice", 0),
    }
    return json.dumps(output)

# Definição da Tool para obter informações de ações
tools = [
    {
        "type": "function",
        "name": "get_stock",
        "description": "Retorna informações básicas de uma ação.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Símbolo da ação (ex: AAPL, NVDA)",
                },
            },
            "required": ["ticker"],
        },
    },
]

# Lista de mensagens de entrada para o modelo, incluindo a pergunta do usuário
input_list = [{"role": "user", "content": "Qual o preço da ação da Apple?"}]

# Primeira chamada ao modelo para obter informações da ação
response = client.responses.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    tools=tools,
    input=input_list,
)

response.model_dump()

# Processa a resposta do modelo para verificar se ele fez uma chamada à Tool e obter o resultado
for item in response.output:
    if item.type == "function_call":
        args = json.loads(item.arguments)
        result = get_stock(**args)
        input_list.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": result,
            }
        )

# A resposta da Tool é adicionada à lista de mensagens de entrada para a próxima chamada ao modelo
input_list[1]["output"]

# Segunda chamada ao modelo para obter uma análise baseada nos dados retornados pela função
final_response = client.responses.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    instructions="Responsa com uma análise baseada nos dados retornados pela função.",
    tools=tools,
    input=input_list,
)

print(final_response.output_text)