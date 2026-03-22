from mem0 import MemoryClient # instalar direto do repositório git
from dotenv import load_dotenv

load_dotenv()

client = MemoryClient()

messages = [
   {
       "role": "user",
       "content": "Meu nome é Luiz e estou em transição de carreira para Ciência de Dados",
   },
   {
       "role": "assistant",
       "content": "Oi Luiz! Anotei que você está em transição de carreira para Ciência de Dados! Vou manter isso em mente para recomendações e discussões relacionadas.",
   },
]

client.add(messages, user_id="luiz")

query = "Qual o meu nome?"
response = client.search(query, filters={"user_id": "luiz"})
response
response["results"][0]["memory"]

# Maneira mais simples de adicionar a mensagem
client.add("Sou o Luiz e gosto de física!", user_id="luiz")