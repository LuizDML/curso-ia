from dotenv import load_dotenv
load_dotenv()     

import openai

cliente = openai.OpenAI(base_url="https://api.groq.com/openai/v1")

response = cliente.responses.create(
    model="llama-3.1-8b-instant",
    instructions="Responda de forma simples em apenas um paragrafo curto",
    input="O que é machine learning?",
    temperature= 0.5,
)

print(response.output_text)