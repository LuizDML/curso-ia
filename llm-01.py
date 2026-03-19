from groq import Groq

client = Groq()

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role":"system","content":"Atue como um especialista em machine learning."},
        {"role":"user","content":"De forma simples, o que é machine learning?"}
    ],
    temperature=1,
    top_p=1
)

print(response.choices[0].message.content)