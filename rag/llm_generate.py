import openai
from retrieve import retrieve_context

# 🔑 Set your API key
openai.api_key = "YOUR_OPENAI_API_KEY"

def generate_answer(question: str) -> str:
    context = retrieve_context(question)

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()
