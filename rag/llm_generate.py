from __future__ import annotations

import os
import re

try:
    from .retrieve import retrieve_context
except ImportError:
    from retrieve import retrieve_context


def _extractive_answer(question: str, context: str) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", context)
        if sentence.strip()
    ]
    if not sentences:
        return "I could not find relevant information in the available documents."

    question_terms = set(re.findall(r"\w+", question.lower()))
    scored = []
    for sentence in sentences:
        terms = set(re.findall(r"\w+", sentence.lower()))
        scored.append((len(question_terms & terms), sentence))
    scored.sort(key=lambda item: item[0], reverse=True)

    selected = [sentence for score, sentence in scored if score > 0][:2]
    if not selected:
        selected = [sentences[0]]
    return " ".join(selected)


def generate_answer(question: str) -> str:
    question = question.strip()
    if not question:
        return "Please enter a question."

    context = retrieve_context(question)
    if not context:
        return "I could not find relevant information in the available documents."

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _extractive_answer(question, context)

    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not present, say that it is not in the documents.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )

    try:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": "You are a precise document question-answering assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            answer = response.choices[0].message.content
        except ImportError:
            import openai

            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            answer = response.choices[0].message.content
        return answer.strip() if answer else _extractive_answer(question, context)
    except Exception as exc:
        raise RuntimeError(f"Unable to generate answer with OpenAI: {exc}") from exc
