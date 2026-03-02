from endee import Endee
from sentence_transformers import SentenceTransformer

endee = Endee(index_name="documents")
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_context(query: str, top_k: int = 5) -> str:
    query_vector = model.encode(query).tolist()

    results = endee.search(
        vector=query_vector,
        k=top_k
    )

    return "\n".join([r["metadata"]["text"] for r in results])
