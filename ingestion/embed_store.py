from endee import Endee
from sentence_transformers import SentenceTransformer

# Initialize Endee
endee = Endee(index_name="documents")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load document
with open("data/sample.txt", "r") as f:
    text = f.read()

# Simple chunking
chunks = text.split(".")

for chunk in chunks:
    if len(chunk.strip()) > 20:
        vector = model.encode(chunk).tolist()

        endee.add(
            vector=vector,
            metadata={"text": chunk.strip()}
        )

print("✅ Documents embedded and stored in Endee")
