from openai import OpenAI

from retriever import config

client = OpenAI(api_key=config.OPENAI_API_KEY)
resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input="hello world")
print(f"OK: {config.EMBEDDING_MODEL} returned {len(resp.data[0].embedding)} dimensions")
