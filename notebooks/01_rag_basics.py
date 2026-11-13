# %% [markdown]
# # Stage 1: RAG basics, step by step
# Run cell by cell in VS Code (Jupyter extension). The goal is to SEE each
# stage of RAG before hiding it inside `copilot ask`.

# %% [markdown]
# ## 1. Embeddings: text -> vectors
# Similar meaning = nearby vectors. Cosine similarity measures "nearby".

# %%
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import numpy as np

embed = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
sentences = [
    "The contractor must complete work within 18 months.",
    "Project duration shall not exceed one and a half years.",
    "The cafeteria serves lunch at noon.",
]
vecs = np.array([embed.get_text_embedding(s) for s in sentences])
print("vector dim:", vecs.shape[1])

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

print("sentence 0 vs 1 (same meaning):", round(cos(vecs[0], vecs[1]), 3))
print("sentence 0 vs 2 (unrelated):   ", round(cos(vecs[0], vecs[2]), 3))

# %% [markdown]
# ## 2. Chunking: what does the splitter actually produce?
# Change chunk_size / chunk_overlap and re-run. Stage 2 turns this into a
# proper experiment.

# %%
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

import os
_data_dir = os.path.join(os.path.dirname(globals().get("__file__", "notebooks/placeholder")), "../data")
docs = SimpleDirectoryReader(_data_dir).load_data()
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
nodes = splitter.get_nodes_from_documents(docs)
print(len(docs), "pages ->", len(nodes), "chunks")
print(nodes[0].get_content()[:500])
print(nodes[0].metadata)

# %% [markdown]
# ## 3. Retrieval only (no LLM yet)
# Look at what comes back BEFORE the LLM touches it. Most RAG bugs are
# retrieval bugs, not generation bugs.

# %%
from llama_index.core import VectorStoreIndex, Settings

Settings.embed_model = embed
Settings.llm = None  # retrieval only
index = VectorStoreIndex(nodes, show_progress=True)
retriever = index.as_retriever(similarity_top_k=5)
for r in retriever.retrieve("What is the project completion period?"):
    print(round(r.score, 3), r.node.metadata.get("page_label"), r.node.get_content()[:120].replace("\n", " "))

# %% [markdown]
# ## 4. Generation with a local LLM
# Needs `ollama pull llama3.2:3b` first.

# %%
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="llama3.2:3b", request_timeout=180.0, temperature=0.1)
engine = index.as_query_engine(similarity_top_k=5)
print(engine.query("What is the project completion period?"))

# %% [markdown]
# ## Things to try (write what you observe in the README)
# 1. Ask something the documents do NOT contain. What does the model do?
# 2. Set chunk_size to 128, then 1024. How do the retrieved chunks change?
# 3. Rephrase one question three ways. Does retrieval stay stable?
