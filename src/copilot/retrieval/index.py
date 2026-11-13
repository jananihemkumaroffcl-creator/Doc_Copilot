"""Stage 1 retrieval: build / load a vector index and answer questions with citations."""
import qdrant_client
from llama_index.core import (
    PromptTemplate,
    Settings as LISettings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore

from copilot.config import settings

QA_PROMPT = PromptTemplate(
    "You answer questions using ONLY the context below.\n"
    "If the context does not contain the answer, say exactly: "
    "\"I can't find that in the documents.\"\n"
    "---------------------\n{context_str}\n---------------------\n"
    "Question: {query_str}\nAnswer:"
)


def configure_models() -> None:
    LISettings.llm = Ollama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        request_timeout=180.0,
        temperature=0.1,
    )
    LISettings.embed_model = HuggingFaceEmbedding(model_name=settings.embed_model)
    LISettings.node_parser = SentenceSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )


def _vector_store() -> QdrantVectorStore:
    if settings.qdrant_url:
        client = qdrant_client.QdrantClient(url=settings.qdrant_url)
    else:
        client = qdrant_client.QdrantClient(path=settings.qdrant_path)
    return QdrantVectorStore(client=client, collection_name=settings.collection)


def build_index(documents) -> VectorStoreIndex:
    configure_models()
    storage = StorageContext.from_defaults(vector_store=_vector_store())
    return VectorStoreIndex.from_documents(
        documents, storage_context=storage, show_progress=True
    )


def load_index() -> VectorStoreIndex:
    configure_models()
    return VectorStoreIndex.from_vector_store(_vector_store())


def ask(index: VectorStoreIndex, question: str):
    engine = index.as_query_engine(
        similarity_top_k=settings.top_k, text_qa_template=QA_PROMPT
    )
    return engine.query(question)
