# doc-copilot

A multimodal voice document copilot, built stage by stage to learn the modern LLM stack
(RAG, LangGraph, MCP, VLMs, quantized local serving, LoRA). One PR per stage.

## Progress

| Stage | Topic | Status |
|---|---|---|
| 1 | Basic RAG (LlamaIndex + Ollama + Qdrant) | COMPLETED|
| 2 | Chunking + reranking, eval harness | |
| 3 | Prompt design, summarization | |
| 4 | LangGraph agent + tools | |
| 5 | Streaming chatbot API | |
| 6 | Guardrails | |
| 7 | Voice (STT/TTS) | |
| 8 | VLM for scanned pages | |
| 9 | MCP server | |
| 10 | CPU serving + quantization benchmarks | |
| 11 | LoRA fine-tune | |

## Quickstart (Stage 1)

```bash
# 1. Local LLM
ollama pull llama3.2:3b

# 2. Python env
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env

# 3. Add a few public PDFs to data/, then:
copilot ingest data/
copilot ask "What is the project completion period?"
```

Embedded Qdrant (on-disk) is used by default, so Docker is not required yet.
To use the server instead: `docker compose up -d` and set `QDRANT_URL=http://localhost:6333`.

## Learning notes
Experiments and findings per stage go in `notebooks/` and are summarized here.
