"""Stage 1 ingestion: a file or a folder of PDFs/text -> LlamaIndex Documents."""
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document


def load_documents(path: str | Path) -> list[Document]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    if path.is_file():
        reader = SimpleDirectoryReader(input_files=[str(path)])
    else:
        # One Document per PDF page, so page numbers survive into citations.
        reader = SimpleDirectoryReader(
            input_dir=str(path),
            required_exts=[".pdf", ".txt", ".md"],
            recursive=True,
        )
    return reader.load_data()