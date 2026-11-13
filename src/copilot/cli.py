"""CLI: `copilot ingest data/` then `copilot ask "your question"`."""
import typer
from rich.console import Console
from rich.table import Table

from copilot.ingestion.loader import load_documents
from copilot.retrieval.index import ask as ask_index
from copilot.retrieval.index import build_index, load_index

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def ingest(data_dir: str = typer.Argument("data")):
    """Parse, chunk, embed and store every document in DATA_DIR."""
    docs = load_documents(data_dir)
    console.print(f"Loaded [bold]{len(docs)}[/bold] pages/documents")
    build_index(docs)
    console.print("[green]Index built.[/green]")


@app.command()
def ask(question: str):
    """Ask a question and show the answer with its source chunks."""
    response = ask_index(load_index(), question)
    console.print(f"\n[bold]Answer:[/bold] {response}\n")

    table = Table(title="Sources")
    table.add_column("Score", justify="right")
    table.add_column("File")
    table.add_column("Page")
    table.add_column("Snippet")
    for n in response.source_nodes:
        meta = n.node.metadata
        table.add_row(
            f"{n.score:.3f}" if n.score is not None else "-",
            str(meta.get("file_name", "?")),
            str(meta.get("page_label", "-")),
            n.node.get_content()[:90].replace("\n", " ") + "...",
        )
    console.print(table)


if __name__ == "__main__":
    app()
