"""Запросы к PDF из командной строки, с общей историей в интерактивном режиме."""

from argparse import ArgumentParser
from os import getenv
from pathlib import Path

from experiments import PDF_PATH, make_models
from rag import PDFLoader, RAG, format_context, split_pages


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Вопрос на английском")
    parser.add_argument("--pdf", type=Path, default=PDF_PATH, help="PDF для поиска")
    args = parser.parse_args()
    base_url = getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    llm, encoder, generate_grounded = make_models(base_url)
    chunks = split_pages(PDFLoader(args.pdf).load(), size=1000, overlap=200)
    rag = RAG(encoder.index(chunks), llm.invoke, generate_grounded)

    def show(question, session=None):
        result = rag.ask(question, session=session)
        print(f"Вопрос: {result.question}\nЗапрос: {result.query}\n"
              f"{format_context(result.hits)}\n\nОтвет: {result.answer}")

    if args.question:
        show(args.question)
        return
    print("Введите вопрос. Команды /clear и /exit очищают историю и завершают сеанс.")
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            break
        if question == "/exit":
            break
        if question == "/clear":
            rag.clear("cli")
        elif question:
            show(question, "cli")


if __name__ == "__main__":
    main()
