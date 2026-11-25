"""Таблицы и раскрываемые фрагменты для учебного ноутбука."""

from html import escape

from IPython.display import HTML, display


def table(headers, rows):
    head = "".join(f"<th>{escape(str(item))}</th>" for item in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{escape(str(item))}</td>" for item in row) + "</tr>"
        for row in rows
    )
    display(HTML(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"))


def show_rows(rows):
    blocks = []
    for row in rows:
        fragments = "\n\n".join(
            f"{rank}. {hit.chunk.citation} | {hit.chunk.source} | "
            f"позиция {hit.chunk.start} | cos={hit.score:.4f}\n{hit.chunk.text}"
            for rank, hit in enumerate(row["hits"], 1)
        )
        header = f'{row["qid"]} — {row["question"]}'
        parts = [
            f'Запрос: {row["query"]}',
            f'Позиции с эталонным доказательством: {row["evidence_ranks"]}',
        ]
        if "answer" in row:
            parts.append(f'Ответ: {row["answer"]}')
        parts.append(f'Фрагменты:\n{fragments or "Выдача пуста."}')
        if row.get("history"):
            parts.append("История:\n" + "\n".join(f"{q}\n{a}" for q, a in row["history"]))
        blocks.append(
            f"<details><summary>{escape(header)}</summary>"
            f"<pre style='white-space:pre-wrap'>{escape(chr(10).join(parts))}</pre></details>"
        )
    display(HTML("\n".join(blocks)))
