"""Сценарии семи экспериментов. Эталоны используются только при оценке."""

import hashlib
from itertools import combinations
from os import getenv
from pathlib import Path
from importlib.metadata import version
from urllib.request import urlopen
import json

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from ollama import Client

from questions import (
    ABSENT, BANK, BOUNDARY, CITATION_IDS, DIALOGUES, EARLY,
    FOLLOW_UPS, LATE, PROMPT_IDS, QUESTIONS,
)
from rag import (
    BASE_PROMPT, CITATION, REFUSAL, STRICT_PROMPT,
    DenseEncoder, PDFLoader, RAG, binary_index, split_pages,
)


ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "data" / "LLM instruct (2).pdf"
LLM_MODEL = "qwen2.5:1.5b"
EMBEDDING_MODEL = "nomic-embed-text:v1.5"


def make_models(base_url):
    options = dict(temperature=0, seed=42, num_ctx=4096, num_predict=110, num_thread=6)
    llm = OllamaLLM(
        model=LLM_MODEL, base_url=base_url, keep_alive=1800, **options,
    )
    encoder = DenseEncoder(
        OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=base_url, num_thread=6, keep_alive=1800),
        document_prefix="search_document: ", query_prefix="search_query: ",
    )
    client = Client(host=base_url)

    def generate_grounded(prompt, hits):
        if not hits:
            return REFUSAL
        sources = {hit.chunk.id: hit.chunk.citation for hit in hits}
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "source_fragment_id": {"type": "integer", "enum": [0, *sources]},
            },
            "required": ["answer", "source_fragment_id"],
            "additionalProperties": False,
        }
        response = client.generate(
            model=LLM_MODEL, prompt=prompt, format=schema,
            options={**options, "num_predict": 256}, keep_alive=1800,
        )
        data = json.loads(response.response)
        answer = data["answer"].strip()
        source_id = data["source_fragment_id"]
        if not answer or source_id not in sources or answer == REFUSAL:
            return REFUSAL
        return f"{answer} {sources[source_id]}"

    return llm, encoder, generate_grounded


def citation_check(result):
    sources = {hit.chunk.id: hit.chunk.page for hit in result.hits}
    citations = [(int(chunk), int(page)) for chunk, page in CITATION.findall(result.answer)]
    return [(chunk, page, chunk in sources, sources.get(chunk) == page) for chunk, page in citations]


def overlapping(hits):
    for left, right in combinations((hit.chunk for hit in hits), 2):
        shared = min(left.end, right.end) - max(left.start, right.start)
        if left.page == right.page and shared >= 0.2 * min(len(left.text), len(right.text)):
            return True
    return False


class Experiments:
    def __init__(self, pdf=PDF_PATH):
        base_url = getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        client = Client(host=base_url)
        tags = {m.model: m.digest for m in client.list().models}
        missing = {LLM_MODEL, EMBEDDING_MODEL} - tags.keys()
        if missing:
            raise RuntimeError(f"Install the missing Ollama models: {', '.join(sorted(missing))}")
        server_url = base_url.removesuffix("/")
        if "://" not in server_url:
            server_url = f"http://{server_url}"
        with urlopen(f"{server_url}/api/version", timeout=10) as response:
            server_version = json.load(response)["version"]
        self.environment = {
            "Ollama": server_version,
            **{name: tags[name] for name in (LLM_MODEL, EMBEDDING_MODEL)},
            **{name: version(name) for name in ("langchain-ollama", "ollama", "numpy", "pypdf")},
            "PDF SHA-256": hashlib.sha256(Path(pdf).read_bytes()).hexdigest(),
        }
        self.llm, self.encoder, generate_grounded = make_models(base_url)
        self.pages = PDFLoader(pdf).load()
        self.evidence = {}
        for question in QUESTIONS.values():
            spans = []
            for page, quote in question.sources:
                start = self.pages[page - 1].text.index(quote)
                spans.append((page, start, start + len(quote)))
            self.evidence[question.id] = spans
        for qid, page, boundary in BOUNDARY:
            assert any(p == page and start < boundary < end for p, start, end in self.evidence[qid])
        self.chunks = split_pages(self.pages, 1000, 200)
        assert len(self.chunks) > 5
        assert all(any(self.supports(qid, chunk) for chunk in self.chunks[:5]) for qid in EARLY)
        assert not any(self.supports(qid, chunk) for qid in LATE for chunk in self.chunks[:5])
        self.index = self.encoder.index(self.chunks)
        self.rag = RAG(self.index, self.llm.invoke, generate_grounded)

    def supports(self, qid, chunk):
        return any(
            page == chunk.page and chunk.start <= start and end <= chunk.end
            for page, start, end in self.evidence.get(qid, ())
        )

    def search_row(self, qid, query, hits):
        ranks = [rank for rank, hit in enumerate(hits, 1) if self.supports(qid, hit.chunk)]
        return {
            "qid": qid, "question": query, "query": query, "hits": hits,
            "hit": bool(ranks), "evidence_ranks": ranks, "overlap": overlapping(hits),
        }

    def answer_row(self, qid, result):
        return {
            **self.search_row(qid, result.query, result.hits),
            "question": result.question, "answer": result.answer, "history": result.history,
            "citations": citation_check(result),
        }

    def baseline(self):
        return self.answer_row("Q5", self.rag.ask(QUESTIONS["Q5"].text))

    def completeness(self):
        subset = RAG(self.encoder.index(self.chunks[:5]), self.llm.invoke)
        return {
            label: [
                self.answer_row(qid, rag.ask(QUESTIONS[qid].text, prompt=STRICT_PROMPT))
                for qid in EARLY + LATE
            ]
            for label, rag in (("Первые 5", subset), ("Полный", self.rag))
        }

    def configurations(self, settings, qids):
        runs = {}
        for size, overlap in settings:
            chunks = split_pages(self.pages, size, overlap)
            index = self.encoder.index(chunks)
            runs[size, overlap] = (
                len(chunks),
                [self.search_row(qid, QUESTIONS[qid].text, index.search(QUESTIONS[qid].text))
                 for qid in qids],
            )
        return runs

    def sizes(self):
        return self.configurations(((500, 10), (1000, 10), (2000, 10)), [q.id for q in BANK])

    def overlaps(self):
        return self.configurations(((1000, 0), (1000, 100), (1000, 200)), [q[0] for q in BOUNDARY])

    def search_methods(self):
        binary = binary_index(self.chunks)
        assert binary.search("zzqxxvvzz") == []
        return {
            label: {
                form: [
                    self.search_row(q.id, query, index.search(query))
                    for q in BANK if q.alternative
                    for query in [q.text if form == "Точные термины" else q.alternative]
                ]
                for form in ("Точные термины", "Переформулировки")
            }
            for label, index in (("Мешок слов", binary), ("Эмбеддинги", self.index))
        }

    def citations(self, baseline):
        return [
            baseline if qid == baseline["qid"] else self.answer_row(qid, self.rag.ask(QUESTIONS[qid].text))
            for qid in CITATION_IDS
        ]

    def prompt_comparison(self):
        present = [(qid, QUESTIONS[qid].text) for qid in PROMPT_IDS]
        absent = [(qid, question) for qid, question, _ in ABSENT]
        contexts = [(qid, text, self.index.search(text)) for qid, text in present + absent]
        return {
            name: [self.answer_row(qid, self.rag.answer(text, text, hits, prompt))
                   for qid, text, hits in contexts]
            for name, prompt in (("Исходный", BASE_PROMPT), ("Строгий", STRICT_PROMPT))
        }

    def conversations(self):
        runs = []
        for number, dialogue in enumerate(DIALOGUES, 1):
            for mode in ("А", "Б"):
                session = f"{number}-{mode}"
                self.rag.clear(session)
                for step, qid in enumerate(dialogue, 1):
                    utterance = FOLLOW_UPS.get(qid, QUESTIONS[qid].text)
                    result = self.rag.ask(utterance, session=session, rewrite=mode == "Б")
                    assert len(result.history) == step - 1
                    runs.append({"dialogue": number, "mode": mode, "step": step, **self.answer_row(qid, result)})
        return runs
