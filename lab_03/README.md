# Лабораторная работа № 3. RAG-система для PDF

Цыканов Артём Эдуардович, МПИ-25-1-1.

- [Ноутбук с кодом и сохранёнными результатами](rag_lab.ipynb)
- [Отчёт PDF](report.pdf), [Markdown](report.md), [LaTeX](report.tex)
- [Исследуемый PDF](data/LLM%20instruct%20%282%29.pdf)

Ноутбук адаптирован из [учебного шаблона Colab](https://colab.research.google.com/drive/17Wcx3SEeMSfzIa3xvE_r8Egmv3S21UFS). Он использует модули из этой директории. В результатах сохранены вопросы, поисковые запросы, полные фрагменты, метаданные и ответы модели. Фрагменты можно раскрывать по одному.

## Запуск

Из директории `lab_03` создайте окружение Python 3.11 или новее.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Установите Ollama и запустите сервер. Если он уже работает, достаточно загрузить модели.

```bash
ollama serve
```

В другом терминале активируйте то же окружение и выполните команды.

```bash
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text:v1.5
jupyter-execute rag_lab.ipynb --inplace --timeout=1800
```

Для интерактивного запуска можно установить JupyterLab в то же окружение и открыть `rag_lab.ipynb`. Параметр `OLLAMA_HOST` позволяет указать другой адрес действующего сервера, например `http://127.0.0.1:12434`. Версии, хеши моделей и SHA-256 PDF выводятся в первой ячейке.

## Режим вопрос–ответ

```bash
python main.py 'What two sources provided the training tasks in section 3.3?'
python main.py
python main.py --pdf path/to/document.pdf
```

Без аргумента запускается диалог с памятью трёх обменов и переформулированием уточнений. `/clear` очищает историю, `/exit` завершает диалог.

Проверки логики выполняются без сервера Ollama.

```bash
python -m unittest discover -s tests
```

Для сборки отчёта из исходника требуются XeLaTeX и шрифты Liberation Serif, DejaVu Sans и DejaVu Sans Mono. Два запуска `xelatex report.tex` формируют оглавление.
