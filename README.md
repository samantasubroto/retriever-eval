# retriever-eval

A small, step-by-step project to **build a RAG retriever and measure how good it is**, using the OpenAI API.

## Goal

Retrieval is the part of a RAG (retrieval-augmented generation) system that most often decides whether an answer is right or wrong. If the right evidence is not retrieved, even a strong LLM cannot answer correctly. Yet retrievers are often tuned by eyeballing a few queries.

This project aims to replace that with a repeatable evaluation:

1. Build a deliberately simple retriever over a PDF corpus.
2. Define a **golden dataset** of questions with trusted expected answers and the corpus sections that support them.
3. Score the retriever against that dataset with objective metrics.
4. Use the scores to compare design choices, changing one thing at a time, instead of guessing.

The retriever is a baseline. The evaluation harness is the main deliverable.

## Approach

### 1. Retriever

```
PDF -> pages -> chunks -> embeddings -> vector store -> top-k search
```

- **Ingest** (`retriever/ingest.py`): load the PDF page by page, split into chunks, embed with OpenAI, and store in a local [Chroma](https://www.trychroma.com/) database. Each chunk carries `source`, `page` and `section_id` metadata.
- **Search** (`retriever/search.py`): embed the query and return the top-k chunks ranked by cosine similarity.

Each step is a separate function, so any one of them (chunking strategy, embedding model, vector store, ranking) can be swapped and re-evaluated in isolation.

### 2. Golden dataset

`data/eval/rag_golden_dataset_15.json` holds questions, each with:

- a reference answer (`ground_truth`)
- the corpus sections needed to answer it (`source_section_ids`)
- a question type and difficulty label

Many questions need information combined from several sections, so the evaluation measures how much of the required evidence is retrieved, not only whether one relevant chunk appears.

Ground truth is labelled by **section ID**, not chunk ID, so the labels stay valid when the chunking strategy changes. In the corpus, `section_id` is `RAG-{page:03d}`.

### 3. Evaluation

Two complementary kinds of metric:

**Section-ID metrics** (deterministic, free, repeatable). Retrieved chunks are mapped to section IDs and compared with `source_section_ids`:

| Metric | Question it answers |
|---|---|
| Recall@k | What fraction of the required sections were retrieved? |
| Hit rate@k | Was at least one required section retrieved? |
| Precision@k | What fraction of retrieved chunks came from required sections? |
| MRR | How high does the first correct chunk rank? |

**LLM-judged metrics** (semantic, uses API calls, slightly noisy). Section IDs are only a proxy: a chunk can come from the right section without containing the needed passage. A judge model checks the actual text:

| Metric | Question it answers |
|---|---|
| Contextual recall | What fraction of the reference answer's key claims are supported by the retrieved chunks? |
| Contextual precision | What fraction of the retrieved chunks are actually useful for the question? |

Use the section-ID metrics as a fast regression gate, and the judged metrics for the final comparison.

### 4. Experiments (planned)

Change one variable at a time and re-run the same evaluation:

- chunk size and overlap
- recursive or structure-aware splitting instead of fixed-size
- stripping the `chunk_id=...` metadata lines from the corpus text before embedding
- embedding model (`text-embedding-3-small` vs `-large`) and vector dimensions
- top-k
- retrieval strategies: MMR, multi-query, hybrid (lexical and semantic) search, reranking

Out of scope for now: answer generation, and answer-level metrics such as faithfulness and correctness. The same golden dataset can be reused for these later.

## Project layout

```
retriever/
  config.py       settings (models, chunking, paths, k values)
  ingest.py       load, chunk, embed, store
  search.py       top-k retrieval
evals/
  metrics.py      section-ID metrics
  judge.py        LLM-judged contextual recall / precision
  run.py          evaluation runner
data/
  *.pdf           corpus
  eval/           golden dataset
check_setup.py    checks the OpenAI key and connection
```

## Setup

Requires Python 3.10+ and an OpenAI API key.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # then set OPENAI_API_KEY in .env
python check_setup.py    # should print the embedding dimensions
```

## Usage

```bash
python -m retriever.ingest        # chunk, embed and store the corpus (rebuilds the DB)
python -m evals.run               # section-ID metrics
python -m evals.run --judge       # also run LLM-judged metrics
```

Re-run ingestion whenever you change chunking settings, the embedding model, or the corpus. Per-question results are written to `evals/results.json`.

## Configuration

Set in `.env` or in `retriever/config.py`:

| Setting | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | none | API key |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | embedding model |
| `JUDGE_MODEL` | `gpt-4o-mini` | model for LLM-judged metrics |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 200 characters | chunking (in `config.py`) |
| `K_VALUES` / `JUDGE_K` | (3, 5, 10) / 5 | cut-offs for scoring (in `config.py`) |
