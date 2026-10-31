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

### 4. Experiments

Change one variable at a time and re-run the same evaluation. Why we keep going: the first baseline showed the retriever reliably finds *a* correct section (hit rate 1.00) but misses most of the *other* sections a question needs (recall well below its ceiling). Each experiment below targets that gap.

Done: embedding model, recursive chunking, diversity-aware search (MMR, per-section cap). Planned:

- multi-query / query decomposition (questions need ~6.5 sections each)
- larger k (6 to 10), combined with the per-section cap
- stripping the `chunk_id=...` metadata lines from the corpus text before embedding
- hybrid (lexical and semantic) search and reranking
- repeated judge runs, to measure how noisy the LLM-judged scores are

Out of scope for now: answer generation, and answer-level metrics such as faithfulness and correctness. The same golden dataset can be reused for these later.

## Results so far

All runs use the 15-question golden dataset, 1000-character chunks with 200 overlap, and `gpt-4o-mini` as judge. Section-ID recall has a hard ceiling because each chunk covers at most one section and questions need 4 to 10 sections: the best possible recall is **0.50 at k=3, 0.66 at k=4, 0.81 at k=5 and 1.00 at k=10**. Judge-based scores are noted as "ctx".

### Step 1: baseline (`text-embedding-3-small`, fixed-size chunks, plain similarity)

| k | Recall (ceiling) | Hit rate | Precision |
|---|---|---|---|
| 3 | 0.27 (0.50) | 0.93 | 0.71 |
| 5 | 0.40 (0.81) | 1.00 | 0.72 |
| 10 | 0.58 (1.00) | 1.00 | 0.57 |

MRR 0.92. At k=5: ctx recall 0.34, ctx precision 0.39.

**Takeaway:** finding one correct section is easy, covering all the needed sections is the problem. Recall is only about half of its ceiling at every k. Small k values are not a fair test for this dataset, so later runs use k=4 (the setting we adopted for quicker iteration).

### Step 2: later experiments, all at k=4 (recall ceiling 0.66)

| Run | Recall | Hit rate | Precision | MRR | ctx recall | ctx precision |
|---|---|---|---|---|---|---|
| Small embeddings, fixed chunks, similarity | 0.36 | 1.00 | 0.75 | 0.92 | 0.31 | 0.38 |
| Large embeddings (`text-embedding-3-large`) | 0.36 | 1.00 | 0.75 | 0.92 | 0.53 | 0.48 |
| + recursive chunking | 0.34 | 1.00 | 0.78 | 0.92 | 0.52 | 0.50 |
| + MMR search (lambda 0.5) | 0.35 | 0.93 | 0.70 | 0.89 | 0.51 | 0.48 |
| + per-section cap (1 chunk per section) | **0.44** | 1.00 | 0.65 | 0.93 | 0.47 | 0.52 |

Each row after the first changes one thing from the row above, except the last two, which both replace plain similarity search on top of recursive chunking.

**What we learned:**
- **Embedding model:** the larger model left the section-ID metrics unchanged. The judge scores rose (ctx recall 0.31 to 0.53), probably because it picks better chunks within the same sections, but this comes from one judge run on 15 questions, so treat the size of the gap with caution.
- **Recursive chunking:** no meaningful change. Each page is only about 2,000 characters, so a 1000-character chunk gives 3 chunks per page whichever method is used. Chunking is not the bottleneck on this corpus.
- **Why diversity:** plain similarity often spends several of its four slots on one section (for example three chunks from `RAG-017`). That wastes slots when a question needs many sections.
- **MMR:** no gain. It picks chunks that differ in wording but still come from the same sections.
- **Per-section cap:** the best section recall so far (0.44, about 67% of the ceiling, up from about 52%), with hit rate and MRR unchanged. **Caveat:** this strategy directly optimises the section-ID metric, so part of the gain is by construction. Judge ctx recall did not improve (0.52 to 0.47), and that difference is within the judge's likely noise. We do not yet know whether it truly gives better evidence for answering.

**Current best configuration for coverage:** large embeddings, recursive chunking, per-section cap, k=4. Next: multi-query retrieval and larger k, then a repeated judge run to see how much to trust small differences.

## Project layout

```
retriever/
  config.py       settings (models, chunking, paths, k values)
  ingest.py       load, chunk, embed, store
  search.py       top-k retrieval (similarity, MMR, per-section cap)
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

# try another search strategy without editing code
SEARCH_STRATEGY=section_cap python -m evals.run --judge --out evals/results_section_cap.json
```

Re-run ingestion whenever you change chunking settings, the embedding model, or the corpus. Per-question results are written to `evals/results.json`.

## Configuration

Set in `.env` or in `retriever/config.py`:

| Setting | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | none | API key |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | embedding model (a value in `.env` overrides this default) |
| `JUDGE_MODEL` | `gpt-4o-mini` | model for LLM-judged metrics |
| `CHUNKING` | `recursive` | `recursive` or `fixed` chunking |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 200 characters | chunking (in `config.py`) |
| `SEARCH_STRATEGY` | `similarity` | `similarity`, `mmr` or `section_cap` |
| `FETCH_K` / `MMR_LAMBDA` / `MAX_PER_SECTION` | 20 / 0.5 / 1 | candidate pool and diversity settings for `mmr` and `section_cap` |
| `K_VALUES` / `JUDGE_K` | (4,) / 4 | cut-offs for scoring (in `config.py`) |
