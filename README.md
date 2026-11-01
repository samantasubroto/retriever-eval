# retriever-eval

A small, step-by-step project to **build a RAG retriever and measure how good it is**, using the OpenAI API.

## Goal

Retrieval is the part of a RAG (retrieval-augmented generation) system that most often decides whether an answer is right or wrong. If the right evidence is not retrieved, even a strong LLM cannot answer correctly. Yet retrievers are often tuned by eyeballing a few queries.

This project aims to replace that with a repeatable evaluation:

1. Build a deliberately simple retriever over a document corpus (first a PDF, then a news-article multi-hop dataset).
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

A second dataset, a 20-question sample of [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) (news articles, each question needs evidence from 2 to 4 articles), is selected with `DATASET=multihop`. Its labels are document-level (`relevant_docs`, the article URLs) plus the exact evidence sentences (`facts`), which allows an exact, free check of whether retrieved chunks contain the evidence. See Step 4 below.

In the first dataset, ground truth is labelled by **section ID**, not chunk ID, so the labels stay valid when the chunking strategy changes. In the corpus, `section_id` is `RAG-{page:03d}`.

### 3. Evaluation

Two complementary kinds of metric:

**Section-ID metrics** (deterministic, free, repeatable). Retrieved chunks are mapped to section IDs and compared with `source_section_ids`:

| Metric | Question it answers |
|---|---|
| Recall@k | What fraction of the required sections were retrieved? |
| Hit rate@k | Was at least one required section retrieved? |
| Precision@k | What fraction of retrieved chunks came from required sections? |
| MRR | How high does the first correct chunk rank? |

**Evidence-fact recall** (multihop only, deterministic): the fraction of a question's evidence sentences that appear verbatim in at least one retrieved chunk. It is stricter than document recall, because retrieving the right article is not enough if the chunk with the evidence is missed.

**LLM-judged metrics** (semantic, uses API calls, slightly noisy). Section IDs are only a proxy: a chunk can come from the right section without containing the needed passage. A judge model checks the actual text:

| Metric | Question it answers |
|---|---|
| Contextual recall | What fraction of the reference answer's key claims are supported by the retrieved chunks? |
| Contextual precision | What fraction of the retrieved chunks are actually useful for the question? |

Use the section-ID metrics as a fast regression gate, and the judged metrics for the final comparison.

### 4. Experiments

Change one variable at a time and re-run the same evaluation. Why we keep going: the first baseline showed the retriever reliably finds *a* correct section (hit rate 1.00) but misses most of the *other* sections a question needs (recall well below its ceiling). Each experiment below targets that gap.

Done: embedding model, recursive chunking, diversity-aware search (MMR, per-section cap), hybrid search with reranking, an audit of the golden dataset, and a switch to the MultiHop-RAG dataset (Step 4). The multihop result below is our final one. Ideas not pursued:

- multi-query / query decomposition (questions need ~6.5 sections each)
- larger k (6 to 10), combined with the per-section cap
- stripping the `chunk_id=...` metadata lines from the corpus text before embedding
- reranking the dense-only pool (to separate the effect of hybrid from reranking), and reranking followed by the per-section cap
- a different model for reranking and judging (they currently share `gpt-4o-mini`)
- a trimmed label set and hand-verified questions for the first dataset (see dataset caveats)
- on multihop: a larger sample (`prepare_multihop.py --n 60`), `ARTICLE_HEADER=0`, and chunk-size ablations (articles are long, so chunking should matter there)
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

### Step 3: hybrid search + RRF, measured at k=20, reranked to 5

Dense and BM25 keyword results (30 each) are fused with reciprocal rank fusion (RRF), the top 20 are kept as a pool, and an LLM reranker scores each chunk from 0 to 10 and keeps the top 5. Recall ceilings: 0.81 at k=5, 1.00 at k=20.

| | similarity | section cap | hybrid RRF (before rerank) | hybrid + rerank |
|---|---|---|---|---|
| Recall@5 | 0.37 | **0.47** | 0.37 | 0.41 |
| Precision@5 | 0.72 | 0.56 | 0.76 | 0.73 |
| MRR | 0.92 | 0.93 | n/a | **0.97** |
| ctx recall @5 | 0.54 | 0.47 | n/a | **0.57** |
| ctx precision @5 | 0.43 | 0.43 | n/a | **0.49** |
| Recall@20 | 0.76 | n/a (same as similarity by construction) | **0.66** | n/a |

**What we learned:**
- Adding BM25 made the candidate pool *worse* (recall@20 0.66 vs 0.76 for dense alone). The pages share a lot of boilerplate, so keyword matching is likely noisy here (not verified).
- Reranking helped a little at the final stage (recall@5 0.37 to 0.41, MRR 0.97, best judge scores), but it can only choose from its pool.
- The reranker and the judge are the same model, so the judge gains are probably flattered.
- Gaps of a few points are within the noise of 15 questions and a single run.

### Step 4: MultiHop-RAG (final result)

The first dataset could not separate the strategies well (hit rate was 1.00 for all of them, and its labels are noisy). We switched to a small sample of [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) ([paper](https://arxiv.org/pdf/2401.15391.pdf), licence ODC-BY, please credit the source if you reuse the data). `evals/prepare_multihop.py` builds the sample reproducibly (seed 42):

- **20 questions**: 7 inference, 7 comparison, 6 temporal (null queries left out, they have no evidence). Each needs 2 to 4 articles.
- **89 articles**: the 39 evidence articles plus 50 random distractors (about 1,270 chunks of 1000 characters).
- **Labels are verifiable**: every evidence article is in the corpus and every evidence sentence appears verbatim in its article. Each chunk is prefixed with `title | source | date`, since the questions refer to outlets and dates.

Document recall can reach 1.0 here at k=5, so scores are easier to read than before.

**Run A: `text-embedding-3-large`, k=5** (pool of 20 reranked to 5 for the hybrid run):

| | similarity | doc cap | hybrid RRF (before rerank) | hybrid + rerank |
|---|---|---|---|---|
| Recall@5 (documents) | 0.51 | **0.75** | 0.66 | 0.67 |
| Hit rate@5 | 0.85 | 1.00 | 1.00 | 1.00 |
| Fact recall@5 | 0.33 | 0.38 | 0.48 | **0.53** |
| Precision@5 | 0.68 | 0.58 | 0.75 | 0.76 |
| MRR | 0.80 | 0.85 | n/a | **0.87** |
| Judge precision@5 | 0.55 | **0.67** | n/a | **0.67** |
| Recall@20 | 0.80 | 0.80 | **0.88** | n/a |
| Fact recall@20 | 0.70 | 0.38 | **0.81** | n/a |

**Run B (final): `text-embedding-3-small`, k=10** (pool of 20 reranked to 10):

| | similarity | doc cap | hybrid RRF (before rerank) | hybrid + rerank |
|---|---|---|---|---|
| Recall@10 (documents) | 0.59 | 0.73 | **0.76** | 0.74 |
| Hit rate@10 | 0.90 | 1.00 | 1.00 | 1.00 |
| Fact recall@10 | 0.47 | 0.35 | 0.62 | **0.65** |
| Precision@10 | 0.62 | 0.55 | 0.67 | **0.68** |
| MRR | 0.76 | 0.82 | n/a | **0.85** |
| Judge precision@10 | 0.44 | **0.57** | n/a | 0.50 |

Pool at k=20: recall 0.83, fact recall 0.78.

**Conclusions:**
- **Hybrid search (dense + BM25 fused with RRF) is the main win.** Fact recall rose from 0.47 to about 0.62 to 0.65, and hit rate from 0.90 to 1.00, with both embedding models. News articles are full of names and dates, so keyword matching helps here, unlike on the PDF corpus where it made the pool worse.
- **The LLM reranker adds little.** Most of the benefit is already there before reranking, and at k=10 from a pool of 20 it changes almost nothing (recall 0.76 to 0.74, fact recall 0.62 to 0.65). It costs one extra LLM call per query, so it may not be worth it.
- **The per-document cap looks good on document recall but is weak on evidence.** It keeps one chunk per article, which is often not the chunk that contains the evidence (fact recall 0.35 to 0.38).
- **Embedding model:** the two runs use different k, so they are not directly comparable. We did not run large at k=10, so we cannot say how much the larger model helps here.

**Caveats:** 20 questions is a small sample, so differences of a few points are noise (the hybrid vs similarity gap is large enough to be more believable). The 50 distractors are random, so this is easier than the full 609-article corpus. The reranker and the judge are the same model (`gpt-4o-mini`), so judge scores are probably flattered. The judge's contextual recall is not used for this dataset, because the short answers ("Yes", "no") contain no claims to split. Fact recall replaces it.

## Dataset caveats

This section is about the first (PDF) dataset. The multihop labels are described in Step 4.

The golden dataset was generated by an LLM asked to synthesise answers across several corpus sections, so its labels may be wrong. `evals/audit.py` checks this by giving a judge the full corpus and the reference answer, and asking which sections support each claim (one `gpt-4o-mini` pass, 15 questions):

| Check | Result |
|---|---|
| Reference-answer claims supported by some section | 80% |
| Listed sections that support at least one claim | 59% |
| Listed sections that support no claim | 2.5 per question |

So about 40% of the listed sections may be padding, which would hold section recall down even for a good retriever (a perfect retriever might score about 0.6). The audit is itself noisy: some flagged claims look like paraphrases the judge missed, and one question (GOLD-013) matched 31 unlisted sections because the pages share boilerplate. Neither the labels nor the audit should be trusted without a human check.

Consequences for reading the results:
- Treat section-ID recall as a relative measure for comparing runs, not an absolute score.
- Lean on the judge-based contextual recall, which does not depend on the section lists.
- Planned: a trimmed label set (only sections that support a claim), plus questions written or verified by hand, including single-section factual ones.

## Project layout

```
retriever/
  config.py       settings (models, chunking, paths, k values)
  ingest.py       load, chunk, embed, store
  search.py       top-k retrieval (similarity, MMR, per-section cap)
  hybrid.py       BM25 + dense search fused with RRF
  rerank.py       LLM reranker
evals/
  metrics.py      section-ID metrics
  judge.py        LLM-judged contextual recall / precision
  run.py          evaluation runner
  audit.py        checks golden-dataset labels against the corpus
  prepare_multihop.py   builds the small MultiHop-RAG sample (data/multihop/, not committed)
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
python -m evals.run               # document-level metrics (plus evidence-fact recall on multihop)
python -m evals.run --judge       # also run LLM-judged metrics

# try another search strategy without editing code
SEARCH_STRATEGY=doc_cap python -m evals.run --judge --out evals/results_doc_cap.json
SEARCH_STRATEGY=hybrid_rerank python -m evals.run --judge

python -m evals.audit             # audit the golden dataset labels (PDF dataset)
```

Re-run ingestion whenever you change chunking settings, the embedding model, or the corpus. Per-question results are written to `evals/results.json`.

### MultiHop-RAG sample (the final experiment)

```bash
mkdir -p /tmp/mhrag_raw && cd /tmp/mhrag_raw
curl -sLO https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json
curl -sLO https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json
cd -                              # back to the project root

python evals/prepare_multihop.py --raw-dir /tmp/mhrag_raw       # writes data/multihop/ (20 questions, 89 articles)
export DATASET=multihop EMBEDDING_MODEL=text-embedding-3-small
python -m retriever.ingest
SEARCH_STRATEGY=hybrid_rerank K_VALUES=10 JUDGE_K=10 FINAL_K=10 python -m evals.run --judge
```

Each dataset has its own collection in `chroma_db/`. A collection built with one embedding model cannot be queried with another (the vector sizes differ), so re-ingest after changing `EMBEDDING_MODEL`.

## Configuration

Set in `.env` or in `retriever/config.py`:

| Setting | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | none | API key |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | embedding model (a value in `.env` overrides this default) |
| `JUDGE_MODEL` | `gpt-4o-mini` | model for LLM-judged metrics |
| `CHUNKING` | `recursive` | `recursive` or `fixed` chunking |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 200 characters | chunking (in `config.py`) |
| `DATASET` | `pdf` | `pdf` or `multihop` |
| `ARTICLE_HEADER` | `1` | multihop: prefix each chunk with `title \| source \| date` |
| `SEARCH_STRATEGY` | `similarity` | `similarity`, `mmr`, `doc_cap` (alias `section_cap`) or `hybrid_rerank` |
| `FETCH_K` / `MMR_LAMBDA` / `MAX_PER_DOC` | 20 / 0.5 / 1 | candidate pool and diversity settings for `mmr` and `doc_cap` |
| `K_VALUES` / `JUDGE_K` | 4 / 4 | comma-separated cut-offs for scoring, e.g. `K_VALUES=5,20` |
| `POOL_K` / `FINAL_K` | 20 / 5 (`FINAL_K` can be set by environment variable) | pool size after fusion, and chunks kept after reranking (`hybrid_rerank`) |
| `RERANK_MODEL` | same as `JUDGE_MODEL` | model used for reranking |
