# Multimodal AI Decision Support System for Customer Experience Management in Banking

**A build specification for an AI coding agent.**
Read this whole file before writing any code. It defines what to build, how the
pieces fit together, and — importantly — what *not* to build. Where a decision is
left open, prefer the simplest option that satisfies the acceptance criteria.

---

## 1. What this project is

A decision-support tool for a bank's customer-service team. It reads three
streams of evidence about a customer — **written text** (complaints, chat, email),
**voice** (contact-centre call recordings), and **transaction/behavioural data**
(failed transfers, declines, app usage) — and combines them into a single,
ranked recommendation telling a human officer *which customers need attention,
how urgently, and what to do next*.

It is an academic MSc prototype. The goal is to **demonstrate that combining the
three modalities produces better decisions than using any one alone**, and to do
so in a way that fits Nigerian banking realities and data-protection law.

### The one sentence the whole system must honour
> The system **recommends**; a human **decides**. It never takes an action that
> affects a customer on its own.

This is not a style preference. It is a legal requirement (see §3).

---

## 2. What to build vs. what NOT to build

**BUILD:**
- Three modality processors (text, voice, transaction).
- A fusion layer that combines their outputs into one risk score + ranking + recommended action.
- Three **unimodal baseline** models (text-only, voice-only, transaction-only) for comparison.
- An evaluation harness that compares multimodal vs. the three baselines.
- A simple **Streamlit** interface showing a ranked customer list with the reasons behind each flag.

**DO NOT BUILD (out of scope — do not attempt, even if it seems helpful):**
- ❌ **Acoustic emotion recognition** (pitch/energy/tone models). The voice path is
  **transcription → text model only**. See §6.2 for why. This is a firm decision.
- ❌ Any component that autonomously refunds, blocks, closes, or messages a customer.
- ❌ Image or video modalities.
- ❌ Local-language / code-switching speech support (English + Nigerian-accented English only).
- ❌ Real bank data integrations. Use public or synthetic data only (see §5).
- ❌ A production auth system, user accounts, or cloud deployment. Keep it runnable locally.

If a requirement seems to need one of the above, stop and leave a clearly marked
`# TODO(scope):` comment instead of building it.

---

## 3. Hard constraints (these shape the architecture)

1. **Human-in-the-loop (NDPA 2023, s.37).** Every output is advisory. The UI must
   always let the officer see *why* a customer was flagged and must present an
   explicit "these are recommendations" framing. No auto-actions.
2. **Robust to missing modalities.** Real customers rarely have all three data
   types. The system must produce a valid result when 1 or 2 modalities are
   missing — never crash, never treat "missing" as "fine." This forces a
   **late/hybrid fusion** design (§6.4).
3. **Interpretability over raw accuracy.** For every recommendation, the system
   must expose the contributing signal from each modality. A slightly less
   accurate but explainable result beats a black box.
4. **Reproducibility.** Fixed random seeds everywhere. Same data + same code =
   same numbers. No hidden state.
5. **Privacy.** Strip any personally identifying information (names, emails,
   phone numbers, account numbers) during preprocessing. Never print raw PII to
   logs or the UI.

---

## 4. Tech stack (use these unless there is a strong reason not to)

| Purpose | Choice |
|---|---|
| Language | Python 3.10+ |
| Text model | Hugging Face `transformers` — a BERT-family model (e.g. `distilbert-base-uncased`) fine-tuned for classification + sentiment |
| Speech-to-text | OpenAI **Whisper** (`openai-whisper` or `faster-whisper`); default to a small model so it runs on CPU |
| Classical ML | `scikit-learn` (transaction model, baselines, metrics, stratified split) |
| Data | `pandas`, `numpy` |
| Interface | **Streamlit** |
| Plots | `matplotlib`, `seaborn` |
| Config | a single `config.yaml` |
| Env | `requirements.txt` + `venv`; must run on a normal laptop CPU. GPU optional, never required. |

Keep dependencies minimal. Pin versions in `requirements.txt`.

---

## 5. Data

> Confidential bank data cannot be used. Everything is **public** or **synthetic**.
> For each dataset the agent proposes, add a one-line note on its licence and
> confirm it permits research use. Do not hardcode a dataset that hasn't been verified.

### 5.1 Suggested starting sources (verify before use)
- **Text** — a public consumer-complaint corpus of banking complaints (e.g. a
  bank-complaints dataset with a free-text narrative + a product/issue label).
  If public banking text is thin, **generate synthetic complaints** reflecting
  the dominant Nigerian failure types: *failed transaction, unauthorised
  deduction, unfair charge, delayed refund, card declined, app downtime.*
- **Voice** — a small set of English / Nigerian-accented English audio clips for
  testing transcription. If none is readily licensable, **synthesise speech**
  from the text complaints using a TTS library, or record a few volunteer clips.
  The point is to prove the transcription→text path works, not to collect a corpus.
- **Transaction** — a public bank-customer-churn tabular dataset, and/or
  **synthetic** records with the behavioural features in §6.3.

> **This build's data decision:** no internet access was available/verified in
> this environment for a third-party banking-complaints or churn dataset, so
> `src/ingest.py` generates a fully **synthetic** dataset from hand-written
> Nigerian-banking complaint/call templates and engineered transaction
> features (see the module docstring for the exact generative process and
> why it lets the fusion vs. unimodal comparison be meaningful). No licence
> applies since nothing is downloaded. Swapping in a verified public dataset
> later only requires changing `src/ingest.py`'s output — the canonical
> schema below is unaffected.

### 5.2 Canonical internal schema
After ingestion, normalise everything to **one row per customer per time window**,
keyed by a synthetic `customer_id`. Any modality may be null.

```
customer_id            : str   (synthetic, no real identifiers)
window_start / window_end : date
text_records           : list[str]   (complaint / chat / email text; may be empty)
voice_transcripts      : list[str]   (Whisper output of calls; may be empty)
txn_features           : dict        (see §6.3; may be null)
label                  : int         (1 = dissatisfied / churn-risk, 0 = not) — for training/eval only
```

### 5.3 Preprocessing (shared)
- Remove PII (regex for emails/phones/account numbers + name stripping).
- Normalise text (lowercase, strip, collapse whitespace); keep a raw copy for display.
- **Deliberately construct records with missing modalities** in the eval set, so
  robustness (§3.2) is *tested*, not assumed.

---

## 6. Modules

Each module is independently buildable and testable (this maps to the Agile
sprint plan in §9). Every module gets its own file, a `fit`/`predict` (or
`process`) interface, and a unit test.

### 6.1 Text modality — `src/text_model.py`
- **Input:** a customer's `text_records`.
- **Do:** classify complaint type and score dissatisfaction/sentiment using a
  fine-tuned BERT-family model.
- **Output:** `{text_score: float in [0,1], text_label: str, confidence: float}`.
  `text_score` = how strongly this text signals dissatisfaction.
- **Acceptance:** on held-out text, beats a majority-class baseline on F1; returns
  a null-safe result when `text_records` is empty.

### 6.2 Voice modality — `src/voice_model.py`
- **Input:** a path to a call-audio file (or a pre-made transcript).
- **Do:** transcribe audio to text with **Whisper**, then pass that transcript
  **through the exact same text model from 6.1** to infer frustration/urgency.
- **Output:** `{voice_score: float in [0,1], transcript: str, confidence: float}`.
- **WHY transcript-only (do not deviate):** open acoustic-emotion datasets are
  trained on scripted, non-Nigerian speech and don't generalise to
  Nigerian-accented, code-switched call audio within this project's scope.
  Transcribe-then-reuse-the-text-model is the deliberate, defensible design.
  Acoustic emotion recognition is **future work**, not part of this build.
- **Acceptance:** given an audio clip, produces a transcript and a score without
  a separate emotion model; runs on CPU with a small Whisper model.

### 6.3 Transaction modality — `src/txn_model.py`
- **Input:** `txn_features` dict. Engineer features such as:
  `failed_transfers_count`, `declines_count`, `txn_frequency_change`,
  `app_sessions_change`, `days_since_last_complaint`, `avg_resolution_delay`.
- **Do:** train a scikit-learn classifier (start with logistic regression or
  gradient boosting) to output a behavioural risk score.
- **Output:** `{txn_score: float in [0,1], top_features: list[(name, contribution)], confidence: float}`.
- **Acceptance:** beats majority-class F1 on held-out data; exposes the top
  contributing features (for interpretability).

### 6.4 Fusion + decision layer — `src/fusion.py`  ⭐ core contribution
- **Input:** the three module outputs (any may be `None`).
- **Strategy:** **late / hybrid fusion**. Each modality is scored independently;
  combine at the decision stage.
  - Combine available scores with **learned weights** (learn weights from
    training data — e.g. a small logistic-regression meta-model over the
    modality scores — rather than hardcoding them).
  - **Missing modality handling:** redistribute a missing modality's weight
    across the present ones. Never substitute 0. Absence of a signal ≠ a good signal.
- **Output:**
  ```
  {
    risk_score      : float in [0,1],
    priority_rank   : int,          # relative to other customers in the batch
    next_best_action: str,          # chosen from an action map keyed on dominant signal
    contributing    : {text: ..., voice: ..., txn: ...}  # the per-modality signals
  }
  ```
- **`next_best_action`:** a simple, transparent rule/lookup from the dominant
  contributing signal to a suggested action (e.g. dominant = repeated failed
  transfers → "verify and process pending reversal"; dominant = negative text
  sentiment about charges → "review recent charges and call customer"). Keep the
  action map in `config.yaml` so it's editable without code changes.
- **Acceptance:** produces valid output with 1, 2, or 3 modalities present;
  `contributing` always populated for the interpretability requirement.

### 6.5 Interface — `app/streamlit_app.py`
- A ranked table of customers (highest risk first).
- For each row: risk score, priority, recommended next action, and an
  **expandable "why"** panel showing each modality's contribution + the raw text
  snippet / transcript excerpt that drove it.
- A visible banner: *"Recommendations only. A customer-service officer reviews
  and decides. No action is taken automatically."*
- An **override / dismiss** control per row (records the officer's decision;
  no downstream effect needed — it exists to embody human authority).
- No login, no database required; reading from the processed dataset is fine.

---

## 7. Evaluation harness — `src/evaluate.py`  ⭐ this is where the thesis is proven

Build this as a first-class deliverable, not an afterthought.

- Train and evaluate **four models on the same split**:
  1. text-only baseline
  2. voice-only baseline
  3. transaction-only baseline
  4. **multimodal fusion**
- **Split:** stratified train/test on the `label` (dissatisfied is the minority
  class, so stratify; fix the seed).
- **Metrics:** accuracy, precision, recall, **F1** (lead with F1 given class
  imbalance). Produce a comparison table + bar chart.
- **Fairness:** where a segment attribute exists, report metrics disaggregated by
  segment and flag large gaps.
- **Interpretability check:** for a sample of predictions, confirm the
  `contributing` signals are present and human-readable.
- **Headline success criterion:** multimodal F1 > each unimodal baseline's F1.
  Print this comparison explicitly.

Save all outputs (table as CSV, charts as PNG) to `results/`.

---

## 8. Repository layout

```
multimodal-cx-dss/
├── README.md                  # this file
├── requirements.txt
├── config.yaml                # paths, model names, action map, seed, weights config
├── data/
│   ├── raw/                   # untouched source data (gitignored)
│   ├── synthetic/             # generated complaints / transactions
│   └── processed/             # canonical schema (§5.2)
├── src/
│   ├── ingest.py              # load sources → canonical schema
│   ├── preprocess.py          # PII strip, normalise, build missing-modality eval cases
│   ├── text_model.py          # 6.1
│   ├── voice_model.py         # 6.2  (Whisper → text_model)
│   ├── txn_model.py           # 6.3
│   ├── fusion.py              # 6.4  ⭐
│   ├── evaluate.py            # 7    ⭐
│   └── utils.py               # seeding, PII regex, IO helpers
├── app/
│   └── streamlit_app.py       # 6.5
├── tests/                     # one test file per src module
├── results/                   # metrics CSVs + charts (generated)
└── notebooks/                 # optional exploration
```

---

## 9. Build plan (Agile / Scrum — 5 sprints)

Deliver in this order; each sprint ends with something runnable and tested.

- **Sprint 1 — Data foundation.** `ingest.py`, `preprocess.py`, canonical schema,
  synthetic-data generators, PII stripping, seed handling. *Done when:* a
  processed dataset with some deliberately-missing-modality rows exists and loads.
- **Sprint 2 — Text modality.** `text_model.py` + tests. *Done when:* it scores
  complaints and beats the majority baseline on F1.
- **Sprint 3 — Voice modality.** `voice_model.py` = Whisper → the Sprint-2 text
  model. *Done when:* an audio clip yields a transcript + score, no emotion model.
- **Sprint 4 — Transaction + fusion.** `txn_model.py`, then `fusion.py` with
  learned weights and missing-modality redistribution. *Done when:* fusion emits
  score + rank + action + contributing signals for 1/2/3 present modalities.
- **Sprint 5 — Interface + evaluation.** `streamlit_app.py` and `evaluate.py`.
  *Done when:* the app shows a ranked, explainable list and the harness prints the
  multimodal-vs-baselines comparison with charts in `results/`.

---

## 10. Setup & run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/ingest.py           # build raw -> processed
python src/preprocess.py       # clean + build eval cases
python src/evaluate.py         # train all 4 models, write results/
streamlit run app/streamlit_app.py
```

### What each command does

| Command | Reads | Writes | Notes |
|---|---|---|---|
| `python src/ingest.py` | nothing (generative) | `data/synthetic/*.jsonl`, `data/synthetic/transactions.csv`, `data/audio/*.wav`, `data/raw/customers.jsonl` | Generates the synthetic dataset (see §5.1). Deterministic for a fixed `config.yaml: seed`. Demo call-audio clips are synthesised with offline TTS (`pyttsx3`); if no TTS voice is available on the machine it logs a warning and skips audio (the rest of the pipeline is unaffected, since bulk voice scoring uses pre-made transcripts, not audio). |
| `python src/preprocess.py` | `data/raw/customers.jsonl` | `data/processed/customers.jsonl` | Strips PII, normalises text, and deliberately nulls out a configured fraction of each modality per customer (`config.yaml: data.missing_*_frac`) so the missing-modality path is actually exercised in evaluation. |
| `python src/evaluate.py` | `data/processed/customers.jsonl` | `results/model_comparison.{csv,png}`, `results/fairness_report.csv`, `results/interpretability_samples.json`, `results/scored_customers.csv`, `models/*` | Trains `TextModel` (fine-tuned DistilBERT), `TxnModel` (logistic regression), and `FusionModel` (logistic-regression meta-model) on a stratified split, prints and saves the headline multimodal-vs-baselines F1 comparison, a fairness breakdown by customer `segment`, and an interpretability spot-check. Also saves the trained models and a fully-scored, ranked customer list for the app. Takes a few minutes on CPU (fine-tuning DistilBERT is the slow step). |
| `streamlit run app/streamlit_app.py` | `results/scored_customers.csv` | (session-only officer decisions, not persisted) | Ranked worklist UI. Requires `evaluate.py` to have been run at least once. Never retrains a model and never acts on a customer automatically. |

Run `pytest` (or `pytest -m "not slow"` to skip the one test that fine-tunes a
tiny real model) from the project root to run the unit test suite.

### Running with Docker (recommended for non-developers)

No Python setup needed — just [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```bash
docker compose up --build
```

Then open **http://localhost:8502** in a browser (the **Live Triage** page is in
the sidebar). Port 8502 is used because 8501 is often taken by other Streamlit
apps; set `APP_PORT` (e.g. `APP_PORT=8600 docker compose up`) to change it.

On Windows, non-developers can simply double-click **`START_APP.bat`** (it checks
that Docker Desktop is running, starts the app, waits until it is ready and opens
the browser) and **`STOP_APP.bat`** when done. `READ ME FIRST.txt` explains this in
plain language for a client.

On first run the container generates the synthetic dataset and trains the models
whenever `models/`, `results/` or `data/processed/` is missing (this can take
several minutes on CPU and needs internet to fetch the base language model — the
container logs will show progress). On every run after that it reuses what's
already in `data/`, `models/`, and `results/` (mounted from the host) and starts
the app immediately. Delete those folders (or run `docker compose build --no-cache`)
to force a full retrain.

Stop the app with `Ctrl+C`, or `docker compose down` to remove the container.

#### Sending the project to a client as a zip

Run `powershell -ExecutionPolicy Bypass -File make_client_zip.ps1` from the project
folder. It writes `dist/mariam-app.zip` (about 240 MB), leaving out `.venv` and git
history but including the trained models, so the client's first start does not have
to train anything. The client only needs Docker Desktop and an internet connection
for the first build (Docker downloads PyTorch and the other packages).

---

## 11. Coding standards & guardrails for the agent

- Fix and centralise the random seed (`config.yaml`); import it everywhere.
- Every `src/` module: a clear function signature, a docstring, and a `tests/` file.
- Null-safe everywhere: a missing modality is normal input, not an error.
- Never log or display raw PII.
- Keep the `next_best_action` logic transparent and config-driven (no opaque model
  deciding actions).
- If you must deviate from this spec, add a `# TODO(spec):` note explaining why,
  rather than silently changing scope.
- Small Whisper + small BERT by default so the whole thing runs on a CPU laptop;
  expose model size in `config.yaml` for anyone with a GPU.
- Prefer clarity over cleverness — this is an academic prototype that must be
  read, defended, and extended by a student, not shipped to production.

---

## 12. Glossary (so the agent uses the project's language)

- **Modality** — one type of evidence: text, voice, or transaction data.
- **Fusion** — combining modality outputs into one decision.
- **Late/hybrid fusion** — score each modality separately, combine at the end;
  chosen because it tolerates missing modalities.
- **Decision support** — the system advises; a human decides.
- **Churn / dissatisfaction risk** — likelihood a customer is unhappy / about to leave.
- **Next-best-action** — the single suggested step for the officer, derived from
  the dominant contributing signal.

---

*End of specification. Build in sprint order (§9). When unsure, choose the
simplest thing that satisfies the acceptance criteria and preserves the
human-in-the-loop and missing-modality guarantees.*
