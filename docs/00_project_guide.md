# FilingLens: complete step-by-step guide

This is the full plan, from an empty folder to a deployed app on your resume. Tick the boxes as you go (VS Code and GitHub both show `- [ ]` as checkboxes).

**Total time:** about 6 weeks at 1–2 hours a day. One step per week.

```
 Step 0  Setup                       DONE
 Step 1  Data pipeline               DONE
 Step 2  Read the reports + RAG      <- in progress (code built; your checks next)
 Step 3  Text-to-SQL + router        <- in progress (code built; your checks next)
 Step 4  Research study              <- in progress (code built; your checks next)
 Step 5  Evaluation + better search
 Step 6  Web app + deployment
 Final   Resume, demo, interview prep
```

## How we work on each step

1. **Write the code** for the step, with an explanation of what each part does and why, the same way as Step 1.
2. **You run it** on your laptop and check the results look right.
3. **You read** that step's guide in `docs/` and do its short exercises. This is what lets you explain the project in an interview, so don't skip it.
4. **You ask** about anything unclear, e.g. "explain `chunk_text` line by line".
5. **Commit to GitHub**, then move to the next step.

Work on your laptop, not Google Colab. Your RTX 4060 GPU handles every AI model in this plan, and your files stay put between sessions.

---

## Step 0: Setup (done)

- [x] Project folder `E:\FilingLens` with a private Python environment (`.venv`)
- [x] Settings in `config/companies.yaml`, SEC contact email in `.env` (never uploaded)
- [ ] Install the VS Code extensions **Python** and **Jupyter**
- [ ] Create a GitHub account (if you don't have one), set up the repository and make the first commit

---

## Step 1: Data pipeline (done)

**Built:** one command (`python -m filinglens.pipeline`) downloads SEC annual reports, financial numbers and daily stock prices for 14 companies over 6 years into one DuckDB database file.

### How to run a command (do this once)

1. Open **VS Code**. Click **File > Open Folder…**, choose `E:\FilingLens`, click **Select Folder**. If it asks "Do you trust the authors?", click **Yes**.
2. Open the terminal: **Terminal > New Terminal** (or press ``Ctrl + ` ``, the key left of `1`). A panel opens at the bottom with a line ending in `PS E:\FilingLens>`. That means it's ready and in the right folder.
3. Type (or paste) a command, then press **Enter**.

### Run the pipeline

```
.\.venv\Scripts\python.exe -m filinglens.pipeline
```

This means "use the project's own Python to run the pipeline". You'll see one line per company, then a table of revenue (in billions of dollars), gross margin, net margin and growth. `NaN` means the company doesn't report that number. It takes a few seconds, because the downloads are already saved.

### Run the tests

```
.\.venv\Scripts\python.exe -m pytest
```

Tests are small automatic checks that the code gives the right answers. Success looks like `17 passed`.

### Read the explanation

In the left sidebar, open the `docs` folder and click `01_data_pipeline.md`. Press **Ctrl + Shift + V** to see it nicely formatted.

Your checklist:
- [ ] Run the pipeline and see the table
- [ ] Run the tests and see `17 passed`
- [ ] Read `docs/01_data_pipeline.md`
- [ ] Do its 4 exercises (especially #3: check one revenue figure against a real 10-K)
- [ ] Be able to explain, in your own words, the three data traps and the Salesforce bug

---

## Step 2: Read the reports + first RAG system (week 2)

**Goal:** ask a question about the annual reports in plain English and get an answer with citations, such as *"What supply chain risks did Apple describe in 2024?"*

**Why:** this is the core "AI" skill of the project. RAG (retrieval-augmented generation) means *find the relevant passages first, then let the AI answer using only those passages*. That reduces made-up answers (hallucinations) and lets every answer cite its source.

### What gets built

| # | Part | What it does | Main tools |
|---|---|---|---|
| 2.1 | Clean the HTML | Turns each 10-K web page into plain text, removing hidden XBRL data, page numbers and navigation | BeautifulSoup, lxml |
| 2.2 | Split into sections | Finds the standard parts: Item 1 Business, 1A Risk Factors, 7 MD&A (management's discussion), 7A Market Risk. The table of contents mentions these too, so it has to find the real ones. | Regular expressions |
| 2.3 | Chunking | Cuts each section into passages of up to 300 words (about 400 tokens), with a small overlap so sentences aren't cut off from their context. Each chunk remembers its company, year and section. | Python |
| 2.4 | Embeddings | Turns each chunk into a list of numbers (a vector) that captures its meaning, so similar meanings get similar vectors. Runs on your GPU. | sentence-transformers, PyTorch (GPU version) |
| 2.5 | Vector database | Stores the vectors so the closest matches to a question can be found in milliseconds, filtered by company and year | ChromaDB |
| 2.6 | Answering | Finds the top chunks, gives them to an LLM with instructions to answer only from them and cite them as [1], [2]… | An LLM (see decision below) |
| 2.7 | Command-line tool | `python -m filinglens.ask "your question"` | Python |

New database tables: `sections` (one row per report section) and `chunks` (one row per passage).

### A decision you'll make: which LLM?

| Option | Cost | Quality | Note |
|---|---|---|---|
| **API** (Claude, GPT or Gemini) | Pay per use; small for a project this size. Some providers have free tiers. | Best | Needs an API key in `.env`. Also works in the deployed app. |
| **Local model with Ollama** (7–8B model) | Free | Good | Runs on your RTX 4060. Can't run in the free deployed app (no GPU there). |

The code will hide the choice behind one small `llm.py` file, so you can switch at any time.

**Decided:** the Gemini API (free tier, `gemini-3.8-flash`), with Ollama kept as a free local option. See `docs/02_rag.md`, section 2.6.

### Your checklist
- [ ] Run the parsing step and open a few sections in DuckDB: do they start and end in the right places?
- [ ] Ask 10 questions of your own; check the citations point to passages that really answer them
- [ ] Read `docs/02_rag.md` and do its exercises
- [ ] Commit to GitHub

**Done when:** you get correct, cited answers for at least 8 of your 10 questions.

**Interview topics you'll be able to answer:** What is an embedding? Why chunk documents, and how did you pick the size? What is cosine similarity? How does RAG reduce hallucination, and when does it still fail?

---

## Step 3: Text-to-SQL, the router and charts (week 3)

**Goal:** handle number questions accurately (*"Compare NVIDIA's and AMD's gross margins since 2021"*) and mixed questions (*"How did Apple's R&D spending change, and why?"*).

**Why:** LLMs are bad at arithmetic and at remembering exact figures. For numbers, the AI should write a database query and let the database answer. Deciding which path a question needs is called routing.

**Built:** the `ask` command now routes every question itself. Try the three kinds:

```
.\.venv\Scripts\python.exe -m filinglens.ask "Compare NVIDIA's and AMD's gross margins since 2021"
.\.venv\Scripts\python.exe -m filinglens.ask "What supply chain risks did Apple describe in 2024?"
.\.venv\Scripts\python.exe -m filinglens.ask "How did Apple's R&D spending change since 2021, and why?"
```

Each answer prints the route, the SQL, the result table and the time of each stage; trend questions open a chart in your browser. Everything is explained in `docs/03_text_to_sql.md`.

### What gets built

| # | Part | What it does |
|---|---|---|
| 3.1 | Schema description | A clear description of the tables, columns, units and metric definitions for the LLM to read |
| 3.2 | Text-to-SQL | The LLM writes a SQL query; the database runs it |
| 3.3 | Safety checks | Read-only connection, SELECT statements only, allowed tables only, row limit. The query is parsed and checked before it runs (`sqlglot`). If it fails, the error goes back to the LLM for one retry. |
| 3.4 | Router | The LLM labels each question *numbers*, *text* or *both* |
| 3.5 | Combined answers | For *both*: runs SQL and RAG, then writes one answer with the numbers and the cited explanation |
| 3.6 | Charts | Trend questions automatically get a Plotly line chart |

### Your checklist
- [ ] Ask 20 number questions; check each answer against `financials_wide` yourself
- [ ] Try to break it: ask for something not in the data, or something that would delete data. It should refuse or say it doesn't know.
- [ ] Read `docs/03_text_to_sql.md`, commit

**Done when:** at least 17 of 20 number questions are exactly right, and unsafe or impossible questions are handled gracefully.

**Interview topics:** Why not let the LLM answer numbers directly? How do you stop an LLM from running a dangerous query? What happens when the router picks the wrong path?

---

## Step 4: Research study (week 4)

**Goal:** answer a real research question with data: *does a change in the wording of a company's annual report predict its stock returns?* This repeats, at small scale, the finance paper **"Lazy Prices"** (Cohen, Malloy & Nguyen, 2020), which found that companies whose reports change a lot tend to underperform.

**Why:** this is the data science half of the project: hypothesis, features, statistics and honest conclusions.

### Before you start: get more companies
14 companies × 5 years is only about 70 data points, too few for statistics. Add 50–90 more companies from different industries to `config/companies.yaml` and re-run the pipeline (about 30–60 minutes the first time).

### What gets built

| # | Part | What it does |
|---|---|---|
| 4.1 | Text change score | For each company-year: how different is this year's Risk Factors section from last year's? (embedding similarity, plus simple word overlap as in the paper) |
| 4.2 | Tone score | FinBERT, a model trained on financial text, rates the sentiment of the MD&A section. Runs on your GPU. |
| 4.3 | Future returns | Stock return over 3, 6 and 12 months **starting the day after** each report was published, minus the market's return (SPY) over the same period |
| 4.4 | Statistics | Correlations, a regression (statsmodels), and "high change vs low change" group comparisons |
| 4.5 | Topic trends | How often reports mention "artificial intelligence", "tariffs", "supply chain", "cybersecurity", by year, with charts |
| 4.6 | Write-up | A notebook and `docs/04_research_findings.md` with charts, results and limitations |

### Pitfalls you'll learn to avoid
- **Look-ahead bias:** using information that wasn't public yet. That's why returns start after the publication date.
- **Survivorship bias:** picking companies that are big and successful *today* makes past returns look better than they really were. Mention it as a limitation.
- **Small samples and p-values:** with few data points, a result can appear by chance. Report your results honestly, even "no clear effect". Interviewers respect that.

### Your checklist
- [ ] Expand the company list and re-run the pipeline
- [ ] Run the notebook; make sure you can explain every chart
- [ ] Write the "Limitations" section yourself, in your own words
- [ ] Commit

**Done when:** you have a findings write-up with at least 2 charts, a statistics table and a limitations section.

**Interview topics:** What is look-ahead bias? What is survivorship bias? What does a p-value mean? What would you do with 10× more data?

---

## Step 5: Evaluation and better search (week 5)

**Goal:** *measure* how good the system is, then improve it with evidence. This step produces the numbers for your resume.

**Why:** most student projects say "it works". Saying "I measured it, here's the before and after" is what makes yours stand out.

### What gets built

| # | Part | What it does |
|---|---|---|
| 5.1 | Test set | About 100 questions with known answers: 40 text (with the passage that answers it), 40 numbers (value checked against the 10-K), 20 mixed. An LLM can draft questions from real passages, but **you check every one by hand**. Saved in `eval/questions.jsonl`. |
| 5.2 | Search quality | **Recall@5**: how often the right passage is in the top 5 results. **MRR**: how high it ranks. |
| 5.3 | Hybrid search | Adds BM25 keyword search (good at exact terms like "Item 1A" or product names) to vector search, and merges the two lists (Reciprocal Rank Fusion) |
| 5.4 | Reranker | A cross-encoder model re-scores the top ~20 results more carefully and puts the best first |
| 5.5 | Answer quality | Faithfulness (does the answer stick to the sources?) via an LLM grader, exact match for numbers, router accuracy, plus speed and cost per question |
| 5.6 | Results table | Each search setup compared side by side in the README |

Example of the table you'll produce (your real numbers will differ):

| Search setup | Recall@5 | MRR |
|---|---|---|
| Vector only | ? | ? |
| BM25 only | ? | ? |
| Hybrid | ? | ? |
| Hybrid + reranker | ? | ? |

### Your checklist
- [ ] Check every test question by hand
- [ ] Look at 10 failures and write down *why* each one failed
- [ ] Commit, with the results table in the README

**Done when:** you have a results table with at least 3 setups compared, and you can explain the differences.

**Interview topics:** What is Recall@k? Why does hybrid search beat vector-only search on some questions? What are the risks of using an LLM to grade an LLM?

---

## Step 6: Web app and deployment (week 6)

**Goal:** a live link anyone can open: this is what goes on your resume.

### What gets built

| # | Part | What it does |
|---|---|---|
| 6.1 | API | FastAPI backend: an `/ask` endpoint that runs the whole system |
| 6.2 | Web interface | Streamlit app with 3 tabs: **Ask** (answer, citations, chart), **Explore** (pick a company and metric, see the chart), **Research** (your findings) |
| 6.3 | Docker | Packages the app so it runs the same on any computer (needs Docker Desktop on Windows) |
| 6.4 | Deployment | Hugging Face Spaces (free). Uses an LLM API, since there's no GPU there. Your API key goes in the Space's secret settings, never in the code. |
| 6.5 | Automatic tests | GitHub Actions runs `pytest` every time you push code |
| 6.6 | Polished README | Demo GIF, architecture diagram, results table, live link |

### Your checklist
- [ ] Open the live link on your phone and ask 5 questions
- [ ] Send the link to a friend; fix whatever confuses them
- [ ] Commit

**Done when:** a stranger can open the link and get a good answer without your help.

---

## Final: resume, demo and interview prep

- [ ] **Resume bullets.** Use this pattern, with your real numbers from Step 5:
  > - Built **FilingLens**, an LLM analyst over 250+ SEC 10-K filings that routes each question to hybrid-search RAG (BM25 + embeddings + reranker) or text-to-SQL over DuckDB. Raised Recall@5 from **X% to Y%** on a 100-question test set I built.
  > - Tested whether changes in 10-K wording predict stock returns across **N** companies and 6 years (FinBERT sentiment, embedding similarity, regression): **your result**.
  > - Deployed with FastAPI, Streamlit and Docker on Hugging Face Spaces, with cited answers and automated tests (**live link**).
- [ ] **2-minute demo video** (screen recording): problem, live demo, results table, one thing you learned
- [ ] **LinkedIn post** with the video and the link
- [ ] **Practice explaining**: the 30-second version, the architecture diagram, and one hard bug (the Salesforce fiscal-year bug from Step 1 is a good one)
- [ ] **Prepare for these interview questions:**
  - Walk me through what happens when a user asks a question.
  - How do you prevent hallucinations? When does your system still get it wrong?
  - Why did you choose that chunk size / embedding model / LLM?
  - How did you evaluate it? What would you improve next?
  - How would you scale this to all 5,000+ US public companies?
  - What is look-ahead bias, and how did you avoid it?

---

## Libraries added along the way

| Step | Libraries |
|---|---|
| 2 | beautifulsoup4, lxml, sentence-transformers, torch (GPU build), chromadb, an LLM client or Ollama |
| 3 | sqlglot, plotly |
| 4 | transformers (FinBERT), statsmodels, scikit-learn, scipy, matplotlib, ipykernel (notebooks) |
| 5 | rank-bm25, a cross-encoder reranker model |
| 6 | fastapi, uvicorn, streamlit, Docker Desktop |
