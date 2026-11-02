# Step 5: measuring the search and the answers

Part 1 measures the search and improves it. [Part 2](#part-2-measuring-the-answers) measures the answers. [Part 3](#part-3-fixing-what-part-2-found), at the end, fixes what Part 2 found and measures again on a free local model.

## Part 1: the search

**The question:** when someone asks about a company's annual report, does the search find the passage that answers it? Until now the honest answer was "it seems to". This step replaces that with a number, then uses the number to choose between four ways of searching.

**What was built:**

| #   | Part                    | What it does                                                                                                            |
| --- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 5.1 | Test set                | 50 questions, each with the passage(s) that answer it: `eval/questions.jsonl` (read them in `eval/questions_review.md`) |
| 5.2 | Search quality          | Recall@5 and MRR for every search setup, one row per question in `eval/search_results.csv`                              |
| 5.3 | Keyword search + hybrid | BM25 keyword search, merged with the vector search by Reciprocal Rank Fusion                                            |
| 5.4 | Reranker                | A cross-encoder that re-scores the 20 best results                                                                      |
| 5.6 | Results table           | Below, and in the README                                                                                                |

Part 2 measures the _answers_: 40 number questions checked against the database, faithfulness, router accuracy, cost per question.

---

## The answer

| Search setup                  | Recall@5 | MRR  | Right passage ranked 1st | In the top 8 (what the LLM sees) | Time per question |
| ----------------------------- | -------- | ---- | ------------------------ | -------------------------------- | ----------------- |
| Vector only (Step 2)          | 0.86     | 0.61 | 21 of 50                 | 0.94                             | 323 ms            |
| Keyword only (BM25)           | 0.90     | 0.80 | 35 of 50                 | 0.94                             | 16 ms             |
| **Hybrid (vector + keyword)** | **0.94** | 0.77 | 32 of 50                 | 0.94                             | 345 ms            |
| Hybrid + reranker             | 0.94     | 0.81 | 37 of 50                 | 0.94                             | 454 ms            |

**Hybrid search is now the default.** Compared with vector search alone, it put the right passage *higher* on 22 questions and lower on 1. If the two setups were equally good, a split this lopsided would happen less than 0.1% of the time (a sign test), so the better ranking is real. In the top 5 it found 4 answers vector missed and missed none; that alone would happen by luck about 13% of the time, so it is promising but not proven with 50 questions. (Before the second review it was 6 to 0; see "The test set was wrong, too".)

**What it does _not_ show:** that the LLM now sees more right passages. The app hands the LLM 8 passages, and within the top 8 every setup finds 94% of answers. The gain is in _ranking_: the right passage is now first for 32 questions instead of 21. That matters because the LLM leans on the first sources, because the web app will show the top few, and because a smaller k (cheaper prompts) only works if the best passage comes first.

**Why not the reranker?** It ties hybrid on Recall@5 and is slightly ahead on MRR (0.81 vs 0.77), but that difference is one or two questions, which is noise at this size (head to head in the top 5: 1 question better, 1 worse; ranked higher on 11, lower on 7). It costs about 100 ms more per question and another model to load, and it made one question much worse (below). It stays available (`mode="rerank"`) for when a bigger test set can settle it.

**Why is keyword search so strong here?** Partly real: 10-K questions are full of exact names (GUARD, Steamboat Willie, RIDEA, Cerro Negro) that BM25 matches perfectly and embeddings blur. Partly the test set: an LLM wrote the questions while reading the passage, so they share its words even when told not to. Real users phrase things more loosely (see the Apple R&D question below, written by hand, where keyword search fails). Read the keyword numbers as an upper bound.

---

## How it was measured

### The test set (5.1)

1. **Pick passages.** One random passage per company (`seed 5`), from Business, Risk Factors or MD&A, at least 150 words long. Companies are taken in random order until there are 50 questions (one, t38, was dropped later).
2. **Draft a question for each** with the LLM (gpt-oss-120b on Groq). It was told: name the company and fiscal year, ask about something specific in _this_ passage (not "what risks does Apple face?"), use your own words, and reply SKIP for boilerplate. It skipped none.
3. **Check every question against its passage.** All 50 answers really were in their passages. 16 questions were edited in the end (14 at this stage, 1 more after the failure analysis, 1 in the second review), and every edit is recorded in the file (`draft` = the LLM's version, `edit` = why):
   - 6 named no year (the rule said to). Without a year the search covers all six reports, and Risk Factors repeat almost word for word every year, so the search could find the 2023 copy of the right paragraph and be marked wrong.
   - 2 named no company the filters know (one said only "GUARD"), and 1 named the wrong company (Vivint for Vivmark Residential).
   - 1 put the event in the wrong year (a 2022 gain asked about "in fiscal 2023").
   - 4 copied long phrases from the passage, which flatters keyword search. One of these rewrites (Realty Income) turned out too vague and was rewritten again, and one question that hadn't been edited (Duke Energy) was too broad; see "the test set was wrong" below.
   - One question was deliberately left as drafted although the filters miss it: it calls Booking Holdings "Booking", which the name matcher ignores because it's an everyday word. Real users say "Booking", so this is a fair test of the whole system.
4. **One question written by hand** (`h01`): _"Why did Apple's R&D spending increase in fiscal 2022?"_, the retrieval failure found in Step 2 (docs/02_rag.md, failure mode 1).

5. **A second review (28 Oct 2026)** re-read all 51 against their passages. Every answer was there. Three changes:
   - **t38 dropped** (Boeing): the question restated the risk factor's heading, generic legal text every setup ranked 1st. It tested nothing.
   - **t33 rewritten** (American Express): it asked whether credit-loss reserves need judgment. The auditor's note in the same report says "a significant amount of judgment" too, so the question had two right answers. It now asks about the passage's specific point (what happens if the models are wrong).
   - **t41 got a second right passage** (JPMorgan): its list of unrest and hostilities is also in the neighbouring chunk. The verbatim-overlap rule missed it because the drafted answer rephrased the list.

   Every question now has `checked: true`. No finance expert reviewed them (Limitations).

### The metrics (5.2)

Each setup returns its top 10 passages for each question. The **rank** is the position of the first relevant passage (1 = top; none = not in the top 10).

- **Recall@5**: the share of questions with a relevant passage in the top 5. Easy to explain: "it found the answer for 94% of questions".
- **MRR (mean reciprocal rank)**: the average of 1/rank (rank 1 → 1, rank 2 → 0.5, rank 4 → 0.25, missed → 0). It rewards putting the answer _first_, not just somewhere near the top.

The company and year filters come from the question, exactly as in the app, so a filter mistake counts against the search too. Timings are for the second of two identical searches: the first search of a company-year reads its vectors from disk, and in the first run whichever setup went first paid for that (vector search looked 2.5× slower than hybrid, which includes it). Times on a laptop also vary between runs (an earlier run was about twice as fast for every setup), so compare the setups with each other, not with other systems.

---

## The four search setups

**Vector search (Step 2).** Question and passages become vectors; the closest vectors win. Matches meaning ("chip shortages" ≈ "constrained semiconductor supply"), but treats a rare name like "GUARD" as just another word.

**Keyword search, BM25 (`rag/keyword_search.py`).** Finds passages containing the question's words and scores each match by three ideas:

- _rare words count more_ ("cybersecurity" says more than "company"): idf;
- _more mentions help, with diminishing returns_ (the 10th "tariff" adds little): tf with saturation (k1 = 1.2);
- _long passages are scored down a bit_, since they contain more words by chance (b = 0.75).

The index is a big table, one row per chunk and one column per word (41,007 words appearing in at least 2 of the 190,309 chunks), stored as a sparse matrix (only non-zero cells are kept). A search adds up the columns of the question's words: about 9 ms for all 190,000 chunks. It's built in 28 seconds by `build_index` and saved as a 170 MB file. "R&D", "AT&T" and "P&G" are kept as single words; very common words ("the", "how", "did") are dropped. There's no stemming yet, so "spending" doesn't match "spend" (one of the failures below).

_Why not DuckDB's built-in full-text search?_ It would need an extension download, and the database connection the LLM's SQL runs on is locked so it can't load extensions (a Step 3 safety rule). Building BM25 from scikit-learn and scipy, both already installed, kept that lock intact.

**Hybrid: Reciprocal Rank Fusion.** Each search returns its top 20. Every passage scores 1 / (60 + its rank) in each list, summed. Only ranks are used, because a BM25 score (0 to ~50) and a cosine similarity (0 to 1) aren't comparable. A passage both searches rank well comes out on top. (The 60 is from the paper that introduced RRF; it makes rank 1 and rank 2 count almost the same, so one search can't dominate.)

**Reranker (`rag/rerank.py`).** A cross-encoder reads the question and one passage _together_ and scores how well the passage answers it. More careful than comparing two separate vectors, but one model run per passage, so it only re-scores the hybrid top 20. Model: `cross-encoder/ms-marco-MiniLM-L6-v2` (90 MB), trained on Bing search questions and web passages.

---

## Ten failures, and why each happened

| #   | Question (setup that failed)                                                                | Why                                                                                                                                                                                                                                            |
| --- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Berkshire: why did **GUARD's** premiums fall? (vector: 6th)                                 | The decisive word is a rare name. The vector ranked passages about _other_ Berkshire insurers' falling premiums higher. Keyword search ranked it 1st. **This is the case for hybrid search.**                                                          |
| 2   | Sempra: who becomes CFO in March 2025? (vector: 8th)                                        | The answer is a footnote under a table of executives. Names and job titles in a table make a vague vector.                                                                                                                                     |
| 3   | Philip Morris: what offset the Americas' gains in 2024? (vector: 8th)                       | The same report also discusses 2023 vs 2022 in nearly the same words, and the vector picked the older comparison. The year filter can't help: it's the same report.                                                                            |
| 4   | Martin Marietta: utilization of the Texas cement plants? (vector: 7th)                      | One specific figure (80%) inside a passage mostly about something else.                                                                                                                                                                        |
| 5   | Realty Income: interest rate on its borrowings in 2023 vs 2022? (all setups: missed)        | The answer is one row of a table, and a real estate company's report uses "weighted average interest rate" in dozens of places.                                                                                                                |
| 6   | PepsiCo: what investment does it need to keep up with consumer tastes? (all setups: missed) | The risk factor is split across two chunks: the heading and "consumer preferences" are in one, "data analytics" is in the next. And "tastes" ≠ "preferences".                                                                                  |
| 7   | Gilead: what acquisition was announced in December 2020? (vector and hybrid: missed)        | The report describes several deals; the deciding detail is a month. Only the reranker (2nd) picked the right one.                                                                                                                              |
| 8   | Apple: why did R&D spending increase in 2022? (keyword: missed; reranker: 10th)             | Keyword: the passage says "expense" and "growth", the question says "spending" and "increase" (no stemming). Reranker: it preferred risk factors _about_ investing in R&D over the passage that _explains_ the change. Vector and hybrid: 3rd. |
| 9   | Freeport: how did it change operations when COVID began? (keyword: 7th)                     | A paraphrase: the question's words aren't the passage's words. This is the case for keeping the vector search in the mix.                                                                                                                      |
| 10  | Seven questions where the "wrong" passage was right (all setups)                            | See below. The test set was wrong, not the search.                                                                                                                                                                                             |

**Patterns:** vector search fails on rare names and on answers buried in tables; keyword search fails on paraphrases; the two fail on _different_ questions, which is exactly why merging them helps. What no setup fixes: answers split across chunk boundaries (6), and answers that are one row of a table (5). Those point at the chunking, not the search.

## The test set was wrong, too

The first run gave vector search Recall@5 = 0.73. Reading the failures showed that for 7 questions the search had found a **different passage that answers the question just as well**: a note to the financial statements repeating an MD&A figure (AMD's $200 million loan, Phillips 66's $2,831 million gain), the MD&A repeating a Business section fact (Gilead's MYR deal), or a Risk Factors passage that answers a broad question better than the drafted one (Monster Beverage). Those passages were added to the question's `relevant` list, each with a note saying why.

Two more fixes, applied to every question, not just the failures:

- **Chunk overlap.** Neighbouring chunks share a few sentences (Step 2 chunking), so a sentence near a boundary is in two chunks. Any chunk of the same report containing the answer sentence word for word now counts too. That added 6 passages. (Short, generic answers like "$57.3" or "higher marketing costs" were excluded: they match unrelated passages.)
- **Too-broad questions were narrowed.** "Why might Duke Energy's sales decline?" has several right answers in the report (weather, the economy, grid access). It now asks about grid access only.

After these fixes vector search scored 0.82, not 0.73. The second review found two more of the same kind (t33, t41 above), and vector went to 0.86: two of hybrid's six "wins" over vector had been label mistakes. **Lesson: when a search "fails", first check the label.** A test set with one right passage per question undercounts, and it can hurt some setups more than others.

---

## Limitations (say these before an interviewer does)

1. **50 questions is small.** Only two differences are clearly beyond luck: hybrid and keyword both rank the answer higher than vector (22 vs 1, and 23 vs 5). In the top 5, hybrid vs vector (4 vs 0), hybrid vs keyword (2 vs 0) and reranker vs hybrid (1 vs 1) are not.
6. **The questions were checked twice, but not by a finance expert.** The checks were: is the answer in the passage, is the question specific, does another passage answer it too. A domain expert might still call some questions unrealistic.
2. **The LLM wrote the questions while reading the answer**, which favours keyword search (above). Hand-written questions from real users would be the better test.
3. **Some right passages are still unlabelled.** Only failures were checked for alternative answers. One more turned up when the app answered a question (Berkshire's GUARD question cited a second passage with the same reason). So every setup probably scores a little better than shown.
4. **One reranker model was tried**, a small one trained on web search. A larger one trained on more varied text (e.g. bge-reranker-base, 1.1 GB) might do better.
5. **No stemming** in the keyword search, and only one setting each for k1, b, the RRF constant and the 20 candidates. Tuning them on these same 50 questions would overfit: that needs a separate set of questions.

## What next

**Later:** a stronger reranker; stemming; 20 hand-written questions in the way people really ask.

**Rerun everything:**

```
python -m filinglens.build_index --keywords-only        # the keyword index (~30 s)
python -m filinglens.evaluation.search_eval             # the table above (~1 min)
python -m filinglens.evaluation.draft_questions --n 50  # new drafts (refuses to overwrite checked ones)
python -m filinglens.evaluation.draft_questions --review-only   # rewrite the review file after edits
python -m filinglens.ask --search-only --search vector "In Berkshire Hathaway's fiscal 2025 report, why did GUARD's premiums written fall sharply?"
                                                        # see one setup's results; try --search hybrid
```

---

## Interview questions

- **What is Recall@k?** The share of test questions where a correct passage is in the top k results. Recall@5 = 0.94 means 47 of 50 questions had the answer in the top 5.
- **Why MRR as well?** Recall@5 treats rank 1 and rank 5 the same. MRR averages 1/rank, so it rewards putting the answer first. Hybrid and vector differ more on ranking than on recall@8.
- **Why does hybrid search beat vector search?** They fail on different questions. Embeddings blur rare names and exact figures ("GUARD", "80% utilization"); keywords miss paraphrases ("changed operations" vs "curtailing mining"). Merging the ranked lists keeps each one's hits.
- **Why merge by rank (RRF) and not by score?** BM25 scores and cosine similarities are on different scales, and the scale changes from question to question. Ranks are comparable, and RRF needs no tuning.
- **How do you know the improvement isn't luck?** On the same 50 questions, hybrid ranked the right passage higher than vector on 22 and lower on 1: a sign test gives p < 0.001. For the top 5 alone it's 4 vs 0, p ≈ 0.13, so I call that promising, not proven. It used to be 6 vs 0 (p ≈ 0.03) until a second review of the labels showed two of those wins were labelling mistakes.
- **How did you build the test set, and what are its weaknesses?** An LLM drafted questions from random passages; every one was checked against its passage twice; 16 were edited and 1 dropped; one hard question came from a real failure. Weaknesses: the LLM's questions share the passages' words, 50 is small, some correct passages are unlabelled, and no finance expert reviewed them.
- **What surprised you?** Seven "failures" were the search finding a different correct passage. The labels were wrong, not the search. I now check the label before blaming the model.
- **Why didn't you use the reranker, when rerankers are standard?** On this test set it didn't beat hybrid (1 question better, 1 worse) and cost 100 ms more. It even pushed the Apple R&D answer from 3rd to 10th, preferring passages _about_ R&D over the one _explaining_ the change. I'd retest it with a bigger model and more questions rather than add it because it's standard.
- **What are the risks of using an LLM to write or grade the test?** It shares blind spots with the system it tests (here: wording that favours keyword search), it can be confidently wrong (it named the wrong company once), and it needs checking by hand. Part 2's grader scored 25 of 25 on its test and still let a real mistake through, because grading a whole answer at once made it lenient.

---

# Part 2: measuring the answers

**The question:** a good search doesn't prove the answers are right. Are the numbers correct, does the router pick the right path, do the sources say what the answers claim, and what does a question cost?

| What was measured | Result | On what |
| --- | --- | --- |
| Number answers with every expected value | **38 of 40** | Values from the database, not from an LLM |
| SQL result tables with every expected value | 39 of 40 | The same questions |
| Mixed answers with the right figure | **20 of 20** | "What happened, and why?" questions |
| Router chose the right path | **107 of 110** | Labelled questions. Two misses went to both paths, which still reads the reports; one went to the database alone |
| Cited sentences a second AI model confirmed | **96 of 105 (91%)** | 39 text and mixed answers, one sentence at a time |
| The same, text answers only | 25 of 25 | |
| The same, mixed answers only | 71 of 80 (89%) | |
| Cost per question at list price | **$0.0013** | $1.27 per 1,000: Groq, gpt-oss-120b, $0.15 in and $0.60 out per million tokens |

| Route | LLM calls | Tokens | Cost | Median time |
| --- | --- | --- | --- | --- |
| Numbers | 3 | 5,800 | $0.0011 | 5 s |
| Text | 2 | 4,100 | $0.0008 | 3 s |
| Both | 3 | 9,900 | $0.0020 | 20 s |

All of this was measured on the full system: hybrid search, on gpt-oss-120b. The online version uses keyword search and Gemini 3.5 Flash Lite. Its search and figures are measured in `docs/06_web_app.md`, and its faithfulness hasn't been graded.

## How it was measured

**Number questions (`eval/answer_questions.jsonl`).** 40 hand-written questions across lookups, ratios, comparisons, multi-year calculations, calendar-year traps, rankings and stock prices, plus 20 mixed ones. Each has its own SQL, so the expected value comes from the database. Writing them caught three bad questions at once: two asked for figures the database doesn't hold (Amazon's R&D, Caterpillar's liabilities).

**Is the database itself right?** A test against the database can't catch a wrong database. So 26 reported figures were searched for in the 10-K text: 25 appear word for word. The last is Netflix's free cash flow, which is calculated, not printed in that form.

**Scoring a number.** A value counts if the answer shows it rounded to as many decimals as it displays: "$212 billion" passes for 211.915, and "$211.92 million" fails. A number must also be within 1%, so a list number "1" can't stand for 1.2. The same check on the SQL's table shows which step failed: a bad query, or a good query reported wrongly. Scores are computed when results are read, so a fix to the scoring costs no LLM tokens. That paid off three times, because my scoring had bugs too (see below).

**Faithfulness.** The answers came from Groq, and a different model (Gemini 3.5 Flash Lite) graded them, so no model marks its own work. Each cited sentence is shown with only the sources it cites, and the grader says whether every fact in it is supported.

**Testing the grader first.** `eval/grader_check.jsonl` has 25 sentences with a known verdict: 12 checked as correct, 11 made wrong on purpose (a changed figure, a reversed direction, the wrong passage, an added claim) and 2 real mistakes by the app. It scored 25 of 25.

## What the failures showed

| Question | What happened | Why it matters |
| --- | --- | --- |
| Eli Lilly's total net income, 2021 to 2023 | Both SQL attempts failed, and the app said so | A visible failure beats a wrong number |
| The three biggest R&D spenders in tech | The table was right ($16.05bn); the answer rounded it to $16.1bn | The LLM was told to copy figures exactly. Rules in a prompt aren't guarantees |
| Goldman's Global Markets revenue | Sent to the database, which correctly said it doesn't hold segments, and the app stopped | The reports had the answer, so a wrong route was a dead end. Now a "can't answer" falls back to the reports, and the router's mistake still counts in its score |
| Merck's 2023 net income | Right figure, wrong reason: it blamed the Organon spin-off, a 2021 event | **A right number doesn't mean a right explanation** |

Three number answers also did arithmetic against the rule (Coca-Cola's lead over PepsiCo, for one). All three were correct, but only because the sums were easy.

**Text answers were all supported, and mixed answers weren't.** All 9 unsupported sentences are in mixed answers. When a question gives the LLM a figure and asks why, it wants to connect them, and it reaches further than the sources go. That's the system's clearest weakness.

**The grader's own story.** Graded one answer at a time, it reported 97% and passed both known mistakes, although it had caught them when tested alone. Asked about many sentences at once it was lenient, so it was changed to one sentence per call, and the score fell to 91%. Merck is now caught. The exact Chevron sentence still isn't: the passage says the absence of a charge offset the decline, and the answer says it reduced earnings. Treat 91% as a ceiling.

**Three bugs in my own scoring,** all found by reading failures: a table's "Visa" check looked for a name in a column of tickers, CSV commas turned `2024-12-31,248.62` into 31,248.62, and "$17.528 billion" was flagged as new because the table had 17528000000.

## Limitations

1. **Small and self-written.** I wrote the number questions knowing the schema, so they're fairer to the SQL than real users are.
2. **The grader isn't the truth.** It passed 25 of 25 in testing and still missed a real mistake. It also can't tell a correct explanation from one that's merely supported by a passage.
3. **43 claims carry no citation.** Mostly bullets under a cited sentence, but nothing checks them.
4. **Sign errors can pass.** The number check ignores minus signs, because answers write "a loss of $2 billion". I read all 40 number answers by hand and found none.
5. **Earnings per share wasn't tested.** It was wrong across stock splits when these questions were written, so it was kept out of them. It's now fixed (`docs/01_data_pipeline.md`, Trap 4), but no question covers it yet.
6. **Free-tier limits.** Only 20 of the 50 text questions were answered in full, to stay within the daily tokens. The other 30 only test the router.

**Rerun part 2:**

```
python -m filinglens.evaluation.answer_eval               # asks whatever is missing, then the summary
python -m filinglens.evaluation.answer_eval --summary     # no LLM calls
python -m filinglens.evaluation.faithfulness --check-grader
python -m filinglens.evaluation.faithfulness
```

## Interview questions, part 2

- **How do you check a number answer automatically?** The expected value comes from my own SQL. I compare it with the answer, allowing only for the rounding the answer shows, and run the same check on the SQL's table to separate a bad query from a bad write-up.
- **What's faithfulness, and why isn't accuracy enough?** It asks whether the sources support what the answer says. An answer can have the right number and an invented reason, as my Merck answer did, and a number check won't see that.
- **How did you know the grader could be trusted?** I tested it on sentences with known verdicts, including wrong ones I planted. It still wasn't enough: it passed real mistakes when grading whole answers, so I changed it to one sentence at a time, and the score dropped from 97% to 91%. I reported the lower number.
- **What's the weakest part of the system?** Explaining why something changed. Text answers were fully supported, but all the unsupported sentences were in mixed answers, where the LLM links a figure to a reason.
- **What does it cost?** About $1.27 per 1,000 questions at list price. A mixed question costs about twice a number question, because it reads eight passages too.
- **What would you fix first?** Have the answer step quote its reason, so an explanation can be checked word for word.

---

# Part 3: fixing what Part 2 found

**The question:** Part 2 found weaknesses. Did the fixes hold up, on questions nobody tuned them on?

## What changed

| Fix | Why |
| --- | --- |
| A figure check on every answer, with one rewrite if a figure isn't in the evidence (`combine.py`) | Telling the model to copy figures wasn't a guarantee: it rounded $16.05bn to $16.1bn |
| Stricter rules for explaining a change | Every unsupported sentence in Part 2 was in an answer that linked a figure to a reason |
| SQL rules, a worked example, and one retry when a column comes back empty | `WHERE` runs before `LAG`, so a filter on one year blanked the growth figure |
| Earnings per share put on one share basis across stock splits | `docs/01_data_pipeline.md`, Trap 4 |
| Income statement totals preferred over contract revenue | `docs/01_data_pipeline.md`, Trap 1. It corrected 18 values at American Express, NextEra and Duke |

**Tried and dropped:** forcing "why" questions to search only the MD&A. Stated causes fell from 113 to 108, and Intel, Chevron and JPMorgan came back almost empty.

## The result

Groq's free tier allows about 200,000 tokens a day and a full run needs about 330,000, so these runs used free models on the laptop's GPU. `qwen2.5:7b` came first. `qwen3:8b` replaced it after [the work below](#making-the-local-model-better). Both are far smaller than Part 2's `gpt-oss-120b`.

| What was measured | Part 2: before the fixes, `gpt-oss-120b` | After the fixes, `qwen2.5:7b` | After the fixes, `qwen3:8b` |
| --- | --- | --- | --- |
| Number answers with every expected value | 38 of 40 | 27 of 40 | **36 of 40** |
| SQL result tables with every expected value | 39 of 40 | 27 of 40 | 37 of 40 |
| Mixed answers with the right figure | 20 of 20 | 19 of 20 | **19 of 20** |
| Router chose the right path | 107 of 110 | 99 of 110, 5 of them harmful | 106 of 110, 3 of them harmful |
| Fresh mixed questions with the right figure | not asked | 18 of 19 | **18 of 19** |
| 12 new hard number questions | not asked | 6 of 12 | **12 of 12** |
| Cited sentences a second model confirmed | 96 of 105 (91%) | grading not finished | **66 of 76 (87%)**, and 41 of 46 (89%) on the fresh questions |
| Median time for a mixed question | 20 s, with free-tier waits | 28 s on an 8 GB laptop GPU | 56 s on the same GPU |

**This is not a before-and-after of the fixes.** The model changed as well as the code. Comparing the fixes fairly needs the same model on both sides, and that run hasn't been done. What this table does show: the system works on a free model that runs offline, and how much accuracy that costs. The README's headline numbers stay Part 2's, because the online app doesn't run these models.

**Grading.** Gemini graded Part 2 because Groq wrote those answers. These answers came from a local model, so Groq can grade them without marking its own work, and it doesn't share the live app's quota. It scored 25 of 25 on the grader test, the same as Gemini. Groq's daily limit stopped it after 13 of the 19 fresh answers, so Gemini graded the last 6. The two graders' scores aren't strictly comparable, so read 87% next to 91% as "about the same", not as a 4-point fall.

**Reading the unsupported sentences.** The grader marked 10 of the 76. I read them against their sources. Four are clearly real mistakes: Chevron's fall in income used to explain a fall in revenue (m05), a reason cited to a table that doesn't show it (m12), a figure more precise than the table gives (m19), and Merck's wrong reason again, with a different reason from Part 2's (m20). The other six aren't settled.

## Making the local model better

All 13 number failures on `qwen2.5:7b` were SQL: 10 queries that ran but were wrong, and 3 that never ran. One change was made at a time, and each was measured.

**1. A model that can think, for SQL only.** `qwen3:8b` can reason before it answers. On the SQL step alone, on those 13 questions:

| | Right | Median time per query |
| --- | --- | --- |
| Thinking off | 6 of 13 | 5 s |
| Thinking on | **11 of 13** | 45 s |

Thinking lost nothing and fixed five more. It stays off for the router and the written answer, because that's where it would add minutes and nothing failed there. Online models are untouched: they already think, at the levels set in `llm.py`.

**2. Thinking overflowed the context window.** The SQL prompt was about 5,330 tokens, and long thinking took two questions past the 8,192 window. Ollama doesn't report that, and the old check only looked at the prompt. The obvious fix was a bigger window, so I measured it:

| Window | On the GPU | Speed |
| --- | --- | --- |
| 8,192 | 100% | 38 tokens/s |
| 10,240 | 92% | 29 tokens/s |
| 12,288 | 87% | 19.5 tokens/s |

Once any of the model spills to the CPU, speed falls quickly. So the prompt shrank instead. The list of every company was 3,123 of its 5,315 tokens, and a question needs only the companies it names. On Ollama the prompt is now about 2,270 tokens, and the check counts the reply too. Groq's and Gemini's prompts are unchanged, because they can't be measured again within their free limits.

**3. A limit on replies.** During the full run one SQL call was still running after 10 minutes, and the timeout crashed the run. Rerun alone, that question took 37 s and came back right, so **the cause isn't known**. It could be endless thinking or a stall in Ollama. Replies are now capped at 4,096 tokens (the longest of 49 normal ones was 3,554), a reply that never finishes thinking is asked again without thinking, and a timeout shows a clear error instead of a crash.

**4. Dropped, with reasons.** Database views for calendar years and returns would have changed the live data to help at most 2 questions. Choosing worked examples per question isn't needed now the prompt fits. Treating `[1]` as `[DB]` isn't needed either: invalid citations fell from 33 answers to 1 of 80 without it.

**The 5 remaining misses.** Two are the scoring being strict, and they stay counted as wrong: 377.3% for a ratio of 3.77 (n18), and $0.37bn for $0.365bn (m20). Three are real: rounding before averaging (n30), the wrong fiscal year for a calendar year (n33), and a right table that the answer step said didn't hold the figure (n32).

## The fresh questions

`eval/fresh_questions.jsonl` was written after the fixes, so they couldn't be tuned on it. That only stayed true for part of it, and the order matters:

1. **The first 11 scored 3 of 11.** That's how the `WHERE` before `LAG` bug was found. After the fix they scored 10 of 11, but they aren't unseen any more.
2. **Eight more were written, unseen.** They scored 7 of 8. **That's the honest number.**
3. **The miss was the database's fault.** The answer said American Express grew 17%, and the expected value said 26%. Tracing it found the revenue bug, and the corrected figure is 17.4%. After the fix the score is 18 of 19, but that one point came from fixing what the question exposed, so it doesn't count as unseen.

The remaining miss is Freeport: 60.6% where the database gives 60.9%.

**A test can be wrong in either direction.** Here the expected value came from the database, and the database was wrong. A wrong answer may be a wrong test.

**These 19 can't tell the two local models apart.** Both scored 18 of 19, because every one is a simple one-year growth question. And 36 of 40 isn't an unseen score, because 13 of those 40 chose the thinking setup. So I wrote 12 new number questions, `f20` to `f31`, with expected values from hand-written SQL: 3 calendar-year, 3 multi-year, 2 ranking, 2 comparison, 1 ratio and 1 stock return. No change was tuned on them, and both models ran them on the same code:

| | Answers right | Result tables right | Median seconds |
| --- | --- | --- | --- |
| `qwen2.5:7b` | 6 of 12 | 3 of 12 | 21 |
| `qwen3:8b` | **12 of 12** | 12 of 12 | 61 |

**That's the number to quote for the local model, and 12 is a small test.** I also chose the question types after seeing where the small model was weak, so they're hard on purpose, not a random sample.

**The score only checks figures, so I read all 12 answers.** `qwen3:8b`'s tables and figures are all right, with no figure outside the table, but 3 of the 12 answers are worded badly:

- f20 gives Oracle's right figure, then says the database doesn't include calendar 2022.
- f22 gives Lowe's right figure, then says it's for the fiscal year and not the calendar year.
- f31 gets Microsoft's return right, then adds a reason "as noted in the annual report", with no report behind it. The number route has no sources, so nothing checks that sentence.

So 12 of 12 means the figures, not the answers. Calendar questions are the weak spot: the answer step doesn't always accept that a fiscal year can stand for a calendar year, which is also what went wrong on n32.

`qwen2.5:7b`'s 6 of 12 flatters it the other way. Three of those passed without a right table: one took its figure from the report text (f22), one did the subtraction itself from a broken table (f29), and one had the right difference next to two wrong revenue figures (f28), which the score doesn't look at.

## Bugs the small model found

A weaker model makes more mistakes, so it reaches code paths a good model never does. Both of these crashed the app:

| Bug | Fix |
| --- | --- |
| SQL the parser couldn't even split into words raised an error nobody caught | `check_sql` now treats it as an unsafe query, which retries |
| Two result columns with one name crashed pandas | A clear error goes back to the model for a retry |

## The damaged search index

The run died at one test question, the same one every time, with no Python error, only `access violation` inside ChromaDB. Each step of the diagnosis removed one suspect:

| Test | Result | So the cause isn't |
| --- | --- | --- |
| The same filter with a random vector | Worked | The filter |
| The real vector with no filter | Crashed | Any filter |
| Other questions, with the model loaded | Worked | The model, or the program's state |
| That one vector, with PyTorch not loaded | Crashed | Anything but the index |

Reading the stored vectors back then failed with "Error getting embedding". The first unreadable rows were Indian passages, the last ones added. So only searches whose path through the index reached the damage crashed, which is why it stayed hidden. The index had processed about 350,000 writes for 229,484 live passages, because each rebuild deleted and re-added vectors in place. I didn't trace the exact fault inside Chroma, so that cause is likely, not proven.

**The repair.** `build_index` writes into the existing index, so re-running it would have kept the damage. Instead the readable vectors were copied into a new folder and the Indian passages embedded again from their text. Before the swap: all 229,484 passages present, none differing from the database, copied vectors unchanged apart from rounding (lowest cosine 0.9999996), and the crashing question working. Recall@5 was the same for all four search setups, and every vector search rank was unchanged.

Part 1's table is from before the Indian passages were added. On the full index (`eval/search_results_after_india.csv`), keyword MRR is 0.79 and hybrid MRR is 0.79, because more passages change how rare each word counts as. Recall@5 didn't move.

**The online app was never affected:** it doesn't use Chroma. To rebuild from nothing, delete `data/processed/chroma` and run `build_index`.

## Limitations

1. **The fixes aren't proven on the original model.** Only a run on `gpt-oss-120b` can show that.
2. **18 of 19 isn't a clean score.** Use 7 of 8 for unseen questions on `qwen2.5:7b`, and 12 of 12 on `qwen3:8b`. Both samples are small, and the second counts figures only: 3 of those 12 answers are worded badly. The calendar wording problem isn't fixed, because the fix would change the prompt the live app uses, and that can't be measured again within the free limits.
3. **Faithfulness used two graders,** and six of the ten sentences marked unsupported weren't read to a conclusion. `qwen2.5:7b`'s grading was left unfinished, because that model was replaced.
4. **It needs a lot of memory.** With the model and the search index loaded, about 1 GB of 15.6 GB stayed free, and the first resumed run was stopped for low memory. `llama-server` reserved 14.2 GB for a 5.2 GB model, and I didn't find out why.
5. **One hang wasn't explained.** The limits stop it from blocking a run, but it didn't happen again, so they weren't tested on a real hang.
6. **Not every odd case was traced.** On this Windows laptop the app's server also crashes with an access violation when it's stopped after serving a visitor. It stayed healthy for 40 seconds after a visit, so visitors shouldn't see it, but the cause wasn't found, and the live app runs on a different system.

**Rerun part 3:**

```
$env:LLM_PROVIDER="ollama"; $env:LLM_MODEL="qwen3:8b"
python -m filinglens.evaluation.answer_eval --tag qwen3
python -m filinglens.evaluation.answer_eval --fresh --tag qwen3

$env:LLM_PROVIDER="groq"; $env:LLM_MODEL="openai/gpt-oss-120b"
python -m filinglens.evaluation.faithfulness --check-grader
python -m filinglens.evaluation.faithfulness --tag qwen3
python -m filinglens.evaluation.faithfulness --fresh --tag qwen3
```

## Interview questions, part 3

- **Your score dropped from 38 to 27. Did your fixes make it worse?** I can't tell from this run, because I changed the model too. The drop is mostly a 7B model failing at SQL. To measure the fixes I'd need the same model on both sides, so I don't claim they raised the score.
- **What's the difference between tuning data and unseen data here?** The first 11 fresh questions found a bug, so once I fixed it they stopped being unseen. Later, 13 of the 40 number questions chose the thinking setup, so 36 of 40 isn't unseen either. I wrote 12 harder questions and ran both models on them: 12 of 12 against 6 of 12. It's a small sample, I chose the hard question types on purpose, and when I read the answers 3 of the 12 were worded badly even though the figures were right.
- **How did you get a laptop model from 27 to 36 of 40?** Every failure was SQL, so I targeted that step only. I moved to a model that can reason first, and let it reason only when writing SQL, because that's where the errors were and thinking costs ten times the time.
- **Thinking overflowed the context window. Why not just make the window bigger?** I measured that: going from 8,192 to 12,288 tokens pushed part of the model onto the CPU and halved the speed. The list of every company was over half the prompt, so the local model now sees only the companies a question names.
- **Why did you switch grader?** The rule is that a model never grades its own work. Gemini was chosen because Groq wrote Part 2's answers. A local model wrote these, so that reason no longer held. I tested Groq on the same 25 known sentences before trusting it.
- **Tell me about a bug your tests found in your data.** A test marked a correct answer wrong. The model read 17% from American Express's report and the database said 26%, because the code had picked a revenue tag that leaves out interest income. Auditing all 99 companies found two more with the same cause.
- **How did you debug a crash with no error message?** I removed one suspect at a time: the filter, the query, the model, then everything but the index. It came down to one vector crashing an index with nothing else loaded.
- **Why not just re-run the index build?** It writes into the existing index, so the damage would stay. I built a new index in a separate folder, checked it against the database, and only then swapped it in.
- **Why use a weaker model at all?** It's free and has no daily limit, and its mistakes reached code paths a stronger model never hit. Two of the crashes it found could also have happened to a real user.
