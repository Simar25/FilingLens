# Step 5 (part 1): measuring the search, then improving it

**The question:** when someone asks about a company's annual report, does the search find the passage that answers it? Until now the honest answer was "it seems to". This step replaces that with a number, then uses the number to choose between four ways of searching.

**What was built:**

| #   | Part                    | What it does                                                                                                            |
| --- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 5.1 | Test set                | 50 questions, each with the passage(s) that answer it: `eval/questions.jsonl` (read them in `eval/questions_review.md`) |
| 5.2 | Search quality          | Recall@5 and MRR for every search setup, one row per question in `eval/search_results.csv`                              |
| 5.3 | Keyword search + hybrid | BM25 keyword search, merged with the vector search by Reciprocal Rank Fusion                                            |
| 5.4 | Reranker                | A cross-encoder that re-scores the 20 best results                                                                      |
| 5.6 | Results table           | Below, and in the README                                                                                                |

Part 2 (still to do) measures the _answers_: 40 number questions checked against the database, faithfulness, router accuracy, cost per question.

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

   Every question now has `checked: true`. The review was done with an AI assistant's help, not by a finance expert (Limitations).

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
6. **The questions were checked with AI help, twice, but not by a finance expert.** The checks were: is the answer in the passage, is the question specific, does another passage answer it too. A domain expert might still call some questions unrealistic.
2. **The LLM wrote the questions while reading the answer**, which favours keyword search (above). Hand-written questions from real users would be the better test.
3. **Some right passages are still unlabelled.** Only failures were checked for alternative answers. One more turned up when the app answered a question (Berkshire's GUARD question cited a second passage with the same reason). So every setup probably scores a little better than shown.
4. **One reranker model was tried**, a small one trained on web search. A larger one trained on more varied text (e.g. bge-reranker-base, 1.1 GB) might do better.
5. **No stemming** in the keyword search, and only one setting each for k1, b, the RRF constant and the 20 candidates. Tuning them on these same 50 questions would overfit: that needs a separate set of questions.

## What next

1. **Step 5 part 2: answer quality.** 40 number questions with values checked against the database, router accuracy on labelled questions, faithfulness (does each cited source support its sentence?) with an LLM grader, and cost per question.
2. **Later:** a stronger reranker; stemming; 20 hand-written questions in the way people really ask.

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
- **What are the risks of using an LLM to write or grade the test?** It shares blind spots with the system it tests (here: wording that favours keyword search), it can be confidently wrong (it named the wrong company once), and it needs checking by hand. Part 2 uses an LLM grader and has to measure that grader against hand labels first.
