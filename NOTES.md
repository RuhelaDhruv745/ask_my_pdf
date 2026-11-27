1,600/200 gave 34 chunks with 23 over the model's limit, and 800/120 gave 59 chunks with none over it. In step 5 you'll replace this with measured hit-rate numbers, which become your failure story.

Chunk size experiment (2 PDFs, 20 golden questions, k=6, cutoff 0.72)
- 1,600/200: 31 chunks, hit rate 16/16, 0/4 unanswerable below cutoff
- 800/120: 53 chunks, hit rate 15/16, 2/4 unanswerable below cutoff
- Expected 800/120 to be better because it fits the embedding model's token limit; the test did not show that.
- One question separates the hit rates, so the difference is not meaningful at n=16.
- A single distance cutoff cannot separate answerable from unanswerable at 800/120 (0.6880 vs 0.6633).
- Chose 800/120 for lower context size (about 4,150 vs 6,614 characters per query).
- Next: larger golden set, and a critic step to catch what the cutoff misses.