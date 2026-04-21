| correctness | evidence | settings |
| :--: | :--: | :--: |
| 39 | 0.2365 | bge-base-en-v1.5, parent = 512, child = 128, overlap = 100, ms-marco-MiniLM-L-6-v2 |
| 36 | 0.2265 | bge-large-en-v1.5, parent = 512, child = 128, overlap = 100, mxbai-rerank-base-v1, HyDE_256 |
| 40 | 0.2269 | bge-large-en-v1.5, parent = 512, child = 128, overlap = 100, mxbai-rerank-base-v1, HyDE_256 |
| 37 | 0.2064 | bge-base-en-v1.5, parent = 320, child = 128, overlap = 80, bge-reranker-base, HyDE_32 |
| 47 | 0.1768 | bge-large-en-v1.5, parent = 320, child = 128, overlap = 80, bge-reranker-base, retrieve_15_of_50, HyDE_with_augmented_question |
| 29 | 0.2311 | bge-large-en-v1.5, (320, 128, 80), bge-reranker-base, retrieve_5_of_50, HyDEv2_with_augmented_question, 仔細打磨prompt |
| 27 | 0.2389 | bge-large-en-v1.5, (0.75, 1024, 256, 64), bge-reranker-base, retrieve_5_of_50, HyDEv2_with_augmented_question, 仔細打磨prompt |
| 19 | 0.2131 | bge-large-en-v1.5, (0.85, 1024, 256, 100), bge-reranker-base, retrieve>=3, HyDEv2_with_augmented_question, 仔細打磨prompt, penalty=1.5 |
||| bge-large-en-v1.5, (0.85, 1024, 256, 100), bge-reranker-base, retrieve>=5_of_40, HyDEv2_128 |
