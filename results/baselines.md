| Método | Consultas | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|---|
| BM25 | val · auto | 1396 | 95.8% | 98.9% | 99.1% | 0.971 |
| BM25 | teste · auto | 1410 | 95.1% | 98.5% | 99.1% | 0.966 |
| BM25 | teste · manual EN | 30 | 36.7% | 70.0% | 76.7% | 0.488 |
| BM25 | teste · manual PT | 30 | 0.0% | 0.0% | 0.0% | 0.004 |
| Embeddings (all-MiniLM-L6-v2) | val · auto | 1396 | 53.3% | 66.2% | 71.3% | 0.593 |
| Embeddings (all-MiniLM-L6-v2) | teste · auto | 1410 | 53.5% | 66.3% | 70.1% | 0.596 |
| Embeddings (all-MiniLM-L6-v2) | teste · manual EN | 30 | 16.7% | 30.0% | 40.0% | 0.233 |
| Embeddings (all-MiniLM-L6-v2) | teste · manual PT | 30 | 0.0% | 0.0% | 0.0% | 0.004 |
