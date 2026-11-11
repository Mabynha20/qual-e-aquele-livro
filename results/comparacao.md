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
| Embeddings (minilm-livros) | val · auto | 1396 | 86.3% | 93.1% | 94.8% | 0.894 |
| Embeddings (minilm-livros) | teste · auto | 1410 | 87.3% | 93.8% | 95.5% | 0.903 |
| Embeddings (minilm-livros) | teste · manual EN | 30 | 33.3% | 46.7% | 56.7% | 0.407 |
| Embeddings (minilm-livros) | teste · manual PT | 30 | 0.0% | 0.0% | 0.0% | 0.008 |
| Híbrido (BM25 + minilm-livros) | val · auto | 1396 | 95.1% | 98.7% | 99.3% | 0.967 |
| Híbrido (BM25 + minilm-livros) | teste · auto | 1410 | 95.5% | 99.1% | 99.8% | 0.970 |
| Híbrido (BM25 + minilm-livros) | teste · manual EN | 30 | 40.0% | 70.0% | 86.7% | 0.544 |
| Híbrido (BM25 + minilm-livros) | teste · manual PT | 30 | 0.0% | 0.0% | 0.0% | 0.005 |
| Híbrido (BM25 + minilm-livros) + tradução | val · auto | 1396 | 95.1% | 98.7% | 99.3% | 0.967 |
| Híbrido (BM25 + minilm-livros) + tradução | teste · auto | 1410 | 95.5% | 99.1% | 99.8% | 0.970 |
| Híbrido (BM25 + minilm-livros) + tradução | teste · manual EN | 30 | 40.0% | 70.0% | 86.7% | 0.544 |
| Híbrido (BM25 + minilm-livros) + tradução | teste · manual PT | 30 | 43.3% | 70.0% | 80.0% | 0.561 |
