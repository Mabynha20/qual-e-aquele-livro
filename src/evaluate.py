"""
Avalia os buscadores com as consultas de validação e teste.

Métricas (para cada consulta, olhamos a posição do livro certo no ranking dos
~9,5 mil livros):
    Recall@k: % das consultas em que o livro certo aparece entre os k primeiros
    MRR:      média de 1/posição (1,0 se sempre em 1º; 0,5 se sempre em 2º...)

Os resultados são separados por tipo de consulta:
    auto (val/teste): frases dos resumos, sem nomes, mais fáceis
    manual EN/PT:     descrições vagas escritas à mão, mais realistas

Uso (na raiz do projeto, com a .venv ativa):
    python src/evaluate.py                                  # baselines
    python src/evaluate.py --model models/minilm-livros     # + modelo treinado, híbrido e híbrido com tradução

Salva a tabela em results/baselines.{csv,md} (ou results/comparacao.{csv,md} com --model).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from search import PROCESSED, BM25Searcher, EmbeddingSearcher, HybridSearcher, load_books
from translate import TranslatingSearcher

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
KS = (1, 5, 10)


def ranks_of_targets(scores, targets):
    """Posição (1 = primeiro) do livro certo de cada consulta."""
    target_scores = scores[np.arange(len(targets)), targets]
    return 1 + (scores > target_scores[:, None]).sum(axis=1)


def metrics(ranks):
    row = {f"Recall@{k}": (ranks <= k).mean() for k in KS}
    row["MRR"] = (1 / ranks).mean()
    row["n"] = len(ranks)
    return row


def query_groups(queries):
    """Subconjuntos de consultas avaliados separadamente."""
    return {
        "val · auto": (queries["split"] == "val") & (queries["kind"] == "auto"),
        "teste · auto": (queries["split"] == "test") & (queries["kind"] == "auto"),
        "teste · manual EN": (queries["kind"] == "manual") & (queries["lang"] == "en"),
        "teste · manual PT": (queries["kind"] == "manual") & (queries["lang"] == "pt"),
    }


def evaluate(searcher, queries):
    scores = searcher.score(queries["query"].tolist())
    ranks = ranks_of_targets(scores, queries["book_id"].to_numpy())
    rows = []
    for group, mask in query_groups(queries).items():
        rows.append({"método": searcher.name, "consultas": group, **metrics(ranks[mask.to_numpy()])})
    return pd.DataFrame(rows), ranks


def to_markdown(table):
    lines = ["| Método | Consultas | n | " + " | ".join(f"Recall@{k}" for k in KS) + " | MRR |",
             "|---|---|---|" + "---|" * (len(KS) + 1)]
    for r in table.to_dict("records"):
        values = " | ".join(f"{r[f'Recall@{k}']:.1%}" for k in KS)
        lines.append(f"| {r['método']} | {r['consultas']} | {r['n']} | {values} | {r['MRR']:.3f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Avalia os buscadores.")
    parser.add_argument("--model", help="pasta de um modelo treinado (Parte 4), comparado com as baselines")
    args = parser.parse_args()
    out_name = "comparacao" if args.model else "baselines"

    books = load_books()
    queries = pd.read_parquet(PROCESSED / "queries.parquet")
    print(f"{len(books)} livros no índice, {len(queries)} consultas\n")

    bm25 = BM25Searcher(books)
    searchers = [bm25, EmbeddingSearcher(books)]
    if args.model:
        tuned = EmbeddingSearcher(books, model_name=args.model)
        hybrid = HybridSearcher(books, bm25=bm25, embeddings=tuned)
        searchers += [tuned, hybrid, TranslatingSearcher(hybrid)]

    tables, all_ranks = [], {}
    for searcher in searchers:
        print(f"Avaliando {searcher.name}...")
        table, ranks = evaluate(searcher, queries)
        tables.append(table)
        all_ranks[searcher.name] = ranks

    results = pd.concat(tables, ignore_index=True)
    RESULTS.mkdir(exist_ok=True)
    results.to_csv(RESULTS / f"{out_name}.csv", index=False)
    markdown = to_markdown(results)
    (RESULTS / f"{out_name}.md").write_text(markdown + "\n", encoding="utf-8")
    print("\n" + markdown)

    # Consultas manuais: em que posição cada método colocou o livro certo
    manual = queries["kind"].eq("manual").to_numpy()
    detail = queries.loc[manual, ["lang", "query"]].assign(
        livro=books.loc[queries.loc[manual, "book_id"], "title"].to_numpy(),
        **{name: ranks[manual] for name, ranks in all_ranks.items()},
    )
    detail.to_csv(RESULTS / f"{out_name}_manual_detail.csv", index=False)
    print(f"\nPosição do livro certo em cada consulta manual: {RESULTS / f'{out_name}_manual_detail.csv'}")


if __name__ == "__main__":
    main()
