"""
Fine-tuning do all-MiniLM-L6-v2 para a tarefa de "lembrança vaga" -> livro.

Pares de treino (só livros do TREINO), no estilo Inverse Cloze Task:
    consulta: uma frase do resumo, sem nomes próprios e em minúsculas
              (igual às consultas automáticas da Parte 2);
    alvo:     o contexto ao redor dessa frase (frases vizinhas), SEM a frase.
O modelo não pode simplesmente achar o texto copiado: precisa ligar a
lembrança ao resto da história.

Perda contrastiva (InfoNCE) com negativos do próprio lote: num lote de N
pares, cada consulta deve ficar mais parecida com o seu alvo do que com os
outros N-1 alvos. É uma classificação com N classes:
    logits = (consultas · alvos^T) / temperatura,   rótulo da linha i = i

Uso (na raiz do projeto, com a .venv ativa):
    python src/train.py --quick      # teste rápido: poucos pares, ~2 min
    python src/train.py              # treino completo

Saídas:
    models/minilm-livros/            melhor modelo (pela validação)
    models/checkpoints/epoch-N/      modelo ao fim de cada época
    results/train_log.csv            perda a cada passo
    results/train_val.csv            métricas de validação por época
    results/train_curves.png         gráficos de treino
"""

import argparse
import math
import os
import random
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

matplotlib.use("Agg")  # gera os gráficos sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402

from evaluate import metrics, ranks_of_targets  # noqa: E402
from search import DEFAULT_MODEL, PROCESSED, EmbeddingSearcher, split_sentences  # noqa: E402
from split import META, MIN_QUERY_WORDS, SENTENCE_WORDS, remove_names  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

SEED = 42
QUERY_MAX_TOKENS = 64
DOC_MAX_TOKENS = 96
CONTEXT_SENTENCES = 2      # frases de contexto de cada lado da frase-consulta
MIN_CONTEXT_WORDS = 15
KEEP_SENTENCE_PROB = 0.1   # às vezes a frase fica no contexto (como no artigo original do ICT)


# ---------------------------------------------------------------------------
# Pares de treino
# ---------------------------------------------------------------------------

def make_pairs(summary, pairs_per_book, rng):
    """Até pairs_per_book pares (consulta, contexto) de um resumo."""
    sentences = split_sentences(summary)
    low, high = SENTENCE_WORDS
    candidates = [i for i, s in enumerate(sentences) if low <= len(s.split()) <= high and not META.search(s)]
    rng.shuffle(candidates)
    pairs = []
    for i in candidates:
        query = remove_names(sentences[i])
        if not query or len(query.split()) < MIN_QUERY_WORDS:
            continue
        before = sentences[max(0, i - CONTEXT_SENTENCES):i]
        after = sentences[i + 1:i + 1 + CONTEXT_SENTENCES]
        middle = [sentences[i]] if rng.random() < KEEP_SENTENCE_PROB else []
        context = " ".join(before + middle + after)
        if len(context.split()) < MIN_CONTEXT_WORDS:
            continue
        pairs.append((query, context))
        if len(pairs) == pairs_per_book:
            break
    return pairs


def build_training_pairs(books, pairs_per_book, rng):
    """
    Monta os pares em "rodadas": a rodada r tem o r-ésimo par de cada livro, embaralhado.
    Assim, um lote quase nunca tem dois pares do mesmo livro, o que evitaria um
    "negativo" que na verdade é positivo.
    """
    per_book = [make_pairs(s, pairs_per_book, rng) for s in books["summary"]]
    ordered = []
    for r in range(pairs_per_book):
        round_pairs = [pairs[r] for pairs in per_book if len(pairs) > r]
        rng.shuffle(round_pairs)
        ordered.extend(round_pairs)
    return ordered


# ---------------------------------------------------------------------------
# Modelo e perda
# ---------------------------------------------------------------------------

def embed(model, texts, max_tokens):
    """Embeddings normalizados de uma lista de textos, COM gradiente (para o treino)."""
    features = model.tokenizer(texts, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt")
    output = model(dict(features))
    return F.normalize(output["sentence_embedding"], dim=-1)


def info_nce(query_emb, doc_emb, temperature):
    """Perda InfoNCE: o alvo certo da consulta i é o documento i; os outros são negativos."""
    logits = query_emb @ doc_emb.T / temperature
    labels = torch.arange(len(logits))
    loss = F.cross_entropy(logits, labels)
    accuracy = (logits.argmax(dim=1) == labels).float().mean()
    return loss, accuracy


# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------

def validate(model, val_books, val_queries):
    """Recall/MRR das consultas automáticas de validação num índice só com os livros de validação."""
    model.eval()
    with torch.no_grad():
        searcher = EmbeddingSearcher(val_books, model=model, use_cache=False)
        scores = searcher.score(val_queries["query"].tolist())
    model.train()
    return metrics(ranks_of_targets(scores, val_queries["target"].to_numpy()))


def load_validation(books, quick):
    splits = pd.read_parquet(PROCESSED / "splits.parquet")
    queries = pd.read_parquet(PROCESSED / "queries.parquet")
    val_books = books[splits["split"].to_numpy() == "val"].reset_index(drop=True)
    if quick:
        val_books = val_books.head(150)
    new_id = {old: new for new, old in enumerate(val_books["book_id"])}
    val_queries = queries[(queries["split"] == "val") & queries["book_id"].isin(new_id)].copy()
    val_queries["target"] = val_queries["book_id"].map(new_id)
    val_books = val_books.assign(book_id=range(len(val_books)))
    return val_books, val_queries


# ---------------------------------------------------------------------------

def save_curves(log, val_log):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    smooth = log["loss"].rolling(20, min_periods=1).mean()
    axes[0].plot(log["step"], log["loss"], alpha=0.25, color="#4C72B0", label="por passo")
    axes[0].plot(log["step"], smooth, color="#4C72B0", label="média móvel (20)")
    axes[0].set(title="Perda de treino (InfoNCE)", xlabel="Passo", ylabel="Perda")
    axes[0].legend()
    for column, color in (("Recall@1", "#4C72B0"), ("Recall@10", "#55A868"), ("MRR", "#C44E52")):
        axes[1].plot(val_log["epoch"], val_log[column], marker="o", color=color, label=column)
    axes[1].set(title="Validação (consultas automáticas)", xlabel="Época", ylim=(0, 1))
    axes[1].set_xticks(val_log["epoch"])
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "train_curves.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning contrastivo do MiniLM.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--pairs-per-book", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="teste rápido com poucos pares")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))  # núcleos físicos

    from sentence_transformers import SentenceTransformer

    books = pd.read_parquet(PROCESSED / "books.parquet")
    splits = pd.read_parquet(PROCESSED / "splits.parquet")
    train_books = books[splits["split"].to_numpy() == "train"]
    if args.quick:
        train_books = train_books.sample(200, random_state=SEED)
        args.epochs = 1

    pairs = build_training_pairs(train_books, args.pairs_per_book, random.Random(SEED))
    print(f"{len(pairs)} pares de treino de {len(train_books)} livros. Exemplo:")
    print(f"  consulta: {pairs[0][0]}\n  alvo:     {pairs[0][1][:200]}...\n")

    loader = DataLoader(pairs, batch_size=args.batch_size, shuffle=False, drop_last=True,
                        collate_fn=lambda batch: tuple(zip(*batch)))
    total_steps = len(loader) * args.epochs

    model = SentenceTransformer(DEFAULT_MODEL, device="cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = max(1, int(0.1 * total_steps))
    # learning rate sobe linearmente no aquecimento (10% dos passos) e depois cai até zero
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup, max(0.0, (total_steps - step) / max(1, total_steps - warmup))))

    val_books, val_queries = load_validation(books, args.quick)
    print(f"Validação: {len(val_queries)} consultas, índice com {len(val_books)} livros")
    RESULTS.mkdir(exist_ok=True)
    out_dir = MODELS / ("minilm-livros-quick" if args.quick else "minilm-livros")

    print("Validação antes do treino (modelo original)...")
    val_rows = [{"epoch": 0, **validate(model, val_books, val_queries)}]
    print(f"  época 0: R@1 {val_rows[0]['Recall@1']:.1%} | R@10 {val_rows[0]['Recall@10']:.1%} | MRR {val_rows[0]['MRR']:.3f}")
    best_mrr = val_rows[0]["MRR"]

    log, step, start = [], 0, time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        for queries, docs in loader:
            loss, accuracy = info_nce(embed(model, list(queries), QUERY_MAX_TOKENS),
                                      embed(model, list(docs), DOC_MAX_TOKENS), args.temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # evita passos grandes demais
            optimizer.step()
            scheduler.step()

            step += 1
            log.append({"step": step, "epoch": epoch, "loss": loss.item(), "batch_acc": accuracy.item(),
                        "lr": scheduler.get_last_lr()[0]})
            if step % 20 == 0 or step == total_steps:
                recent = log[-20:]
                elapsed = time.time() - start
                eta = elapsed / step * (total_steps - step)
                print(f"  passo {step}/{total_steps} | perda {np.mean([r['loss'] for r in recent]):.3f} | "
                      f"acerto no lote {np.mean([r['batch_acc'] for r in recent]):.0%} | "
                      f"faltam ~{math.ceil(eta / 60)} min")

        checkpoint = MODELS / "checkpoints" / f"epoch-{epoch}"
        model.save(str(checkpoint))
        row = {"epoch": epoch, **validate(model, val_books, val_queries)}
        val_rows.append(row)
        print(f"  época {epoch}: R@1 {row['Recall@1']:.1%} | R@10 {row['Recall@10']:.1%} | MRR {row['MRR']:.3f}")
        if row["MRR"] > best_mrr:
            best_mrr = row["MRR"]
            model.save(str(out_dir))
            print(f"  -> melhor até agora, salvo em {out_dir.relative_to(ROOT)}")

        log_df, val_df = pd.DataFrame(log), pd.DataFrame(val_rows)
        log_df.to_csv(RESULTS / "train_log.csv", index=False)
        val_df.to_csv(RESULTS / "train_val.csv", index=False)
        save_curves(log_df, val_df)

    print(f"\nTreino concluído em {(time.time() - start) / 60:.0f} min. Gráficos: results/train_curves.png")
    if best_mrr == val_rows[0]["MRR"]:
        print("Atenção: nenhuma época superou o modelo original na validação; nada foi salvo em", out_dir.name)


if __name__ == "__main__":
    main()
