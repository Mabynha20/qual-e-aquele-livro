"""
Buscadores de livros: BM25 (palavras-chave), embeddings (busca semântica) e
híbrido (os dois juntos).

Todos têm a mesma interface:
    searcher.score(queries) -> matriz (n_consultas x n_livros) com a nota de cada livro
    searcher.search(query, k) -> os k livros mais bem avaliados

Busca rápida pelo terminal (na raiz do projeto, com a .venv ativa):
    python src/search.py "girl becomes a dragon rider at a deadly war college"
    python src/search.py "girl becomes a dragon rider" --method bm25
"""

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROCESSED / "embeddings"
FINETUNED = ROOT / "models" / "minilm-livros"  # modelo treinado na Parte 4

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Tamanho dos pedaços de resumo. Testado com 256/128/64: pedaços menores "diluem"
# menos o significado de cada trecho e deram os melhores resultados (ver PROJETO.md).
CHUNK_TOKENS = 64

STOPWORDS = set("""
a about after again against all also an and any are as at be because been before being between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself
him himself his how i if in into is it its itself just me more most my no nor not of off on once only or other
our out over own same she should so some such than that the their them then there these they this those through
to too under until up very was we were what when where which while who whom why will with would you your
""".split())


def load_books():
    books = pd.read_parquet(PROCESSED / "books.parquet")
    # O book_id é a posição do livro: a coluna j das matrizes de notas é o livro de book_id j.
    assert (books["book_id"].to_numpy() == np.arange(len(books))).all()
    return books


def top_k(scores, k):
    """Índices das k maiores notas, em ordem decrescente."""
    best = np.argpartition(-scores, k)[:k]
    return best[np.argsort(-scores[best])]


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def tokenize(text):
    """Minúsculas, só letras/números, sem stopwords."""
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


class BM25Searcher:
    """
    BM25: nota cada livro pelas palavras da consulta que aparecem no resumo,
    dando mais peso às palavras raras (idf) e descontando resumos longos.
    Não entende sinônimos: "wizard" não encontra "sorcerer".
    """

    name = "BM25"

    def __init__(self, books):
        self.books = books
        self.bm25 = BM25Okapi([tokenize(s) for s in books["summary"]])

    def score(self, queries):
        return np.vstack([self.bm25.get_scores(tokenize(q)) for q in queries])

    def search(self, query, k=5):
        scores = self.score([query])[0]
        return self.books.iloc[top_k(scores, k)].assign(score=scores[top_k(scores, k)])


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def chunk_text(text, tokenizer, max_tokens):
    """
    Junta frases inteiras em pedaços de até max_tokens tokens.
    Uma frase sozinha maior que o limite vira um pedaço próprio (e o modelo a trunca).
    """
    budget = max_tokens - 2  # o modelo acrescenta os tokens especiais [CLS] e [SEP]
    chunks, current, size = [], [], 0
    for sentence in split_sentences(text):
        n = len(tokenizer.tokenize(sentence))
        if current and size + n > budget:
            chunks.append(" ".join(current))
            current, size = [], 0
        current.append(sentence)
        size += n
    if current:
        chunks.append(" ".join(current))
    return chunks


class EmbeddingSearcher:
    """
    Busca semântica: transforma consulta e resumos em vetores (embeddings) e
    compara pelo cosseno. Textos com o mesmo sentido ficam próximos mesmo sem
    palavras em comum.

    Os resumos são divididos em pedaços (chunks) de frases inteiras com até
    chunk_tokens tokens; a nota do livro é a do seu pedaço mais parecido com a consulta.
    """

    def __init__(self, books, model_name=DEFAULT_MODEL, chunk_tokens=CHUNK_TOKENS, batch_size=32,
                 model=None, use_cache=True):
        """
        model_name: modelo do Hugging Face ou pasta de um modelo salvo (ex.: o treinado na Parte 4).
        model:      modelo já carregado (usado na validação durante o treino); dispensa model_name.
        use_cache:  guarda/reaproveita os embeddings dos pedaços em data/processed/embeddings/.
        """
        from sentence_transformers import SentenceTransformer  # importa só quando usado (é pesado)

        self.books = books
        self.model_name = model_name
        self.name = f"Embeddings ({Path(model_name).name})"
        self.model = model if model is not None else SentenceTransformer(model_name, device="cpu")
        self.batch_size = batch_size
        self.use_cache = use_cache

        max_tokens = min(chunk_tokens, self.model.max_seq_length)
        chunk_book, chunk_text_ = [], []
        for book_id, summary in zip(books["book_id"], books["summary"]):
            for chunk in chunk_text(summary, self.model.tokenizer, max_tokens):
                chunk_book.append(book_id)
                chunk_text_.append(chunk)
        self.chunk_book = np.array(chunk_book)
        self.chunk_embeddings = self._encode_chunks(chunk_text_)

    def _encode_chunks(self, chunks):
        """Calcula os embeddings dos pedaços, ou carrega do cache se nada mudou."""
        if not self.use_cache:
            return self.model.encode(chunks, batch_size=self.batch_size, normalize_embeddings=True,
                                     convert_to_numpy=True).astype(np.float32)
        # A chave do cache muda se mudarem o modelo, os pesos (modelo salvo em pasta) ou os pedaços.
        version = self.model_name
        model_dir = Path(self.model_name)
        if model_dir.is_dir():
            # mesmo modelo com o caminho escrito de outro jeito (absoluto/relativo) = mesmo cache
            try:
                version = model_dir.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                version = model_dir.resolve().as_posix()
            version += str(max(f.stat().st_mtime for f in model_dir.rglob("*") if f.is_file()))
        digest = hashlib.sha1((version + "\x00" + "\x00".join(chunks)).encode()).hexdigest()[:12]
        cache = EMBEDDINGS_DIR / f"{Path(self.model_name).name}_{digest}.npy"
        if cache.exists():
            print(f"Embeddings carregados do cache: {cache.name}")
            return np.load(cache)
        print(f"Calculando embeddings de {len(chunks)} pedaços ({len(self.books)} livros). Na CPU, leva alguns minutos...")
        embeddings = self.model.encode(chunks, batch_size=self.batch_size, normalize_embeddings=True,
                                       convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, embeddings)
        return embeddings

    def score(self, queries, batch=256):
        query_embeddings = self.model.encode(list(queries), batch_size=self.batch_size,
                                             normalize_embeddings=True, convert_to_numpy=True)
        scores = np.full((len(queries), len(self.books)), -np.inf, dtype=np.float32)
        for start in range(0, len(queries), batch):  # em blocos, para não estourar a memória
            sims = query_embeddings[start:start + batch] @ self.chunk_embeddings.T  # cosseno (vetores normalizados)
            block = scores[start:start + batch]
            for j in range(block.shape[0]):
                np.maximum.at(block[j], self.chunk_book, sims[j])  # nota do livro = melhor pedaço
        return scores

    def search(self, query, k=5):
        scores = self.score([query])[0]
        return self.books.iloc[top_k(scores, k)].assign(score=scores[top_k(scores, k)])


# ---------------------------------------------------------------------------
# Híbrido
# ---------------------------------------------------------------------------

def to_ranks(scores):
    """Posição (1 = melhor) de cada livro no ranking de cada consulta."""
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, scores.shape[1] + 1), axis=1)
    return ranks


class HybridSearcher:
    """
    Junta BM25 e embeddings por Reciprocal Rank Fusion (RRF):
        nota(livro) = 1 / (k + posição no BM25) + 1 / (k + posição nos embeddings)
    Usa só as posições, não as notas, porque as escalas dos dois métodos não se
    comparam. Um livro bem colocado em qualquer um dos dois sobe no resultado.
    k = 60 é o valor padrão do artigo original (Cormack et al., 2009).
    """

    def __init__(self, books, model_name=DEFAULT_MODEL, rrf_k=60, bm25=None, embeddings=None):
        """bm25/embeddings: buscadores já criados (evita recalcular o índice)."""
        self.books = books
        self.rrf_k = rrf_k
        self.bm25 = bm25 if bm25 is not None else BM25Searcher(books)
        self.embeddings = embeddings if embeddings is not None else EmbeddingSearcher(books, model_name)
        self.name = f"Híbrido (BM25 + {Path(self.embeddings.model_name).name})"

    def score(self, queries, batch=256):
        queries = list(queries)
        scores = np.empty((len(queries), len(self.books)), dtype=np.float32)
        for start in range(0, len(queries), batch):  # em blocos, para não estourar a memória
            block = queries[start:start + batch]
            fused = 1 / (self.rrf_k + to_ranks(self.bm25.score(block)))
            fused += 1 / (self.rrf_k + to_ranks(self.embeddings.score(block)))
            scores[start:start + batch] = fused
        return scores

    def search(self, query, k=5):
        scores = self.score([query])[0]
        return self.books.iloc[top_k(scores, k)].assign(score=scores[top_k(scores, k)])


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Busca livros por uma descrição vaga.")
    parser.add_argument("query", help="descrição do livro (em inglês, por enquanto)")
    parser.add_argument("--method", choices=["hybrid", "embeddings", "bm25"], default="hybrid")
    parser.add_argument("--model", default=None,
                        help="modelo de embeddings (padrão: models/minilm-livros se existir, senão o original)")
    parser.add_argument("-k", type=int, default=5, help="quantos livros mostrar")
    args = parser.parse_args()

    model = args.model or (str(FINETUNED) if FINETUNED.is_dir() else DEFAULT_MODEL)
    books = load_books()
    if args.method == "bm25":
        searcher = BM25Searcher(books)
    elif args.method == "embeddings":
        searcher = EmbeddingSearcher(books, model)
    else:
        searcher = HybridSearcher(books, model)
    results = searcher.search(args.query, args.k)
    print(f"\n{searcher.name} — \"{args.query}\"\n")
    for rank, row in enumerate(results.itertuples(), start=1):
        print(f"{rank}. {row.title} — {row.author or 'autor desconhecido'} ({row.year})  [nota {row.score:.3f}]")


if __name__ == "__main__":
    main()
