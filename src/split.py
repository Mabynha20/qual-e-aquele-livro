"""
Divide os livros em treino/validação/teste e monta as consultas de avaliação.

Divisão (70/15/15), agrupada por AUTOR:
    todos os livros de um mesmo autor caem na mesma parte. Assim, uma série
    inteira fica junta, e o modelo não "decora" nomes de personagens e lugares
    no treino para depois reconhecê-los no teste. Sem autor, o grupo é a série;
    sem série, o próprio livro. Os livros das consultas manuais (e, portanto,
    os outros livros de seus autores) vão obrigatoriamente para o teste.

Consultas de avaliação:
    - manuais: lists/consultas_manuais.csv (sempre no teste);
    - automáticas: uma frase sorteada do resumo de cada livro de validação e
      teste, sem os nomes próprios e em minúsculas, imitando uma lembrança vaga.

Uso (na raiz do projeto, com a .venv ativa):
    python src/split.py

Gera data/processed/splits.parquet e data/processed/queries.parquet.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MANUAL_QUERIES = ROOT / "lists" / "consultas_manuais.csv"

SEED = 42
VAL_SHARE = 0.15
TEST_SHARE = 0.15
MIN_QUERY_WORDS = 6  # consulta automática mínima, depois de tirar os nomes
SENTENCE_WORDS = (10, 40)  # tamanho das frases sorteadas

# Palavras com inicial maiúscula que NÃO são nomes próprios (início de frase).
CAPITALIZED_KEEP = {
    "a", "an", "the", "in", "on", "at", "as", "after", "before", "when", "while", "during",
    "he", "she", "they", "it", "his", "her", "their", "its", "i", "we", "one", "two",
    "meanwhile", "later", "eventually", "however", "soon", "then", "there", "this", "that",
    "with", "without", "despite", "although", "upon", "years", "once", "now", "back",
}
# Palavras que ficam soltas quando o nome seguinte é removido ("travels to [Paris]").
ORPHANS = {"the", "a", "an", "of", "to", "from", "in", "at", "on", "by", "for", "with", "and",
           "named", "called", "into", "near", "between", "s"}
MAX_NAME_SHARE = 0.3  # frases com mais nomes que isso viram consultas sem sentido
# Frases que falam do livro, e não da história.
META = re.compile(r"\b(novel|book|chapters?|epilogue|prologue|sequel|prequel|series|trilogy|"
                  r"author|narrat\w*|sections?|volumes?|installment|story is told)\b", re.I)


# ---------------------------------------------------------------------------
# Divisão
# ---------------------------------------------------------------------------

def group_key(row):
    """Grupo do livro: autor; sem autor, a série; sem série, o próprio livro."""
    if isinstance(row["author"], str):
        return "autor:" + row["author"].lower().strip()
    if isinstance(row["series"], str):
        return "serie:" + row["series"].lower().strip()
    return f"livro:{row['book_id']}"


def assign_splits(books, forced_test_groups):
    """Sorteia grupos inteiros para teste e validação até atingir as proporções."""
    sizes = books.groupby("group").size()
    rng = np.random.default_rng(SEED)
    order = rng.permutation([g for g in sizes.index if g not in forced_test_groups])

    split_of = {g: "test" for g in forced_test_groups}
    n_test = sizes[list(forced_test_groups)].sum()
    n_val = 0
    target = {"test": TEST_SHARE * len(books), "val": VAL_SHARE * len(books)}
    for group in order:
        if n_test < target["test"]:
            split_of[group] = "test"
            n_test += sizes[group]
        elif n_val < target["val"]:
            split_of[group] = "val"
            n_val += sizes[group]
        else:
            split_of[group] = "train"
    return books["group"].map(split_of)


# ---------------------------------------------------------------------------
# Consultas
# ---------------------------------------------------------------------------

def sentences(text):
    """Quebra o resumo em frases."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def remove_names(sentence):
    """
    Tira palavras com inicial maiúscula (nomes de pessoas e lugares), junto com
    a preposição/artigo que ficaria solto antes delas, e põe em minúsculas.
    Retorna None se a frase tiver nomes demais.
    """
    words = sentence.split()
    is_name = []
    for word in words:
        core = word.strip("\"'“”‘’(),.;:!?").split("'")[0]
        is_name.append(core[:1].isupper() and core.lower() not in CAPITALIZED_KEEP)
    if sum(is_name) > MAX_NAME_SHARE * len(words):
        return None

    removed = list(is_name)
    for i in range(len(words) - 2, -1, -1):  # de trás para frente, para pegar "to the [Nome]"
        if removed[i + 1] and words[i].lower().strip(",") in ORPHANS:
            removed[i] = True
    kept = [w for w, r in zip(words, removed) if not r]
    text = " ".join(kept).lower()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)  # espaço antes de pontuação
    text = re.sub(r"([,;:])(\s*[,;:])+", r"\1", text)  # pontuação repetida
    text = re.sub(r"(^|\s)'s\b", " ", text)  # posse sem o nome ("'s")
    return " ".join(text.split()).strip(" ,;:")


def auto_query(summary, rng):
    """Sorteia uma frase do resumo e a transforma numa consulta vaga (ou None)."""
    low, high = SENTENCE_WORDS
    candidates = [s for s in sentences(summary)
                  if low <= len(s.split()) <= high and not META.search(s)]
    rng.shuffle(candidates)
    for sentence in candidates:
        query = remove_names(sentence)
        if query and len(query.split()) >= MIN_QUERY_WORDS:
            return query, sentence
    return None, None


def build_queries(books, manual):
    """Junta as consultas manuais e gera as automáticas para validação e teste."""
    rng = np.random.default_rng(SEED)
    rows = [
        {"query": q.consulta, "book_id": q.book_id, "kind": "manual", "lang": q.idioma, "source_sentence": None}
        for q in manual.itertuples()
    ]
    for book in books[books["split"] != "train"].itertuples():
        query, sentence = auto_query(book.summary, rng)
        if query:
            rows.append({"query": query, "book_id": book.book_id, "kind": "auto", "lang": "en",
                         "source_sentence": sentence})

    queries = pd.DataFrame(rows).merge(books[["book_id", "split"]], on="book_id")
    queries.insert(0, "query_id", range(len(queries)))
    return queries


# ---------------------------------------------------------------------------

def main():
    books = pd.read_parquet(PROCESSED / "books.parquet")
    books["group"] = books.apply(group_key, axis=1)

    manual = pd.read_csv(MANUAL_QUERIES, sep=";")
    manual = manual.merge(books[["book_id", "title", "author"]],
                          left_on=["titulo", "autor"], right_on=["title", "author"], how="left")
    missing = manual[manual["book_id"].isna()]
    if len(missing):
        raise SystemExit(f"Consultas manuais sem livro correspondente na base:\n{missing[['titulo', 'autor']]}")
    manual["book_id"] = manual["book_id"].astype(int)

    forced = set(books.loc[books["book_id"].isin(manual["book_id"]), "group"])
    books["split"] = assign_splits(books, forced)

    queries = build_queries(books, manual)

    books[["book_id", "group", "split"]].to_parquet(PROCESSED / "splits.parquet", index=False)
    queries.to_parquet(PROCESSED / "queries.parquet", index=False)

    print("Livros por parte:")
    counts = books["split"].value_counts().reindex(["train", "val", "test"])
    for split, n in counts.items():
        print(f"  {split:<5} {n:>5} livros ({n / len(books):.1%}), {books.loc[books['split'] == split, 'group'].nunique()} grupos")
    print(f"  grupos forçados para o teste (autores das consultas manuais): {len(forced)}")
    print("\nConsultas:")
    print(queries.groupby(["split", "kind"]).size().to_string())
    print("\nExemplos de consultas automáticas:")
    for q in queries[queries["kind"] == "auto"].sample(3, random_state=SEED).itertuples():
        print(f"  frase:    {q.source_sentence}\n  consulta: {q.query}\n")


if __name__ == "__main__":
    main()
