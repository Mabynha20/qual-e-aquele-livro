"""
Complementa a base com os livros da lista curada (lists/booktok_titles.txt).

Recupera livros que a coleta por ano (collect_wikipedia.py) não encontra:
  - livros cuja página tem outro nome (ex.: "Babel" -> "Babel, or the Necessity of Violence");
  - livros fora das categorias por ano;
  - livros que só aparecem na página da SÉRIE (ex.: ACOTAR, Throne of Glass,
    The Cruel Prince). Nessas páginas, cada livro fica num bloco {{Book list}}
    com título, data e resumo; todos os livros da série são aproveitados.

Uso (na raiz do projeto, com a .venv ativa):
    python src/collect_extra.py

Gera data/raw/extra_books.jsonl (refeito a cada execução) e mostra, para cada
título da lista, se ele já estava na base, foi adicionado ou não foi encontrado.
"""

import json
import re
from pathlib import Path

import mwparserfromhell

from collect_wikipedia import (
    BOOKS_FILE, MIN_WORDS, RAW_DIR, api_get, clean_wikitext,
    extract_categories, extract_infobox, fetch_batch, parse_page,
)

LIST_FILE = Path(__file__).resolve().parents[1] / "lists" / "booktok_titles.txt"
EXTRA_FILE = RAW_DIR / "extra_books.jsonl"


def normalize(title):
    """Simplifica um título para comparação: minúsculas, sem '(novel)', sem 'the', sem pontuação."""
    title = re.sub(r"\(.*?\)", "", title.lower())
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"^\s*the\s+", "", title)
    return " ".join(title.split())


def titles_match(query, page_title):
    """True se um título contém o outro (ex.: 'babel' e 'babel or the necessity of violence')."""
    q, p = normalize(query), normalize(page_title)
    return bool(q) and (q in p or p in q)


def read_list():
    """Lê a lista curada: retorna pares (título, autor ou None)."""
    entries = []
    for line in LIST_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        title, _, author = line.partition("|")
        entries.append((title.strip(), author.strip() or None))
    return entries


def find_year(*texts):
    """Primeiro ano (1900-2099) encontrado nos textos."""
    for text in texts:
        match = re.search(r"\b(19|20)\d{2}\b", text or "")
        if match:
            return int(match.group())
    return None


def year_from_page(code, categories):
    """Ano de publicação: pelas categorias '{ano} ... novels' ou pela data do infobox."""
    years = [int(m.group(1)) for c in categories if (m := re.match(r"^(\d{4}) .*novels$", c))]
    if years:
        return min(years)
    for template in code.filter_templates():
        if str(template.name).strip().lower().startswith("infobox"):
            for name in ("release_date", "pub_date", "published", "publication_date"):
                if template.has(name):
                    return find_year(str(template.get(name).value))
    return None


def books_from_series(page, code):
    """Extrai os livros dos blocos {{Book list}} de uma página de série."""
    info = extract_infobox(code)
    categories = extract_categories(code)
    books = []
    for template in code.filter_templates():
        if str(template.name).strip().lower() != "book list" or not template.has("short_summary"):
            continue
        title = clean_wikitext(str(template.get("title").value)) if template.has("title") else ""
        plot = clean_wikitext(str(template.get("short_summary").value))
        if not title or len(plot.split()) < MIN_WORDS:
            continue
        date = str(template.get("publish_date").value) if template.has("publish_date") else ""
        books.append({
            "pageid": page["pageid"],
            "title": title,
            "year": find_year(date),
            "author": info["author"],
            "genre": info["genre"],
            "series": page["title"],
            "categories": categories,
            "plot": plot,
            "n_words": len(plot.split()),
            "url": "https://en.wikipedia.org/wiki/" + page["title"].replace(" ", "_"),
        })
    return books


def is_novel_page(code):
    """True se a página é de um romance, e não de um filme ou série de TV adaptados dele."""
    categories = " | ".join(extract_categories(code)).lower()
    return "novel" in categories and not re.search(r"films|television", categories)


def books_from_page(page, query, author):
    """Livros da página que correspondem à consulta (lista vazia se a página não servir)."""
    wikitext = page["revisions"][0]["slots"]["main"]["content"]
    if re.search(r"\{\{\s*(disambiguation|dab|disambig)", wikitext, re.I):
        return []
    # Com autor informado, a página precisa citar o sobrenome (evita filmes de mesmo nome).
    if author and author.split()[-1].lower() not in wikitext.lower():
        return []
    code = mwparserfromhell.parse(wikitext)

    series_books = books_from_series(page, code)
    if any(titles_match(query, b["title"]) for b in series_books):
        return series_books

    if titles_match(query, page["title"]) and is_novel_page(code):
        book = parse_page(page, None)
        if book:
            book["year"] = year_from_page(code, book["categories"])
            return [book]
    return []


def search_titles(query, author):
    """Títulos de páginas candidatas, pela busca da Wikipédia."""
    terms = f"{query} {author or ''} novel"
    result = api_get({"action": "query", "list": "search", "srsearch": terms, "srlimit": 5})
    return [hit["title"] for hit in result.get("query", {}).get("search", [])]


def find_books(query, author):
    """Tenta a página com o título exato e, se não servir, os resultados da busca."""
    tried = set()
    for candidates in ([query], None):
        if candidates is None:
            candidates = [t for t in search_titles(query, author) if t not in tried]
        tried.update(candidates)
        if not candidates:
            continue
        pages, _ = fetch_batch(candidates)
        for page in pages:
            books = books_from_page(page, query, author)
            if books:
                return books
    return []


def same_author(author, known_authors):
    """True se o autor não foi informado/registrado ou se o sobrenome aparece no autor registrado."""
    if not author:
        return True
    surname = author.split()[-1].lower()
    return any(a is None or surname in a.lower() for a in known_authors)


def main():
    known = {}  # título normalizado -> autores registrados (o mesmo título pode ter vários)
    with BOOKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            book = json.loads(line)
            known.setdefault(normalize(book["title"]), []).append(book["author"])

    def in_base(title, author):
        return normalize(title) in known and same_author(author, known[normalize(title)])

    entries = read_list()
    print(f"{len(entries)} títulos na lista\n")

    added = []
    for query, author in entries:
        if in_base(query, author):
            print(f"  já na base      {query}")
            continue
        found = find_books(query, author)
        books = [b for b in found if not in_base(b["title"], b["author"])]
        if not books:
            if found:  # achou, mas com outro título que já estava na base
                print(f"  já na base      {query} (como \"{found[0]['title']}\")")
            else:
                print(f"  NÃO ENCONTRADO  {query}")
            continue
        for book in books:
            book["source"] = "lista_curada"
            known.setdefault(normalize(book["title"]), []).append(book["author"])
        added.extend(books)
        extra = f" (+{len(books) - 1} da série)" if len(books) > 1 else ""
        print(f"  adicionado      {query}{extra}")

    with EXTRA_FILE.open("w", encoding="utf-8") as out:
        for book in added:
            out.write(json.dumps(book, ensure_ascii=False) + "\n")
    print(f"\nLivros adicionados: {len(added)} -> {EXTRA_FILE.relative_to(RAW_DIR.parents[1])}")


if __name__ == "__main__":
    main()
