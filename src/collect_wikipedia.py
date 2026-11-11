"""
Coleta resumos de enredo de romances da Wikipédia em inglês.

Etapa 1 - lista de livros:
    percorre `Category:{ano} novels` e suas subcategorias (ex.: "2023 American
    novels", "2023 speculative fiction novels" -> "2023 fantasy novels") e salva
    os títulos encontrados em data/raw/titles.json.

Etapa 2 - conteúdo:
    baixa o wikitexto das páginas em lotes de 50, extrai a seção de enredo
    ("Plot", "Plot summary", "Synopsis"...), o infobox (autor, gênero, série) e
    as categorias, e grava um livro por linha em data/raw/wikipedia_books.jsonl.

Uso (na raiz do projeto, com a .venv ativa):
    python src/collect_wikipedia.py --years 2023 --limit 100   # teste rápido
    python src/collect_wikipedia.py                             # 2000-2026

A coleta pode ser interrompida com Ctrl+C e retomada depois: as páginas já
processadas ficam registradas e são puladas.
"""

import argparse
import json
import re
import time
from pathlib import Path

import mwparserfromhell
import requests
from tqdm import tqdm

API = "https://en.wikipedia.org/w/api.php"
# A Wikimedia pede um User-Agent que identifique o projeto e um contato.
USER_AGENT = "qual-e-aquele-livro/0.1 (https://github.com/Mabynha20/qual-e-aquele-livro)"
PAUSE = 0.5  # segundos entre requisições, para não sobrecarregar a API
BATCH_SIZE = 50  # máximo de páginas por requisição de conteúdo permitido pela API
MAX_DEPTH = 3  # profundidade máxima ao descer nas subcategorias
MIN_WORDS = 30  # corte mínimo na coleta; o corte definitivo é decidido na EDA

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
TITLES_FILE = RAW_DIR / "titles.json"
BOOKS_FILE = RAW_DIR / "wikipedia_books.jsonl"
PROCESSED_FILE = RAW_DIR / "processed_titles.txt"

# Subcategorias que não são romances em prosa.
SKIP_CATEGORY = re.compile(r"graphic novels|light novels|comics|manga|webcomics", re.I)
# Títulos de seção que contêm o enredo.
PLOT_HEADING = re.compile(r"^(plot|synopsis|summary|story|storyline|premise|overview|narrative)\b", re.I)
# Templates de lista cujo conteúdo deve ser mantido (comuns no campo "genre" do infobox).
LIST_TEMPLATES = {"hlist", "flatlist", "plainlist", "ubl", "unbulleted list", "plain list"}

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT


# ---------------------------------------------------------------------------
# Acesso à API
# ---------------------------------------------------------------------------

def api_get(params):
    """Faz uma requisição à API, esperando e tentando de novo se o servidor pedir."""
    params = {**params, "format": "json", "formatversion": 2, "maxlag": 5}
    for attempt in range(5):
        response = session.get(API, params=params, timeout=60)
        if response.status_code == 429:  # muitas requisições
            time.sleep(10 * (attempt + 1))
            continue
        response.raise_for_status()
        data = response.json()
        if data.get("error", {}).get("code") == "maxlag":  # servidor sobrecarregado
            time.sleep(5 * (attempt + 1))
            continue
        time.sleep(PAUSE)
        return data
    raise RuntimeError(f"A API falhou 5 vezes seguidas: {params}")


def api_query(params):
    """Gera as páginas de resultado de uma consulta, seguindo a paginação ('continue')."""
    cont = {}
    while True:
        data = api_get({**params, **cont})
        yield data.get("query", {})
        if "continue" not in data:
            break
        cont = data["continue"]


# ---------------------------------------------------------------------------
# Etapa 1: lista de livros por ano
# ---------------------------------------------------------------------------

def list_titles_for_year(year):
    
    root = f"Category:{year} novels"
    queue = [(root, 0)]
    seen = {root}
    titles = set()
    while queue:
        category, depth = queue.pop()
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "page|subcat",
            "cmlimit": "max",
        }
        for result in api_query(params):
            for member in result.get("categorymembers", []):
                if member["ns"] == 0:  
                    titles.add(member["title"])
                elif (member["ns"] == 14  
                      and depth < MAX_DEPTH
                      and member["title"] not in seen
                      and not SKIP_CATEGORY.search(member["title"])):
                    seen.add(member["title"])
                    queue.append((member["title"], depth + 1))
    return titles


def collect_titles(years):
    
    if TITLES_FILE.exists():
        state = json.loads(TITLES_FILE.read_text(encoding="utf-8"))
    else:
        state = {"years_done": [], "titles": {}}  

    for year in years:
        if year in state["years_done"]:
            continue
        found = list_titles_for_year(year)
        for title in found:
            state["titles"].setdefault(title, year)  
        state["years_done"].append(year)
        TITLES_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {year}: {len(found)} artigos")

    return {t: y for t, y in state["titles"].items() if y in years}


# ---------------------------------------------------------------------------
# Etapa 2: conteúdo das páginas
# ---------------------------------------------------------------------------

def clean_wikitext(wikitext):
    """Converte wikitexto em texto puro: remove referências, templates, imagens e títulos."""
    wikitext = re.sub(r"<ref[^>]*/>", "", wikitext)
    wikitext = re.sub(r"<ref[^>]*>.*?</ref>", "", wikitext, flags=re.S)
    wikitext = re.sub(r"<br\s*/?>", "\n", wikitext, flags=re.I)
    code = mwparserfromhell.parse(wikitext)
    for template in code.filter_templates(recursive=False):
        if str(template.name).strip().lower() in LIST_TEMPLATES:
            items = "\n".join(str(p.value).strip() for p in template.params)
            code.replace(template, "\n" + items + "\n")
        else:
            code.remove(template)
    for heading in code.filter_headings(recursive=False):
        code.remove(heading)
    for link in code.filter_wikilinks(recursive=False):
        if str(link.title).strip().lower().startswith(("file:", "image:", "category:")):
            code.remove(link)
    text = code.strip_code(normalize=True, collapse=True)
    text = re.sub(r"^\s*[*#]+\s*", "", text, flags=re.M)  # marcadores de lista
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def extract_plot(code):
    """Retorna o texto da primeira seção de enredo (com subseções) ou None."""
    for section in code.get_sections(levels=[2]):
        heading = section.filter_headings()[0].title.strip_code().strip()
        if PLOT_HEADING.match(heading):
            return clean_wikitext(str(section))
    return None


def extract_infobox(code):
    """Lê autor, gênero e série do infobox do livro, se houver."""
    for template in code.filter_templates():
        if str(template.name).strip().lower().startswith("infobox"):
            def field(*names):
                for name in names:
                    if template.has(name):
                        value = clean_wikitext(str(template.get(name).value))
                        if value:
                            return ", ".join(v.strip(" ,") for v in value.split("\n") if v.strip(" ,"))
                return None
            return {
                "author": field("author", "authors"),
                "genre": field("genre", "genres"),
                "series": field("series"),
            }
    return {"author": None, "genre": None, "series": None}


def extract_categories(code):
    """Lista as categorias declaradas no wikitexto (sem o prefixo 'Category:')."""
    categories = []
    for link in code.filter_wikilinks():
        title = str(link.title).strip()
        if title.lower().startswith("category:"):
            categories.append(title.split(":", 1)[1].strip())
    return categories


def parse_page(page, year):
    """Monta o registro de um livro, ou None se a página não tiver resumo de enredo."""
    wikitext = page["revisions"][0]["slots"]["main"]["content"]
    code = mwparserfromhell.parse(wikitext)
    plot = extract_plot(code)
    if not plot or len(plot.split()) < MIN_WORDS:
        return None
    return {
        "pageid": page["pageid"],
        "title": page["title"],
        "year": year,
        **extract_infobox(code),
        "categories": extract_categories(code),
        "plot": plot,
        "n_words": len(plot.split()),
        "url": "https://en.wikipedia.org/wiki/" + page["title"].replace(" ", "_"),
    }


def fetch_batch(titles):
    """Baixa o wikitexto de até 50 títulos; retorna (páginas, mapa título pedido -> título final)."""
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "titles": "|".join(titles),
        "redirects": 1,
    }
    data = api_get(params)
    query = data.get("query", {})
    # A API pode normalizar o título ou seguir um redirecionamento.
    final = {t: t for t in titles}
    for step in query.get("normalized", []) + query.get("redirects", []):
        for asked, current in final.items():
            if current == step["from"]:
                final[asked] = step["to"]
    pages = [p for p in query.get("pages", []) if "revisions" in p]
    return pages, final


def collect_books(titles, limit=None):
    """Etapa 2: processa os títulos pendentes e grava os livros com enredo."""
    processed = set()
    if PROCESSED_FILE.exists():
        processed = set(PROCESSED_FILE.read_text(encoding="utf-8").splitlines())
    saved_ids = set()
    if BOOKS_FILE.exists():
        with BOOKS_FILE.open(encoding="utf-8") as f:
            saved_ids = {json.loads(line)["pageid"] for line in f}

    pending = sorted(t for t in titles if t not in processed)
    if limit:
        pending = pending[:limit]
    print(f"Etapa 2: {len(pending)} páginas pendentes ({len(processed)} já processadas)")

    new_books = 0
    with BOOKS_FILE.open("a", encoding="utf-8") as books_out, \
         PROCESSED_FILE.open("a", encoding="utf-8") as processed_out:
        for i in tqdm(range(0, len(pending), BATCH_SIZE), unit="lote"):
            batch = pending[i:i + BATCH_SIZE]
            pages, final = fetch_batch(batch)
            year_of = {final[t]: titles[t] for t in batch}
            for page in pages:
                if page["pageid"] in saved_ids:  # dois títulos que levam à mesma página
                    continue
                book = parse_page(page, year_of.get(page["title"]))
                if book:
                    books_out.write(json.dumps(book, ensure_ascii=False) + "\n")
                    saved_ids.add(page["pageid"])
                    new_books += 1
            processed_out.write("\n".join(batch) + "\n")
            # Grava no disco a cada lote, para não perder nada se a coleta for interrompida.
            books_out.flush()
            processed_out.flush()

    print(f"Livros novos com enredo: {new_books}. Total salvo: {len(saved_ids)}")


# ---------------------------------------------------------------------------

def parse_years(text):
    """Converte '2000-2026', '2023' ou '2019,2023' em uma lista de anos."""
    years = []
    for part in text.split(","):
        if "-" in part:
            start, end = part.split("-")
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(part))
    return years


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", default="2000-2026", help="anos a coletar (padrão: 2000-2026)")
    parser.add_argument("--limit", type=int, help="processa no máximo N páginas (para testes)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    years = parse_years(args.years)

    print(f"Etapa 1: listando romances de {years[0]} a {years[-1]}")
    titles = collect_titles(years)
    print(f"Total de artigos listados: {len(titles)}")

    collect_books(titles, limit=args.limit)


if __name__ == "__main__":
    main()
