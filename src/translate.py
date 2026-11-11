"""
Consultas em português: detecta o idioma e traduz para o inglês antes da busca.

Os resumos da base estão em inglês. Em vez de trocar o modelo de busca, a
consulta em PT é traduzida (modelo MarianMT opus-mt-ROMANCE-en, ~300 MB, roda
na CPU) e segue para o buscador de sempre. Consultas em inglês passam direto.

Uso como "embrulho" de qualquer buscador:
    searcher = TranslatingSearcher(HybridSearcher(books, ...))
    searcher.search("garota que vira cavaleira de dragão")
"""

import re

TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-ROMANCE-en"

# Palavras muito frequentes em cada idioma (e raras no outro)
PT_WORDS = set("""
que não uma um uns umas para com por pelo pela como mas seu sua seus suas dela dele ele ela eles elas são
está estão foi era muito também numa num nos nas no na os das dos do da ao aos à às é se quando onde depois
porque garota menina mulher homem livro história
""".split())
EN_WORDS = set("""
the and of to in is that with for his her she he who on by an at from they their it its was were has have
into after when where about girl woman man book story
""".split())
PT_CHARS = re.compile(r"[ãõçáéíóúâêôà]")


def detect_language(text):
    """'pt' ou 'en', contando palavras típicas de cada idioma e letras acentuadas."""
    words = re.findall(r"\w+", text.lower())
    pt = sum(w in PT_WORDS for w in words) + sum(bool(PT_CHARS.search(w)) for w in words)
    en = sum(w in EN_WORDS for w in words)
    return "pt" if pt > en else "en"


class Translator:
    """Tradução PT -> EN com MarianMT (carregado só quando usado)."""

    def __init__(self, model_name=TRANSLATION_MODEL, num_beams=4):
        from transformers import MarianMTModel, MarianTokenizer  # importa só quando usado (é pesado)

        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).eval()
        self.num_beams = num_beams

    def translate(self, texts, batch_size=16):
        import torch

        out = []
        for start in range(0, len(texts), batch_size):
            features = self.tokenizer(texts[start:start + batch_size], return_tensors="pt",
                                      padding=True, truncation=True, max_length=256)
            with torch.no_grad():
                generated = self.model.generate(**features, num_beams=self.num_beams, max_new_tokens=128)
            out += self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return out


class TranslatingSearcher:
    """Traduz as consultas em português e repassa tudo para o buscador embrulhado."""

    def __init__(self, searcher, translator=None):
        self.searcher = searcher
        self.books = searcher.books
        self.translator = translator if translator is not None else Translator()
        self.name = f"{searcher.name} + tradução"

    def prepare(self, queries):
        """Consultas prontas para a busca (as em PT, traduzidas)."""
        queries = list(queries)
        pt = [i for i, q in enumerate(queries) if detect_language(q) == "pt"]
        for i, translated in zip(pt, self.translator.translate([queries[i] for i in pt])):
            queries[i] = translated
        return queries

    def score(self, queries):
        return self.searcher.score(self.prepare(queries))

    def search(self, query, k=5):
        return self.searcher.search(self.prepare([query])[0], k)
