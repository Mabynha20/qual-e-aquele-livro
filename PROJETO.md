# Projeto de Deep Learning com Livros: "Qual é aquele livro?"

## Contexto

**Objetivo:** criar um projeto de deep learning para aprender na prática e publicar no portfólio/LinkedIn. O tema deve ser algo de interesse pessoal: **livros** (a autora é leitora).

**Ideias levantadas (2026-10-18):**
1. Classificador de gênero pela capa (CNN).
2. Classificador de gênero pela sinopse (BERT).
3. Identificação de autoria pelo estilo (literatura de domínio público).
4. Recomendador de livros (Goodreads / Book-Crossing).
5. Busca semântica "qual é aquele livro em que..." (embeddings). **← escolhida**

**Perfil da autora:** estudante de Ciência de Dados com especialização em IA (4º semestre, Faculdade Donaduzzi). Já domina Python, pandas e outras bibliotecas e já treinou modelos de ML. O passo a passo pode partir de nível intermediário, com foco em deep learning/PyTorch.

**Ambiente:** Windows 11, Ryzen 5 5600GT com GPU integrada (sem CUDA), Python 3.13 e Git instalados. Pasta do projeto: `C:\Users\amabi\Documents\PESSOAL\projeto_busca_semantica`.

**Decisões tomadas:**
- **Ideia escolhida:** busca semântica de livros por descrição vaga (ideia 5). A 4 (recomendador) foi descartada por ser muito comum em portfólios; a 5 é mais original.
- **Framework:** PyTorch (recomendado; falta a confirmação final).
- **Dados:** **base própria coletada da Wikipédia** (opção A), com foco em livros **contemporâneos** (YA, fantasia/romantasy, romance, thriller, os populares do BookTok). Clássicos e literatura antiga podem aparecer, mas não são o foco. O CMU Book Summary Dataset (~16 mil resumos, até 2012) entra só como complemento para livros mais antigos. O texto usado é o **resumo de enredo** (seção "Plot"), não a sinopse de contracapa, porque o leitor lembra de detalhes da história toda.
- **Hardware:** **tudo roda localmente na CPU**, inclusive o treino (Parte 4). O Colab foi descartado por preferência da autora. Estratégias para a CPU: modelo pequeno (ex.: all-MiniLM-L6-v2), truncar textos (~256 tokens nos resumos, ~64 nas consultas), validar primeiro com um subconjunto, salvar checkpoints e ajustar `torch.set_num_threads()`. `torch-directml` (GPU integrada AMD) foi avaliado e não será usado por ora: é experimental e o ganho é incerto.

## Roteiro (do começo à publicação)

### Parte 0: Preparação do ambiente
- **Objetivo:** deixar o projeto pronto para começar a codar.
- **Tarefas:** criar a estrutura de pastas (`data/`, `notebooks/`, `src/`, `app/`); criar o ambiente virtual e instalar PyTorch (CPU), sentence-transformers, pandas e Jupyter; fazer `git init`, `.gitignore` (ignorar `data/` e modelos) e o repositório no GitHub.
- **Entregável:** repositório no GitHub com a estrutura e o `requirements.txt`.

### Parte 1: Coleta, dados e análise exploratória
- **Objetivo:** montar a base de livros contemporâneos e conhecê-la.
- **Tarefas:**
  - **1a. Coleta (`src/collect_wikipedia.py`):** listar romances pelas categorias da Wikipédia por ano (`Category:{ano} novels` e subcategorias, ex.: `2023 American novels`, `2023 fantasy novels`) e por gênero (ex.: `American young adult novels`, `Young adult fantasy novels`); baixar a seção "Plot"/"Plot summary" de cada página pela API (identificando o projeto no User-Agent e com pausa entre as requisições); guardar título, autor, ano, categorias e resumo; salvar o bruto em `data/raw/`.
  - **1b. Complemento:** baixar o CMU Book Summary Dataset (`http://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz`) e juntar, removendo duplicados.
  - **1c. EDA (`notebooks/01_eda.ipynb`):** tamanho dos resumos em palavras/tokens, distribuição por ano e gênero, nulos, duplicados; limpar e salvar em Parquet.
- **Entregável:** script de coleta + `notebooks/01_eda.ipynb` + dataset limpo.
- **Aprendizado:** coleta de dados via API (paginação, *rate limit*, boas práticas), tokenização e o porquê de truncar textos.

### Parte 2: Divisão dos dados e conjunto de avaliação
- **Objetivo:** definir *como* vamos medir se a busca funciona, antes de construir qualquer modelo.
- **Tarefas:** separar treino/validação/teste **por livro** (para evitar vazamento de dados); criar consultas "vagas" de teste: algumas escritas à mão, como um leitor lembraria, e outras geradas automaticamente a partir de frases dos resumos.
- **Entregável:** arquivos de split + conjunto de consultas de teste.
- **Aprendizado:** avaliação em sistemas de busca (*information retrieval*) e *data leakage*.

### Parte 3: Versões base (baselines)
- **Objetivo:** ter algo funcionando e números para comparar.
- **Tarefas:** baseline 1, busca por palavras-chave (BM25); baseline 2, embeddings pré-treinados (all-MiniLM-L6-v2) + similaridade de cosseno; implementar as métricas Recall@1/5/10 e MRR; comparar os dois.
- **Entregável:** `src/search.py`, `src/evaluate.py` + tabela de resultados.
- **Aprendizado:** o que são embeddings, busca vetorial e por que a semântica supera palavras-chave.

### Parte 4: Treino próprio em PyTorch (o coração do projeto)
- **Objetivo:** fazer *fine-tuning* do modelo para a tarefa de "lembrança vaga" e superar a baseline.
- **Tarefas:** montar os pares (consulta = frase do resumo, alvo = resumo completo) com `Dataset`/`DataLoader`; escrever o loop de treino à mão com a perda contrastiva (InfoNCE, usando os outros exemplos do lote como negativos); rodar um teste rápido com um subconjunto pequeno; treinar com o conjunto completo na CPU, salvando checkpoints; acompanhar perda e Recall na validação; comparar com as baselines.
- **Entregável:** `src/train.py`, modelo treinado e gráficos de treino.
- **Aprendizado:** loop de treino, otimizador, *learning rate*, aprendizado contrastivo e otimização para CPU.

### Parte 5: Busca em português
- **Objetivo:** permitir consultas em PT ("menina que entra num guarda-roupa...").
- **Tarefas:** testar um modelo multilíngue (ex.: paraphrase-multilingual-MiniLM); traduzir parte das consultas de teste para PT e avaliar; opcionalmente aplicar o fine-tuning da Parte 4 ao modelo multilíngue.
- **Entregável:** resultados PT vs. EN.
- **Aprendizado:** modelos multilíngues e transferência entre idiomas.

### Parte 6: Demo interativa
- **Objetivo:** qualquer pessoa poder testar pelo navegador.
- **Tarefas:** interface em Gradio (campo de busca → top 5 livros com título, autor, gênero e trecho do resumo); pré-calcular os embeddings para a busca ser instantânea; publicar no Hugging Face Spaces (CPU gratuita).
- **Entregável:** link público da demo.

### Parte 7: Documentação e acabamento
- **Objetivo:** deixar o repositório com cara de projeto profissional.
- **Tarefas:** README com problema, abordagem, resultados (tabela + gráficos), GIF da demo, como rodar e limitações; limpar o código e os notebooks; publicar o modelo no Hugging Face Hub (opcional).
- **Entregável:** repositório final no GitHub.

### Parte 8: Publicação no LinkedIn
- **Objetivo:** mostrar o projeto.
- **Tarefas:** escrever o post (história pessoal de leitora + problema + resultado em números + aprendizados, como treinar na CPU); preparar as imagens (GIF da demo, gráfico de resultados); incluir os links do GitHub e da demo; adicionar o projeto na seção "Projetos" do perfil; opcionalmente escrever um artigo mais longo explicando a técnica.
- **Entregável:** post publicado.

## Log de mudanças

| Data | Mudança | Motivo |
|------|---------|--------|
| 2026-10-18 | Criação do arquivo e levantamento de 5 ideias de projeto | Início do projeto; registrar o histórico das conversas |
| 2026-10-18 | Escolhida a ideia 5 (busca semântica de livros); escopo proposto em 5 fases | A autora achou a 5 mais original que o recomendador (4) |
| 2026-10-18 | Registrados o perfil da autora e o hardware; definido CPU local + Colab para o treino | A máquina não tem GPU NVIDIA; a autora já tem base de Python/ML |
| 2026-10-18 | Treino passa a ser 100% local na CPU (Colab descartado) | A autora prefere treinar no próprio PC; o fine-tuning de um modelo pequeno é viável na CPU |
| 2026-10-18 | As 5 fases viraram um roteiro detalhado de 9 partes (0 a 8), até a publicação no LinkedIn; incluídos uma baseline BM25 e uma etapa de avaliação antes dos modelos | A autora pediu a divisão completa do trabalho; medir antes de modelar torna os resultados confiáveis |
| 2026-10-18 | Parte 0 iniciada: os comandos são executados à mão, passo a passo; confirmados PyTorch CPU e repositório sugerido `qual-e-aquele-livro` | A autora prefere rodar os comandos para fixar o aprendizado |
| 2026-10-18 | Pasta do projeto renomeada de `projeto legal` para `projeto_busca_semantica` (feito pela autora, com o VS Code fechado) | Nome mais descritivo e sem espaço; o Windows bloqueava a renomeação com a pasta aberta |
| 2026-10-18 | Parte 0: pastas e `.venv` criadas; `Set-ExecutionPolicy RemoteSigned` (CurrentUser) para ativar a venv; instalados torch 2.14.1+cpu, sentence-transformers 6.1.0, pandas 3.0.6 e demais libs | Ambiente pronto para começar |
| 2026-10-18 | `requirements.txt` refeito à mão (enxuto, só dependências diretas, com `--extra-index-url` da CPU do PyTorch) em vez do `pip freeze` | O `>` do PowerShell 5.1 gerou UTF-16; o freeze listava 130+ pacotes; o índice CPU evita baixar o torch com CUDA no HF Spaces |
| 2026-10-18 | **Parte 0 concluída**: Git configurado, primeiro commit e repositório público criado com `gh repo create` (https://github.com/Mabynha20/qual-e-aquele-livro) | O GitHub CLI já estava instalado e logado; criar o repositório pelo terminal foi mais rápido |
| 2026-10-18 | Fonte de dados trocada: base própria da Wikipédia focada em livros contemporâneos (opção A); CMU vira complemento. Parte 1 ganhou a etapa de coleta | O CMU vai só até 2012 e não tem os livros atuais do BookTok (foco da autora). Teste: 7 de 8 títulos do BookTok (Fourth Wing, It Ends with Us, Evelyn Hugo, The Cruel Prince etc.) têm seção "Plot" na Wikipédia. Goodreads (opção B) foi descartado por trazer sinopse de contracapa e ir só até ~2017-2020 |
| 2026-11-11 | Escopo da coleta: romances de 2000 a 2026, **todos os gêneros** (gênero vira etiqueta/filtro, não critério de coleta) | A Wikipédia classifica gêneros de forma irregular (ex.: Evelyn Hugo está como "historical fiction"); filtrar na coleta perderia livros do foco |
| 2026-11-11 | Escrito e testado `src/collect_wikipedia.py`: percorre `Category:{ano} novels` + subcategorias (até 3 níveis, sem graphic/light novels), baixa o wikitexto em lotes de 50, extrai a seção de enredo, o infobox e as categorias; retomável. Novas libs: `requests`, `mwparserfromhell`, `tqdm` | Teste com 2023: 313 romances listados, 77% com enredo; Fourth Wing, Evelyn Hugo, Six of Crows etc. extraídos corretamente. Corte mínimo de 30 palavras na coleta; o corte final fica para a EDA. User-Agent usa a URL do repositório como contato |
| 2026-11-11 | Limitação encontrada: livros que só têm página de **série** na Wikipédia (ACOTAR, The Cruel Prince → "The Folk of the Air") não entram pela coleta por ano | Fica para a Parte 1b, com uma lista curada de títulos do BookTok |
| 2026-11-11 | **Coleta completa (Parte 1a)**: 14.199 romances listados (2000-2026), 10.552 com enredo (74%), em ~9 min. Sem duplicados; resumo mediano de 220 palavras (mín. 30, máx. 10.117); 13% sem autor e 21% sem gênero no infobox. Dados em `data/raw/` (fora do Git) | Base ~2x maior que o estimado. Checagem com 38 títulos populares do BookTok: 30 presentes (Fourth Wing, Onyx Storm, It Ends with Us, Evelyn Hugo, The Housemaid, Lessons in Chemistry...); faltam ACOTAR, Throne of Glass, The Cruel Prince, The Inheritance Games, Babel, The Atlas Six, Ninth House e Haunting Adeline |
| 2026-11-11 | Parte 1b: criados `lists/booktok_titles.txt` (56 títulos, formato `Título \| Autor`, editável) e `src/collect_extra.py`. Ele tenta a página exata e depois a busca da Wikipédia; valida pelo sobrenome do autor; lê os blocos `{{Book list}}` das páginas de série (pega a série inteira); descarta páginas de filme/TV; não duplica (título + autor). Saída: `data/raw/extra_books.jsonl` | Teste: +19 livros (ACOTAR 1-5, Throne of Glass 0-7, Crescent City 1-3, The Folk of the Air 1-3); 34 já estavam na base (Babel e Mistborn com o título completo); 22 sem página de enredo na Wikipédia (ex.: Caraval, Divine Rivals, Haunting Adeline, Ugly Love) ou só com página da adaptação (The Hating Game, Reminders of Him). Ninth House tem a seção de sinopse vazia. Resumos das séries são mais curtos (44-475 palavras) |
| 2026-11-11 | Parte 1c: `notebooks/01_eda.ipynb`. Decisões: (1) remover 815 *light novels* japonesas (entraram via `Japanese novels`); (2) deduplicar por título normalizado + autor, mantendo o maior resumo (3 pares); (3) mínimo de **40 palavras** por resumo, sem máximo (-231 livros); (4) guardar o resumo completo e usar ***chunks* de ~256 tokens** na Parte 3; (5) gêneros simplificados multirrótulo (fantasia, romance, jovem adulto, distopia etc.) a partir do infobox + categorias. Base final: **9.522 livros** em `data/processed/books.parquet` | Light novels fogem do foco (BookTok/YA ocidental). 56% dos resumos passam de 256 tokens (limite do all-MiniLM-L6-v2); truncar perderia o meio e o fim das histórias. Mediana de 1,28 token por palavra. "Distopia" separada de "ficção científica" (Fourth Wing tem "Dystopian fiction" no infobox). 23% dos livros ficam sem rótulo de gênero |
| 2026-11-11 | Parte 2: split **70/15/15** (treino/validação/teste), proposto pela autora (~6.665 / 1.428 / 1.428 livros). Ajustes: (1) o índice de busca contém **todos** os livros; o split define de quais livros saem os pares de treino e as consultas de avaliação; (2) divisão **agrupada por série**; (3) livros alvo das consultas escritas à mão vão para o teste; (4) semente fixa | 1.428 livros de teste dão uma estimativa estável do Recall@k (margem de ~±2-3 p.p.). Agrupar por série evita vazamento: personagens e nomes de um volume do treino "entregam" outro volume do teste. O treino na CPU é limitado por tempo, não por quantidade de dados |
| 2026-11-11 | Parte 2: criado `lists/consultas_manuais.csv` (separado por `;`): **60 consultas vagas para 30 livros populares** (BookTok, YA, thriller, romance), uma em PT e uma em EN por livro, sem título nem nomes de personagens, variando o tipo de lembrança (trama, final/reviravolta, cenário, personagem). Todas casam com a base por título + autor | As consultas em PT servem para a Parte 5. Esses 30 livros (e suas séries) vão obrigatoriamente para o teste |
| 2026-11-11 | Parte 2: `src/split.py`. Agrupamento mudou de **série** para **autor** (sem autor → série → o próprio livro). Resultado (semente 42): treino 6.630 (69,6%), validação 1.442 (15,1%), teste 1.450 (15,2%); 26 autores forçados para o teste; nenhum autor em mais de uma parte. Consultas: 60 manuais (teste) + 1 automática por livro de validação/teste (1.396 val, 1.410 teste): uma frase sorteada do resumo (10-40 palavras), sem nomes próprios (com a preposição solta antes deles), em minúsculas; pula frases "sobre o livro" (novel, chapter, sequel...) e frases com mais de 30% de nomes. Saídas: `data/processed/splits.parquet` e `queries.parquet` | O campo `series` é inconsistente (Jogos Vorazes, Em Chamas e A Esperança estão sem série), então agrupar por série deixaria a trilogia dividida entre treino e teste; agrupar por autor é mais robusto (maior grupo: 35 livros). Tirar os nomes impede que a consulta automática seja trivial (achar pelo nome do personagem). As automáticas continuam mais fáceis que as manuais (a frase vem do próprio resumo), por isso os resultados serão reportados separadamente |
| 2026-11-11 | Parte 3: `src/search.py` (BM25Okapi com stopwords; embeddings all-MiniLM-L6-v2, resumos em *chunks* de frases inteiras, nota do livro = melhor *chunk*, cache dos embeddings em `data/processed/embeddings/`) e `src/evaluate.py` (Recall@1/5/10 e MRR por tipo de consulta; salva `results/baselines.{csv,md}` e o detalhe das manuais). 1ª rodada completa (índice com 9.522 livros, *chunks* de 256 tokens): BM25 teste·auto R@1 95,1% / manual EN R@10 76,7%; embeddings teste·auto R@1 20,4% / manual EN R@10 33,3%; manual PT ~0% nos dois | Embeddings muito abaixo do BM25. Checado que não era bug (vetores normalizados, notas coerentes, ACOTAR em 1º). Hipótese: diluição, pois um vetor por *chunk* de 256 tokens mistura muitos acontecimentos, e o MiniLM foi treinado com até 128 tokens |
| 2026-11-11 | Experimento de tamanho de *chunk* (índice só com os 1.450 livros de teste, consultas EN). R@1 auto / R@10 manual: 256 → 32% / 63%; 128 → 40% / 63%; **64 → 64% / 67%**; BM25 → 98% / 93%. Padrão mudou para **64 tokens** (`CHUNK_TOKENS`) | Pedaços menores melhoram muito a busca semântica. Mesmo assim, o modelo pré-treinado perde para o BM25: ele não foi treinado para ligar lembranças vagas a resumos de enredo, e o BM25 aproveita palavras-chave marcantes. Essa é a motivação do fine-tuning (Parte 4). As consultas automáticas favorecem o BM25 (frase copiada do resumo) e *chunks* pequenos, por isso o peso maior na decisão vai para as manuais (n=30, ruidosas). Consultas em PT ficam para a Parte 5 |
| 2026-11-11 | **Baselines oficiais (Parte 3)**, índice completo (9.522 livros; 98.590 *chunks* de 64 tokens; embeddings em ~9 min na CPU). Teste · auto, R@1 / R@10 / MRR: BM25 95,1% / 99,1% / 0,966; MiniLM 53,5% / 70,1% / 0,596. Teste · manual EN (n=30): BM25 36,7% / 76,7% / 0,488; MiniLM 16,7% / 40,0% / 0,233. Manual PT: ~0% nos dois | *Chunks* de 64 tokens mais que dobraram o R@1 automático dos embeddings (20,4% → 53,5%). Nas manuais, o R@10 subiu (33% → 40%) e o R@1 caiu (27% → 17%): com n=30, cada consulta vale 3,3 p.p., então a variação é ruído. Meta da Parte 4: superar o BM25 nas consultas manuais |
| 2026-11-11 | Parte 4: `src/train.py`. Pares no estilo **Inverse Cloze Task**, só de livros de treino (até 3 por livro): consulta = frase do resumo sem nomes (mesma função da Parte 2); alvo = 2 frases de contexto de cada lado, **sem** a frase (fica em 10% dos casos). Pares em "rodadas", para um lote quase nunca ter dois pares do mesmo livro (falso negativo). Loop PyTorch escrito à mão: InfoNCE com negativos do lote (temperatura 0,05), lote 32, AdamW (lr 2e-5, weight decay 0,01), aquecimento linear de 10% + decaimento linear, *gradient clipping* 1,0, consulta ≤64 e alvo ≤96 tokens, 2 épocas, semente 42. Validação por época (consultas automáticas de validação, índice só com os livros de validação), checkpoint por época, melhor modelo pelo MRR em `models/minilm-livros/` (fora do Git); log e gráficos em `results/`. `search.py` passou a aceitar um modelo em memória e `evaluate.py` ganhou `--model` (gera `results/comparacao.*`). Teste `--quick` (200 livros, 524 pares, 16 passos, ~1 min): perda 2,4 → 1,95; MRR de validação (150 livros) 0,850 → 0,925 | Remover a frase do alvo obriga o modelo a ligar a lembrança ao contexto da história, em vez de reconhecer texto copiado. Estimativa do treino completo: ~17 mil pares, ~540 passos por época, ~1 h no total na CPU |
| 2026-11-11 | **Resultado do fine-tuning (Parte 4)**: 17.807 pares de 6.630 livros, 1.112 passos (2 épocas), 43 min na CPU. Perda ~2,05 → ~1,3; acerto no lote 44% → ~63%. MRR de validação (índice de 1.442 livros): 0,711 → **0,940** (época 1) → 0,935 (época 2); salvo o modelo da época 1. Índice completo, teste · auto, R@1 / R@10 / MRR: MiniLM original 53,5% / 70,1% / 0,596 → **ajustado 87,3% / 95,5% / 0,903** (BM25 95,1% / 99,1% / 0,966). Teste · manual EN: original 16,7% / 40,0% / 0,233 → **ajustado 33,3% / 56,7% / 0,407** (BM25 36,7% / 76,7% / 0,488). Manual PT: ~0% em todos | O fine-tuning quase dobrou o MRR dos embeddings, mas **ainda não supera o BM25** nas manuais. Consulta a consulta (EN): ajustado melhor em 7, BM25 em 15, empate em 8. O BM25 vence quando a lembrança tem palavras marcantes (vampire, assassin, amnesia, cane); o modelo vence nas paráfrases (The Midnight Library: BM25 412º → ajustado 1º; Fourth Wing 39º → 3º; It Ends with Us 16º → 3º). Os dois se complementam: em 86,7% das consultas EN pelo menos um deles põe o livro no top 10 (BM25 sozinho: 76,7%), o que motiva uma busca **híbrida**. A 2ª época não melhorou a validação (perda ainda caindo): sinal de saturação nas consultas automáticas |
| 2026-11-11 | Parte 4b: **busca híbrida** `HybridSearcher` em `src/search.py`, por *Reciprocal Rank Fusion*: nota = 1/(60 + posição no BM25) + 1/(60 + posição no modelo ajustado). k = 60 é o padrão da literatura, fixado **sem ajuste** nas consultas de teste. Entrou no `evaluate.py --model` e virou o método padrão de `python src/search.py` (usa `models/minilm-livros` se existir). Corrigida a chave do cache de embeddings: o caminho do modelo é normalizado, então caminho absoluto e relativo usam o mesmo cache. Resultado (teste): auto R@1 / R@10 / MRR **95,5% / 99,8% / 0,970** (BM25: 95,1% / 99,1% / 0,966); manual EN **40,0% / 86,7% / 0,544** (BM25: 36,7% / 76,7% / 0,488). Manual PT segue ~0% | **Primeira vez que a meta foi batida: o híbrido supera o BM25 em todas as métricas.** Consulta a consulta (EN): melhor que o BM25 em 11, pior em 8. Limitação do RRF: um livro em 1º num método e muito mal no outro pode não chegar ao top 5 (ex.: The Midnight Library, 1º nos embeddings e 412º no BM25). Pesos entre os métodos não foram ajustados, para não sobreajustar às 30 consultas manuais |

## Próximos passos

- [x] Parte 0: pastas, `.venv`, instalação, `requirements.txt`, `.gitignore`, primeiro commit e push. Repositório: https://github.com/Mabynha20/qual-e-aquele-livro
- [x] Parte 1a: escopo definido (2000-2026, todos os gêneros) e `src/collect_wikipedia.py` escrito e testado.
- [x] Parte 1a: coleta completa rodada: 14.199 romances listados, **10.552 com enredo**.
- [x] Parte 1b: `src/collect_extra.py` + `lists/booktok_titles.txt` escritos e testados.
- [x] Parte 1b: `python src/collect_extra.py` rodado: +19 livros em `data/raw/extra_books.jsonl` (base total: 10.571). A lista pode crescer depois, inclusive com títulos do BookTok BR.
- [ ] Parte 1b (opcional): complemento com o CMU para livros anteriores a 2000.
- [x] Parte 1c: `notebooks/01_eda.ipynb` escrito e testado.
- [x] Parte 1c: a autora roda o notebook no VS Code (gera `data/processed/books.parquet`).
- [x] Parte 2: `src/split.py` escrito e testado (split 70/15/15 agrupado por autor + consultas de avaliação).
- [x] Parte 2: `python src/split.py` rodado.
- [x] Parte 3: `src/search.py` (BM25 + embeddings com *chunks* de 64 tokens) e `src/evaluate.py` (Recall@1/5/10, MRR) escritos e testados.
- [x] Parte 3: `python src/evaluate.py` rodado; baselines oficiais em `results/baselines.md`.
- [x] Parte 4: `src/train.py` escrito e testado (`--quick`).
- [x] Parte 4: treino completo rodado (43 min) e comparado em `results/comparacao.md`: MRR manual EN 0,233 → 0,407 (BM25: 0,488).
- [x] Parte 4b: busca híbrida (RRF) escrita e testada: MRR manual EN 0,544 (BM25: 0,488).
- [ ] Parte 4b: a autora roda `python src/evaluate.py --model models/minilm-livros` (~4 min, usa o cache) e faz o commit.
- [ ] Parte 5: modelo multilíngue para as consultas em PT (hoje ~0%).
