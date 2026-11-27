import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODELO = "paraphrase-multilingual-MiniLM-L12-v2"
COLUNAS = ["title", "score", "genres", "themes", "synopsis", "image_url", "authors"]

# Termos bloqueados em gêneros/temas (busca por trecho)
TERMOS_BLOQUEADOS = [
    "hentai", "erotica", "ecchi", "smut", "gore", 
    "sexual violence", "incest", "harem", "reverse harem"
]

TITULOS_BLOQUEADOS = {"h wa suki? kirai?"}

# Palavras-chave na sinopse (usa \b para buscar palavras inteiras)
REGEX_SINOPSE = re.compile(
    r"\b(hentai|erotic|erotica|porn|pornographic|explicit sex|sexual assault|rape|incest|brothel|prostitution|nsfw|18\+)\b",
    re.IGNORECASE
)

def carregar():
    """Lê o dataset, o modelo e os embeddings (embeddings.npz em float16; gera só se não existir)."""
    df = pd.read_parquet("manga_limpo.parquet").reset_index(drop=True)
    model = SentenceTransformer(MODELO)
    model.max_seq_length = 128
    if os.path.exists("embeddings.npz") and len(np.load("embeddings.npz")["emb"]) == len(df):
        emb = np.load("embeddings.npz")["emb"]
    else:
        emb = model.encode(df["synopsis"].tolist(), batch_size=64,
                           normalize_embeddings=True, show_progress_bar=True)
        np.savez_compressed("embeddings.npz", emb=emb.astype(np.float16))
    return df, model, emb.astype(np.float32)

def _eh_inapropriado(row):
    """Verifica se genres, themes ou synopsis contêm conteúdo +18."""
    generos = str(row.get("genres", "")).lower()
    temas = str(row.get("themes", "")).lower()
    sinopse = str(row.get("synopsis", ""))
    # 1. Checa gêneros e temas
    texto_tags = f"{generos}, {temas}"
    if any(termo in texto_tags for termo in TERMOS_BLOQUEADOS):
        return True
    # 2. Checa sinopse com regex de palavra inteira
    if REGEX_SINOPSE.search(sinopse):
        return True
    # 3. Checa lista manual de títulos
    if str(row.get("title", "")).lower() in TITULOS_BLOQUEADOS:
        return True
    return False

def _formatar(res):
    for col in ["genres", "themes"]:
        if col in res.columns:
            res[col] = res[col].apply(
                lambda g: ", ".join(g) if isinstance(g, (list, tuple, np.ndarray)) else g
            )
    
    # Cria a coluna booleana (True/False) para o front-end/CSS
    if not res.empty:
        res["mais_18"] = res.apply(_eh_inapropriado, axis=1)
    else:
        res["mais_18"] = pd.Series(dtype=bool)
    return res

def buscar_titulo(busca, df, k=5):
    """Mangás com título correspondente."""
    b = busca.lower()
    t = df["title"].fillna("").str.lower()
    te = df["title_english"].fillna("").str.lower() if "title_english" in df.columns else t
    mask = t.str.contains(b, regex=False) | te.str.contains(b, regex=False)
    res = df[mask].copy()
    if res.empty:
        return _formatar(res[[c for c in COLUNAS if c in df.columns]].copy().assign(_idx=lambda d: d.index))
    exato = ((t == b) | (te == b))[mask].values
    comeca = (t.str.startswith(b) | te.str.startswith(b))[mask].values
    res["_ordem"] = np.where(exato, 0, np.where(comeca, 1, 2))
    res["_nota"] = pd.to_numeric(res["score"], errors="coerce").fillna(0) if "score" in res.columns else 0
    res = res.sort_values(["_ordem", "_nota"], ascending=[True, False]).head(k)
    sel = res[[c for c in COLUNAS if c in res.columns]].copy().assign(_idx=lambda d: d.index)
    return _formatar(sel).fillna("")

def recomendar(busca, df, model, emb, k=10):
    """Recomendações baseadas em similaridade."""
    b = busca.lower()
    igual = (df["title"].fillna("").str.lower() == b) | (df["title_english"].fillna("").str.lower() == b if "title_english" in df.columns else False)
    if igual.any():
        i = int(df.index[igual][0])
        q, pular, rotulo = emb[i], i, f"Mais parecidos com: {df.loc[i, 'title']}"
    else:
        q, pular, rotulo = model.encode(busca, normalize_embeddings=True).astype(np.float32), None, "Mais parecidos com o que você digitou:"
    sims = emb @ q
    top = [int(j) for j in sims.argsort()[::-1] if j != pular][:k]
    sel = df.loc[top, [c for c in COLUNAS if c in df.columns]].copy().assign(_idx=lambda d: d.index)
    res = _formatar(sel)
    res["similaridade"] = sims[top].round(3)
    return rotulo, res.fillna("")

def _itens(v):
    """Lista de tags de uma célula (lista/array ou texto separado por | ou ,)."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return [str(t).strip() for t in v if str(t).strip()]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    return [t.strip() for t in re.split(r"[|,]", str(v)) if t.strip()]

def matriz_tags(df):
    """Matriz (n, V) com 1 se o mangá tem o gênero/tema; linhas com norma 1.
    Calcule uma vez e reutilize (no app, com st.cache_resource)."""
    linhas = []
    vocab = {}
    for g, t in zip(df["genres"] if "genres" in df.columns else [None] * len(df),
                    df["themes"] if "themes" in df.columns else [None] * len(df)):
        tags = set(_itens(g)) | set(_itens(t))
        linhas.append([vocab.setdefault(x, len(vocab)) for x in tags])
    m = np.zeros((len(df), len(vocab)), dtype=np.float32)
    for i, cols in enumerate(linhas):
        m[i, cols] = 1.0
    norma = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norma == 0, 1, norma)

def recomendar_por_indices(indices, df, emb, k=1000, peso_max=0.0, peso_tags=0.0, tags=None):
    """Recomenda a partir de um ou mais mangás (índices de linha de df/emb).

    Pontuação de cada mangá j, com q = média normalizada dos vetores escolhidos:
      emb_j  = (1 - peso_max) * cos(j, q) + peso_max * max_i cos'(j, favorito_i)
      (cos' = cosseno reescalado pela densidade dos vizinhos de cada favorito)
      score  = (1 - peso_tags) * emb_j + peso_tags * cos(tags_j, perfil_de_tags)
    - cos com a média captura o gosto "geral" dos favoritos;
    - o máximo captura quem parece muito com UM favorito, mesmo que os gostos
      sejam diferentes (a média cairia num ponto "no meio");
    - o perfil de tags é a média dos vetores de gênero/tema dos favoritos.
    Com um único índice, média e máximo coincidem (cosseno puro de sinopses).
    `df` deve ter as linhas na mesma ordem de `emb` (índice = posição).
    """
    vazio = df.iloc[0:0].assign(similaridade=pd.Series(dtype=float))
    idx = sorted({int(i) for i in indices})
    if not idx:
        return vazio
    q = emb[idx].mean(axis=0)
    norma = np.linalg.norm(q)
    if norma == 0:
        return vazio
    score = emb @ (q / norma)
    if peso_max > 0 and len(idx) > 1:
        S = emb @ emb[idx].T  # (n, m): cosseno de cada mangá com cada favorito, alguns favoritos têm vizinhos naturalmente mais próximos que outros
        # (grupos mais "densos"). recoloca cada coluna pelo nível dos 10 vizinhos
        # mais próximos do favorito, senão o grupo mais denso domina o máximo.
        viz = np.partition(S, -11, axis=0)[-11:]          # 11 maiores por coluna
        ref = (np.sort(viz, axis=0)[:-1]).mean(axis=0)    # tira o próprio favorito
        S = S * (ref.mean() / ref)
        score = (1 - peso_max) * score + peso_max * S.max(axis=1)
    if peso_tags > 0 and tags is not None:
        perfil = tags[idx].mean(axis=0)
        n_p = np.linalg.norm(perfil)
        if n_p > 0:
            score = (1 - peso_tags) * score + peso_tags * (tags @ (perfil / n_p))
    score[idx] = -np.inf  # não recomenda os próprios mangás de origem
    n = min(k, len(emb) - len(idx))
    top = np.argsort(-score)[:n]
    res = df.iloc[top].copy()
    if "_idx" not in res.columns:
        res["_idx"] = res.index
    res["similaridade"] = score[top].round(3)
    return res
