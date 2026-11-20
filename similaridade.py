import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODELO = "paraphrase-multilingual-MiniLM-L12-v2"
COLUNAS = ["title", "score", "genres", "synopsis", "image_url"]


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


def recomendar(busca, df, model, emb, k=10):
    """Devolve (rótulo, DataFrame com os k mais parecidos com a busca)."""
    b = busca.lower()
    igual = df["title"].fillna("").str.lower() == b
    if "title_english" in df.columns:
        igual |= df["title_english"].fillna("").str.lower() == b

    if igual.any():
        i = int(df.index[igual][0])
        q, pular = emb[i], i
        rotulo = f"Mais parecidos com: {df.loc[i, 'title']}"
    else:
        q = model.encode(busca, normalize_embeddings=True).astype(np.float32)
        pular = None
        rotulo = "Mais parecidos com o que você digitou:"

    sims = emb @ q  # normalizado: produto escalar = cosseno
    top = [int(j) for j in sims.argsort()[::-1] if j != pular][:k]

    res = df.loc[top, [c for c in COLUNAS if c in df.columns]].copy()
    if "genres" in res.columns:
        res["genres"] = res["genres"].apply(
            lambda g: ", ".join(g) if isinstance(g, (list, tuple, np.ndarray)) else g)
    res["similaridade"] = sims[top].round(3)
    return rotulo, res.fillna("")
