import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from similaridade import buscar_titulo, carregar, recomendar

MODELO = "paraphrase-multilingual-MiniLM-L12-v2"
COLUNAS = ["title", "score", "genres", "themes", "synopsis", "image_url"]

# Termos bloqueados em gêneros/temas (correspondência exata ou palavra inteira)
TERMOS_BLOQUEADOS = [
    "hentai", "erotica", "ecchi", "smut", "adult", "gore", 
    "sexual violence", "incest", "harem", "reverse harem"
]

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
