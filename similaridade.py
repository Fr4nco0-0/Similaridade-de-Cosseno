import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_parquet("manga_limpo.parquet")

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
model.max_seq_length = 128

# embeddings: gera só se ainda não existirem
if os.path.exists("embeddings.npy") and len(np.load("embeddings.npy", mmap_mode="r")) == len(df):
    emb = np.load("embeddings.npy")
else:
    emb = model.encode(df["synopsis"].tolist(), batch_size=64,
                       normalize_embeddings=True, show_progress_bar=True)
    np.save("embeddings.npy", emb)

# busca livre: título, trecho de sinopse, palavra ou descrição (português ou inglês)
while True:
    busca = input("\nDigite um título, trecho da sinopse ou descrição (ou 'sair'): ").strip()
    if busca.lower() == "sair":
        break
    if not busca:
        continue

    # se for exatamente o título de um mangá, usa o vetor dele e tira ele mesmo da lista
    igual = (df["title"].str.lower() == busca.lower()) | (df["title_english"].str.lower() == busca.lower())
    if igual.any():
        i = df.index[igual][0]
        print(f"\nMais parecidos com: {df.loc[i, 'title']}")
        q = emb[i]
        pular = i
    else:
        print("\nMais parecidos com o que você digitou:")
        q = model.encode(busca, normalize_embeddings=True)
        pular = None

    sims = emb @ q                       # normalizado: produto escalar = cosseno
    top = [j for j in sims.argsort()[::-1] if j != pular][:10]
    res = df.loc[top, ["title", "score"]].copy()
    res["similaridade"] = sims[top].round(3)
    print(res.to_string())