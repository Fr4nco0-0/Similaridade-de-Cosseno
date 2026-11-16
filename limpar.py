import pandas as pd

df = pd.read_parquet("manga_dataset_export.parquet")

# 1. tira créditos no fim da sinopse
df["synopsis"] = (
    df["synopsis"]
    .str.replace(r"\s*\[Written by[^\]]*\]?", "", regex=True)
    .str.replace(r"\s*\(Source:[^)]*\)?", "", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# 2. remove linhas sem sinopse ou com sinopse muito curta
df = df.dropna(subset=["synopsis"])
df = df[df["synopsis"].str.split().str.len() >= 15]

# 3. remove duplicatas (mesmo mal_id ou mesma sinopse, mantendo a de maior nota)
df = df.sort_values("score", ascending=False)
df = df.drop_duplicates("mal_id").drop_duplicates("synopsis")

df = df.reset_index(drop=True)
print(len(df), "linhas")
df.to_parquet("manga_limpo.parquet", index=False)
df.to_csv("manga_limpo.csv", index=False)