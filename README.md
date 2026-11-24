## Dataset / Similaridade de Cosseno / Informações gerais

- [Base filtrada (Parquet)](manga_limpo.parquet): Base para a Similaridade de Cosseno
- Fonte do Dataset: [Anime & Manga Analytics Dataset](https://www.kaggle.com/datasets/patelris/anime-and-manga-dataset-2026)
- Último update: 24 de abril de 2026
- Fonte dos dados: [MyAnimeList](https://myanimelist.net) via [Jikan API](https://jikan.moe)

---------

## Site

https://recomendador-manga.streamlit.app/
# Recomendador de Mangás — Similaridade de Cosseno

## 1. Sobre o projeto

Este projeto implementa um sistema de recomendação de mangás utilizando Processamento de Linguagem Natural (PLN) e Similaridade de Cosseno.

A aplicação analisa as sinopses das obras para identificar conteúdos semanticamente semelhantes e recomendar mangás relacionados à busca realizada pelo usuário.

O sistema também permite pesquisar títulos e filtrar resultados por gêneros e temas.

## 2. Modelo de linguagem utilizado

O projeto utiliza o modelo:

`paraphrase-multilingual-MiniLM-L12-v2`

O modelo é disponibilizado pela biblioteca Sentence Transformers e transforma textos em embeddings, ou seja, vetores numéricos que representam características semânticas das sinopses.

A geração dos embeddings utiliza a opção `normalize_embeddings=True`, garantindo que os vetores gerados tenham norma igual a 1.

## 3. Cálculo da Similaridade de Cosseno

A Similaridade de Cosseno é utilizada para medir o grau de proximidade entre os vetores que representam as sinopses dos mangás.

A fórmula tradicional é:

cos(A, B) = (A · B) / (||A|| × ||B||)

Como os embeddings são normalizados, as normas dos vetores são iguais a 1. Dessa forma, o cálculo pode ser simplificado para o produto escalar:

similaridade(A, B) = A · B

No código, essa operação é realizada por meio de:

`emb @ q`

O resultado é utilizado para ordenar os mangás de acordo com a proximidade semântica em relação à consulta do usuário.

Quanto mais próximo de 1 for o resultado, maior será a similaridade entre os conteúdos.

## 4. Tecnologias utilizadas

- Python
- Streamlit
- Pandas
- NumPy
- PyArrow
- Sentence Transformers

## 5. Estrutura dos arquivos

- `app.py`: interface e funcionamento da aplicação Streamlit.
- `similaridade.py`: carregamento dos dados, geração de embeddings e cálculo da similaridade.
- `manga_limpo.parquet`: base de dados utilizada pelo recomendador.
- `embeddings.npz`: armazenamento dos embeddings pré-calculados.
- `estilo.css`: estilos visuais da aplicação.
- `requirements.txt`: dependências necessárias para executar o projeto.

## 6. Como executar o projeto localmente

### Pré-requisitos

- Python instalado.
- Git instalado.
- Acesso à internet para baixar o modelo de linguagem, caso ainda não esteja disponível no ambiente.

### Instalação

**1. Clone o repositório:**

```bash
git clone https://github.com/Fr4nco0-0/Similaridade-de-Cosseno.git
```

**2. Acesse a pasta do projeto:**

```bash
cd Similaridade-de-Cosseno
```

**3. Instale as dependências:**

```bash
pip install -r requirements.txt
```

**4. Inicie a aplicação:**

```bash
streamlit run app.py
```

Após a inicialização, a aplicação poderá ser acessada pelo endereço local informado no terminal, normalmente `http://localhost:8501`.

## 7. Deploy

O projeto está preparado para execução no Streamlit Community Cloud.

Para realizar o deploy, selecione o repositório GitHub, a branch `main` e o arquivo principal `app.py`.

O arquivo `requirements.txt` deve permanecer na raiz do repositório para que o ambiente instale automaticamente as bibliotecas necessárias.

O primeiro carregamento pode levar mais tempo devido à inicialização do modelo de linguagem e à leitura dos dados.

## 8. Fonte dos dados

A base de dados utilizada é o Anime & Manga Analytics Dataset, com informações provenientes do MyAnimeList por meio da Jikan API.

- Dataset: https://www.kaggle.com/datasets/patelris/anime-and-manga-dataset-2026
- MyAnimeList: https://myanimelist.net
- Jikan API: https://jikan.moe

## 9. Aplicação online

[Recomendador de Mangás — Streamlit](https://recomendador-manga.streamlit.app/)
