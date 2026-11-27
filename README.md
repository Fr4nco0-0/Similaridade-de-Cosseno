# Recomendador de Mangás — Similaridade de Cosseno

Aplicação web interativa para busca e recomendação semântica de mangás utilizando Processamento de Linguagem Natural (PLN).

- **Aplicação online:** [recomendador-manga.streamlit.app](https://recomendador-manga.streamlit.app/)
- **Base filtrada:** [`manga_limpo.parquet`](manga_limpo.parquet)
- **Fonte do Dataset:** [Anime & Manga Analytics Dataset](https://www.kaggle.com/datasets/patelris/anime-and-manga-dataset-2026) *(último update: 24 de abril de 2026)*
- **Origem dos dados:** [MyAnimeList](https://myanimelist.net) via [Jikan API](https://jikan.moe)

---

## 1. Sobre o projeto

Este projeto implementa um sistema de recomendação de mangás utilizando Processamento de Linguagem Natural (PLN) e Similaridade de Cosseno. A aplicação analisa as sinopses das obras para identificar conteúdos semanticamente semelhantes e recomendar mangás relacionados à busca realizada pelo usuário.

### Principais funcionalidades
### Principais funcionalidades
- **Busca por título e descrição:** Pesquisa direta por título (original ou em inglês) ou por trechos/descrições livres em português ou inglês.
- **Filtros combinados:** Refinamento por múltiplos gêneros e temas simultaneamente.
- **Vitrine paginada:** Exibição dos mangás mais bem avaliados quando nenhuma busca está ativa.
- **Moderação de conteúdo (+18):** Verificação de idade na entrada (salva no navegador com opção para alterar a preferência na barra lateral) e detecção automática de conteúdo sensível em gêneros, temas e sinopses, aplicando desfoque (`blur`) clicável nas capas.
- **Sinopses expansíveis:** Corte limpo na última palavra completa com opção de expandir para ler a prévia integral.
- **Favoritos:** Permite adicionar títulos na lista pessoal (salva localmente no navegador via `localStorage`), visualizar os itens salvos, limpar a seleção ou gerar recomendações personalizadas com base nas obras favoritas.
- **Parecidos:** Exibe sugestões de obras semelhantes diretamente a partir do card de cada título.

---

## 2. Modelo de linguagem utilizado

O projeto utiliza o modelo pré-treinado:

`paraphrase-multilingual-MiniLM-L12-v2`

Disponibilizado pela biblioteca **Sentence Transformers**, o modelo transforma textos em *embeddings* (vetores numéricos densos que capturam o significado semântico das sinopses em múltiplos idiomas).

A geração dos embeddings utiliza `normalize_embeddings=True`, garantindo que todos os vetores tenham norma euclidiana igual a 1.

---

## 3. Cálculo da Similaridade de Cosseno

A Similaridade de Cosseno mede o grau de proximidade entre o vetor da consulta e os vetores das sinopses do catálogo:

$$\text{cos}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|}$$

Como os embeddings estão normalizados ($\|A\| = \|B\| = 1$), o cálculo se resume ao produto escalar:

$$\text{similaridade}(A, B) = A \cdot B$$

No código, essa operação é executada de forma vetorizada com NumPy por meio de `emb @ q`, ordenando os mangás do mais próximo de `1` (maior semelhança) ao menor.

---

## 4. Tecnologias utilizadas

- **Python** (incluindo módulos nativos como `os`, `re`, `html`, `json` e `time`)
- **Streamlit**
- **Streamlit Local Storage**
- **Pandas**
- **NumPy**
- **PyArrow**
- **Sentence Transformers**

---

## 5. Estrutura dos arquivos

- `app.py`: interface web, vitrine paginada, filtros, favoritos salvos no navegador (`localStorage`), modal e controle de preferência +18 e rodapé.
- `similaridade.py`: carregamento dos dados, geração/leitura de embeddings, cálculo da similaridade e detecção de conteúdo +18.
- `limpar.py`: script de limpeza, remoção de duplicatas e pré-processamento do dataset original.
- `manga_limpo.parquet`: base de dados tratada utilizada pelo recomendador.
- `embeddings.npz`: armazenamento compactado (`float16`) dos embeddings pré-calculados.
- `estilo.css`: estilização personalizada da interface, cards, desfoque de capas +18 e responsividade.
- `requirements.txt`: dependências externas necessárias para executar o projeto.

---

## 6. Como executar o projeto localmente

### Pré-requisitos
- Python 3.9+ instalado
- Git instalado
- Conexão com a internet no primeiro carregamento (para download do modelo de linguagem)

### Instalação e execução

**1. Clone o repositório:**
```bash
git clone https://github.com/Fr4nco0-0/Similaridade-de-Cosseno.git
cd Similaridade-de-Cosseno
```

**2. Instale as dependências:**
```bash
pip install -r requirements.txt
```

**3. Inicie a aplicação:**
```bash
streamlit run app.py
```

Após a inicialização, acesse o endereço exibido no terminal (geralmente `http://localhost:8501`).

---

## 7. Deploy

O projeto está configurado para execução no **Streamlit Community Cloud**, apontando para a branch `main` e o arquivo principal `app.py`. O arquivo `requirements.txt` na raiz garante a instalação automática das bibliotecas necessárias.

---

## 8. Autores

- **Amanda da Silva Gonçalves**
- **Gustavo Franco Mendonça Vasques**
- **Isabelly Raizer**
