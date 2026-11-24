import html
import os
import pandas as pd
import streamlit as st
from similaridade import _formatar, buscar_titulo, carregar, recomendar

st.set_page_config(page_title="Recomendador de Mangá", layout="wide")
if os.path.exists("estilo.css"):
    st.markdown(f"<style>{open('estilo.css', encoding='utf-8').read()}</style>", unsafe_allow_html=True)

@st.cache_resource
def dados():
    return carregar()

def capa(onde, r):
    img_url = str(r.get("image_url", ""))
    if not img_url.startswith("http"):
        return
    if r.get("mais_18", False):
        onde.markdown(
            f"""
            <div class="capa-wrap">
                <label class="capa-nsfw" title="Clique para revelar">
                    <input type="checkbox">
                    <img src="{img_url}" alt="{html.escape(str(r.get('title', '')), quote=True)}">
                    <span class="badge-18">+18</span>
                </label>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        onde.image(img_url)

def cartao(r, similaridade=True):
    with st.container(border=True):
        col1, col2, col3 = st.columns([1.2, 4, 1.4])
        capa(col1, r)
        col2.markdown(f"### {r['title']}")
        info = f"Nota: {r.get('score', '')}"
        if similaridade:
            info = f"Similaridade: {float(r['similaridade']):.3f} · " + info
        col2.write(info)
        col2.caption(str(r.get("genres", "")).replace("|", ", "))

        autores = str(r.get("authors", "")).strip()
        if autores:
            col3.markdown("**Autor**")
            for a in autores.split("|"):
                col3.write(a.strip())

        sinopse = str(r.get("synopsis", "")).strip()
        if not sinopse:
            st.write("Sinopse não disponível.")
        else:
            incompleta = sinopse[-1] not in ".!?\"'"
            if incompleta:
                sinopse = sinopse.rsplit(" ", 1)[0] + "... <em>(fim da prévia)</em>"
            if len(sinopse) > 400:
                resumo = sinopse[:400].rsplit(" ", 1)[0]
                st.markdown(
                    f"""
                    <details class="sinopse-toggle">
                        <summary>{resumo}<span class="ver-mais">... (ler mais)</span></summary>
                        <p>{sinopse}</p>
                    </details>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(sinopse, unsafe_allow_html=True)

def _txt(valor, n, sep):
    itens = [t.strip() for t in str(valor).split("|") if t.strip()]
    return html.escape(sep.join(itens[:n]))

def cartao_compacto(r):
    with st.container(border=True, height=360):
        capa(st, r)
        nota = r.get("score", "")
        nota = "—" if str(nota) in ("", "nan", "None") else nota
        st.markdown(
            f"""
            <div class="cc-titulo">{html.escape(str(r['title']))}</div>
            <div class="cc-nota">Nota: {nota}</div>
            <div class="cc-meta">{_txt(r.get('genres', ''), 3, ', ')}</div>
            <div class="cc-meta">{_txt(r.get('authors', ''), 2, ' · ')}</div>
            """,
            unsafe_allow_html=True,
        )

def grade(res, por_linha=5):
    if len(res) == 0:
        st.info("Nenhum mangá encontrado.")
        return
    if st.session_state.maior_18 and res["mais_18"].astype(bool).any():
        st.warning("Alguns resultados têm conteúdo +18. As capas ficam borradas até você clicar nelas.")
    elif not st.session_state.maior_18:
        st.info("Resultados +18 estão ocultos. Recarregue a página para responder ao aviso de idade novamente.")
    for i in range(0, len(res), por_linha):
        cols = st.columns(por_linha)
        for col, (_, r) in zip(cols, res.iloc[i:i + por_linha].iterrows()):
            with col:
                cartao_compacto(r)

df, model, emb = dados()

@st.cache_resource
def df_completo():
    return df if "mais_18" in df.columns else _formatar(df.copy())

# AVISO +18
if "maior_18" not in st.session_state:
    st.session_state.maior_18 = None

@st.dialog("Você tem 18 anos ou mais?")
def aviso_idade():
    st.write("Este site pode exibir capas e sinopses de mangás com conteúdo adulto (+18).")
    st.markdown(
        """
        <div class="termo-box">
            <strong>Termo de responsabilidade</strong><br>
            Ao aceitar, você declara ter 18 anos ou mais e assume a responsabilidade por
            visualizar esse conteúdo. As obras vêm de um dataset público do MyAnimeList e este
            é um trabalho acadêmico: os autores não produzem nem recomendam o conteúdo.
            Se você não aceitar, os resultados +18 serão ocultados.
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    if c1.button("Sim, tenho 18+ e aceito", type="primary", use_container_width=True):
        st.session_state.maior_18 = True
        st.rerun()
    if c2.button("Não aceito", use_container_width=True):
        st.session_state.maior_18 = False
        st.rerun()

def filtrar(d):
    if st.session_state.maior_18:
        return d
    return d[~d["mais_18"].astype(bool)]

if st.session_state.maior_18 is None:
    aviso_idade()

# FILTROS DE GÊNERO E TEMA
@st.cache_data
def opcoes(_df, coluna):
    if coluna not in _df.columns:
        return []
    itens = set()
    for v in _df[coluna].dropna().astype(str):
        for t in v.split("|"):
            t = t.strip()
            if t:
                itens.add(t)
    return sorted(itens)

def contem_todos(d, coluna, escolhidos):
    if not escolhidos or coluna not in d.columns:
        return d

    def ok(v):
        itens = {t.strip() for t in str(v).split("|")}
        return all(e in itens for e in escolhidos)

    return d[d[coluna].apply(ok)]

def aplicar_filtros(d, generos, temas):
    return contem_todos(contem_todos(d, "genres", generos), "themes", temas)

def limpar_filtros():
    for k in list(st.session_state):
        if k.startswith(("g_", "t_")):
            st.session_state[k] = False

def checklist(titulo, itens, prefixo):
    st.markdown(f"**{titulo}**")
    escolhidos = []
    with st.container(height=220):
        for item in itens:
            if st.checkbox(item, key=f"{prefixo}_{item}"):
                escolhidos.append(item)
    return escolhidos

def lista(res, similaridade=True):
    if len(res) == 0:
        st.info("Nenhum mangá encontrado com esses filtros.")
        return
    if st.session_state.maior_18 and res["mais_18"].astype(bool).any():
        st.warning("Alguns resultados têm conteúdo +18. As capas ficam borradas até você clicar nelas.")
    elif not st.session_state.maior_18:
        st.info("Resultados +18 estão ocultos. Recarregue a página para responder ao aviso de idade novamente.")
    for _, r in res.iterrows():
        cartao(r, similaridade=similaridade)

st.title("Recomendador de Mangá")
st.caption("As recomendações são ordenadas por sua similaridade entre as sinopses. Use os filtros de gêneros e temas para refinar a sua busca.")
busca = st.text_input(
    "Digite um título, trecho de sinopse ou descrição",
    placeholder="Dica: quanto mais detalhada a descrição, mais precisa será a recomendação. Valores perto de 0,4 a 0,6 já indicam boa semelhança.",
).strip()

col_filtros, col_resultados = st.columns([1, 3], gap="large")
with col_filtros:
    with st.container(border=True):
        st.markdown("### Gêneros e temas")
        generos = checklist("Gêneros", opcoes(df, "genres"), "g")
        temas = checklist("Temas", opcoes(df, "themes"), "t")
        st.button("Limpar filtros", on_click=limpar_filtros, use_container_width=True)
        st.caption("Os resultados precisam ter todos os itens escolhidos.")

with col_resultados:
    if busca:
        achados = aplicar_filtros(filtrar(buscar_titulo(busca, df)), generos, temas)
        if len(achados):
            st.subheader("Títulos encontrados")
            for _, r in achados.iterrows():
                cartao(r, similaridade=False)
        rotulo, res = recomendar(busca, df, model, emb, k=1000)
        res = aplicar_filtros(filtrar(res), generos, temas)
        if len(achados):
            res = res[~res["title"].isin(achados["title"])]
        res = res.head(10)
        st.subheader(rotulo)
        lista(res)
    elif generos or temas:
        res = aplicar_filtros(filtrar(df_completo()), generos, temas)
        res = (
            res.assign(_nota=pd.to_numeric(res["score"], errors="coerce"))
            .sort_values("_nota", ascending=False)
            .head(10)
        )
        st.subheader("Melhores notas nesses filtros")
        grade(res)
    else:
        base = filtrar(df_completo())
        destaque = (
            base.assign(_nota=pd.to_numeric(base["score"], errors="coerce"))
            .sort_values("_nota", ascending=False)
            .head(10)
        )
        st.subheader("Mangás mais bem avaliados")
        grade(destaque)

st.markdown(
    """
    <div class="rodape">
        Recomendador de Mangá · Similaridade do cosseno com embeddings de sinopses<br>
        Dados e capas via MyAnimeList (<a href="https://myanimelist.net" target="_blank">ver site</a>) · Autores: Amanda da Silva Gonçalves, Gustavo Franco Mendonça Vasques e Isabelly Raizer ·
        <a href="https://github.com" target="_blank">GitHub</a> · © 2026<br>
        Alguns resultados podem conter conteúdo adulto (+18), indicado para maiores de 18 anos.
    </div>
    """,
    unsafe_allow_html=True,
)
