import html
import json
import os
import pandas as pd
import streamlit as st
from streamlit_local_storage import LocalStorage
from similaridade import _formatar, buscar_titulo, carregar, recomendar, recomendar_por_indices, matriz_tags

st.set_page_config(page_title="Recomendador de Mangá", layout="wide")
if os.path.exists("estilo.css"):
    st.markdown(f"<style>{open('estilo.css', encoding='utf-8').read()}</style>", unsafe_allow_html=True)

# CONEXÃO COM O LOCALSTORAGE DO NAVEGADOR
localS = LocalStorage()

if "favoritos" not in st.session_state:
    st.session_state.favoritos = []
if "favs_sincronizados" not in st.session_state:
    st.session_state.favs_sincronizados = False

# Recupera os favoritos salvos assim que o componente carregar no navegador
if not st.session_state.favs_sincronizados:
    favs_salvos = localS.getItem("favoritos_manga")
    if favs_salvos is not None:
        try:
            lista_ids = json.loads(favs_salvos) if isinstance(favs_salvos, str) else favs_salvos
            st.session_state.favoritos = [int(x) for x in lista_ids]
        except Exception:
            st.session_state.favoritos = []
        st.session_state.favs_sincronizados = True

# ESTADO: "visão" mostrada na coluna da direita
st.session_state.setdefault("visao", "vitrine")  # vitrine | parecidos | recs_fav | lista_fav
st.session_state.setdefault("ref_idx", None)

VISOES_ESPECIAIS = ("parecidos", "recs_fav", "lista_fav")


def ir_para(visao, ref=None):
    st.session_state.visao = visao
    st.session_state.ref_idx = ref
    for v in VISOES_ESPECIAIS:
        st.session_state[f"pag_{v}"] = 1


def alternar_fav(idx):
    favs = st.session_state.favoritos
    if idx in favs:
        favs.remove(idx)
    else:
        favs.append(idx)
    st.session_state["pag_recs_fav"] = 1
    st.session_state.favs_sincronizados = True
    localS.setItem("favoritos_manga", json.dumps(favs))


def limpar_favoritos():
    st.session_state.favoritos = []
    st.session_state.favs_sincronizados = True
    localS.deleteItem("favoritos_manga")
    if st.session_state.visao in ("recs_fav", "lista_fav"):
        ir_para("vitrine")


def botoes_acao(onde, r, chave):
    """Botões ♡ Favoritar e 🔍 Parecidos. `chave` evita chaves duplicadas entre seções."""
    if r.get("_idx") is None or r.get("_idx") == "":
        return
    idx = int(r["_idx"])
    fav = idx in st.session_state.favoritos
    onde.button(
        "♥ Favorito" if fav else "♡ Favoritar",
        key=f"fav_{chave}_{idx}",
        on_click=alternar_fav,
        args=(idx,),
        type="primary" if fav else "secondary",
        use_container_width=True,
        help="Remover dos favoritos" if fav else "Adicionar aos favoritos",
    )
    onde.button(
        "🔍 Parecidos",
        key=f"sim_{chave}_{idx}",
        on_click=ir_para,
        args=("parecidos", idx),
        use_container_width=True,
        help="Ver mangás com sinopse parecida com esta",
    )


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


def cartao(r, similaridade=True, chave="c"):
    with st.container(border=True):
        col1, col2, col3 = st.columns((1.2, 4, 1.4))
        capa(col1, r)
        col2.markdown(f"### {r['title']}")
        nota = r.get("score", "")
        nota = "—" if str(nota).strip() in ("", "nan", "None", "0", "0.0") else nota
        info = f"Nota: {nota}"
        if similaridade:
            info = f"Similaridade: {float(r['similaridade']):.3f} · " + info
        col2.write(info)
        col2.caption(str(r.get("genres", "")).replace("|", ", "))

        autores = str(r.get("authors", "")).strip()
        if autores:
            col3.markdown("**Autor**")
            for a in autores.split("|"):
                col3.write(a.strip())
        botoes_acao(col3, r, chave)

        sinopse = str(r.get("synopsis", "")).strip()
        if not sinopse:
            st.write("Sinopse não disponível.")
        else:
            incompleta = sinopse[-1] not in ".!?\"'"
            if incompleta and " " in sinopse:
                antes, _, _ = sinopse.rpartition(" ")
                sinopse = antes + "... <em>(fim da prévia)</em>"
            if len(sinopse) > 400:
                resumo, _, _ = sinopse[:400].rpartition(" ")
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


def cartao_compacto(r, chave="g"):
    with st.container(border=True, height=430):
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
        botoes_acao(st, r, chave)


def grade(res, chave="vitrine", por_pagina=10, por_linha=5):
    total = len(res)
    if total == 0:
        st.info("Nenhum mangá encontrado.")
        return

    if st.session_state.maior_18 and res["mais_18"].astype(bool).any():
        st.warning("Alguns resultados têm conteúdo +18. As capas ficam borradas até você clicar nelas.")
    elif not st.session_state.maior_18:
        st.info("Resultados +18 estão ocultos. Recarregue a página para responder ao aviso de idade novamente.")

    total_paginas = max(1, (total - 1) // por_pagina + 1)
    estado_pag = f"pag_{chave}"
    if estado_pag not in st.session_state or st.session_state[estado_pag] > total_paginas:
        st.session_state[estado_pag] = 1

    pag = st.session_state[estado_pag]
    inicio = (pag - 1) * por_pagina
    fatia = res.iloc[inicio:inicio + por_pagina]

    for i in range(0, len(fatia), por_linha):
        cols = st.columns(por_linha)
        for col, (_, r) in zip(cols, fatia.iloc[i:i + por_linha].iterrows()):
            with col:
                cartao_compacto(r, chave)

    if total_paginas > 1:
        c_esq, c_meio, c_dir = st.columns((1.2, 2, 1.2))
        if c_esq.button("◀ Anterior", key=f"prev_{chave}", disabled=(pag <= 1), use_container_width=True):
            st.session_state[estado_pag] -= 1
            st.rerun()
        c_meio.markdown(
            f"<div style='text-align:center; padding-top:6px; color:var(--muted);'>Página <strong>{pag}</strong> de {total_paginas}</div>",
            unsafe_allow_html=True,
        )
        if c_dir.button("Próxima ▶", key=f"next_{chave}", disabled=(pag >= total_paginas), use_container_width=True):
            st.session_state[estado_pag] += 1
            st.rerun()


df, model, emb = dados()


@st.cache_resource
def df_completo():
    d = df if "mais_18" in df.columns else _formatar(df.copy())
    # df tem índice 0..n-1 (reset_index), igual à linha de `emb`
    return d if "_idx" in d.columns else d.assign(_idx=d.index)


@st.cache_resource
def tags_matriz():
    return matriz_tags(df_completo())


# Pesos da recomendação por favoritos (ver recomendar_por_indices)
PESO_MAX = 0.7   # 0 = só a média dos favoritos; 1 = só o favorito mais parecido
PESO_TAGS = 0.2  # peso de gêneros/temas em comum com os favoritos


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
        if k.startswith(("g_", "t_", "pag_")):
            if k.startswith("pag_"):
                st.session_state[k] = 1
            else:
                st.session_state[k] = False


def checklist(titulo, itens, prefixo):
    st.markdown(f"**{titulo}**")
    escolhidos = []
    with st.container(height=220):
        for item in itens:
            if st.checkbox(item, key=f"{prefixo}_{item}"):
                escolhidos.append(item)
    return escolhidos


def lista(res, similaridade=True, chave="l"):
    if len(res) == 0:
        st.info("Nenhum mangá encontrado com esses filtros.")
        return
    if st.session_state.maior_18 and res["mais_18"].astype(bool).any():
        st.warning("Alguns resultados têm conteúdo +18. As capas ficam borradas até você clicar nelas.")
    elif not st.session_state.maior_18:
        st.info("Resultados +18 estão ocultos. Recarregue a página para responder ao aviso de idade novamente.")
    for _, r in res.iterrows():
        cartao(r, similaridade=similaridade, chave=chave)


st.title("Recomendador de Mangá")
st.caption("As recomendações são ordenadas por sua similaridade entre as sinopses. Use os filtros de gêneros e temas para refinar a sua busca.")
busca = st.text_input(
    "Digite um título, trecho de sinopse ou descrição",
    placeholder="Dica: quanto mais detalhada a descrição, mais precisa será a recomendação. Valores perto de 0,4 a 0,6 já indicam boa semelhança.",
    key="busca",
    on_change=ir_para,
    args=("vitrine",),  # digitar uma nova busca sai das telas de parecidos/favoritos
).strip()

col_filtros, col_resultados = st.columns((1, 3), gap="large")
with col_filtros:
    with st.container(border=True):
        st.markdown("### Gêneros e temas")
        generos = checklist("Gêneros", opcoes(df, "genres"), "g")
        temas = checklist("Temas", opcoes(df, "themes"), "t")
        st.button("Limpar filtros", on_click=limpar_filtros, use_container_width=True)
        st.caption("Os resultados precisam ter todos os itens escolhidos.")
    with st.container(border=True):
        n_fav = len(st.session_state.favoritos)
        st.markdown(f"### Favoritos ({n_fav})")
        st.button("Ver meus favoritos", key="ver_fav", on_click=ir_para, args=("lista_fav",),
                  disabled=n_fav == 0, use_container_width=True)
        st.button("Recomendar com base nos favoritos", key="rec_fav", on_click=ir_para,
                  args=("recs_fav",), disabled=n_fav == 0, use_container_width=True)
        st.button("Limpar favoritos", key="limpar_fav", on_click=limpar_favoritos,
                  disabled=n_fav == 0, use_container_width=True)


def tela_especial(generos, temas):
    visao = st.session_state.visao
    base = df_completo()
    st.button("← Voltar", key="voltar", on_click=ir_para, args=("vitrine",))

    if visao == "parecidos":
        ref = st.session_state.ref_idx
        st.subheader(f"Mais parecidos com: {base.at[ref, 'title']}")
        st.caption("Cosseno entre a sinopse deste mangá e a de todos os outros.")
        res = recomendar_por_indices([ref], base, emb)
    elif visao == "recs_fav":
        favs = st.session_state.favoritos
        if not favs:
            st.info("Você ainda não tem favoritos. Use o botão ♡ nos cartões.")
            return
        st.subheader(f"Recomendados a partir dos seus {len(favs)} favoritos")
        st.caption(
            "Nota de cada mangá: 80% sinopse (30% cosseno com a média dos favoritos, "
            "70% pelo favorito mais parecido) + 20% gêneros e temas em comum com seus favoritos."
        )
        res = recomendar_por_indices(favs, base, emb, peso_max=PESO_MAX,
                                     peso_tags=PESO_TAGS, tags=tags_matriz())
    else:  # lista_fav
        favs = st.session_state.favoritos
        if not favs:
            st.info("Você ainda não tem favoritos. Use o botão ♡ nos cartões.")
            return
        st.subheader("Meus favoritos")
        res = base.iloc[favs]

    res = aplicar_filtros(filtrar(res), generos, temas)
    grade(res, chave=visao)

with col_resultados:
    if st.session_state.visao in VISOES_ESPECIAIS:
        tela_especial(generos, temas)
    elif busca:
        achados = aplicar_filtros(filtrar(buscar_titulo(busca, df)), generos, temas)
        if len(achados):
            st.subheader("Títulos encontrados")
            for _, r in achados.iterrows():
                cartao(r, similaridade=False, chave="t")
        rotulo, res = recomendar(busca, df, model, emb, k=1000)
        res = aplicar_filtros(filtrar(res), generos, temas)
        if len(achados):
            res = res[~res["title"].isin(achados["title"])]
        res = res.head(10)
        st.subheader(rotulo)
        lista(res, chave="r")
    elif generos or temas:
        res = aplicar_filtros(filtrar(df_completo()), generos, temas)
        res = res.assign(_nota=pd.to_numeric(res["score"], errors="coerce")).sort_values("_nota", ascending=False)
        st.subheader("Melhores notas nesses filtros")
        grade(res, chave="filtros")
    else:
        base = filtrar(df_completo())
        destaque = base.assign(_nota=pd.to_numeric(base["score"], errors="coerce")).sort_values("_nota", ascending=False)
        st.subheader("Mangás mais bem avaliados")
        grade(destaque, chave="destaque")
        
st.markdown(
    """
    <div style="margin-top: 50px; padding: 22px 20px; background-color: #171717; border: 1px solid #303030; border-radius: 12px; text-align: center; color: #999999; font-size: 13px; line-height: 1.7;">
        <strong style="color: #f5f5f5;">Recomendador de Mangá</strong> · Similaridade do cosseno com embeddings de sinopses<br>
        Dados e capas via MyAnimeList (<a href="https://myanimelist.net" target="_blank" style="color: #ffffff; font-weight: 600; text-decoration: none;">ver site</a>) · Autores: Amanda da Silva Gonçalves, Gustavo Franco Mendonça Vasques e Isabelly Raizer ·
        <a href="https://github.com" target="_blank" style="color: #ffffff; font-weight: 600; text-decoration: none;">GitHub</a> · © 2026<br>
        Alguns resultados podem conter conteúdo adulto (+18), indicado para maiores de 18 anos.
    </div>
    """,
    unsafe_allow_html=True,
)
