import os

import streamlit as st

from similaridade import buscar_titulo, carregar, recomendar

st.set_page_config(page_title="Recomendador de Mangá", layout="wide")

if os.path.exists("estilo.css"):
    st.markdown(f"<style>{open('estilo.css', encoding='utf-8').read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def dados():
    return carregar()


def cartao(r, similaridade=True):
    with st.container(border=True):
        col1, col2 = st.columns([1, 4])
        img_url = str(r.get("image_url", ""))
        if img_url.startswith("http"):
            if r.get("mais_18", False):
                col1.markdown(
                    f"""
                    <div class="capa-wrap">
                        <label class="capa-nsfw" title="Clique para revelar">
                            <input type="checkbox">
                            <img src="{img_url}" alt="{r.get('title', '')}">
                            <span class="badge-18">+18</span>
                        </label>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                col1.image(img_url)

        col2.markdown(f"### {r['title']}")
        info = f"Nota: {r.get('score', '')}"
        if similaridade:
            info = f"Similaridade: {float(r['similaridade']):.3f} · " + info
        col2.write(info)
        col2.caption(str(r.get("genres", "")).replace("|", ", "))
        col2.write(str(r.get("synopsis", ""))[:400])


df, model, emb = dados()

st.title("Recomendador de Mangá")
busca = st.text_input("Digite um título, trecho de sinopse ou descrição (português ou inglês)").strip()

if busca:
    achados = buscar_titulo(busca, df)
    if len(achados):
        st.subheader("Títulos encontrados")
        for _, r in achados.iterrows():
            cartao(r, similaridade=False)

    rotulo, res = recomendar(busca, df, model, emb)
    st.subheader(rotulo)
    for _, r in res.iterrows():
        cartao(r)
