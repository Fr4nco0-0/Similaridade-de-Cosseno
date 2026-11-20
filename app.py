import os

import streamlit as st

from similaridade import carregar, recomendar

st.set_page_config(page_title="Recomendador de Mangá", layout="wide")

if os.path.exists("estilo.css"):
    st.markdown(f"<style>{open('estilo.css', encoding='utf-8').read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def dados():
    return carregar()


df, model, emb = dados()

st.title("📚 Recomendador de Mangá")
busca = st.text_input("Digite um título, trecho de sinopse ou descrição (português ou inglês)")

if busca.strip():
    rotulo, res = recomendar(busca.strip(), df, model, emb)
    st.subheader(rotulo)
    for _, r in res.iterrows():
        col1, col2 = st.columns([1, 4])
        if str(r.get("image_url", "")).startswith("http"):
            col1.image(r["image_url"])
        col2.markdown(f"### {r['title']}")
        col2.write(f"Similaridade: {r['similaridade']} · Nota: {r.get('score', '')}")
        col2.caption(r.get("genres", ""))
        col2.write(str(r.get("synopsis", ""))[:400])
