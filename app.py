def cartao(r, similaridade=True):
    col1, col2 = st.columns([1, 4])
    img_url = str(r.get("image_url", ""))
    if img_url.startswith("http"):
        if r.get("mais_18", False):
            col1.markdown(
                f"""
                <label class="capa-nsfw" title="Clique para revelar">
                    <input type="checkbox">
                    <img src="{img_url}" alt="{r.get('title', '')}">
                    <span class="badge-18">+18</span>
                </label>
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
