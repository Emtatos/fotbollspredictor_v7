# streamlit_app_v7.py
# v7 – startpunkt för ny version (frikopplad från v6 live)

import os
import streamlit as st

st.set_page_config(page_title="Fotboll v7", layout="wide")

st.title("⚽ Fotboll v7 — ny gren")
st.write("""
Detta är en tom start-app för v7. Den är frikopplad från v6 och kan deployas separat.
Byt ut innehållet här med din v7-kod när du är redo.
""")

# Miljö/nycklar (säkert): bara läs env om det finns
openai_key = os.getenv("OPENAI_API_KEY")
st.sidebar.header("Status")
st.sidebar.write("OPENAI:", "OK" if openai_key else "—")
st.sidebar.write("Miljö:", os.getenv("RENDER", "lokal/GitHub preview"))

st.success("Appen kör! Du kan nu utveckla v7 i detta repo utan att röra v6.")
