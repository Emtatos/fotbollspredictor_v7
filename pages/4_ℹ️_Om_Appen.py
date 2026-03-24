"""
Sida: Om Appen — information om verktyget och teknisk bakgrund.

Visar huvudfunktioner, teknisk information och versionsinfo.
"""

import streamlit as st

from app_helpers import get_model_and_data
from schema import FEATURE_COLUMNS

# Ladda metadata
_, _, model_metadata, _, _ = get_model_and_data()

st.header("Om appen")

st.markdown("""
**Odds & Value-verktyget** hjälper dig analysera 1X2-odds, beräkna fair
probabilities, hitta value (edge) och jämföra mot folkets streckprocent.

Appen innehåller också en **modellbaserad prediktion** (XGBoost) som
komplement — se sidorna *Enskild Match* och *Flera Matcher*.
""")

st.divider()

st.subheader("Huvudfunktioner")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Odds & Value (huvudflöde)**
    - Fair probabilities från bookmaker-odds
    - Value-analys: edge & expected return
    - Streckjämförelse: under-/överstreckade utfall
    - Ranking av mest intressanta utfall & matcher
    """)

with col2:
    st.markdown("""
    **Modellprediktion (komplement)**
    - Enskild match eller hel omgång
    - Halvgarderingsförslag
    - Trust score & AI-analys (valfritt)
    """)

st.divider()

with st.expander("Teknisk information"):
    st.markdown("""
    - **Frontend:** Streamlit
    - **Modell:** XGBoost + CalibratedClassifierCV
    - **Data:** football-data.co.uk (engelska ligor)
    - **Testning:** pytest, CI på Python 3.9/3.10/3.11
    - **Deployment:** Render / Docker
    """)

    if model_metadata:
        feats = model_metadata.get("features", [])
        n_feats = len(feats)
    else:
        n_feats = len(FEATURE_COLUMNS)
    st.caption(f"Modellen använder {n_feats} features.")

st.divider()

if model_metadata:
    ver = model_metadata.get("model_version", "v8.0")
    st.caption(f"Version: **{ver}**")
else:
    st.caption("Version: **v8.0**")

st.caption("Utvecklad av **Emtatos**.")
