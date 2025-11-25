import streamlit as st
import pandas as pd
import numpy as np
from markowitz import markowitz_view
from downside import downside_view
from robust import robust_view
from helpers import load_sample_data

st.set_page_config(page_title="Optimizador de Portafolios - TMP", layout="wide")
st.title("Optimizador de Portafolios — basado en la Teoría Moderna de Portafolio")

st.sidebar.markdown("## Recursos")
st.sidebar.markdown("- Documento base (subido):")
st.sidebar.markdown(f"- `/mnt/data/Teoria_moderna_de_portafolio_desarrollos_fundament.pdf`")
st.sidebar.markdown("---")

data_source = st.sidebar.selectbox("Cargar datos de precios:", ["Subir CSV", "Usar datos de ejemplo"])
prices = None

if data_source == "Subir CSV":
    uploaded = st.sidebar.file_uploader("Sube un CSV con columnas: Date, asset1, asset2, ...", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded, parse_dates=True, index_col=0)
        prices = df.sort_index()
else:
    st.sidebar.markdown("Usando datos sintéticos de ejemplo (simulados).")
    prices = load_sample_data(n_assets=6, n_obs=1000)
    st.sidebar.markdown("Datos de ejemplo cargados (simulados).")

if prices is None:
    st.warning("No hay datos cargados. Carga un CSV o selecciona datos de ejemplo.")
    st.stop()

st.sidebar.markdown("---")
menu = st.sidebar.selectbox(
    "Seleccione un módulo:",
    [
        "Modelo Media-Varianza (Markowitz)",
        "Downside Risk (Semivarianza, VaR, CVaR)",
        "Portafolios Robustos (Black-Litterman, Risk Parity, Shrinkage)"
    ]
)

if menu == "Modelo Media-Varianza (Markowitz)":
    markowitz_view(prices)
elif menu == "Downside Risk (Semivarianza, VaR, CVaR)":
    downside_view(prices)
elif menu == "Portafolios Robustos (Black-Litterman, Risk Parity, Shrinkage)":
    robust_view(prices)

st.sidebar.markdown("---")
st.sidebar.markdown("Proyecto generado a partir del documento 'Teoría moderna de portafolio' (archivo subido).")
