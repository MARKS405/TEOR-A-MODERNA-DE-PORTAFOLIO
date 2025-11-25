"""
datos.py
Funciones para descargar precios, calcular retornos y estadísticas básicas.
Todo pensado para ser usado desde Streamlit.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st


@st.cache_data
def descargar_precios(tickers, fecha_inicio, fecha_fin):
    """
    Descarga precios ajustados de Yahoo Finance para una lista de tickers.

    Parámetros
    ----------
    tickers : list[str]
        Lista de símbolos (por ejemplo ["AAPL", "MSFT"]).

    Retorna
    -------
    DataFrame con precios ajustados.
    """
    data = yf.download(tickers, start=fecha_inicio, end=fecha_fin)["Adj Close"]

    # Si solo hay un ticker, yfinance devuelve una Serie
    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.dropna()
    return data


def calcular_retornos(precios, metodo="log"):
    """
    Calcula retornos a partir de precios.

    metodo = "log"    -> retornos logarítmicos
    metodo = "simple" -> retornos porcentuales simples
    """
    if metodo == "log":
        retornos = np.log(precios / precios.shift(1))
    else:
        retornos = precios.pct_change()

    retornos = retornos.dropna()
    return retornos


def calcular_estadisticas_basicas(retornos, factor_anual=252):
    """
    Calcula media, volatilidad y covarianza anualizadas.

    Parámetros
    ----------
    retornos : DataFrame
        Retornos diarios (o periódicos).
    factor_anual : int
        Número de periodos en un año (252 para datos diarios).

    Retorna
    -------
    media : Series
    volatilidad : Series
    covarianza : DataFrame
    """
    media = retornos.mean() * factor_anual
    volatilidad = retornos.std() * np.sqrt(factor_anual)
    covarianza = retornos.cov() * factor_anual
    return media, volatilidad, covarianza
