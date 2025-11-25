import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from helpers import returns_from_prices

def semivariance(returns, target=0.0):
    # returns: series or array of returns (not annualized)
    diff = np.minimum(returns - target, 0)
    return np.mean(diff**2)

def var_historical(returns, alpha=0.05):
    return np.percentile(returns, 100*alpha)

def cvar_historical(returns, alpha=0.05):
    var = var_historical(returns, alpha)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return var
    return tail.mean()

def optimize_cvar(rets: pd.DataFrame, alpha=0.05):
    n = rets.shape[1]
    init = np.repeat(1/n, n)
    bounds = [(0,1)]*n
    cons = ({'type':'eq', 'fun': lambda w: np.sum(w)-1})
    # minimize historical CVaR of portfolio returns
    def obj(w):
        port_rets = rets.values @ w
        return cvar_historical(port_rets, alpha)
    res = minimize(obj, init, bounds=bounds, constraints=cons, options={'maxiter':500})
    return res

def downside_view(prices: pd.DataFrame):
    st.header("Downside Risk — Semivarianza, VaR y CVaR")
    rets = returns_from_prices(prices)
    st.write("Muestra de retornos (primeras filas):")
    st.dataframe(rets.head())

    st.subheader("Medidas")
    alpha = st.slider("Nivel de confianza para VaR/CVaR (alpha)", 0.01, 0.20, 0.05)
    col1, col2 = st.columns(2)
    with col1:
        st.write("VaR (histórico) por activo:")
        var_table = rets.apply(lambda x: var_historical(x, alpha))
        st.write(var_table)
    with col2:
        st.write("CVaR (histórico) por activo:")
        cvar_table = rets.apply(lambda x: cvar_historical(x, alpha))
        st.write(cvar_table)

    st.subheader("Optimizar CVaR (restricción 0<=w<=1)")
    if st.button("Optimizar CVaR histórico"):
        res = optimize_cvar(rets, alpha)
        if res.success:
            st.success("Optimización completada")
            st.write("Pesos óptimos (min CVaR):")
            st.write(pd.Series(res.x, index=prices.columns))
            port_rets = rets.values @ res.x
            st.write("CVaR (portafolio):", cvar_historical(port_rets, alpha))
        else:
            st.error("La optimización falló.")
