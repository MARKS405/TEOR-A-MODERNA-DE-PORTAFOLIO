import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from utils.helpers import returns_from_prices

def portfolio_variance(w, cov):
    return w.T @ cov @ w

def global_min_variance(cov):
    n = cov.shape[0]
    x0 = np.repeat(1/n, n)
    bounds = [(0,1)]*n
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w)-1})
    res = minimize(lambda w: portfolio_variance(w, cov), x0, bounds=bounds, constraints=cons)
    return res.x

def efficient_frontier(means, cov, points=50):
    n = len(means)
    mus = np.linspace(means.min(), means.max(), points)
    frontier = []
    for mu in mus:
        # minimize variance s.t. expected return = mu and weights sum to 1 and 0<=w<=1
        x0 = np.repeat(1/n, n)
        bounds = [(0,1)]*n
        cons = (
            {'type':'eq', 'fun': lambda w: np.sum(w)-1},
            {'type':'eq', 'fun': lambda w, mu=mu: w.dot(means)-mu}
        )
        res = minimize(lambda w: w.T @ cov @ w, x0, bounds=bounds, constraints=cons)
        if res.success:
            frontier.append((res.fun, res.x, mu))
    return frontier

def sharpe_portfolio(means, cov, rf=0.0):
    n = len(means)
    x0 = np.repeat(1/n, n)
    bounds = [(0,1)]*n
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w)-1})
    def neg_sharpe(w):
        port_ret = w.dot(means)
        port_std = np.sqrt(w.T @ cov @ w)
        return - (port_ret - rf) / (port_std + 1e-10)
    res = minimize(neg_sharpe, x0, bounds=bounds, constraints=cons)
    return res.x

def markowitz_view(prices: pd.DataFrame):
    st.header("Modelo Media-Varianza (Markowitz)")
    st.write("Calculando retornos y matriz de covarianzas.")
    rets = returns_from_prices(prices)
    means = rets.mean()
    cov = rets.cov().values

    st.subheader("Inputs")
    st.write("Activos:", list(prices.columns))
    st.write("Retorno esperado anualizado (aprox):")
    st.write(means * 252)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Calcular portafolio de mínima varianza global (con shorting restringido)"):
            w_gmv = global_min_variance(cov)
            st.write("Pesos GMV:")
            st.write(pd.Series(w_gmv, index=prices.columns))
    with col2:
        rf = st.number_input("Tasa libre de riesgo (rf)", value=0.0, step=0.001, format="%.4f")
        if st.button("Calcular portafolio de Sharpe (máx Sharpe)"):
            w_sh = sharpe_portfolio(means.values, cov, rf)
            st.write("Pesos portafolio Sharpe:")
            st.write(pd.Series(w_sh, index=prices.columns))

    st.subheader("Frontera eficiente (restricción 0<=w<=1)")
    points = st.slider("Puntos en la frontera", 10, 200, 60)
    frontier = efficient_frontier(means.values, cov, points=points)
    if len(frontier) > 0:
        variances = [f[0] for f in frontier]
        returns = [f[2] for f in frontier]
        fig, ax = plt.subplots()
        ax.plot(np.sqrt(variances), returns, '-o')
        ax.set_xlabel("Volatilidad (std)")
        ax.set_ylabel("Retorno esperado")
        ax.set_title("Frontera eficiente")
        st.pyplot(fig)
        # show a sample allocation
        idx = st.slider("Seleccionar punto en la frontera (índice)", 0, max(0, len(frontier)-1), 0)
        st.write("Pesos en la selección:")
        st.write(pd.Series(frontier[idx][1], index=prices.columns))
    else:
        st.info("No se pudo calcular la frontera con los parámetros actuales.")
