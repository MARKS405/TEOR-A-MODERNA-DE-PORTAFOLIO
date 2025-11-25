import streamlit as st
import numpy as np
import pandas as pd
from helpers import returns_from_prices
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def shrinkage_cov(cov, shrink=0.1):
    # Shrink covariance towards a diagonal matrix (simple)
    diag = np.diag(np.diag(cov))
    return (1 - shrink) * cov + shrink * diag

def risk_parity_weights(cov, tol=1e-8, max_iter=1000):
    # Iterative algorithm to compute risk parity weights (by solving x_i * (Σ x)_i = constant)
    n = cov.shape[0]
    x = np.repeat(1/n, n)
    for _ in range(max_iter):
        portfolio_var = cov.dot(x)
        risk_contrib = x * portfolio_var
        target = np.sum(risk_contrib) / n
        # update
        x_new = x * (target / risk_contrib)
        x_new = x_new / np.sum(x_new)
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x

def black_litterman_simple(prior_weights, cov, tau=0.05, P=None, Q=None, omega=None):
    # Very simplified Black-Litterman: prior (equilibrium) weights -> implied returns
    # using pi = delta * Sigma * w
    delta = 2.5  # risk aversion parameter (default)
    pi = delta * cov.dot(prior_weights)
    # if no views provided, return equilibrium portfolio (normalized)
    if P is None:
        return prior_weights
    # otherwise perform BL update (simplified)
    # omega: uncertainty of views
    Sigma_inv = np.linalg.inv(tau*cov)
    if omega is None:
        omega = np.eye(Q.shape[0]) * 0.1
    middle = np.linalg.inv(Sigma_inv + P.T @ np.linalg.inv(omega) @ P)
    mu_bl = middle.dot(Sigma_inv.dot(pi) + P.T.dot(np.linalg.inv(omega)).dot(Q))
    # obtain new optimal weights via mean-variance (no shorts, normalized)
    from scipy.optimize import minimize
    n = cov.shape[0]
    x0 = np.repeat(1/n, n)
    bounds = [(0,1)]*n
    cons = ({'type':'eq', 'fun': lambda w: np.sum(w)-1},
            {'type':'eq', 'fun': lambda w: w.dot(mu_bl)-w.dot(mu_bl)})  # dummy to keep shape
    res = minimize(lambda w: w.T @ cov @ w - 2*w.dot(mu_bl)/delta, x0, bounds=bounds, constraints=({'type':'eq', 'fun': lambda w: np.sum(w)-1},))
    return res.x

def robust_view(prices: pd.DataFrame):
    st.header("Portafolios Robustos")
    rets = returns_from_prices(prices)
    means = rets.mean()
    cov = rets.cov().values

    st.subheader("Shrinkage de la matriz de covarianzas")
    shrink = st.slider("Factor de shrinkage", 0.0, 0.9, 0.1)
    cov_shrunk = shrinkage_cov(cov, shrink=shrink)
    st.write("Covarianza original (shape):", cov.shape)
    st.write("Covarianza shrunk (shape):", cov_shrunk.shape)

    st.subheader("Risk Parity")
    if st.button("Calcular pesos Risk Parity"):
        w_rp = risk_parity_weights(cov_shrunk)
        st.write(pd.Series(w_rp, index=prices.columns))

    st.subheader("Black-Litterman (simple)")
    if st.button("Usar Black-Litterman (equilibrium)"):
        # equilibrium prior: market cap neutral = inverse volatility
        prior = 1 / np.sqrt(np.diag(cov))
        prior = prior / np.sum(prior)
        w_bl = black_litterman_simple(prior, cov_shrunk, tau=0.05)
        st.write(pd.Series(w_bl, index=prices.columns))
