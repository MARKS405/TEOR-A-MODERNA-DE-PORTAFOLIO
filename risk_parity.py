"""
risk_parity.py
Implementación sencilla de un portafolio de paridad de riesgo (risk parity).
"""

import numpy as np


def calcular_pesos_risk_parity(covarianza, max_iter=1000, tol=1e-6, paso=0.01):
    """
    Heurística simple para encontrar pesos de paridad de riesgo.
    Ajusta pesos iterativamente para igualar contribuciones de riesgo.
    """
    cov = np.asarray(covarianza)
    n = cov.shape[0]

    w = np.ones(n) / n  # empezar con equiponderado

    for _ in range(max_iter):
        sigma_port, contribuciones = contribuciones_riesgo(cov, w)
        objetivo = sigma_port / n  # contribución deseada igual para todos

        diferencia = contribuciones - objetivo

        if np.max(np.abs(diferencia)) < tol:
            break

        # Paso simple de gradiente
        w = w - paso * diferencia
        w = np.clip(w, 0, None)  # sin short
        if w.sum() == 0:
            w = np.ones(n) / n
        else:
            w = w / w.sum()

    return w


def contribuciones_riesgo(covarianza, pesos):
    """
    Calcula la volatilidad del portafolio y las contribuciones de riesgo
    aproximadas de cada activo.
    """
    cov = np.asarray(covarianza)
    w = np.asarray(pesos).reshape(-1)

    var_port = float(w.T @ cov @ w)
    sigma_port = np.sqrt(var_port) if var_port > 0 else 0.0

    if sigma_port == 0:
        contribuciones = np.zeros_like(w)
    else:
        mrc = (cov @ w) / sigma_port  # marginal risk contribution
        contribuciones = w * mrc      # risk contribution

    return sigma_port, contribuciones
