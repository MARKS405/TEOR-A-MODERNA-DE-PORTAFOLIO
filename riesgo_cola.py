"""
riesgo_cola.py
Cálculo de VaR y CVaR históricos para una serie de retornos.
"""

import numpy as np


def var_cvar_historico(retornos_portafolio, alpha=0.95):
    """
    Calcula VaR y CVaR históricos para una serie de retornos del portafolio.

    Parámetros
    ----------
    retornos_portafolio : array-like
        Serie de retornos (por ejemplo diarios).
    alpha : float
        Nivel de confianza (por ejemplo 0.95 para 95%).

    Retorna
    -------
    var : float
        Valor en Riesgo (VaR).
    cvar : float
        Valor en Riesgo Condicional (CVaR o Expected Shortfall).
    """
    r = np.sort(np.asarray(retornos_portafolio))
    n = len(r)
    if n == 0:
        return np.nan, np.nan

    indice = int((1 - alpha) * n)
    indice = max(min(indice, n - 1), 0)

    var = -r[indice]
    if indice > 0:
        cvar = -r[:indice].mean()
    else:
        cvar = np.nan

    return var, cvar
