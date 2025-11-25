"""
downside_risk.py
Optimización heurística por Sortino (downside risk) usando búsqueda aleatoria.
"""

import numpy as np
from optimizacion_media_varianza import calcular_metricas_portafolio


def pesos_aleatorios(n_activos):
    """
    Genera pesos aleatorios que suman 1 (sin permitir short).
    """
    w = np.random.rand(n_activos)
    w = w / w.sum()
    return w


def optimizar_sortino(
    media,
    covarianza,
    retornos_hist,
    h_anual=0.0,
    tasa_libre_riesgo=0.0,
    iteraciones=5000,
    permitir_short=False,
    factor_anual=252,
):
    """
    Búsqueda aleatoria para maximizar el Sortino ratio.

    No es una optimización exacta, pero es muy útil de forma pedagógica
    para mostrar cómo cambia el portafolio cuando el criterio es downside risk.
    """
    mu = np.asarray(media).reshape(-1)
    cov = np.asarray(covarianza)
    rets_hist = np.asarray(retornos_hist)
    n = len(mu)

    mejor_pesos = None
    mejor_sortino = -np.inf

    for _ in range(int(iteraciones)):
        if permitir_short:
            # Pesos aleatorios con posible short (centrados en 0)
            w = np.random.randn(n)
            w = w / np.sum(np.abs(w))
        else:
            w = pesos_aleatorios(n)

        ret_p, vol_p, sharpe_p, sortino_p = calcular_metricas_portafolio(
            w,
            mu,
            cov,
            tasa_libre_riesgo=tasa_libre_riesgo,
            retornos_hist=rets_hist,
            h_anual=h_anual,
            factor_anual=factor_anual,
        )

        if sortino_p > mejor_sortino:
            mejor_sortino = sortino_p
            mejor_pesos = w

    return mejor_pesos, mejor_sortino
