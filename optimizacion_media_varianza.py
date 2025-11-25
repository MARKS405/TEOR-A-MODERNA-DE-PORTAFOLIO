"""
optimizacion_media_varianza.py
Implementa el modelo clásico de Media–Varianza de Markowitz:
- Métricas del portafolio
- Portafolio de mínima varianza
- Portafolio para retorno objetivo
- Portafolio de máxima razón de Sharpe
- Frontera eficiente
"""

import numpy as np
import cvxpy as cp


def calcular_metricas_portafolio(
    pesos,
    media,
    covarianza,
    tasa_libre_riesgo=0.0,
    retornos_hist=None,
    h_anual=0.0,
    factor_anual=252,
):
    """
    Calcula retorno esperado, volatilidad, Sharpe y Sortino de un portafolio.

    Notas sobre Sortino:
    - h_anual: retorno mínimo aceptable anual (cut-off).
    - Para los retornos diarios se aproxima h_diario = h_anual / factor_anual.
    """
    w = np.asarray(pesos).reshape(-1)
    mu = np.asarray(media).reshape(-1)
    cov = np.asarray(covarianza)

    # Retorno esperado y volatilidad (anuales)
    retorno_port = float(w @ mu)
    var_port = float(w.T @ cov @ w)
    vol_port = float(np.sqrt(var_port)) if var_port > 0 else 0.0

    # Sharpe ratio
    sharpe = (retorno_port - tasa_libre_riesgo) / vol_port if vol_port > 0 else np.nan

    # Sortino ratio (downside risk)
    sortino = np.nan
    if retornos_hist is not None:
        retornos_hist = np.asarray(retornos_hist)
        retornos_port_diarios = retornos_hist @ w

        h_diario = h_anual / factor_anual
        downside = np.minimum(retornos_port_diarios - h_diario, 0.0)
        semivar = np.mean(downside ** 2)
        semidesv = np.sqrt(semivar)

        # anualizar semidesviación (misma lógica que la volatilidad)
        semidesv_anual = semidesv * np.sqrt(factor_anual)

        sortino = (
            (retorno_port - h_anual) / semidesv_anual
            if semidesv_anual > 0
            else np.nan
        )

    return retorno_port, vol_port, sharpe, sortino


def optimizar_minima_varianza(covarianza, permitir_short=False):
    """
    Encuentra el portafolio de mínima varianza global (GMV).
    """
    cov = np.asarray(covarianza)
    n = cov.shape[0]

    w = cp.Variable(n)
    objetivo = cp.Minimize(cp.quad_form(w, cov))

    restricciones = [cp.sum(w) == 1]
    if not permitir_short:
        restricciones.append(w >= 0)

    problema = cp.Problem(objetivo, restricciones)
    problema.solve()

    return w.value


def optimizar_media_varianza_retorno_objetivo(
    media,
    covarianza,
    retorno_objetivo,
    permitir_short=False,
):
    """
    Minimiza la varianza del portafolio sujeto a alcanzar al menos
    un retorno esperado objetivo.
    """
    mu = np.asarray(media).reshape(-1)
    cov = np.asarray(covarianza)
    n = cov.shape[0]

    w = cp.Variable(n)
    objetivo = cp.Minimize(cp.quad_form(w, cov))
    restricciones = [
        cp.sum(w) == 1,
        mu @ w >= retorno_objetivo,
    ]
    if not permitir_short:
        restricciones.append(w >= 0)

    problema = cp.Problem(objetivo, restricciones)
    problema.solve()

    return w.value


def optimizar_max_sharpe(media, covarianza, tasa_libre_riesgo=0.0, permitir_short=False):
    """
    Encuentra el portafolio de máxima razón de Sharpe:
    - Se impone que el retorno en exceso sobre rf sea 1.
    - Se minimiza la varianza.
    - Luego se normalizan los pesos para que sumen 1.
    """
    mu = np.asarray(media).reshape(-1)
    cov = np.asarray(covarianza)
    n = cov.shape[0]

    w = cp.Variable(n)
    exceso = mu - tasa_libre_riesgo

    objetivo = cp.Minimize(cp.quad_form(w, cov))
    restricciones = [exceso @ w == 1]
    if not permitir_short:
        restricciones.append(w >= 0)

    problema = cp.Problem(objetivo, restricciones)
    problema.solve()

    pesos_crudos = w.value
    pesos_norm = pesos_crudos / pesos_crudos.sum()
    return pesos_norm


def construir_frontera_eficiente(
    media,
    covarianza,
    n_puntos=30,
    permitir_short=False,
):
    """
    Construye un conjunto de portafolios sobre la frontera eficiente
    variando el retorno objetivo entre el mínimo y el máximo de los activos.
    """
    mu = np.asarray(media).reshape(-1)
    cov = np.asarray(covarianza)

    mu_min, mu_max = mu.min(), mu.max()
    objetivos = np.linspace(mu_min, mu_max, n_puntos)

    vols = []
    rets = []
    pesos_lista = []

    for objetivo_ret in objetivos:
        w = optimizar_media_varianza_retorno_objetivo(
            mu,
            cov,
            retorno_objetivo=objetivo_ret,
            permitir_short=permitir_short,
        )

        if w is None:
            continue

        ret_p, vol_p, _, _ = calcular_metricas_portafolio(w, mu, cov)
        vols.append(vol_p)
        rets.append(ret_p)
        pesos_lista.append(w)

    return np.array(vols), np.array(rets), np.array(pesos_lista)
