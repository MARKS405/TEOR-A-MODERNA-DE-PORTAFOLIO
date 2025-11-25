"""
app.py
Aplicación principal en Streamlit para mostrar distintas formas
de optimizar un portafolio de inversión:

- Modelo Media–Varianza (Markowitz)
- Downside risk (Sortino)
- VaR / CVaR históricos
- Portafolio de paridad de riesgo (risk parity)

Texto de la interfaz completamente en español.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from datos import (
    descargar_precios,
    calcular_retornos,
    calcular_estadisticas_basicas,
)
from optimizacion_media_varianza import (
    calcular_metricas_portafolio,
    optimizar_minima_varianza,
    optimizar_media_varianza_retorno_objetivo,
    optimizar_max_sharpe,
    construir_frontera_eficiente,
)
from downside_risk import optimizar_sortino
from riesgo_cola import var_cvar_historico
from risk_parity import calcular_pesos_risk_parity, contribuciones_riesgo


FACTOR_ANUAL = 252  # supuesto: datos diarios


def main():
    st.set_page_config(
        page_title="Optimización de portafolios – Teoría Moderna",
        layout="wide",
    )

    st.title("Optimización de Portafolios en Streamlit")
    st.caption(
        "Demostración interactiva de Teoría Moderna de Portafolio: "
        "Media–Varianza, Downside Risk, VaR/CVaR y Paridad de Riesgo."
    )

    # ==============================
    # SIDEBAR – PARÁMETROS
    # ==============================
    st.sidebar.header("Configuración de datos")

    tickers_str = st.sidebar.text_input(
        "Tickers (separados por coma)",
        value="AAPL,MSFT,GOOGL,AMZN",
        help="Ejemplo: AAPL,MSFT,GOOGL,AMZN",
    )

    tickers = [
        t.strip().upper()
        for t in tickers_str.split(",")
        if t.strip() != ""
    ]

    fecha_inicio = st.sidebar.date_input(
        "Fecha de inicio",
        value=pd.to_datetime("2018-01-01"),
    )
    fecha_fin = st.sidebar.date_input(
        "Fecha de fin",
        value=pd.to_datetime("2024-12-31"),
    )

    metodo_retornos = st.sidebar.selectbox(
        "Tipo de retorno",
        options=["log", "simple"],
        format_func=lambda x: "Logarítmico" if x == "log" else "Simple",
    )

    tasa_libre_riesgo = st.sidebar.number_input(
        "Tasa libre de riesgo anual (decimal)",
        value=0.02,
        step=0.005,
        help="Ejemplo: 0.02 = 2% anual",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Downside Risk (Sortino)")

    h_anual = st.sidebar.number_input(
        "Retorno mínimo aceptable (anual, h)",
        value=0.0,
        step=0.01,
        help="Por ejemplo 0.03 para 3% anual como mínimo aceptable.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("VaR / CVaR")

    alpha = st.sidebar.slider(
        "Nivel de confianza (α)",
        min_value=0.90,
        max_value=0.99,
        value=0.95,
        step=0.01,
    )

    st.sidebar.markdown("---")
    permitir_short = st.sidebar.checkbox(
        "Permitir posiciones cortas (short selling)?",
        value=False,
    )

    # ==============================
    # CARGA Y PREPARACIÓN DE DATOS
    # ==============================
    if len(tickers) == 0:
        st.error("Por favor, escribe al menos un ticker en la barra lateral.")
        return

    precios = descargar_precios(tickers, fecha_inicio, fecha_fin)
    retornos = calcular_retornos(precios, metodo=metodo_retornos)
    media, volatilidad, covarianza = calcular_estadisticas_basicas(
        retornos, factor_anual=FACTOR_ANUAL
    )

    media_vec = media.values
    cov_mat = covarianza.values
    retornos_np = retornos.values

    st.subheader("Resumen de la muestra")
    st.write(f"**Activos:** {', '.join(tickers)}")
    st.write(f"**Período:** {fecha_inicio} a {fecha_fin}")
    st.write(f"**Número de observaciones:** {len(retornos)}")

    # ==============================
    # PESTAÑAS PRINCIPALES
    # ==============================
    (
        tab_datos,
        tab_mv,
        tab_downside,
        tab_var_cvar,
        tab_risk_parity,
    ) = st.tabs(
        [
            "Datos y estadísticas",
            "Media–Varianza (Markowitz)",
            "Downside Risk (Sortino)",
            "VaR / CVaR históricos",
            "Portafolio robusto (Risk Parity)",
        ]
    )

    # --------------------------------------------------
    # TAB 1 – DATOS Y ESTADÍSTICAS
    # --------------------------------------------------
    with tab_datos:
        st.header("Datos y estadísticas básicas")

        st.write("### Precios ajustados (últimas filas)")
        st.dataframe(precios.tail())

        st.write("### Retornos (últimas filas)")
        st.dataframe(retornos.tail())

        st.write("### Media anual de retornos")
        st.dataframe(media.to_frame("Media anual"))

        st.write("### Volatilidad anual")
        st.dataframe(volatilidad.to_frame("Volatilidad anual"))

        st.write("### Matriz de covarianza anual")
        st.dataframe(covarianza)

        st.markdown(
            """
            Estas estadísticas corresponden al enfoque clásico de la Teoría Moderna
            de Portafolio: asumimos que el riesgo se mide por la varianza /
            desviación estándar de los retornos.
            """
        )

    # --------------------------------------------------
    # TAB 2 – MEDIA–VARIANZA (MARKOWITZ)
    # --------------------------------------------------
    with tab_mv:
        st.header("Modelo Media–Varianza de Markowitz")

        # Portafolio de mínima varianza global (GMV)
        pesos_gmv = optimizar_minima_varianza(
            cov_mat,
            permitir_short=permitir_short,
        )

        ret_gmv, vol_gmv, sharpe_gmv, sortino_gmv = calcular_metricas_portafolio(
            pesos_gmv,
            media_vec,
            cov_mat,
            tasa_libre_riesgo=tasa_libre_riesgo,
            retornos_hist=retornos_np,
            h_anual=h_anual,
            factor_anual=FACTOR_ANUAL,
        )

        # Portafolio de máxima razón de Sharpe
        pesos_sharpe = optimizar_max_sharpe(
            media_vec,
            cov_mat,
            tasa_libre_riesgo=tasa_libre_riesgo,
            permitir_short=permitir_short,
        )
        ret_sh, vol_sh, sharpe_sh, sortino_sh = calcular_metricas_portafolio(
            pesos_sharpe,
            media_vec,
            cov_mat,
            tasa_libre_riesgo=tasa_libre_riesgo,
            retornos_hist=retornos_np,
            h_anual=h_anual,
            factor_anual=FACTOR_ANUAL,
        )

        # Portafolio equiponderado
        pesos_eq = np.ones(len(media_vec)) / len(media_vec)
        ret_eq, vol_eq, sharpe_eq, sortino_eq = calcular_metricas_portafolio(
            pesos_eq,
            media_vec,
            cov_mat,
            tasa_libre_riesgo=tasa_libre_riesgo,
            retornos_hist=retornos_np,
            h_anual=h_anual,
            factor_anual=FACTOR_ANUAL,
        )

        # Frontera eficiente
        vols_front, rets_front, pesos_front = construir_frontera_eficiente(
            media_vec,
            cov_mat,
            n_puntos=40,
            permitir_short=permitir_short,
        )

        col_graf, col_tabla = st.columns([2, 1])

        with col_graf:
            st.write("### Frontera eficiente")
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(
                vols_front,
                rets_front,
                s=15,
                label="Frontera eficiente",
            )
            ax.scatter(vol_eq, ret_eq, marker="o", s=70, label="Equiponderado")
            ax.scatter(vol_gmv, ret_gmv, marker="*", s=120, label="GMV")
            ax.scatter(vol_sh, ret_sh, marker="^", s=120, label="Máx. Sharpe")

            ax.set_xlabel("Volatilidad anual")
            ax.set_ylabel("Retorno esperado anual")
            ax.set_title("Frontera eficiente – Modelo Media–Varianza")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

        with col_tabla:
            st.write("### Resumen de portafolios clave")
            df_res = pd.DataFrame(
                {
                    "Portafolio": ["Equiponderado", "GMV", "Máx. Sharpe"],
                    "Retorno anual": [ret_eq, ret_gmv, ret_sh],
                    "Volatilidad anual": [vol_eq, vol_gmv, vol_sh],
                    "Sharpe": [sharpe_eq, sharpe_gmv, sharpe_sh],
                    "Sortino": [sortino_eq, sortino_gmv, sortino_sh],
                }
            )
            st.dataframe(df_res)

            st.write("### Pesos – GMV")
            st.dataframe(
                pd.Series(
                    pesos_gmv,
                    index=media.index,
                    name="GMV",
                )
            )

            st.write("### Pesos – Máx. Sharpe")
            st.dataframe(
                pd.Series(
                    pesos_sharpe,
                    index=media.index,
                    name="Max Sharpe",
                )
            )

        st.markdown(
            """
            En esta pestaña mostramos el enfoque clásico de Markowitz: el riesgo
            se mide con la varianza y buscamos portafolios sobre la frontera
            eficiente, así como el portafolio de mínima varianza y el de máxima
            razón de Sharpe.
            """
        )

    # --------------------------------------------------
    # TAB 3 – DOWNSIDE RISK (SORTINO)
    # --------------------------------------------------
    with tab_downside:
        st.header("Optimización por Downside Risk (Sortino)")

        iteraciones = st.number_input(
            "Número de iteraciones de búsqueda aleatoria",
            min_value=100,
            max_value=50000,
            value=3000,
            step=500,
        )

        if st.button("Ejecutar optimización por Sortino"):
            pesos_sortino, mejor_sortino = optimizar_sortino(
                media_vec,
                cov_mat,
                retornos_np,
                h_anual=h_anual,
                tasa_libre_riesgo=tasa_libre_riesgo,
                iteraciones=int(iteraciones),
                permitir_short=permitir_short,
                factor_anual=FACTOR_ANUAL,
            )

            ret_s, vol_s, sharpe_s, sortino_s = calcular_metricas_portafolio(
                pesos_sortino,
                media_vec,
                cov_mat,
                tasa_libre_riesgo=tasa_libre_riesgo,
                retornos_hist=retornos_np,
                h_anual=h_anual,
                factor_anual=FACTOR_ANUAL,
            )

            st.subheader("Resultado – Portafolio de máximo Sortino")
            st.write(f"**Sortino:** {sortino_s:.3f}")
            st.write(f"**Sharpe:** {sharpe_s:.3f}")
            st.write(f"**Retorno anual:** {ret_s:.2%}")
            st.write(f"**Volatilidad anual:** {vol_s:.2%}")

            st.write("### Pesos – Máx. Sortino")
            st.dataframe(
                pd.Series(
                    pesos_sortino,
                    index=media.index,
                    name="Pesos Sortino",
                )
            )

        st.markdown(
            """
            Aquí el riesgo ya no se mide por la varianza total, sino solo por
            los retornos por debajo de un umbral (h). El ratio de Sortino
            penaliza únicamente la parte negativa, reflejando un enfoque de
            **downside risk**.
            """
        )

    # --------------------------------------------------
    # TAB 4 – VaR / CVaR HISTÓRICOS
    # --------------------------------------------------
    with tab_var_cvar:
        st.header("VaR y CVaR históricos de los portafolios")

        # Recalculamos portafolios básicos
        pesos_gmv = optimizar_minima_varianza(
            cov_mat,
            permitir_short=permitir_short,
        )
        pesos_sharpe = optimizar_max_sharpe(
            media_vec,
            cov_mat,
            tasa_libre_riesgo=tasa_libre_riesgo,
            permitir_short=permitir_short,
        )
        pesos_eq = np.ones(len(media_vec)) / len(media_vec)

        nombres = ["Equiponderado", "GMV", "Máx. Sharpe"]
        lista_pesos = [pesos_eq, pesos_gmv, pesos_sharpe]

        registros = []
        for nombre, w in zip(nombres, lista_pesos):
            retornos_port = retornos_np @ w  # retornos diarios del portafolio
            var, cvar = var_cvar_historico(retornos_port, alpha=alpha)
            registros.append(
                {
                    "Portafolio": nombre,
                    f"VaR {int(alpha * 100)}% (histórico)": var,
                    f"CVaR {int(alpha * 100)}% (histórico)": cvar,
                }
            )

        df_var_cvar = pd.DataFrame(registros)
        st.dataframe(df_var_cvar)

        st.markdown(
            """
            El VaR y el CVaR miran la **cola izquierda** de la distribución de 
            retornos, capturando pérdidas extremas. Son medidas muy utilizadas
            en gestión de riesgos y complementan la visión de la varianza.
            """
        )

    # --------------------------------------------------
    # TAB 5 – PARIDAD DE RIESGO (RISK PARITY)
    # --------------------------------------------------
    with tab_risk_parity:
        st.header("Portafolio robusto de Paridad de Riesgo (Risk Parity)")

        pesos_rp = calcular_pesos_risk_parity(cov_mat)
        ret_rp, vol_rp, sharpe_rp, sortino_rp = calcular_metricas_portafolio(
            pesos_rp,
            media_vec,
            cov_mat,
            tasa_libre_riesgo=tasa_libre_riesgo,
            retornos_hist=retornos_np,
            h_anual=h_anual,
            factor_anual=FACTOR_ANUAL,
        )

        st.write("### Pesos del portafolio de paridad de riesgo")
        st.dataframe(
            pd.Series(
                pesos_rp,
                index=media.index,
                name="Pesos Risk Parity",
            )
        )

        st.write("### Métricas del portafolio Risk Parity")
        st.write(f"**Retorno anual:** {ret_rp:.2%}")
        st.write(f"**Volatilidad anual:** {vol_rp:.2%}")
        st.write(f"**Sharpe:** {sharpe_rp:.3f}")
        st.write(f"**Sortino:** {sortino_rp:.3f}")

        sigma_port, contribuciones = contribuciones_riesgo(cov_mat, pesos_rp)

        st.write("### Contribuciones de riesgo aproximadas")
        st.dataframe(
            pd.Series(
                contribuciones,
                index=media.index,
                name="Contribución al riesgo",
            )
        )

        st.markdown(
            """
            En un portafolio de **paridad de riesgo** buscamos que cada activo
            aporte un porcentaje similar del riesgo total, en lugar de centrarnos
            en los pesos en valor. Esto se relaciona con enfoques robustos de
            construcción de portafolios discutidos en la literatura moderna.
            """
        )


if __name__ == "__main__":
    main()
