Proyecto Streamlit: Optimizador de Portafolios

Contenido:
- app.py : aplicación principal (Streamlit)
- modules/markowitz.py : implementación de Markowitz (frontera eficiente, GMV, Sharpe)
- modules/downside.py : medidas downside (semivarianza, VaR, CVaR) y optimización de CVaR
- modules/robust.py : shrinkage, risk parity y Black-Litterman (simple)
- utils/helpers.py : utilidades y generador de datos de ejemplo
- data/: carpeta para tus series de precios (opcional)
- requirements.txt : dependencias sugeridas

Documento base (subido por el usuario): /mnt/data/Teoria_moderna_de_portafolio_desarrollos_fundament.pdf

Instrucciones rápidas:
1. Instala dependencias (preferible en un entorno virtual):
   pip install -r requirements.txt
2. Ejecuta:
   streamlit run app.py
