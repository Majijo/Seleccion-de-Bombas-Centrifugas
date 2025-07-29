# - - - - - - - - - SELECCIÓN BOMBAS - - - - - - - - -
import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO

st.set_page_config(page_title="Selección de Bombas", layout="wide")

st.title("💧 Selección de Bomba Centrífuga")

# --- Entrada de datos del sistema ---
st.header("Datos del sistema")

# --- Entradas necesarias para el modelo detallado ---
di = st.number_input("Diámetro interno de cañería (m)", min_value=0.001, value=0.027, step=0.01, format="%.3f")
H_geo = st.number_input("ΔZ (m)", value=-1.27, format="%0.2f")
delta_p = st.number_input("ΔP (Pa)", value=20500,format="%g")
L = st.number_input("Longitud de cañería (m)", value=4.0,format="%g")
sum_K = st.number_input("Suma de coeficientes K locales", value=2.2,format="%g")
rugosidad = st.number_input("Rugosidad absoluta (m)", value=0.0003,format="%g")
deltaP_perm = st.number_input("Pérdida permanente del caudalímetro (Pa)",value=0,format="%g")
rho = st.number_input("Densidad del fluido (kg/m³)", value=900.0,format="%g")
mu = st.number_input("Viscosidad dinámica (Pa·s)", value=0.27, format="%0.4f")

# --- Curvas de bomba precargadas ---
curvas_precargadas = {
    "Wilo-VeroLine-IPL 32/110 1450 rpm": pd.DataFrame({"Q": [0, 1, 2, 3, 4, 5, 6], "H": [4.2, 4.1, 3.9, 3.5, 3, 2.3, 1.4]}),
    "Wilo-VeroLine-IPL 32/160 1450 rpm": pd.DataFrame({"Q": [0, 1, 2, 3, 4, 5, 6], "H": [7.3, 6.9, 6.4, 5.7, 4.8,3.6,2.2]}),
    "Wilo-VeroLine-IPL 40/130 1450 rpm": pd.DataFrame({"Q": [0, 4, 6, 8, 12, 16], "H": [5.2, 5.4, 5.3, 5.1, 4.5, 3.2]}),
    "Wilo-VeroLine-IPL 40/160 1450 rpm": pd.DataFrame({"Q": [0, 4, 6, 8, 12, 16, 20], "H": [7, 7.1, 7, 6.8, 6.1, 5, 3.5]}),
    "Wilo-VeroLine-IPL 50/160 1450 rpm": pd.DataFrame({"Q": [0, 4, 8, 12, 16, 20, 24, 28, 32], "H": [ 7.2, 7.3, 7.3, 7.1, 6.8, 6.3, 5.7, 4.9, 4]}),
    "Wilo-VeroLine-IPL 100/135 1450 rpm": pd.DataFrame({"Q": [0, 20, 40, 60, 80, 100, 120], "H": [5, 4.9, 4.65, 4.2, 3.5, 2.6, 1.6]}),
    "Wilo-VeroLine-IPL 100/175 1450 rpm": pd.DataFrame({"Q": [0, 20, 40, 60, 80, 100, 120, 140, 160, 180], "H": [9.1, 9.3, 9.2, 9, 8.6, 8, 7.2, 6.2, 5, 3.8]}),
    "Wilo-VeroLine-IPL 40/150 2900 rpm": pd.DataFrame({"Q": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40], "H": [27, 27.7, 28.2, 28, 27, 26.5, 25.5, 23.6, 22, 18.5, 16.5]})
}

st.header("Seleccionar o cargar curva de bomba", divider='blue')
opcion_curva = st.selectbox("Seleccioná una curva de bomba o subí un archivo CSV", ["Cargar archivo CSV"] + list(curvas_precargadas.keys()))

datos_bomba = None
if opcion_curva == "Cargar archivo CSV":
    archivo = st.file_uploader("Subí un archivo CSV con las columnas Q (caudal) y H (altura)", type="csv")
    if archivo is not None:
        try:
            datos_bomba = pd.read_csv(archivo)
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
else:
    datos_bomba = curvas_precargadas[opcion_curva]
    st.write(f"📈 Curva seleccionada: {opcion_curva}")
    st.write(datos_bomba)

# --- Procesamiento y gráfico ---
if datos_bomba is not None:
    H_sistema = []
    Q_bomba = datos_bomba["Q"].values
    H_bomba = datos_bomba["H"].values

    Q = np.linspace(min(Q_bomba)-0.2, max(Q_bomba)+0.2, 300)
    Q_m3s = Q / 3600 
    v = 4 * Q_m3s / (np.pi * di**2)
    Re = rho * v * di / mu
    e_rel = rugosidad / di

    for i in range(len(Re)):
        def sistec(vars):
            f, H = vars
            eq1 = -1 / np.sqrt(f) - 2 * np.log10((rugosidad / di) / 3.7 + 2.51 / (Re[i] * np.sqrt(f)))
            eq2 = H_geo + (f * L / di + sum_K) * (v[i]**2) / (2 * 9.81) + deltaP_perm / (rho * 9.81) + delta_p / (rho * 9.81) - H
            return [eq1, eq2]

        solution = fsolve(sistec, [0.01, 1.0])
        H_sistema.append(float(solution[1]))

    def error(q):
        q_m3s = q / 3600
        v_q = 4 * q_m3s / (np.pi * di**2)
        Re_q = rho * v_q * di / mu

        def sistec2(vars):
            f, H = vars
            eq1 = -1 / np.sqrt(f) - 2 * np.log10((rugosidad / di) / 3.7 + 2.51 / (Re_q * np.sqrt(f)))
            eq2 = H_geo + (f * L / di + sum_K) * (v_q**2) / (2 * 9.81) + deltaP_perm / (rho * 9.81) + delta_p / (rho * 9.81) - H
            return [eq1, eq2]
        
        solution2 = fsolve(sistec2, [0.01, 1.0])
        h_q = float(solution2[1])
        h_interp = np.interp(q, Q_bomba, H_bomba)

        return h_q - h_interp

    q_min, q_max = max(Q[0], min(Q_bomba)), min(Q[-1], max(Q_bomba))

    try:
        if error(q_min) * error(q_max) < 0:
            q_operacion = brentq(error, q_min, q_max)
            h_operacion = np.interp(q_operacion, Q, H_sistema)
            interseccion = True
        else:
            q_operacion, h_operacion = None, None
            interseccion = False
    except:
        q_operacion, h_operacion = None, None
        interseccion = False

    # - - - PARTE visual - - -
    col1, col2 = st.columns([1.2, 1.8])

    with col2:
        fig, ax = plt.subplots()
        ax.plot(Q, H_sistema, label="Curva del sistema", color='blue')
        ax.plot(Q_bomba, H_bomba, label="Curva de bomba", color='green', marker='o')

        # - - - PUNTO DE INTERSECCIÓN - - -
        if interseccion and q_operacion is not None and h_operacion is not None:
            ax.plot(q_operacion, h_operacion, 'ro', markersize=8, label=f'Punto de operación\nQ={q_operacion:.2f} m³/h\nH={h_operacion:.2f} m')

        ax.set_xlabel("Caudal (m³/h)")
        ax.set_ylabel("Altura (m)")
        ax.set_title("Curvas del Sistema y de la Bomba")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("### Exportar informe")
        buffer = BytesIO()
        with PdfPages(buffer) as pdf:
            pdf.savefig(fig)
            plt.figure(figsize=(8.5, 11))
            plt.axis('off')
            if q_operacion is not None and h_operacion is not None:
                texto = f"RESULTADOS\n\nCaudal de operación: {q_operacion:.2f} m³/h\nAltura total: {h_operacion:.2f} m"
            else:
                texto = "RESULTADOS\n\nNo se encontró un punto de operación válido entre la bomba y el sistema."
            plt.text(0.1, 0.9, texto, fontsize=12)
            pdf.savefig()
        pdf_data = buffer.getvalue()
        st.download_button("📥 Descargar informe PDF", data=pdf_data, file_name="informe_bomba.pdf", mime="application/pdf")

    with col1:
        st.subheader("📘 Ecuaciones utilizadas")

        st.markdown("*1. Ecuación de continuidad:*")
        st.latex(r"v = \frac{4Q}{\pi d^2}")

        st.markdown("*2. Número de Reynolds:*")
        st.latex(r"Re = \frac{\rho v d}{\mu}")

        st.markdown("*3. Coeficiente de fricción (Colebrook-White, implícita):*")
        st.latex(r"\frac{1}{\sqrt{f}} = -2 \log_{10}\left(\frac{\varepsilon / d}{3.7} + \frac{2.51}{Re \sqrt{f}}\right)")

        st.markdown("*4. Pérdidas por fricción y accesorios:*")
        st.latex(r"H_f = \left( f \cdot \frac{L}{d} + \sum K \right) \cdot \frac{v^2}{2g}")

        st.markdown("*5. Balance del sistema:*")
        st.latex(r"H_{\text{sistema}} = H_{geo} + H_f + \frac{\Delta P_{perm}}{\rho g} + \frac{\Delta P}{\rho g}")

else:
    st.info("📂 Esperando archivo CSV o selección de curva precargada...")
