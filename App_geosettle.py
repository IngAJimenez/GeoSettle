import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io # Para manejar la imagen en memoria para el PDF

# Importar fpdf2 si quieres generar PDFs
#try:
#    from fpdf import FPDF
#except ImportError:
#    st.warning("La librería 'fpdf2' no está instalada. Para generar PDFs, por favor instala: pip install fpdf2")
#    FPDF = None # Set FPDF to None if not available

st.set_page_config(layout="wide", page_title="Cálculo de Asentamientos en Cimentaciones Superficiales")

# --- Funciones de Cálculo (Placeholder) ---
# Aquí irían tus funciones para los cálculos de asentamientos
# Estas son solo funciones de ejemplo para que el código funcione.

def calcular_asentamiento_elastico(E, nu, q, B):
    # Fórmula simplificada de asentamiento elástico para ejemplo
    # Asentamiento = q * B * (1 - nu^2) / E * Iw (factor de influencia, simplificado)
    Iw = 0.82 # Factor de influencia para una zapata cuadrada rígida
    return q * B * (1 - nu**2) / E * Iw * 100 # Convertir a cm para ejemplo

def calcular_asentamiento_consolidacion_primaria(Cc, e0, H, dp):
    # Fórmula simplificada de consolidación primaria para ejemplo
    # Asentamiento = Cc / (1 + e0) * H * log10((p0 + dp) / p0)
    # p0 = presión inicial, dp = incremento de presión
    p0 = 100 # kPa, ejemplo de presión inicial
    # Asegurarse de que p0 + dp > 0 para evitar log(0) o log(negativo)
    if p0 + dp <= 0:
        return 0
    return (Cc / (1 + e0)) * H * np.log10((p0 + dp) / p0) * 100 # Convertir a cm

def calcular_asentamiento_consolidacion_secundaria(C_alpha, e0, H, tiempo_primaria, tiempo_final):
    # Fórmula simplificada de consolidación secundaria para ejemplo
    # Asentamiento = C_alpha / (1 + e0) * H * log10(tiempo_final / tiempo_primaria)
    if tiempo_primaria <= 0 or tiempo_final <= 0:
        return 0
    return (C_alpha / (1 + e0)) * H * np.log10(tiempo_final / tiempo_primaria) * 100 # Convertir a cm

def calcular_incremento_esfuerzos_burmister(q, z, B, L):
    # Función placeholder para Burmister.
    # Necesitarías una implementación más completa aquí.
    # Para simplificar, asumiremos un valor de ejemplo.
    # q: carga aplicada, z: profundidad, B, L: dimensiones de la cimentación
    # Este es un ejemplo muy simplificado, no una fórmula real de Burmister.
    # Burmister es para estratos multicapa, requiere propiedades de cada capa.
    # Para una implementación real, buscar tablas o ecuaciones de factores de influencia.
    if z == 0: return q # En la superficie
    return q * (1 - (z / np.sqrt(z**2 + (B/2)**2 + (L/2)**2))**3) # Ejemplo muy simplificado

def calcular_incremento_esfuerzos_boussinesq(q, r, z):
    # Función placeholder para Boussinesq (punto de carga).
    # Para una cimentación, necesitarías integrar o usar una fórmula para área.
    # Para simplificar, asumiremos un valor de ejemplo.
    # q: carga, r: distancia horizontal, z: profundidad
    # Este es un ejemplo muy simplificado, no la fórmula completa de Boussinesq para área.
    return q * (1 - (z / np.sqrt(z**2 + r**2))**3) # Ejemplo muy simplificado

# --- Título principal de la aplicación ---
st.title("🏗️ Cálculo de Asentamientos en Cimentaciones Superficiales")
st.markdown("---")

# --- Configuración de Pestañas ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "ℹ️ Info del Proyecto",
    "⚙️ Ajustes de Cálculo",
    "⛰️ Estratigrafía",
    "🔬 Propiedades del Material",
    "📊 Datos de Cargas",
    "📈 Resultados y Gráficas",
    "📄 Generar Reporte PDF"
])

# --- Pestaña 1: Información del Proyecto ---
with tab1:
    st.header("Información General del Proyecto")
    st.markdown("Aquí puedes ingresar los detalles clave de tu proyecto.")

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Número de Proyecto", "PROYECTO-001", key="project_number")
        st.text_input("Ubicación", "Ciudad de México, México", key="location")
        st.text_input("Ingeniero a Cargo", "Ing. Juan Pérez", key="engineer_name")
    with col2:
        st.info("Sube el logo de tu empresa o proyecto aquí.")
        uploaded_file = st.file_uploader("Subir Logo", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Logo del Proyecto", width=150)
            st.session_state["project_logo"] = uploaded_file.read() # Almacenar para el PDF
        else:
            st.session_state["project_logo"] = None

# --- Pestaña 2: Configuración de Cálculo ---
with tab2:
    st.header("Ajustes y Parámetros de Cálculo")
    st.markdown("Define los tipos de análisis a realizar y las unidades.")

    st.subheader("Tipos de Análisis a Calcular")
    col_analisis1, col_analisis2, col_analisis3 = st.columns(3)
    with col_analisis1:
        st.session_state["analisis_elastico"] = st.checkbox("Asentamiento Elástico", value=True)
    with col_analisis2:
        st.session_state["analisis_consolidacion_primaria"] = st.checkbox("Consolidación Primaria", value=True)
    with col_analisis3:
        st.session_state["analisis_consolidacion_secundaria"] = st.checkbox("Consolidación Secundaria", value=False)

    st.subheader("Método de Cálculo para Incremento de Esfuerzos")
    st.session_state["metodo_incremento_esfuerzos"] = st.selectbox(
        "Selecciona el método",
        ["Boussinesq (Punto)", "Burmister (Estratos)", "2:1 Método Aproximado"],
        index=0
    )

    st.subheader("Unidades")
    col_unidades1, col_unidades2 = st.columns(2)
    with col_unidades1:
        st.session_state["unidades_longitud"] = st.selectbox("Unidades de Longitud", ["m", "cm", "ft", "in"], index=0)
    with col_unidades2:
        st.session_state["unidades_presion"] = st.selectbox("Unidades de Presión/Esfuerzo", ["kPa", "kg/cm²", "psi", "psf", "tsf"], index=0)

    st.subheader("Nivel Freático")
    st.session_state["nivel_freatico_activo"] = st.checkbox("Considerar Nivel Freático", value=False)
    if st.session_state["nivel_freatico_activo"]:
        st.session_state["profundidad_nivel_freatico"] = st.number_input(
            f"Profundidad del Nivel Freático ({st.session_state['unidades_longitud']}) desde la superficie",
            value=1.5, min_value=0.0
        )
    else:
        st.session_state["profundidad_nivel_freatico"] = np.inf # Un valor grande si no hay nivel freático

# --- Pestaña 3: Estratigrafía ---
with tab3:
    st.header("Definición de Estratos del Suelo")
    st.markdown("Agrega los diferentes estratos de suelo que componen el perfil geotécnico.")

    if "estratos_df" not in st.session_state:
        st.session_state["estratos_df"] = pd.DataFrame(
            [
                {"ID Estrato": 1, "Nombre del Estrato": "Arena Superficial", "Tipo de Suelo": "Arena", "Espesor (m)": 2.0},
                {"ID Estrato": 2, "Nombre del Estrato": "Arcilla Blanda", "Tipo de Suelo": "Arcilla", "Espesor (m)": 5.0},
            ]
        )
    
    st.write("Edita la tabla para definir tus estratos:")
    # Usar st.data_editor con key para que Streamlit maneje el estado de la tabla
    edited_estratos_df = st.data_editor(
        st.session_state["estratos_df"],
        column_config={
            "ID Estrato": st.column_config.NumberColumn("ID Estrato", help="Identificador único del estrato", disabled=True),
            "Nombre del Estrato": st.column_config.TextColumn("Nombre del Estrato", help="Nombre descriptivo del estrato"),
            "Tipo de Suelo": st.column_config.SelectboxColumn(
                "Tipo de Suelo",
                options=["Arena", "Arcilla", "Limo", "Grava", "Roca"],
                help="Clasificación principal del suelo"
            ),
            "Espesor (m)": st.column_config.NumberColumn(
                f"Espesor ({st.session_state.get('unidades_longitud', 'm')})",
                help="Espesor del estrato", min_value=0.1
            ),
        },
        num_rows="dynamic",
        use_container_width=True,
        key="estratos_editor" # Clave única para el data_editor
    )
    st.session_state["estratos_df"] = edited_estratos_df # Actualizar el DataFrame en session_state

    if not st.session_state["estratos_df"].empty:
        selected_estratos_indices = st.session_state["estratos_editor"]["edited_rows"]
        if selected_estratos_indices:
            # Obtener los IDs de los estratos seleccionados para eliminar
            rows_to_delete = [idx for idx, selected in selected_estratos_indices.items() if selected.get('_selected', False)]
            
            if st.button("Eliminar Estrato Seleccionado", key="delete_estrato_btn"):
                if rows_to_delete:
                    # Crear un nuevo DataFrame excluyendo las filas seleccionadas
                    st.session_state["estratos_df"] = st.session_state["estratos_df"].drop(index=rows_to_delete).reset_index(drop=True)
                    st.success(f"Se eliminaron {len(rows_to_delete)} estrato(s) seleccionado(s).")
                    st.rerun() # Volver a ejecutar para actualizar la tabla
                else:
                    st.warning("Por favor, selecciona al menos un estrato para eliminar.")
        else:
            st.info("Selecciona filas en la tabla para habilitar la opción de eliminar.")
    else:
        st.info("No hay estratos definidos. Agrega uno para empezar.")

    st.info("Asegúrate de que la suma de los espesores sea suficiente para la profundidad de interés.")

# --- Pestaña 4: Propiedades de Materiales ---
with tab4:
    st.header("Propiedades Geotécnicas de los Materiales")
    st.markdown("Define las propiedades para cada tipo de suelo usado en tu estratigrafía.")

    tipos_de_suelo_en_estratos = st.session_state["estratos_df"]["Tipo de Suelo"].unique()

    if "propiedades_materiales_df" not in st.session_state:
        st.session_state["propiedades_materiales_df"] = pd.DataFrame(columns=[
            "Tipo de Suelo", "Módulo de Elasticidad (E)", "Coeficiente de Poisson (nu)",
            "Índice de Compresión (Cc)", "Relación de Vacíos Inicial (e0)",
            "Coeficiente de Compresión Secundaria (Cα)", "Peso Volumétrico (γ)"
        ])

    st.write("Tabla de Propiedades de Materiales:")
    # Asegurarse de que todos los tipos de suelo en estratos tengan una entrada en propiedades
    for tipo_suelo in tipos_de_suelo_en_estratos:
        if tipo_suelo not in st.session_state["propiedades_materiales_df"]["Tipo de Suelo"].values:
            # Añadir fila por defecto si el tipo de suelo no existe
            new_row = pd.DataFrame([{
                "Tipo de Suelo": tipo_suelo,
                "Módulo de Elasticidad (E)": 10000.0, # kPa
                "Coeficiente de Poisson (nu)": 0.3,
                "Índice de Compresión (Cc)": 0.3,
                "Relación de Vacíos Inicial (e0)": 0.8,
                "Coeficiente de Compresión Secundaria (Cα)": 0.01,
                "Peso Volumétrico (γ)": 18.0 # kN/m3
            }])
            st.session_state["propiedades_materiales_df"] = pd.concat([st.session_state["propiedades_materiales_df"], new_row], ignore_index=True)

    # Filtrar para mostrar solo los tipos de suelo que están en la estratigrafía
    df_to_edit = st.session_state["propiedades_materiales_df"][
        st.session_state["propiedades_materiales_df"]["Tipo de Suelo"].isin(tipos_de_suelo_en_estratos)
    ].copy()

    edited_df = st.data_editor(
        df_to_edit,
        column_config={
            "Tipo de Suelo": st.column_config.TextColumn("Tipo de Suelo", disabled=True),
            "Módulo de Elasticidad (E)": st.column_config.NumberColumn(
                f"Módulo de Elasticidad (E) ({st.session_state.get('unidades_presion', 'kPa')})",
                help="Módulo de Young para cálculo elástico", min_value=1.0
            ),
            "Coeficiente de Poisson (nu)": st.column_config.NumberColumn(
                "Coeficiente de Poisson (nu)", help="Relación de Poisson", min_value=0.0, max_value=0.5
            ),
            "Índice de Compresión (Cc)": st.column_config.NumberColumn(
                "Índice de Compresión (Cc)", help="Para consolidación primaria", min_value=0.0
            ),
            "Relación de Vacíos Inicial (e0)": st.column_config.NumberColumn(
                "Relación de Vacíos Inicial (e0)", help="Para consolidación primaria", min_value=0.0
            ),
            "Coeficiente de Compresión Secundaria (Cα)": st.column_config.NumberColumn(
                "Coeficiente de Compresión Secundaria (Cα)", help="Para consolidación secundaria", min_value=0.0
            ),
            "Peso Volumétrico (γ)": st.column_config.NumberColumn(
                f"Peso Volumétrico (γ) ({st.session_state.get('unidades_presion', 'kN/m²')})", # Ajustar la unidad
                help="Peso volumétrico del suelo", min_value=10.0
            ),
        },
        hide_index=True,
        use_container_width=True,
        key="propiedades_editor"
    )
    # Actualizar solo las filas que fueron editadas
    st.session_state["propiedades_materiales_df"] = pd.concat([
        st.session_state["propiedades_materiales_df"][
            ~st.session_state["propiedades_materiales_df"]["Tipo de Suelo"].isin(tipos_de_suelo_en_estratos)
        ],
        edited_df
    ], ignore_index=True)

    st.subheader("Valores Típicos (Referencia)")
    st.markdown("""
    Esta tabla es solo una referencia. Los valores reales deben obtenerse de ensayos de laboratorio.

    | Tipo de Suelo | E (kPa)     | nu    | Cc      | e0    | Cα      | γ (kN/m³) |
    |---------------|-------------|-------|---------|-------|---------|-----------|
    | Arena Suelta  | 10,000-25,000 | 0.2-0.3 | N/A     | N/A   | N/A     | 16-18     |
    | Arena Densa   | 30,000-80,000 | 0.2-0.3 | N/A     | N/A   | N/A     | 18-20     |
    | Arcilla Blanda| 2,000-10,000  | 0.3-0.4 | 0.2-0.5 | 0.8-1.2 | 0.01-0.03 | 16-18     |
    | Arcilla Media | 10,000-30,000 | 0.3-0.4 | 0.1-0.3 | 0.6-0.9 | 0.005-0.015| 18-20     |
    | Limo          | 5,000-20,000  | 0.3-0.35| 0.15-0.4| 0.7-1.0 | 0.008-0.02 | 17-19     |
    """)

# --- Pestaña 5: Datos de Cargas ---
with tab5:
    st.header("Definición de las Cargas Aplicadas")
    st.markdown("Especifica las cargas y geometrías de las cimentaciones.")

    if "cargas_df" not in st.session_state:
        st.session_state["cargas_df"] = pd.DataFrame(
            [
                {"ID Carga": 1, "Nombre Carga": "Zapata C1", "Tipo de Carga": "Cuadrada", "Longitud (m)": 2.0, "Ancho (m)": 2.0, "Presión (kPa)": 150.0, "Ubicación X (m)": 0.0, "Ubicación Y (m)": 0.0},
            ]
        )

    st.write("Edita la tabla para definir tus cargas:")
    edited_cargas_df = st.data_editor(
        st.session_state["cargas_df"],
        column_config={
            "ID Carga": st.column_config.NumberColumn("ID Carga", disabled=True),
            "Nombre Carga": st.column_config.TextColumn("Nombre Carga"),
            "Tipo de Carga": st.column_config.SelectboxColumn(
                "Tipo de Carga",
                options=["Cuadrada", "Rectangular", "Circular", "Corrida"],
                help="Geometría de la cimentación"
            ),
            "Longitud (m)": st.column_config.NumberColumn(
                f"Longitud ({st.session_state.get('unidades_longitud', 'm')})",
                min_value=0.1
            ),
            "Ancho (m)": st.column_config.NumberColumn(
                f"Ancho ({st.session_state.get('unidades_longitud', 'm')})",
                min_value=0.1, help="Ignorar para cargas circulares y corridas"
            ),
            "Presión (kPa)": st.column_config.NumberColumn(
                f"Presión ({st.session_state.get('unidades_presion', 'kPa')})",
                min_value=0.1
            ),
            "Ubicación X (m)": st.column_config.NumberColumn(
                f"Ubicación X ({st.session_state.get('unidades_longitud', 'm')})",
                help="Coordenada X del centro de la cimentación"
            ),
            "Ubicación Y (m)": st.column_config.NumberColumn(
                f"Ubicación Y ({st.session_state.get('unidades_longitud', 'm')})",
                help="Coordenada Y del centro de la cimentación"
            ),
        },
        num_rows="dynamic",
        use_container_width=True,
        key="cargas_editor" # Clave única para el data_editor
    )
    st.session_state["cargas_df"] = edited_cargas_df # Actualizar el DataFrame en session_state

    if not st.session_state["cargas_df"].empty:
        selected_cargas_indices = st.session_state["cargas_editor"]["edited_rows"]
        if selected_cargas_indices:
            # Obtener los IDs de las cargas seleccionadas para eliminar
            rows_to_delete = [idx for idx, selected in selected_cargas_indices.items() if selected.get('_selected', False)]

            if st.button("Eliminar Carga Seleccionada", key="delete_carga_btn"):
                if rows_to_delete:
                    # Crear un nuevo DataFrame excluyendo las filas seleccionadas
                    st.session_state["cargas_df"] = st.session_state["cargas_df"].drop(index=rows_to_delete).reset_index(drop=True)
                    st.success(f"Se eliminaron {len(rows_to_delete)} carga(s) seleccionada(s).")
                    st.rerun() # Volver a ejecutar para actualizar la tabla
                else:
                    st.warning("Por favor, selecciona al menos una carga para eliminar.")
        else:
            st.info("Selecciona filas en la tabla para habilitar la opción de eliminar.")
    else:
        st.info("No hay cargas definidas. Agrega una para empezar.")

    st.info("Para cargas circulares, la 'Longitud' se interpretará como el diámetro. Para cargas corridas, el 'Ancho' se ignora.")

# --- Pestaña 6: Resultados y Gráficas ---
with tab6:
    st.header("Resultados de Asentamientos y Visualización")
    st.markdown("Aquí se mostrarán los asentamientos calculados y una gráfica de isovalores.")

    st.warning("Nota: Las funciones de cálculo son placeholders. Necesitas implementar tus propias fórmulas geotécnicas.")

    if st.button("Calcular Asentamientos"):
        if st.session_state["estratos_df"].empty or st.session_state["cargas_df"].empty or st.session_state["propiedades_materiales_df"].empty:
            st.error("Por favor, asegúrate de haber definido Estratigrafía, Propiedades de Materiales y Datos de Cargas.")
        else:
            # --- Lógica de Cálculo Principal ---
            profundidad_maxima = st.session_state["estratos_df"]["Espesor (m)"].sum()
            if profundidad_maxima == 0:
                st.error("La suma de los espesores de los estratos no puede ser cero.")
            else:
                # Definir una malla para la gráfica de isovalores
                # Consideraremos un área un poco más grande que el área de las cimentaciones
                # Asegurarse de que haya cargas para calcular min/max
                if not st.session_state["cargas_df"].empty:
                    min_x = st.session_state["cargas_df"]["Ubicación X (m)"].min() - st.session_state["cargas_df"]["Longitud (m)"].max() * 1.5
                    max_x = st.session_state["cargas_df"]["Ubicación X (m)"].max() + st.session_state["cargas_df"]["Longitud (m)"].max() * 1.5
                    min_y = st.session_state["cargas_df"]["Ubicación Y (m)"].min() - st.session_state["cargas_df"]["Ancho (m)"].max() * 1.5
                    max_y = st.session_state["cargas_df"]["Ubicación Y (m)"].max() + st.session_state["cargas_df"]["Ancho (m)"].max() * 1.5
                else:
                    min_x, max_x = -5, 5
                    min_y, max_y = -5, 5

                x = np.linspace(min_x, max_x, 50)
                y = np.linspace(min_y, max_y, 50)
                X, Y = np.meshgrid(x, y)
                asentamientos_totales = np.zeros_like(X, dtype=float) # Para acumular asentamientos en cada punto de la malla

                total_asentamientos_por_carga = {}

                # Iterar sobre cada punto de la malla (X, Y)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        current_x = X[i, j]
                        current_y = Y[i, j]
                        asentamiento_en_punto = 0

                        # Sumar la influencia de cada carga en este punto (current_x, current_y)
                        for index, carga in st.session_state["cargas_df"].iterrows():
                            q = carga["Presión (kPa)"]
                            B_carga = carga["Ancho (m)"]
                            L_carga = carga["Longitud (m)"]
                            x_center_carga = carga["Ubicación X (m)"]
                            y_center_carga = carga["Ubicación Y (m)"]

                            # Calcular la distancia relativa del punto de la malla al centro de la carga
                            r_dist = np.sqrt((current_x - x_center_carga)**2 + (current_y - y_center_carga)**2)

                            # Iterar sobre cada estrato para calcular el asentamiento en el punto (current_x, current_y)
                            profundidad_acumulada = 0
                            for idx_estrato, estrato in st.session_state["estratos_df"].iterrows():
                                H_estrato = estrato["Espesor (m)"]
                                tipo_suelo = estrato["Tipo de Suelo"]
                                propiedades = st.session_state["propiedades_materiales_df"][
                                    st.session_state["propiedades_materiales_df"]["Tipo de Suelo"] == tipo_suelo
                                ]

                                if propiedades.empty:
                                    profundidad_acumulada += H_estrato
                                    continue # Omitir si no hay propiedades

                                E = propiedades["Módulo de Elasticidad (E)"].values[0]
                                nu = propiedades["Coeficiente de Poisson (nu)"].values[0]
                                Cc = propiedades["Índice de Compresión (Cc)"].values[0]
                                e0 = propiedades["Relación de Vacíos Inicial (e0)"].values[0]
                                C_alpha = propiedades["Coeficiente de Compresión Secundaria (Cα)"].values[0]
                                gamma = propiedades["Peso Volumétrico (γ)"].values[0]

                                # Profundidad media del estrato para el cálculo del incremento de esfuerzos
                                z_media = profundidad_acumulada + H_estrato / 2

                                # Cálculo del incremento de esfuerzos (simplificado para la demo)
                                delta_p = 0
                                if st.session_state["metodo_incremento_esfuerzos"] == "Boussinesq (Punto)":
                                    # Para Boussinesq de carga puntual, r_dist es la distancia horizontal
                                    delta_p = calcular_incremento_esfuerzos_boussinesq(q, r_dist, z_media)
                                elif st.session_state["metodo_incremento_esfuerzos"] == "Burmister (Estratos)":
                                    # Burmister es más complejo, aquí una simplificación.
                                    delta_p = calcular_incremento_esfuerzos_burmister(q, z_media, B_carga, L_carga)
                                elif st.session_state["metodo_incremento_esfuerzos"] == "2:1 Método Aproximado":
                                    # Método 2:1, asume distribución de carga
                                    Bx_dist = B_carga + z_media
                                    Lx_dist = L_carga + z_media
                                    delta_p = q * (B_carga * L_carga) / (Bx_dist * Lx_dist) if (Bx_dist * Lx_dist) > 0 else 0

                                # Asegurarse de que delta_p no sea negativo si la fórmula lo permite
                                delta_p = max(0, delta_p)

                                # Calcular asentamientos para este estrato y sumarlos
                                if st.session_state["analisis_elastico"]:
                                    asent_elastico = calcular_asentamiento_elastico(E, nu, q, B_carga) # q y B_carga son de la cimentación
                                    asentamiento_en_punto += asent_elastico

                                if st.session_state["analisis_consolidacion_primaria"]:
                                    if tipo_suelo in ["Arcilla", "Limo"]:
                                        asent_consolidacion_primaria = calcular_asentamiento_consolidacion_primaria(Cc, e0, H_estrato, delta_p)
                                        asentamiento_en_punto += asent_consolidacion_primaria

                                if st.session_state["analisis_consolidacion_secundaria"]:
                                    if tipo_suelo in ["Arcilla", "Limo"]:
                                        tiempo_primaria_ejemplo = 1 # Año
                                        tiempo_final_ejemplo = 50 # Años
                                        asent_consolidacion_secundaria = calcular_asentamiento_consolidacion_secundaria(C_alpha, e0, H_estrato, tiempo_primaria_ejemplo, tiempo_final_ejemplo)
                                        asentamiento_en_punto += asent_consolidacion_secundaria

                                profundidad_acumulada += H_estrato
                        asentamientos_totales[i, j] = asentamiento_en_punto # Asentamiento total en este punto de la malla

                # Calcular asentamientos totales por cimentación (para la tabla de resumen)
                # Esto es diferente a la malla, es el asentamiento promedio o máximo bajo cada cimentación
                # Para simplificar, aquí se usa el cálculo anterior de asentamiento_total_carga
                # En una aplicación real, se podría tomar el asentamiento máximo en el área de la zapata de la malla calculada.
                for index, carga in st.session_state["cargas_df"].iterrows():
                    asentamiento_total_carga_resumen = 0
                    q = carga["Presión (kPa)"]
                    B = carga["Ancho (m)"]
                    L = carga["Longitud (m)"]

                    profundidad_acumulada_resumen = 0
                    for idx_estrato, estrato in st.session_state["estratos_df"].iterrows():
                        H_estrato = estrato["Espesor (m)"]
                        tipo_suelo = estrato["Tipo de Suelo"]
                        propiedades = st.session_state["propiedades_materiales_df"][
                            st.session_state["propiedades_materiales_df"]["Tipo de Suelo"] == tipo_suelo
                        ]
                        if propiedades.empty:
                            profundidad_acumulada_resumen += H_estrato
                            continue

                        E = propiedades["Módulo de Elasticidad (E)"].values[0]
                        nu = propiedades["Coeficiente de Poisson (nu)"].values[0]
                        Cc = propiedades["Índice de Compresión (Cc)"].values[0]
                        e0 = propiedades["Relación de Vacíos Inicial (e0)"].values[0]
                        C_alpha = propiedades["Coeficiente de Compresión Secundaria (Cα)"].values[0]

                        z_media_resumen = profundidad_acumulada_resumen + H_estrato / 2
                        
                        delta_p_resumen = 0
                        if st.session_state["metodo_incremento_esfuerzos"] == "Boussinesq (Punto)":
                            delta_p_resumen = calcular_incremento_esfuerzos_boussinesq(q, 0, z_media_resumen) # r=0 para centro
                        elif st.session_state["metodo_incremento_esfuerzos"] == "Burmister (Estratos)":
                            delta_p_resumen = calcular_incremento_esfuerzos_burmister(q, z_media_resumen, B, L)
                        elif st.session_state["metodo_incremento_esfuerzos"] == "2:1 Método Aproximado":
                            Bx_resumen = B + z_media_resumen
                            Lx_resumen = L + z_media_resumen
                            delta_p_resumen = q * (B * L) / (Bx_resumen * Lx_resumen) if (Bx_resumen * Lx_resumen) > 0 else 0
                        
                        delta_p_resumen = max(0, delta_p_resumen)

                        if st.session_state["analisis_elastico"]:
                            asent_elastico = calcular_asentamiento_elastico(E, nu, q, B)
                            asentamiento_total_carga_resumen += asent_elastico

                        if st.session_state["analisis_consolidacion_primaria"]:
                            if tipo_suelo in ["Arcilla", "Limo"]:
                                asent_consolidacion_primaria = calcular_asentamiento_consolidacion_primaria(Cc, e0, H_estrato, delta_p_resumen)
                                asentamiento_total_carga_resumen += asent_consolidacion_primaria

                        if st.session_state["analisis_consolidacion_secundaria"]:
                            if tipo_suelo in ["Arcilla", "Limo"]:
                                tiempo_primaria_ejemplo = 1
                                tiempo_final_ejemplo = 50
                                asent_consolidacion_secundaria = calcular_asentamiento_consolidacion_secundaria(C_alpha, e0, H_estrato, tiempo_primaria_ejemplo, tiempo_final_ejemplo)
                                asentamiento_total_carga_resumen += asent_consolidacion_secundaria
                        
                        profundidad_acumulada_resumen += H_estrato
                    total_asentamientos_por_carga[carga["Nombre Carga"]] = asentamiento_total_carga_resumen

                st.subheader("Asentamientos Totales por Cimentación")
                asentamientos_df = pd.DataFrame(total_asentamientos_por_carga.items(), columns=["Cimentación", f"Asentamiento Total ({st.session_state.get('unidades_longitud', 'cm')})"])
                st.dataframe(asentamientos_df)

                st.subheader("Gráfica de Isovalores de Asentamiento (2D)")
                fig, ax = plt.subplots(figsize=(10, 8))
                
                contour = ax.contourf(X, Y, asentamientos_totales, levels=20, cmap="viridis_r")
                fig.colorbar(contour, ax=ax, label=f"Asentamiento ({st.session_state.get('unidades_longitud', 'cm')})")

                # Dibujar las cimentaciones
                for index, carga in st.session_state["cargas_df"].iterrows():
                    x_center = carga["Ubicación X (m)"]
                    y_center = carga["Ubicación Y (m)"]
                    length = carga["Longitud (m)"]
                    width = carga["Ancho (m)"]

                    if carga["Tipo de Carga"] == "Cuadrada" or carga["Tipo de Carga"] == "Rectangular":
                        rect = plt.Rectangle((x_center - length/2, y_center - width/2), length, width,
                                             edgecolor='red', facecolor='none', linestyle='--', linewidth=2, label=carga["Nombre Carga"])
                        ax.add_patch(rect)
                        ax.text(x_center, y_center, carga["Nombre Carga"], color='red', ha='center', va='center', fontsize=9)
                    elif carga["Tipo de Carga"] == "Circular":
                        radius = length / 2
                        circle = plt.Circle((x_center, y_center), radius,
                                            edgecolor='red', facecolor='none', linestyle='--', linewidth=2, label=carga["Nombre Carga"])
                        ax.add_patch(circle)
                        ax.text(x_center, y_center, carga["Nombre Carga"], color='red', ha='center', va='center', fontsize=9)

                ax.set_xlabel(f"Coordenada X ({st.session_state.get('unidades_longitud', 'm')})")
                ax.set_ylabel(f"Coordenada Y ({st.session_state.get('unidades_longitud', 'm')})")
                ax.set_title("Mapa de Asentamientos por Isovalores")
                ax.grid(True, linestyle=':', alpha=0.7)
                ax.set_aspect('equal', adjustable='box')
                st.pyplot(fig)
                plt.close(fig) # Cerrar la figura para liberar memoria

                st.session_state["asentamientos_totales_plot_data"] = {
                    "X": X, "Y": Y, "Asentamientos": asentamientos_totales,
                    "cargas": st.session_state["cargas_df"].to_dict('records')
                }
                st.session_state["asentamientos_df_reporte"] = asentamientos_df

                st.subheader("Gráfica de Asentamientos (3D)")
                # Crear la figura 3D con Plotly
                fig_3d = go.Figure(data=[go.Surface(z=asentamientos_totales, x=X, y=Y, colorscale='Viridis')])
                fig_3d.update_layout(
                    title='Asentamientos en 3D (Vista de Superficie)',
                    scene = dict(
                        xaxis_title=f'Coordenada X ({st.session_state.get("unidades_longitud", "m")})',
                        yaxis_title=f'Coordenada Y ({st.session_state.get("unidades_longitud", "m")})',
                        zaxis_title=f'Asentamiento ({st.session_state.get("unidades_longitud", "cm")})',
                        aspectmode='cube' # Para mantener proporciones
                    )
                )
                st.plotly_chart(fig_3d, use_container_width=True)

    else:
        st.info("Presiona 'Calcular Asentamientos' para ver los resultados.")

# --- Pestaña 7: Generar Reporte PDF ---
with tab7:
    st.header("Generar Reporte en PDF")
    st.markdown("Genera un resumen de los datos de entrada y los resultados en formato PDF.")

    if st.button("Generar PDF del Reporte"):
        if FPDF is None:
            st.error("La librería 'fpdf2' no está instalada. Por favor, instálala con: `pip install fpdf2` para generar el PDF.")
        elif "asentamientos_df_reporte" not in st.session_state or st.session_state["asentamientos_df_reporte"].empty:
            st.warning("No se han calculado asentamientos. Por favor, realiza el cálculo primero.")
        else:
            try:
                class PDF(FPDF):
                    def header(self):
                        if "project_logo" in st.session_state and st.session_state["project_logo"]:
                            try:
                                # fpdf2 puede leer de bytes directamente
                                self.image(io.BytesIO(st.session_state["project_logo"]), 10, 8, 33)
                            except Exception as e:
                                st.warning(f"No se pudo cargar la imagen del logo en el PDF: {e}")
                                pass # Continuar sin logo si falla
                        self.set_font("Arial", "B", 15)
                        self.cell(0, 10, "Reporte de Cálculo de Asentamientos", 0, 1, "C")
                        self.set_font("Arial", "", 10)
                        self.cell(0, 5, f"Proyecto: {st.session_state.get('project_number', 'N/A')}", 0, 1, "C")
                        self.ln(10)

                    def footer(self):
                        self.set_y(-15)
                        self.set_font("Arial", "I", 8)
                        self.cell(0, 10, f"Página {self.page_no()}/{{nb}}", 0, 0, "C")

                    # Método para crear tabla desde DataFrame
                    def create_table_from_dataframe(self, df):
                        # Column widths (example, adjust as needed)
                        # Distribute width evenly, adjust if specific columns need more space
                        col_width = (self.w - 2 * self.l_margin) / len(df.columns)
                        row_height = 8

                        # Headers
                        self.set_fill_color(200, 220, 255)
                        self.set_font("Arial", "B", 8)
                        for col in df.columns:
                            self.cell(col_width, row_height, str(col), 1, 0, "C", True)
                        self.ln(row_height)

                        # Data
                        self.set_font("Arial", "", 8)
                        for index, row in df.iterrows():
                            for col in df.columns:
                                val = str(row[col])
                                # Check if text fits in cell, if not, use multi_cell with smaller font or wrap
                                if self.get_string_width(val) > col_width - 2:
                                    # Option 1: Reduce font size for this cell
                                    original_font_size = self.font_size_pt
                                    self.set_font("Arial", "", 6) # Smaller font
                                    self.multi_cell(col_width, row_height / 2, val, border=1, align="C")
                                    self.set_font("Arial", "", original_font_size) # Restore font size
                                    # Need to adjust cursor position after multi_cell
                                    self.set_xy(self.get_x() + col_width, self.get_y() - row_height / 2)
                                else:
                                    self.cell(col_width, row_height, val, 1, 0, "C")
                            self.ln(row_height)

                pdf = PDF()
                pdf.alias_nb_pages()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=15)

                # --- Información del Proyecto ---
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "1. Información del Proyecto", 0, 1, "L")
                pdf.set_font("Arial", "", 10)
                pdf.multi_cell(0, 6, f"Número de Proyecto: {st.session_state.get('project_number', 'N/A')}\n"
                                     f"Ubicación: {st.session_state.get('location', 'N/A')}\n"
                                     f"Ingeniero a Cargo: {st.session_state.get('engineer_name', 'N/A')}\n")
                pdf.ln(5)

                # --- Ajustes de Cálculo ---
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "2. Ajustes de Cálculo", 0, 1, "L")
                pdf.set_font("Arial", "", 10)
                pdf.multi_cell(0, 6, f"Análisis Elástico: {'Sí' if st.session_state.get('analisis_elastico', False) else 'No'}\n"
                                     f"Consolidación Primaria: {'Sí' if st.session_state.get('analisis_consolidacion_primaria', False) else 'No'}\n"
                                     f"Consolidación Secundaria: {'Sí' if st.session_state.get('analisis_consolidacion_secundaria', False) else 'No'}\n"
                                     f"Método Incremento de Esfuerzos: {st.session_state.get('metodo_incremento_esfuerzos', 'N/A')}\n"
                                     f"Unidades de Longitud: {st.session_state.get('unidades_longitud', 'N/A')}\n"
                                     f"Unidades de Presión: {st.session_state.get('unidades_presion', 'N/A')}\n"
                                     f"Nivel Freático: {'Sí' if st.session_state.get('nivel_freatico_activo', False) else 'No'}"
                                     + (f" (Profundidad: {st.session_state.get('profundidad_nivel_freatico', 'N/A')} {st.session_state.get('unidades_longitud', 'm')})" if st.session_state.get('nivel_freatico_activo', False) else "") + "\n"
                                     )
                pdf.ln(5)

                # --- Estratigrafía ---
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "3. Estratigrafía", 0, 1, "L")
                pdf.set_font("Arial", "", 9)
                if not st.session_state["estratos_df"].empty:
                    pdf.create_table_from_dataframe(st.session_state["estratos_df"])
                else:
                    pdf.multi_cell(0, 6, "No se han definido estratos.")
                pdf.ln(5)

                # --- Propiedades de Materiales ---
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "4. Propiedades de Materiales", 0, 1, "L")
                pdf.set_font("Arial", "", 9)
                if not st.session_state["propiedades_materiales_df"].empty:
                    pdf.create_table_from_dataframe(st.session_state["propiedades_materiales_df"])
                else:
                    pdf.multi_cell(0, 6, "No se han definido propiedades de materiales.")
                pdf.ln(5)

                # --- Datos de Cargas ---
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "5. Datos de Cargas", 0, 1, "L")
                pdf.set_font("Arial", "", 9)
                if not st.session_state["cargas_df"].empty:
                    pdf.create_table_from_dataframe(st.session_state["cargas_df"])
                else:
                    pdf.multi_cell(0, 6, "No se han definido cargas.")
                pdf.ln(5)

                # --- Resultados de Asentamiento ---
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "6. Resultados de Asentamientos", 0, 1, "L")
                pdf.set_font("Arial", "", 10)
                if "asentamientos_df_reporte" in st.session_state and not st.session_state["asentamientos_df_reporte"].empty:
                    pdf.multi_cell(0, 6, "Asentamientos Totales por Cimentación:")
                    pdf.ln(2)
                    pdf.set_font("Arial", "", 9)
                    pdf.create_table_from_dataframe(st.session_state["asentamientos_df_reporte"])
                else:
                    pdf.multi_cell(0, 6, "No hay resultados de asentamientos para mostrar.")
                pdf.ln(5)

                # --- Gráfica de Isovalores (Agregar como imagen) ---
                if "asentamientos_totales_plot_data" in st.session_state:
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 10, "Gráfica de Isovalores de Asentamiento (2D)", 0, 1, "L")
                    # Regenerar la gráfica para guardarla como imagen
                    fig_pdf, ax_pdf = plt.subplots(figsize=(10, 8))
                    
                    plot_data = st.session_state["asentamientos_totales_plot_data"]
                    X_pdf = plot_data["X"]
                    Y_pdf = plot_data["Y"]
                    asentamientos_pdf = plot_data["Asentamientos"]
                    cargas_pdf = plot_data["cargas"]

                    contour_pdf = ax_pdf.contourf(X_pdf, Y_pdf, asentamientos_pdf, levels=20, cmap="viridis_r")
                    fig_pdf.colorbar(contour_pdf, ax=ax_pdf, label=f"Asentamiento ({st.session_state.get('unidades_longitud', 'cm')})")

                    for carga in cargas_pdf:
                        x_center = carga["Ubicación X (m)"]
                        y_center = carga["Ubicación Y (m)"]
                        length = carga["Longitud (m)"]
                        width = carga["Ancho (m)"]

                        if carga["Tipo de Carga"] == "Cuadrada" or carga["Tipo de Carga"] == "Rectangular":
                            rect = plt.Rectangle((x_center - length/2, y_center - width/2), length, width,
                                                 edgecolor='red', facecolor='none', linestyle='--', linewidth=2, label=carga["Nombre Carga"])
                            ax_pdf.add_patch(rect)
                            ax_pdf.text(x_center, y_center, carga["Nombre Carga"], color='red', ha='center', va='center', fontsize=9)
                        elif carga["Tipo de Carga"] == "Circular":
                            radius = length / 2
                            circle = plt.Circle((x_center, y_center), radius,
                                                edgecolor='red', facecolor='none', linestyle='--', linewidth=2, label=carga["Nombre Carga"])
                            ax_pdf.add_patch(circle)
                            ax_pdf.text(x_center, y_center, carga["Nombre Carga"], color='red', ha='center', va='center', fontsize=9)

                    ax_pdf.set_xlabel(f"Coordenada X ({st.session_state.get('unidades_longitud', 'm')})")
                    ax_pdf.set_ylabel(f"Coordenada Y ({st.session_state.get('unidades_longitud', 'm')})")
                    ax_pdf.set_title("Mapa de Asentamientos por Isovalores")
                    ax_pdf.grid(True, linestyle=':', alpha=0.7)
                    ax_pdf.set_aspect('equal', adjustable='box')
                    
                    plt.tight_layout()
                    
                    # Guardar la figura en un buffer de bytes
                    img_buffer_2d = io.BytesIO()
                    fig_pdf.savefig(img_buffer_2d, format="png", dpi=300)
                    img_buffer_2d.seek(0)
                    pdf.image(img_buffer_2d, x=pdf.get_x(), y=pdf.get_y(), w=pdf.w - 2 * pdf.l_margin)
                    plt.close(fig_pdf) # Cerrar la figura para liberar memoria
                else:
                    pdf.multi_cell(0, 6, "No hay gráfica de isovalores disponible.")
                pdf.ln(5)

                # --- Gráfica 3D (Agregar como imagen) ---
                if "asentamientos_totales_plot_data" in st.session_state:
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 10, "Gráfica de Asentamientos (3D)", 0, 1, "L")
                    
                    plot_data = st.session_state["asentamientos_totales_plot_data"]
                    X_pdf = plot_data["X"]
                    Y_pdf = plot_data["Y"]
                    asentamientos_pdf = plot_data["Asentamientos"]

                    fig_3d_pdf = go.Figure(data=[go.Surface(z=asentamientos_pdf, x=X_pdf, y=Y_pdf, colorscale='Viridis')])
                    fig_3d_pdf.update_layout(
                        title='Asentamientos en 3D (Vista de Superficie)',
                        scene = dict(
                            xaxis_title=f'Coordenada X ({st.session_state.get("unidades_longitud", "m")})',
                            yaxis_title=f'Coordenada Y ({st.session_state.get("unidades_longitud", "m")})',
                            zaxis_title=f'Asentamiento ({st.session_state.get("unidades_longitud", "cm")})',
                            aspectmode='cube'
                        )
                    )
                    
                    # Guardar la figura 3D como imagen estática
                    # Esto requiere que tengas kaleido instalado: pip install kaleido
                    try:
                        img_buffer_3d = io.BytesIO()
                        fig_3d_pdf.write_image(img_buffer_3d, format="png", scale=2)
                        img_buffer_3d.seek(0)
                        pdf.image(img_buffer_3d, x=pdf.get_x(), y=pdf.get_y(), w=pdf.w - 2 * pdf.l_margin)
                    except Exception as e:
                        st.warning(f"No se pudo generar la imagen 3D para el PDF (requiere 'kaleido'): {e}")
                        pdf.multi_cell(0, 6, "No se pudo generar la gráfica 3D para el PDF (asegúrate de tener 'kaleido' instalado: pip install kaleido).")
                else:
                    pdf.multi_cell(0, 6, "No hay gráfica 3D disponible.")
                pdf.ln(5)

                # Output the PDF
                pdf_output = pdf.output(dest='S').encode('latin1')
                st.download_button(
                    label="Descargar Reporte PDF",
                    data=pdf_output,
                    file_name="Reporte_Asentamientos.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Ocurrió un error al generar el PDF: {e}")
                st.info("Asegúrate de que todos los datos estén completos y los cálculos realizados.")
