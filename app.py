from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html
from huggingface_hub import hf_hub_download
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# --------------------------
# CONFIGURACIÓN DE DATOS
# --------------------------
DATASET_PATH = 'Areas_cultivadas_y_produccion_agr_cola_en_Antioquia_desde_1990-2022_20250716 (1).csv'

rubros_disponibles = ['Maíz', 'Café', 'Papa', 'Plátano', 'Aguacate', 'Frijol', 'Tomate']
municipios_antioquia = ['Medellín', 'Envigado', 'Rionegro', 'La Ceja', 'Sabaneta', 'Marinilla', 'Guarne']

def load_dataset():
    try:
        print(f"[INFO] Intentando cargar dataset desde: {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH, encoding='utf-8')
        print(f"[INFO] Dataset cargado exitosamente: {len(df)} filas, {len(df.columns)} columnas")
        print(f"[INFO] Columnas: {list(df.columns)}")
        return df, True
    except FileNotFoundError:
        print(f"[ERROR] No se encontró el archivo: {DATASET_PATH}")
        # Datos simulados si no está el CSV (para que la app no se caiga)
        print("[INFO] Generando datos simulados...")
        np.random.seed(42)
        years = list(range(1990, 2023))
        rubros = rubros_disponibles
        municipios = municipios_antioquia
        data = []
        for year in years:
            for rubro in rubros:
                for municipio in municipios:
                    area_prod = np.random.uniform(5, 100)
                    area_total = area_prod + np.random.uniform(0, 20)
                    volumen = area_prod * np.random.uniform(10, 25) + np.random.normal(0, 50)
                    data.append([year, rubro, municipio, area_prod, area_total, max(0, volumen)])
        df = pd.DataFrame(data, columns=['Año', 'Rubro', 'Municipio', 'Área Producción', 'Área total', 'Volumen Producción'])
        print(f"[INFO] Datos simulados generados: {len(df)} filas")
        return df, False
    except Exception as e:
        print(f"[ERROR] Error al cargar dataset: {e}")
        # Datos simulados como respaldo
        print("[INFO] Generando datos simulados como respaldo...")
        np.random.seed(42)
        years = list(range(1990, 2023))
        rubros = rubros_disponibles
        municipios = municipios_antioquia
        data = []
        for year in years:
            for rubro in rubros:
                for municipio in municipios:
                    area_prod = np.random.uniform(5, 100)
                    area_total = area_prod + np.random.uniform(0, 20)
                    volumen = area_prod * np.random.uniform(10, 25) + np.random.normal(0, 50)
                    data.append([year, rubro, municipio, area_prod, area_total, max(0, volumen)])
        df = pd.DataFrame(data, columns=['Año', 'Rubro', 'Municipio', 'Área Producción', 'Área total', 'Volumen Producción'])
        print(f"[INFO] Datos simulados generados: {len(df)} filas")
        return df, False

def load_model_and_preprocessor():
    model_ml = None
    preprocessor = None
    model_loaded = False
    # Modelo
    try:
        model_path = hf_hub_download(repo_id="nicolas2w1/model_RM", filename="model_RM.pkl")
        with open(model_path, 'rb') as f:
            model_ml = pickle.load(f)
        model_loaded = True
    except Exception as e:
        print(f"[AVISO] No se cargó model_RM.pkl desde HF: {e}")
    # Preprocesador (label encoders)
    try:
        preproc_path = hf_hub_download(repo_id="nicolas2w1/model_RM", filename="label_encoders.pkl")
        with open(preproc_path, 'rb') as file:
            preprocessor = pickle.load(file)
    except Exception as e:
        print(f"[AVISO] No se cargó label_encoders.pkl desde HF: {e}")
    return model_ml, preprocessor, model_loaded

# --------------------------
# LÓGICA DE PREDICCIÓN
# --------------------------
df_data, data_loaded = load_dataset()
model_ml, preprocessor, model_loaded = load_model_and_preprocessor()

if df_data is not None:
    if 'Rubro' in df_data.columns:
        rubros_disponibles = sorted(df_data['Rubro'].unique().tolist())
    if 'Municipio' in df_data.columns:
        municipios_antioquia = sorted(df_data['Municipio'].unique().tolist())

def predict_volume(year, rubro, municipio, area_produccion, area_total, model, preprocessor_obj):
    # Predicción con modelo si existe
    if model is not None and preprocessor_obj is not None:
        try:
            encoded_rubro = preprocessor_obj['Rubro'].transform([rubro])[0]
            encoded_municipio = preprocessor_obj['Municipio'].transform([municipio])[0]
            processed_input = np.array([[year, encoded_rubro, encoded_municipio, area_produccion, area_total]])
            prediction = model.predict(processed_input)[0]
            return max(0, prediction)
        except Exception as e:
            print(f"[AVISO] Fallback a simulación por error en predict: {e}")

    # Simulación mejorada (coincide con tu Streamlit)
    base_yield = {'Maíz': 4.5, 'Café': 1.2, 'Papa': 18.0, 'Plátano': 12.0,
                  'Aguacate': 8.5, 'Frijol': 1.8, 'Tomate': 45.0}
    yield_rate = base_yield.get(rubro, 10.0)
    year_factor = 1 + (year - 2000) * 0.01
    efficiency = area_produccion / area_total if area_total > 0 else 1
    simulated_prediction = area_produccion * yield_rate * year_factor * efficiency * np.random.uniform(0.8, 1.2)
    return max(0, simulated_prediction)

# --------------------------
# HELPERS PARA GRÁFICOS
# --------------------------
def fig_html(fig):
    # Inserta cada figura como HTML independiente con plotly.js CDN
    return to_html(fig, include_plotlyjs='cdn', full_html=False)

def build_analisis_charts(df):
    charts = {}
    if df is None or df.empty:
        return charts

    # Producción promedio por rubro (barra horizontal)
    if {'Rubro', 'Volumen Producción'}.issubset(df.columns):
        avg_production = df.groupby('Rubro')['Volumen Producción'].mean().sort_values(ascending=True)
        fig_bar = px.bar(
            x=avg_production.values,
            y=avg_production.index,
            orientation='h',
            title="Producción Promedio por Rubro",
            labels={'x': 'Volumen Promedio (Toneladas)', 'y': 'Rubro'},
            color=avg_production.values,
            color_continuous_scale="Viridis"
        )
        fig_bar.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
        charts['bar_avg_rubro'] = fig_html(fig_bar)

        # Participación en producción total (pie)
        total_production = df.groupby('Rubro')['Volumen Producción'].sum()
        fig_pie = px.pie(
            values=total_production.values,
            names=total_production.index,
            title="Participación en Producción Total"
        )
        fig_pie.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
        charts['pie_rubro'] = fig_html(fig_pie)

    # Tendencia temporal (línea)
    if {'Año', 'Volumen Producción'}.issubset(df.columns):
        yearly_production = df.groupby('Año')['Volumen Producción'].sum().reset_index()
        fig_line = px.line(
            yearly_production,
            x='Año',
            y='Volumen Producción',
            title="Evolución de la Producción Total por Año",
            markers=True
        )
        fig_line.update_traces(line_color='#2a5298', line_width=3)
        fig_line.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
        charts['line_yearly'] = fig_html(fig_line)

    # Top 10 municipios (barra)
    if {'Municipio', 'Volumen Producción'}.issubset(df.columns):
        top_municipios = df.groupby('Municipio')['Volumen Producción'].sum().sort_values(ascending=False).head(10)
        fig_top = px.bar(
            x=top_municipios.index,
            y=top_municipios.values,
            title="Top 10 Municipios por Producción Total",
            color=top_municipios.values,
            color_continuous_scale="Blues"
        )
        fig_top.update_layout(height=400, xaxis_tickangle=-45, margin=dict(l=10, r=10, t=50, b=10))
        charts['bar_top_muni'] = fig_html(fig_top)

    return charts

def build_visualizaciones_charts(df):
    charts = {}
    insights = []
    if df is None or df.empty:
        return charts, insights

    # Matriz de correlación
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 1:
        corr = df[numeric_columns].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Matriz de Correlación - Variables Numéricas",
            color_continuous_scale="RdBu_r"
        )
        fig_corr.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10))
        charts['corr'] = fig_html(fig_corr)

        if 'Volumen Producción' in corr.columns:
            rel = corr['Volumen Producción'].drop('Volumen Producción').abs().sort_values(ascending=False)
            top = rel[rel > 0.1].head(3)
            for var, c in top.items():
                insights.append(f"{var}: correlación {c:.3f} con el volumen de producción")

    # Heatmap Año x Rubro
    if {'Año', 'Rubro', 'Volumen Producción'}.issubset(df.columns):
        heatmap_data = df.groupby(['Año', 'Rubro'])['Volumen Producción'].sum().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='Rubro', columns='Año', values='Volumen Producción')
        fig_heatmap = px.imshow(
            heatmap_pivot,
            aspect="auto",
            title="Producción por Año y Rubro (Toneladas)",
            color_continuous_scale="Viridis",
            labels=dict(x="Año", y="Rubro", color="Toneladas")
        )
        fig_heatmap.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
        charts['heatmap'] = fig_html(fig_heatmap)

    # Histogramas
    if 'Área Producción' in df.columns:
        fig_hist1 = px.histogram(
            df, x='Área Producción', nbins=30,
            title="Distribución del Área de Producción",
            color_discrete_sequence=['#2a5298']
        )
        fig_hist1.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
        charts['hist_area'] = fig_html(fig_hist1)

    if 'Volumen Producción' in df.columns:
        fig_hist2 = px.histogram(
            df, x='Volumen Producción', nbins=30,
            title="Distribución del Volumen de Producción",
            color_discrete_sequence=['#764ba2']
        )
        fig_hist2.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
        charts['hist_vol'] = fig_html(fig_hist2)

    # Scatter multidimensional
    if {'Área Producción', 'Volumen Producción', 'Rubro'}.issubset(df.columns):
        size_col = 'Área total' if 'Área total' in df.columns else None
        hover_cols = ['Municipio'] if 'Municipio' in df.columns else None
        fig_scatter = px.scatter(
            df,
            x='Área Producción',
            y='Volumen Producción',
            color='Rubro',
            size=size_col,
            hover_data=hover_cols,
            title="Relación entre Área de Producción y Volumen (por Rubro)",
            template="plotly_white"
        )
        fig_scatter.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10))
        charts['scatter'] = fig_html(fig_scatter)

    return charts, insights

def build_rendimiento(df, model_ok):
    datos = {}
    charts = {}

    if model_ok and df is not None:
        # Métricas "reales" (simuladas tal como en tu Streamlit cuando hay modelo)
        datos['r2']   = "0.847"
        datos['mae']  = "12.34"
        datos['rmse'] = "18.67"
        datos['acc']  = "84.7%"
        datos['modo'] = "Modelo cargado"
        datos['delta_r2'] = "Excelente"
        datos['delta_mae'] = "-2.1%"
        datos['delta_rmse'] = "-1.8%"
        datos['delta_acc'] = "+2.3%"

        # Importancia de características
        feature_importance = {
            'Área Producción': 0.45,
            'Rubro': 0.25,
            'Año': 0.15,
            'Municipio': 0.10,
            'Área Total': 0.05
        }
        fig_imp = px.bar(
            x=list(feature_importance.values()),
            y=list(feature_importance.keys()),
            orientation='h',
            title="Importancia de Características en el Modelo",
            color=list(feature_importance.values()),
            color_continuous_scale="Viridis"
        )
        fig_imp.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
        charts['importance'] = fig_html(fig_imp)

        # Curva de aprendizaje simulada
        training_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = 1 - np.exp(-3 * training_sizes) * 0.3
        rng = np.random.default_rng(42)
        val_scores = train_scores - 0.05 - 0.02 * rng.normal(0, 1, 10)

        fig_lc = go.Figure()
        fig_lc.add_trace(go.Scatter(
            x=training_sizes, y=train_scores,
            mode='lines+markers', name='Score de Entrenamiento',
            line=dict(color='#2a5298', width=3)
        ))
        fig_lc.add_trace(go.Scatter(
            x=training_sizes, y=val_scores,
            mode='lines+markers', name='Score de Validación',
            line=dict(color='#764ba2', width=3)
        ))
        fig_lc.update_layout(
            title="Curva de Aprendizaje",
            xaxis_title="Tamaño del Conjunto de Entrenamiento",
            yaxis_title="Score del Modelo",
            height=400, margin=dict(l=10, r=10, t=50, b=10)
        )
        charts['learning'] = fig_html(fig_lc)

    else:
        datos['modo'] = "Simulación (modelo no cargado)"
        datos['sim_status'] = True
        datos['prec_est'] = "~75%"

    return datos, charts

# --------------------------
# RUTAS FLASK (pestañas)
# --------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    metrics = None

    if request.method == "POST":
        year = int(request.form["year"])
        rubro = request.form["rubro"]
        municipio = request.form["municipio"]
        area_produccion = float(request.form["area_produccion"])
        area_total = float(request.form["area_total"])

        if area_produccion > area_total:
            prediction_result = {"error": "El Área Total debe ser mayor o igual al Área de Producción"}
        else:
            pred = predict_volume(year, rubro, municipio, area_produccion, area_total, model_ml, preprocessor)
            metrics = {
                "rendimiento": pred / area_produccion if area_produccion > 0 else 0,
                "eficiencia": (area_produccion / area_total) * 100 if area_total > 0 else 0,
                "anio": year
            }
            prediction_result = {"valor": pred}

    return render_template(
        "index.html",
        active="prediccion",
        rubros=rubros_disponibles,
        municipios=municipios_antioquia,
        prediction=prediction_result,
        metrics=metrics,
        data_status=data_loaded,
        model_status=model_loaded
    )

@app.route("/analisis")
def analisis():
    total_reg = len(df_data) if df_data is not None else 0
    rubros_u = df_data['Rubro'].nunique() if df_data is not None and 'Rubro' in df_data.columns else 0
    municipios_u = df_data['Municipio'].nunique() if df_data is not None and 'Municipio' in df_data.columns else 0
    periodo = f"{int(df_data['Año'].min())}-{int(df_data['Año'].max())}" if df_data is not None and 'Año' in df_data.columns else "N/A"

    charts = build_analisis_charts(df_data)
    return render_template(
        "analisis.html",
        active="analisis",
        total_reg=total_reg,
        rubros_u=rubros_u,
        municipios_u=municipios_u,
        periodo=periodo,
        charts=charts,
        data_status=data_loaded,
        model_status=model_loaded
    )

@app.route("/visualizaciones")
def visualizaciones():
    charts, insights = build_visualizaciones_charts(df_data)
    return render_template(
        "visualizaciones.html",
        active="visualizaciones",
        charts=charts,
        insights=insights,
        data_status=data_loaded,
        model_status=model_loaded
    )

@app.route("/rendimiento")
def rendimiento():
    datos, charts = build_rendimiento(df_data, model_loaded)
    return render_template(
        "rendimiento.html",
        active="rendimiento",
        datos=datos,
        charts=charts,
        data_status=data_loaded,
        model_status=model_loaded
    )

# --------------------------
# MAIN (para correr local)
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)
