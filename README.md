## AgriPredict — Análisis y Predicción Agrícola (Antioquia)

Aplicación interactiva para explorar datos agrícolas y predecir el volumen de producción por rubro y municipio en Antioquia. Disponible en dos versiones: **Flask** (con plantillas Jinja) y **Streamlit** (interfaz moderna).

### Características

#### Versión Flask
- Formulario de predicción con variables: Año, Rubro, Municipio, Área de Producción y Área Total.
- Vistas de análisis con gráficas integradas (barras, líneas, pastel; correlación/heatmap/dispersion si el backend las provee).
- Diseño moderno con tarjetas, badges y estilos en `css/styles.css`.
- Ignora datasets y modelos pesados en Git via `.gitignore`.

#### Versión Streamlit
- **Predicción** del volumen de producción a partir de 5 variables: Año, Rubro, Municipio, Área de Producción y Área Total.
- **Análisis exploratorio**: métricas generales, tendencias temporales, top municipios y comparativas por rubro.
- **Visualizaciones** avanzadas: correlaciones, mapas de calor, histogramas y gráficos de dispersión.
- **Rendimiento del modelo**: panel con métricas e importancia de características (real o simulada según disponibilidad de modelos).
- **Modo demo**: si no encuentra datos/modelo, usa datos y predicciones simuladas para no bloquear la experiencia.

---

## Estructura del proyecto

### Versión Flask
- `app.py`: servidor Flask (punto de entrada).
- `templates/`
  - `base.html`: layout base (navbar, contenedores, includes de CSS/JS).
  - `index.html`: formulario y resultado de predicción.
  - `analisis.html`: panel visual de análisis (tarjetas y gráficas).
- `css/`
  - `styles.css`: estilos globales (paleta, tarjetas, botones, formularios, tablas).

### Versión Streamlit
- `interfaz.py`: aplicación Streamlit (UI, carga de datos/modelos, predicción y visualizaciones).
- `ground.ipynb`: notebook de trabajo/experimentación (EDA/entrenamiento).

### Archivos compartidos
- `Areas_cultivadas_y_produccion_agr_cola_en_Antioquia_desde_1990-2022_20250716.csv`: dataset principal esperado por la app.
- `model_RM.pkl`: modelo de Machine Learning serializado (por ejemplo, Random Forest).
- `label_encoders.pkl`: preprocesador con `LabelEncoder` para `Rubro` y `Municipio` (diccionario serializado).
- `requirements.txt`: dependencias.
- `.gitignore`: reglas para ignorar archivos pesados (CSV, PKL, etc.).
- `venv/`: entorno virtual (opcional; se recomienda crearlo localmente).

---

## Requisitos

- Python 3.9 – 3.12 (recomendado 3.10+)

### Dependencias Flask
- Flask, pandas, numpy, scikit-learn, plotly, seaborn, matplotlib, huggingface_hub, scipy

### Dependencias Streamlit
- streamlit, pandas, numpy, scikit-learn, plotly, seaborn, matplotlib

Instalación rápida de dependencias:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Instalación (Windows / PowerShell)

1) Entrar a la carpeta del proyecto:

```bash
cd "C:\Users\nicol\OneDrive\Escritorio\maching learning"
```

2) Crear y activar entorno virtual:

```bash
python -m venv venv
./venv/Scripts/Activate.ps1
```

3) Instalar dependencias:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Ejecución

### Versión Flask

Opción 1 (recomendada):
```bash
python app.py
```

Opción 2 (Flask CLI):
```bash
$env:FLASK_APP = "app.py"
flask run
```

Abrir en el navegador: `http://localhost:5000/`

Rutas esperadas (según implementación):
- `/`: formulario de predicción y métricas resumidas.
- `/analisis`: tablero con gráficas (si el backend prepara `charts` y `table_html`).

### Versión Streamlit

Desde la raíz del proyecto:

```bash
streamlit run interfaz.py
```

La app se abrirá en el navegador (por defecto en `http://localhost:8501`).

---

## Uso

### Versión Flask

1) En la página de inicio, completa Año, Rubro, Municipio, Área de Producción y Área Total. Envía el formulario.
2) Se mostrará el volumen estimado, rendimiento (Ton/Ha) y eficiencia (%).
3) En la vista de análisis (si está activa), se muestran tarjetas y gráficas. El backend debe pasar objetos HTML seguros (Plotly) en `charts`:
   - `charts.bar_avg_rubro`, `charts.pie_rubro`, `charts.line_yearly`.
   - Opcional: `charts.corr_heatmap`, `charts.heatmap_year_rubro`, `charts.scatter_multi`.
   - Opcional: `table_html` con una tabla renderizada.

### Versión Streamlit

La interfaz tiene cuatro pestañas principales:

- **🎯 Predicción**: ingresa Año, Rubro, Municipio, Área de Producción y Área Total. Presiona "Predecir Volumen" para obtener el valor estimado y métricas derivadas (rendimiento y eficiencia).
- **📊 Análisis de Datos**: métricas descriptivas, promedio/participación por rubro, tendencias anuales y top municipios.
- **🔍 Visualizaciones**: matriz de correlación, mapa de calor (Año × Rubro), distribuciones e interacción multi-dimensional.
- **📈 Rendimiento del Modelo**: métricas del modelo e importancia de características (si hay modelo); si no, se muestran valores simulados.

Notas:
- Si el CSV o los modelos no están presentes, la app funcionará en "modo demo" con datos/predicciones simuladas.
- Puedes personalizar el archivo de datos editando `DATASET_PATH` en `interfaz.py`.

---

## Datos esperados

El dataset debe incluir, como mínimo, las columnas (nombres exactos):

- `Año` (int)
- `Rubro` (str)
- `Municipio` (str)
- `Área Producción` (float)
- `Área total` (float)
- `Volumen Producción` (float) — usada para análisis/validación; no es necesaria estrictamente para predecir.

La ruta del archivo se define en `interfaz.py` mediante `DATASET_PATH`.

---

## Modelos y preprocesamiento

Para usar predicciones reales (no simuladas), coloca en la raíz del proyecto:

- `model_RM.pkl`: modelo entrenado que reciba como entrada los 5 atributos siguientes, en este orden: `[Año, Rubro_cod, Municipio_cod, Área Producción, Área total]`.
- `label_encoders.pkl`: diccionario con `LabelEncoder` para `Rubro` y `Municipio`, por ejemplo: `{'Rubro': LabelEncoder(), 'Municipio': LabelEncoder()}` ya ajustados.

Ejemplo mínimo para generar y guardar el preprocesador y el modelo:

```python
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# df: DataFrame con las columnas esperadas
encoders = {
    'Rubro': LabelEncoder().fit(df['Rubro']),
    'Municipio': LabelEncoder().fit(df['Municipio'])
}

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

X = pd.DataFrame({
    'Año': df['Año'],
    'Rubro_cod': encoders['Rubro'].transform(df['Rubro']),
    'Municipio_cod': encoders['Municipio'].transform(df['Municipio']),
    'Área Producción': df['Área Producción'],
    'Área total': df['Área total']
})
y = df['Volumen Producción']

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X, y)

with open('model_RM.pkl', 'wb') as f:
    pickle.dump(model, f)
```

La app cargará automáticamente ambos archivos si están disponibles. Si falta alguno, utilizará la simulación interna para producir resultados razonables.

---

## Datos y modelos

- Los archivos pesados están ignorados por Git: `*.csv`, `*.pkl`, `*.joblib` y, específicamente:
  - `Areas_cultivadas_y_produccion_agr_cola_en_Antioquia_desde_1990-2022_20250716.csv`
  - `label_encoders.pkl`, `model_RM.pkl`
- Si ya estaban trackeados, para dejar de versionarlos:
```bash
git rm -r --cached Areas_cultivadas_y_produccion_agr_cola_en_Antioquia_desde_1990-2022_20250716.csv label_encoders.pkl model_RM.pkl
git add .
git commit -m "chore: ignorar dataset y modelos grandes"
```
- Si el modelo se descarga desde Hugging Face con `huggingface_hub`, asegúrate de tener red. Para repos privados, exporta un token (`HF_TOKEN`) y úsalo en tu código.

---

## Estilos (Flask)

El archivo `css/styles.css` define:
- Paleta y sombras (`:root`).
- Tarjetas/metric-cards con hover y bordes redondeados.
- Botones con gradiente y foco accesible.
- Inputs y selects con foco realzado.
- Estilos de tablas y contenedores de gráficos.

---

## Personalización rápida (Streamlit)

- Cambiar el dataset: editar `DATASET_PATH` en `interfaz.py`.
- Ajustar "rendimientos base" del modo simulado: modificar el diccionario `base_yield` en `interfaz.py`.
- Desactivar el modo demo: asegúrate de proveer `model_RM.pkl` y `label_encoders.pkl` válidos.

---

## Solución de problemas

### Flask
- El puerto 5000 está ocupado: `python app.py` con `PORT` alterno (`$env:PORT=5001`).
- No carga el modelo/datos: verifica rutas locales o credenciales/red si vienen de Hugging Face.
- Estilos no se aplican: confirma que `base.html` incluya `css/styles.css` en `<head>`.

### Streamlit
- **Streamlit no se reconoce como comando**: activa el entorno virtual (`./venv/Scripts/Activate.ps1`) e instala dependencias (`pip install streamlit ...`).
- **No encuentra el CSV**: verifica el nombre/ruta en `DATASET_PATH` y que el archivo exista en la carpeta del proyecto.
- **Caracteres extraños en el CSV**: intenta `encoding='utf-8'` o guarda el CSV en UTF-8 desde tu editor/Excel.
- **Modelo no cargado**: asegúrate de que `model_RM.pkl` y `label_encoders.pkl` estén en la raíz y que `label_encoders.pkl` contenga claves `Rubro` y `Municipio` con `LabelEncoder` ajustados.
- **Puerto ocupado (8501)**: ejecuta `streamlit run interfaz.py --server.port 8502` (o cambia el puerto que prefieras).
- **Permisos en PowerShell**: si no puedes activar el entorno, ejecuta PowerShell como Administrador o ajusta la política de ejecución si corresponde a tus políticas de TI.

---

## Licencia

Este proyecto se distribuye con fines educativos. Puedes adaptarlo y ampliarlo según tus necesidades. Añade aquí tu licencia preferida (por ejemplo, MIT) si lo vas a publicar.

## Créditos

Desarrollado como prototipo para optimizar la toma de decisiones en la producción agrícola en Antioquia.
