## AgriPredict — Predicción y Análisis Agrícola (Flask)

Aplicación web en Flask para predecir el volumen de producción agrícola y explorar datos por rubro y municipio (Antioquia). Interfaz con plantillas Jinja y estilos propios.

### Características
- Formulario de predicción con variables: Año, Rubro, Municipio, Área de Producción y Área Total.
- Vistas de análisis con gráficas integradas (barras, líneas, pastel; correlación/heatmap/dispersion si el backend las provee).
- Diseño moderno con tarjetas, badges y estilos en `css/styles.css`.
- Ignora datasets y modelos pesados en Git via `.gitignore`.

---

## Estructura del proyecto

- `app.py`: servidor Flask (punto de entrada).
- `templates/`
  - `base.html`: layout base (navbar, contenedores, includes de CSS/JS).
  - `index.html`: formulario y resultado de predicción.
  - `analisis.html`: panel visual de análisis (tarjetas y gráficas).
- `css/`
  - `styles.css`: estilos globales (paleta, tarjetas, botones, formularios, tablas).
- `requirements.txt`: dependencias.
- `.gitignore`: reglas para ignorar archivos pesados (CSV, PKL, etc.).

---

## Requisitos

- Python 3.9 – 3.12
- Dependencias (instalación recomendada con `requirements.txt`):
  - Flask, pandas, numpy, scikit-learn, plotly, seaborn, matplotlib, huggingface_hub, scipy

Instalación rápida (Windows / PowerShell):

```bash
cd "C:\Users\nicol\OneDrive\Escritorio\maching learning"
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

---

## Ejecución

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

---

## Uso

1) En la página de inicio, completa Año, Rubro, Municipio, Área de Producción y Área Total. Envía el formulario.
2) Se mostrará el volumen estimado, rendimiento (Ton/Ha) y eficiencia (%).
3) En la vista de análisis (si está activa), se muestran tarjetas y gráficas. El backend debe pasar objetos HTML seguros (Plotly) en `charts`:
   - `charts.bar_avg_rubro`, `charts.pie_rubro`, `charts.line_yearly`.
   - Opcional: `charts.corr_heatmap`, `charts.heatmap_year_rubro`, `charts.scatter_multi`.
   - Opcional: `table_html` con una tabla renderizada.

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

## Estilos

El archivo `css/styles.css` define:
- Paleta y sombras (`:root`).
- Tarjetas/metric-cards con hover y bordes redondeados.
- Botones con gradiente y foco accesible.
- Inputs y selects con foco realzado.
- Estilos de tablas y contenedores de gráficos.

---

## Solución de problemas

- El puerto 5000 está ocupado: `python app.py` con `PORT` alterno (`$env:PORT=5001`).
- No carga el modelo/datos: verifica rutas locales o credenciales/red si vienen de Hugging Face.
- Estilos no se aplican: confirma que `base.html` incluya `css/styles.css` en `<head>`.

---

## Licencia

Uso educativo. Ajusta la licencia según tus necesidades (p. ej., MIT).
