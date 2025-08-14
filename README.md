## AgriPredict — Análisis y Predicción Agrícola (Antioquia)

Aplicación interactiva para explorar datos agrícolas y predecir el volumen de producción por rubro y municipio en Antioquia. Implementada con **Flask** y plantillas Jinja, con una interfaz moderna y funcional.

### Características

- **Predicción** del volumen de producción a partir de 5 variables: Año, Rubro, Municipio, Área de Producción y Área Total.
- **Análisis exploratorio**: métricas generales, tendencias temporales, top municipios y comparativas por rubro.
- **Visualizaciones** avanzadas: correlaciones, mapas de calor, histogramas y gráficos de dispersión.
- **Rendimiento del modelo**: panel con métricas e importancia de características.
- **Modo demo**: si no encuentra datos/modelo, usa datos y predicciones simuladas para no bloquear la experiencia.
- Diseño moderno con tarjetas, badges y estilos en `css/styles.css`.
- Ignora datasets y modelos pesados en Git via `.gitignore`.

---

## Estructura del proyecto

- `app.py`: servidor Flask (punto de entrada principal).
- `templates/`
  - `base.html`: layout base (navbar, contenedores, includes de CSS/JS).
  - `index.html`: formulario y resultado de predicción.
  - `analisis.html`: panel visual de análisis (tarjetas y gráficas).
  - `visualizaciones.html`: visualizaciones avanzadas y correlaciones.
  - `rendimiento.html`: métricas del modelo e importancia de características.
- `css/`
  - `styles.css`: estilos globales (paleta, tarjetas, botones, formularios, tablas).
- `ground.ipynb`: notebook de trabajo/experimentación (EDA/entrenamiento).
- `Areas_cultivadas_y_produccion_agr_cola_en_Antioquia_desde_1990-2022_20250716 (1).csv`: dataset principal.
- `model_RM.pkl`: modelo de Machine Learning serializado (Random Forest).
- `model_DM.pkl`: modelo adicional de Machine Learning.
- `label_encoders.pkl`: preprocesador con `LabelEncoder` para `Rubro` y `Municipio`.
- `requirements.txt`: dependencias.
- `.gitignore`: reglas para ignorar archivos pesados (CSV, PKL, etc.).
- `venv/` y `.venv/`: entornos virtuales (opcional; se recomienda crearlo localmente).

---

## Requisitos

- Python 3.9 – 3.12 (recomendado 3.10+)

### Dependencias
- Flask, pandas, numpy, scikit-learn, plotly, seaborn, matplotlib, huggingface_hub, scipy

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

Rutas disponibles:
- `/`: formulario de predicción y métricas resumidas.
- `/analisis`: tablero con gráficas de análisis exploratorio.
- `/visualizaciones`: visualizaciones avanzadas y correlaciones.
- `/rendimiento`: métricas del modelo e importancia de características.

---

## Uso

### Versión Flask

La aplicación tiene cuatro pestañas principales:

1) **🎯 Predicción** (`/`): En la página de inicio, completa Año, Rubro, Municipio, Área de Producción y Área Total. Envía el formulario para obtener el volumen estimado, rendimiento (Ton/Ha) y eficiencia (%).

2) **📊 Análisis de Datos** (`/analisis`): Métricas descriptivas, promedio/participación por rubro, tendencias anuales y top municipios con gráficas interactivas.

3) **🔍 Visualizaciones** (`/visualizaciones`): Matriz de correlación, mapa de calor (Año × Rubro), distribuciones e interacción multi-dimensional.

4) **📈 Rendimiento del Modelo** (`/rendimiento`): Métricas del modelo e importancia de características (si hay modelo); si no, se muestran valores simulados.

Notas:
- Si el CSV o los modelos no están presentes, la app funcionará en "modo demo" con datos/predicciones simuladas.
- El modelo se descarga automáticamente desde Hugging Face Hub si está disponible.

---

## Datos esperados

El dataset debe incluir, como mínimo, las columnas (nombres exactos):

- `Año` (int)
- `Rubro` (str)
- `Municipio` (str)
- `Área Producción` (float)
- `Área total` (float)
- `Volumen Producción` (float) — usada para análisis/validación; no es necesaria estrictamente para predecir.

La ruta del archivo se define en `app.py` mediante `DATASET_PATH`.

---

## Modelos y preprocesamiento

Para usar predicciones reales (no simuladas), la aplicación:

1. **Descarga automática desde Hugging Face**: Intenta descargar `model_RM.pkl` y `label_encoders.pkl` desde el repositorio `nicolas2w1/model_RM`.

2. **Archivos locales**: Si los archivos están en la raíz del proyecto, los carga directamente.

El modelo debe recibir como entrada los 5 atributos siguientes, en este orden: `[Año, Rubro_cod, Municipio_cod, Área Producción, Área total]`.

El `label_encoders.pkl` debe contener un diccionario con `LabelEncoder` para `Rubro` y `Municipio`, por ejemplo: `{'Rubro': LabelEncoder(), 'Municipio': LabelEncoder()}` ya ajustados.

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
  - `Areas_cultivadas_y_produccion_agr_cola_en_Antioquia_desde_1990-2022_20250716 (1).csv`
  - `label_encoders.pkl`, `model_RM.pkl`, `model_DM.pkl`
- Si ya estaban trackeados, para dejar de versionarlos:
```bash
git rm -r --cached "Areas_cultivadas_y_produccion_agr_cola_en_Antioquia_desde_1990-2022_20250716 (1).csv" label_encoders.pkl model_RM.pkl model_DM.pkl
git add .
git commit -m "chore: ignorar dataset y modelos grandes"
```
- El modelo se descarga desde Hugging Face con `huggingface_hub`. Para repos privados, exporta un token (`HF_TOKEN`) y úsalo en tu código.

---

## Estilos (Flask)

El archivo `css/styles.css` define:
- Paleta y sombras (`:root`).
- Tarjetas/metric-cards con hover y bordes redondeados.
- Botones con gradiente y foco accesible.
- Inputs y selects con foco realzado.
- Estilos de tablas y contenedores de gráficos.

---

## Personalización rápida

- Cambiar el dataset: editar `DATASET_PATH` en `app.py`.
- Ajustar "rendimientos base" del modo simulado: modificar el diccionario `base_yield` en `app.py`.
- Desactivar el modo demo: asegúrate de proveer `model_RM.pkl` y `label_encoders.pkl` válidos.

---

## Solución de problemas

### Flask
- El puerto 5000 está ocupado: `python app.py` con `PORT` alterno (`$env:PORT=5001`).
- No carga el modelo/datos: verifica rutas locales o credenciales/red si vienen de Hugging Face.
- Estilos no se aplican: confirma que `base.html` incluya `css/styles.css` en `<head>`.
- **Modelo no cargado**: asegúrate de que `model_RM.pkl` y `label_encoders.pkl` estén en la raíz y que `label_encoders.pkl` contenga claves `Rubro` y `Municipio` con `LabelEncoder` ajustados.
- **Permisos en PowerShell**: si no puedes activar el entorno, ejecuta PowerShell como Administrador o ajusta la política de ejecución si corresponde a tus políticas de TI.

---

## Licencia

Este proyecto se distribuye con fines educativos. Puedes adaptarlo y ampliarlo según tus necesidades. Añade aquí tu licencia preferida (por ejemplo, MIT) si lo vas a publicar.

## Créditos

Desarrollado como prototipo para optimizar la toma de decisiones en la producción agrícola en Antioquia.
