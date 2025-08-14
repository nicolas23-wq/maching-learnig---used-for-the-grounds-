## AgriPredict — Análisis y Predicción Agrícola (Antioquia)

Aplicación interactiva para explorar datos agrícolas y predecir el volumen de producción por rubro y municipio en Antioquia. Construida con Streamlit y Plotly, con soporte para modelos de Machine Learning pre-entrenados.

### Características
- **Predicción** del volumen de producción a partir de 5 variables: Año, Rubro, Municipio, Área de Producción y Área Total.
- **Análisis exploratorio**: métricas generales, tendencias temporales, top municipios y comparativas por rubro.
- **Visualizaciones** avanzadas: correlaciones, mapas de calor, histogramas y gráficos de dispersión.
- **Rendimiento del modelo**: panel con métricas e importancia de características (real o simulada según disponibilidad de modelos).
- **Modo demo**: si no encuentra datos/modelo, usa datos y predicciones simuladas para no bloquear la experiencia.

---

## Estructura del proyecto

- `interfaz.py`: aplicación Streamlit (UI, carga de datos/modelos, predicción y visualizaciones).
- `ground.ipynb`: notebook de trabajo/experimentación (EDA/entrenamiento).
- `Areas_cultivadas_y_produccion_agr_cola_en_Antioquia_desde_1990-2022_20250716.csv`: dataset principal esperado por la app.
- `model_RM.pkl`: modelo de Machine Learning serializado (por ejemplo, Random Forest).
- `label_encoders.pkl`: preprocesador con `LabelEncoder` para `Rubro` y `Municipio` (diccionario serializado).
- `venv/`: entorno virtual (opcional; se recomienda crearlo localmente).

---

## Requisitos

- Python 3.9 – 3.12 (recomendado 3.10+)
- Paquetes Python:
  - streamlit
  - pandas, numpy
  - scikit-learn
  - plotly, seaborn, matplotlib

Instalación rápida de dependencias:

```bash
pip install --upgrade pip
pip install streamlit pandas numpy scikit-learn plotly seaborn matplotlib
```

O bien, usando el archivo `requirements.txt`:

```bash
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
pip install streamlit pandas numpy scikit-learn plotly seaborn matplotlib
```

---

## Ejecución

Desde la raíz del proyecto:

```bash
streamlit run interfaz.py
```

La app se abrirá en el navegador (por defecto en `http://localhost:8501`).

---

## Uso de la aplicación

La interfaz tiene cuatro pestañas principales:

- **🎯 Predicción**: ingresa Año, Rubro, Municipio, Área de Producción y Área Total. Presiona “Predecir Volumen” para obtener el valor estimado y métricas derivadas (rendimiento y eficiencia).
- **📊 Análisis de Datos**: métricas descriptivas, promedio/participación por rubro, tendencias anuales y top municipios.
- **🔍 Visualizaciones**: matriz de correlación, mapa de calor (Año × Rubro), distribuciones e interacción multi-dimensional.
- **📈 Rendimiento del Modelo**: métricas del modelo e importancia de características (si hay modelo); si no, se muestran valores simulados.

Notas:
- Si el CSV o los modelos no están presentes, la app funcionará en “modo demo” con datos/predicciones simuladas.
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

## Personalización rápida

- Cambiar el dataset: editar `DATASET_PATH` en `interfaz.py`.
- Ajustar “rendimientos base” del modo simulado: modificar el diccionario `base_yield` en `interfaz.py`.
- Desactivar el modo demo: asegúrate de proveer `model_RM.pkl` y `label_encoders.pkl` válidos.

---

## Solución de problemas (FAQ)

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

