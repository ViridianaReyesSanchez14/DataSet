# ML Metrics App (Streamlit + Render)

App web que entrena un modelo simple con **N muestras** y muestra **accuracy**, **F1**, **curva ROC**, **curva Precisión-Recall** y **matriz de confusión**.

## 🚀 Estructura
```
ml-metrics-app/
├─ app.py
├─ ml/
│  ├─ data.py
│  ├─ train.py
│  └─ plots.py
├─ scripts/
│  └─ convert_dataset.py
├─ data/
│  └─ (aquí va tu dataset convertido) 
├─ requirements.txt
├─ render.yaml
├─ .gitignore
└─ README.md
```

## 🔧 Uso local (VS Code)
1. Crea un entorno:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Convierte tu dataset grande a **.pkl.gz** (o parquet):
   ```bash
   python scripts/convert_dataset.py --input data/tu_archivo.csv --output data/dataset.pkl.gz --sample 200000
   ```
   > Usa `--sample` para generar una versión recortada que cargue rápido (ideal para Render free).
3. Ejecuta la app:
   ```bash
   streamlit run app.py
   ```
4. En la interfaz ingresa la **columna objetivo** (target) y elige cuántas filas entrenar.

## 🌐 Despliegue en Render
1. Haz **push** de este repositorio a GitHub (sin subir el dataset enorme).
2. En Render, crea un **Web Service** desde el repo. Render leerá `render.yaml`.
3. (Opcional) Define la variable de entorno `DATA_URL` con un **enlace directo** a tu `.pkl.gz` alojado externamente (S3, Hugging Face, etc.). La app lo descargará a `data/dataset.pkl.gz` en el arranque si no existe.
4. Define `TARGET_COL` (por ejemplo, `label`) para precargar el nombre en la UI.
5. En el plan gratuito limita `N` para evitar quedarte sin memoria (1k–5k suele ser razonable).

## 🧠 Notas de diseño
- El pipeline usa `ColumnTransformer` con *imputación* y *OneHotEncoder* + `StandardScaler`.
- Modelo: `LogisticRegression` (binario). Para ROC/PR debes elegir cuál es la **clase positiva**.
- Si tu problema es multiclase, puedes ejecutar la app varias veces eligiendo diferentes clases positivas, o adaptar `train.py`.

## 📦 Conversión de dataset
El script `scripts/convert_dataset.py` soporta:
- Entrada: `.csv`, `.parquet`, `.pkl/.pkl.gz`
- Salida: `.pkl.gz` (por defecto) o `.parquet` (`--parquet`)
- `--sample N` para crear una **muestra** ligera.
- **Sugerencia**: evita subir archivos de >100MB a Git; usa un hosting externo y `DATA_URL`.

¡Listo! Abre `app.py` en VS Code para personalizar el modelo o la UI.
