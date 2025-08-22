import os
import streamlit as st
import pandas as pd

from ml.data import ensure_data_local, load_dataset, get_feature_target
from ml.train import train_and_evaluate
from ml.plots import plot_roc_curve, plot_pr_curve, plot_confusion

st.set_page_config(page_title="Evaluación de Modelos (ROC/PR/Matriz)", layout="wide")

st.title("📊 Evaluación de Modelo con N muestras")
st.write(
    "Sube o apunta a tu dataset (en **.pkl** o **.pkl.gz** preferentemente) y elige "
    "cuántas filas quieres usar para **entrenar**. El app calculará *accuracy*, *F1*, "
    "la **curva ROC**, **curva Precisión-Recall** y la **matriz de confusión**."
)

with st.sidebar:
    st.header("⚙️ Configuración")
    default_path = "data/dataset.pkl.gz"
    data_url = os.environ.get("DATA_URL", "").strip()
    target_env = os.environ.get("TARGET_COL", "").strip()

    path_input = st.text_input("Ruta del dataset (.pkl / .pkl.gz / .parquet)", value=default_path)
    st.caption("Sugerencia: usa un pickle comprimido (.pkl.gz) o parquet para cargar más rápido y consumir menos memoria.")

    if data_url:
        st.write("`DATA_URL` detectada en variables de entorno. Si el archivo no existe localmente, se descargará.")
        if st.button("Descargar/validar dataset desde DATA_URL"):
            ensure_data_local(data_url, path_input)
            st.success(f"Dataset disponible en: {path_input}")

    test_size = st.slider("Proporción de TEST", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    random_state = st.number_input("Random seed", value=42, step=1)

@st.cache_data(show_spinner=True)
def _load_df(path: str) -> pd.DataFrame:
    return load_dataset(path)

df = None
if os.path.exists(path_input):
    try:
        df = _load_df(path_input)
    except Exception as e:
        st.error(f"No se pudo cargar el dataset: {e}")
else:
    st.info("El archivo no existe. Cárgalo con el script de conversión o ajusta la ruta.")

uploaded = st.file_uploader("…o sube un .pkl/.pkl.gz/.parquet/.csv", type=["pkl", "gz", "parquet", "csv"])
if uploaded is not None:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        elif uploaded.name.endswith(".parquet"):
            df = pd.read_parquet(uploaded)
        else:
            df = pd.read_pickle(uploaded)
        st.success(f"Archivo cargado: {uploaded.name} | shape={df.shape}")
    except Exception as e:
        st.error(f"Error leyendo el archivo subido: {e}")

if df is not None:
    st.subheader("📁 Vista rápida del dataset")
    st.write(df.head())

    guess_target = target_env or (df.columns[-1] if df is not None and len(df.columns) > 0 else "")
    target_col = st.text_input("Columna objetivo (target)", value=guess_target)

    if target_col and target_col in df.columns:
        X, y, numeric_cols, categorical_cols = get_feature_target(df, target_col)

        # Clases y clase positiva
        classes = sorted(pd.unique(y))
        if len(classes) < 2:
            st.error("Se requiere un problema de clasificación con al menos 2 clases.")
            st.stop()

        pos_default = classes[-1]
        pos_class = st.selectbox("Clase POSITIVA (para ROC/PR y F1)", options=classes, index=classes.index(pos_default))

        # Tamaño de entrenamiento
        max_train = max(100, min(len(X) - 1, 50000))
        n_train = st.slider("N filas para ENTRENAR", min_value=100, max_value=int(max_train), value=min(5000, int(max_train)), step=100)
        st.caption("Consejo: en Render (free) usa valores modestos (1k-5k) para no agotar memoria.")

        if st.button("🚀 Entrenar y evaluar"):
            with st.spinner("Entrenando…"):
                results = train_and_evaluate(
                    df=df,
                    target_col=target_col,
                    n_train=n_train,
                    positive_class=pos_class,
                    test_size=test_size,
                    random_state=int(random_state),
                )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{results['metrics']['accuracy']:.4f}")
            col2.metric("F1 (pos)", f"{results['metrics']['f1']:.4f}")
            col3.metric("ROC AUC", f"{results['metrics']['roc_auc']:.4f}")
            col4.metric("AP (PR AUC)", f"{results['metrics']['avg_precision']:.4f}")

            # Graficar
            st.subheader("📈 Curva ROC")
            fig_roc = plot_roc_curve(results['y_true_pos'], results['y_score'])
            st.pyplot(fig_roc, use_container_width=True)

            st.subheader("📈 Curva Precisión-Recall")
            fig_pr = plot_pr_curve(results['y_true_pos'], results['y_score'])
            st.pyplot(fig_pr, use_container_width=True)

            st.subheader("🧮 Matriz de confusión")
            fig_cm = plot_confusion(results['y_true'], results['y_pred'], labels_order=[pos_class] + [c for c in classes if c != pos_class])
            st.pyplot(fig_cm, use_container_width=True)

    else:
        st.info("Indica la columna objetivo (target) para continuar.")
else:
    st.stop()
