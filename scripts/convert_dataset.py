"""
Script para convertir un dataset grande (p.ej., CSV) a un Pickle comprimido (.pkl.gz)
o Parquet (.parquet), reduciendo tiempos de carga en producción.

Uso:
    python scripts/convert_dataset.py --input data/tu.csv --output data/dataset.pkl.gz
Opcional:
    --sample 500000   # guarda solo 500k filas para despliegue (útil en Render)
    --parquet         # fuerza .parquet como salida (requiere pyarrow)
"""
import argparse
import pandas as pd
import numpy as np
import os

def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=[np.number]).columns:
        col_min, col_max = df[col].min(), df[col].max()
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='float')
        else:
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Ruta al CSV/Parquet/PKL original")
    parser.add_argument("--output", required=True, help="Ruta destino (.pkl.gz o .parquet)")
    parser.add_argument("--sample", type=int, default=None, help="Número de filas a muestrear (opcional)")
    parser.add_argument("--parquet", action="store_true", help="Forzar salida Parquet")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Cargar
    if args.input.endswith(".csv"):
        df = pd.read_csv(args.input, low_memory=False)
    elif args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    elif args.input.endswith(".arff"):
        import arff
        data = arff.load(open(args.input, "r"))
        df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
    else:
        df = pd.read_pickle(args.input)


    # Optimización básica
    df = downcast_numeric(df)
    for col in df.select_dtypes(include=['object']).columns:
        # Convertir a categoría si cardinalidad moderada
        nunique = df[col].nunique(dropna=True)
        if nunique <= max(50, 0.2 * len(df)):
            df[col] = df[col].astype('category')

    # Guardar
    output = args.output
    if args.parquet or output.endswith(".parquet"):
        df.to_parquet(output, index=False)
        print(f"[OK] Guardado en Parquet: {output}")
    else:
        if not output.endswith(".pkl") and not output.endswith(".pkl.gz"):
            output = output + ".pkl.gz"
        df.to_pickle(output, compression="gzip")
        print(f"[OK] Guardado Pickle comprimido: {output}")

if __name__ == "__main__":
    main()
