import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def cargar_datos(ruta_archivo):
    # Carga el dataset desde un archivo CSV
    try:
        df = pd.read_csv(ruta_archivo)
        print(f"Dataset cargado correctamente. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ruta_archivo}")
        return None
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None

def eliminar_duplicados(df):
    # Elimina filas duplicadas del dataset
    filas_antes = df.shape[0]
    df = df.drop_duplicates()
    filas_despues = df.shape[0]
    duplicados_eliminados = filas_antes - filas_despues
    
    print(f"Duplicados eliminados: {duplicados_eliminados}")
    return df

def manejar_valores_nulos(df):
    # Maneja valores nulos según el tipo de columna
    print("\n--- Manejo de Valores Nulos ---")
    valores_nulos_antes = df.isnull().sum().sum()
    print(f"Valores nulos totales antes: {valores_nulos_antes}")
    
    for columna in df.columns:
        nulos = df[columna].isnull().sum()
        if nulos > 0:
            if df[columna].dtype in ['float64', 'int64']:
                # Para numéricas: reemplazar con mediana
                df[columna].fillna(df[columna].median(), inplace=True)
                print(f"  {columna} (numérica): {nulos} nulos reemplazados con mediana")
            else:
                # Para categóricas: reemplazar con moda
                moda = df[columna].mode()[0] if not df[columna].mode().empty else 'Desconocido'
                df[columna].fillna(moda, inplace=True)
                print(f"  {columna} (categórica): {nulos} nulos reemplazados con moda")
    
    valores_nulos_despues = df.isnull().sum().sum()
    print(f"Valores nulos totales después: {valores_nulos_despues}")
    
    return df

def normalizar_numericas(df):
    # Normaliza las columnas numéricas usando StandardScaler
    columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(columnas_numericas) > 0:
        scaler = StandardScaler()
        df[columnas_numericas] = scaler.fit_transform(df[columnas_numericas])
        print(f"Columnas numéricas normalizadas: {list(columnas_numericas)}")
    else:
        print("No se encontraron columnas numéricas para normalizar")
    
    return df

def codificar_categoricas(df):
    # Codifica variables categóricas usando LabelEncoder
    columnas_categoricas = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    if len(columnas_categoricas) > 0:
        for columna in columnas_categoricas:
            le = LabelEncoder()
            df[columna] = le.fit_transform(df[columna].astype(str))
            label_encoders[columna] = le
            print(f"  {columna} codificada con LabelEncoder")
        
        print(f"Columnas categóricas codificadas: {list(columnas_categoricas)}")
    else:
        print("No se encontraron columnas categóricas para codificar")
    
    return df, label_encoders

def preprocesamiento_completo(ruta_archivo):
    # Función principal que ejecuta el pipeline completo de preprocesamiento
    print("=== INICIANDO PREPROCESAMIENTO COMPLETO ===")
    
    # 1. Cargar datos
    df = cargar_datos(ruta_archivo)
    if df is None:
        return None, None
    
    print(f"Dataset original: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # 2. Eliminar duplicados
    df = eliminar_duplicados(df)
    
    # 3. Manejar valores nulos
    df = manejar_valores_nulos(df)
    
    # 4. Normalizar numéricas
    df = normalizar_numericas(df)
    
    # 5. Codificar categóricas
    df, encoders = codificar_categoricas(df)
    
    print("\n=== PREPROCESAMIENTO COMPLETADO ===")
    print(f"Dataset final: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Tipos de datos finales:\n{df.dtypes}")
    
    return df, encoders


