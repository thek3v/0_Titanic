import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def basic_eda(df):
    print(" Info general:")
    display(df.info())
    
    print(f"\n Filas: {df.shape[0]}  |  Columnas: {df.shape[1]}")
    print(f"\n  Columnas duplicadas: {df.duplicated().sum()}")
    
    print("\n Tipos de columnas:")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    other_cols = df.columns.difference(num_cols + cat_cols).tolist()
    
    print(f"Numéricas ({len(num_cols)}): {num_cols}")
    print(f"Categóricas ({len(cat_cols)}): {cat_cols}")
    print(f"Otras ({len(other_cols)}): {other_cols}")
    
    print("\n Descripción estadística:")
    display(df.describe())
    
    print("\n Valores nulos por columna:")
    display(df.isnull().sum())
    
    # Plot numéricas
    if num_cols:
        print("\n Distribución de columnas numéricas:")
        for col in num_cols:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribución de {col}")
            plt.show()
    
    # Plot categóricas
    if cat_cols:
        print("\n Distribución de columnas categóricas:")
        for col in cat_cols:
            plt.figure(figsize=(8, 5))
            sns.countplot(x=df[col])
            plt.title(f"Distribución de {col}")
            plt.xticks(rotation=45)
            plt.show()

