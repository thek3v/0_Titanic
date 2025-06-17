# 0_Titanic
Este proyecto utiliza técnicas de aprendizaje automático para predecir qué pasajeros del Titanic tenían más probabilidades de sobrevivir, basado en características como la edad, sexo, clase, número de familiares a bordo, entre otros.
# Predicción de Supervivencia en el Titanic con Machine Learning

Este proyecto aplica técnicas de machine learning para predecir qué pasajeros sobrevivieron al hundimiento del Titanic. Se han implementado múltiples modelos, análisis comparativo y optimización para encontrar el mejor clasificador.

---

## Estructura del proyecto

- `notebooks/`
  - `0_titanic.ipynb` → Notebook principal con todo el flujo de trabajo (EDA, preprocesado, modelado, evaluación)
- `data/`
  - `train.csv` → Datos de entrenamiento
  - `test.csv` → Datos de test sin etiquetas
- `resultados/`
  - `submission_rf.csv` → Predicción generada con el modelo Random Forest optimizado
  - `submission_histboost.csv` → Predicción generada con el modelo HistGradientBoosting

---

## Objetivo del proyecto

Predecir la columna `Survived` (1 = sobrevivió, 0 = no sobrevivió) usando variables como clase, edad, sexo, tarifa, embarque, etc.

---

## Pasos seguidos

### 🔹 1. Limpieza y preprocesamiento
- Imputación de `Age` con la mediana por `Sex` y `Pclass`
- Imputación de `Fare` y `Embarked`
- Codificación de variables categóricas (`Sex`, `Embarked`, `Title`, etc.)
- Creación de nuevas features (`FamilySize`, `Title`, `IsAlone`)

### 🔹 2. Modelado
- Modelos evaluados:
  - `Logistic Regression`
  - `Random Forest`
  - `XGBoost`
  - `HistGradientBoostingClassifier`
- Evaluación con `train_test_split`, `cross_val_score` y `cross_val_predict`
- Optimización de hiperparámetros con `GridSearchCV`
- Evaluación por:
  - Accuracy
  - Precision, Recall, F1-score
  - Matriz de confusión
  - Curva ROC + AUC

---

## Resultados

| Modelo                       | Accuracy | Precision | Recall | F1 Score | AUC  |
|-----------------------------|----------|-----------|--------|----------|------|
| Logistic Regression         | 0.8156   | 0.7808    | 0.7703 | 0.7755   | 0.83 |
| Random Forest (GridSearch)  | 0.8324   | 0.8235    | 0.7568 | 0.7887   | 0.90 |
| HistGradientBoosting        | 0.8547   | 0.8429    | 0.7973 | 0.8194   | 0.91 ✅

 **Modelo seleccionado: HistGradientBoostingClassifier (base)**

---

## Curva ROC

Se compararon los mejores modelos visualmente:

- AUC Random Forest: **0.90**
- AUC HistGradientBoosting: **0.91**

---
