# Predicción de Retornos Financieros con Machine Learning

Este proyecto implementa modelos de **Machine Learning** para predecir retornos financieros a partir de datos históricos de precios de activos del data Kaggle, todo con fines educativos.  
El objetivo fue comparar distintos algoritmos de regresión y determinar cuál ofrece el mejor desempeño para la predicción de retornos.

---

## Dataset
- Fuente: Yahoo Finance (descarga de precios históricos).  
- Variables principales:
  - **Date**: Fecha de la observación.
  - **Close**: Precio de cierre.
  - **Returns**: Retornos calculados a partir de los precios.
  - Variables derivadas generadas en el preprocesamiento.

---

## Flujo del Proyecto
1. **EDA (Análisis Exploratorio)**
   - Distribución de los retornos y detección de outliers.
   - Correlación entre variables financieras.
   - Gráficas de series temporales y dispersión.

2. **Preprocesamiento**
   - Tratamiento de valores nulos.
   - Normalización y escalado de variables.
   - Creación de variables derivadas para enriquecer el modelo.

3. **Modelado**
   - Modelos lineales: `LinearRegression`, `ElasticNet`, `HuberRegressor`.
   - Modelos no lineales: `RandomForestRegressor`, `CatBoostRegressor`, `XGBRegressor`.

4. **Evaluación**
   - Métricas utilizadas: `MAE`, `RMSE`, `R² Score`.
   - Visualización de valores reales vs. predichos.
   - Comparación del rendimiento de cada modelo.

---

## Resultados
- Los modelos lineales mostraron limitaciones importantes para la predicción.  
- Los modelos no lineales presentaron mejor capacidad de ajuste, aunque con resultados aún mejorables:
  - **RandomForestRegressor**
    - MAE: 0.0537
    - RMSE: 0.0663
    - R²: -39.51
  - **CatBoostRegressor**
    - MAE: 0.0550
    - RMSE: 0.0671
    - R²: -41.20  

Aunque los resultados reflejan la dificultad del problema, el proyecto demuestra el flujo completo de **EDA -- Preprocesamiento -- Modelado -- Evaluación**, aplicable a problemas financieros.

---

## 🚀 Tecnologías usadas
- Python 3.x  
- Librerías:  
  - `pandas`, `numpy`  
  - `scikit-learn`  
  - `xgboost`, `catboost`  
  - `matplotlib`, `seaborn`  
  - `shap`

---

## 📂 Estructura del Repositorio

financial-returns-prediction/
│── data/              # datasets
│── notebooks/         # notebooks de análisis
│── src/               # scripts de ML
│── docs/              # reportes y documentación
│── images/            # visualizaciones exportadas
│── README.md
│── requirements.txt


---

## ⚙️ Instalación
```bash
git clone https://github.com/AVALLEJOTORRES/financial-returns-prediction.git
cd financial-returns-prediction
pip install -r requirements.txt

---



