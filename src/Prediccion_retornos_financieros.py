# %% [markdown]
# ## Importar librerias necesarias
# 
# #### En este proyecto se utilizan las siguientes librerías fundamentales de Python para el análisis de datos y Machine Learning:
# 
# * pandas: manipulación y análisis de datos tabulares.
# * numpy: operaciones matemáticas y manejo de arreglos multidimensionales.
# * matplotlib y seaborn: visualización de datos (gráficos estadísticos y exploratorios).

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
##
# Estas librerías constituyen la base para la exploración, limpieza y análisis de los datos financieros que se utilizarán a lo largo del proyecto.
##

# %% [markdown]
# ## Cargar el dataset 
# 
# #### Cargar el dataset **SnP_daily_update.csv**
# 
# Para este análisis se utiliza el dataset S&P 500 Daily Update, correspondiente al sector financiero, que contiene información de precios de mercado bursátil.
# El dataset fue obtenido desde Kaggle y está en formato Excel (data_Snp.xlsx).
# El primer paso es cargar los datos en un DataFrame de pandas, al que llamaremos data. Esto nos permitirá trabajar sobre una copia en memoria sin alterar el archivo original.

# %%
data = pd.read_excel("data_Snp.xlsx")

# %% [markdown]
# ## Inspección inicial del dataset
# 
# Inspección inicial del dataset
# 
# El primer paso tras cargar los datos es realizar una exploración básica del dataset para entender su estructura, el tipo de variables disponibles y la calidad de la información.
# Esto nos permite identificar rápidamente posibles problemas de calidad (valores nulos, duplicados, inconsistencias) y también confirmar que los datos cargados corresponden a lo esperado.
# 
# * data.shape : dimensiones del dataset (número de filas y columnas).
# * data.head() : primeras 5 filas para observar la estructura inicial.
# * data.tail() : últimas 5 filas para verificar consistencia.
# * data.nunique() : número de valores únicos por columna (útil para categóricas).
# * data.isnull().sum() : cantidad de valores nulos por columna.
# * data.info() : tipos de datos, conteo de valores no nulos y memoria usada.
# * data.describe() : estadísticas descriptivas de las variables numéricas (media, mediana, cuartiles, mínimos, máximos, desviación estándar).
# 

# %%
## =========================
# Visualización inicial del data
## =========================
data

# %%
## =========================
# Dimensiones del data
## =========================
data.shape

# %%
## =========================
# Visualización de datos unicos
## =========================
data
data.nunique()

# %%
## =========================
# Tipos de datos
## =========================
data
data.info()

# %%
## =========================
# Estadisticas descriptivas
## =========================
data
data.describe()

# %%
## =========================
# Visualización inicial de las primeras 10 filas
## =========================
data
data.head(10)

# %%
## =========================
# Visualización inicial de las ultimas filas
## =========================
data.tail(10)

# %% [markdown]
# ## Preprocesamiento

# %% [markdown]
# Para este ejercicio se seleccionó la acción de Abbott Laboratories (ticker: ABT) dentro del índice S&P 500.
# La elección se realiza únicamente con fines académicos, ya que se trata de una empresa del sector salud con datos históricos consistentes y de interés para análisis de inversión.
# 
# Se filtra el dataset para trabajar únicamente con dos columnas relevantes:
# 
# * Date : fecha de la observación.
# * ABT : precio ajustado de la acción de Abbott.

# %%
## =========================
# Filtramos el dataset para ABT
## =========================
abt = data[["Date","ABT"]].copy()
## =========================
# Visualizamos las primeras 15 filas
## =========================
abt.head(15)

# %% [markdown]
# #### Transformación de fecha
# 
# Un paso esencial en el análisis de series temporales es asegurarse de que la columna de fechas esté en el formato adecuado (datetime) y en orden cronológico.
# Esto facilita la creación de gráficos de evolución y el entrenamiento de modelos que dependen de la secuencia temporal.

# %%
# ==============================
# Transformar "Date" a datetime
# ==============================
abt["Date"] = pd.to_datetime(abt["Date"])

# Ordenar cronológicamente y resetear el índice
abt = abt.sort_values("Date")
abt = abt.reset_index(drop=True)

# Confirmamos la transformación
abt

# %% [markdown]
# ## Creación de nuevas características
# 
# En el análisis de series temporales, una práctica esencial es generar variables derivadas que ayuden a los modelos a capturar patrones históricos y a identificar tendencias en los precios.
# Para este proyecto se construyeron dos tipos principales de variables: rezagos (lags) y medias móviles (SMA/EMA).

# %% [markdown]
# #### Creación de rezagos
# #### Objetivo: Los rezagos representan valores pasados del precio, desplazados cierto número de días. Estas variables permiten a los modelos aprender la dependencia temporal, ya que el precio de un día está influenciado por los valores previos.
# 

# %%
# Crear rezagos de 1, 5, 15, 25, 30, 60 y 90 días
abt["lag_1"] = abt["ABT"].shift(1)
abt["lag_5"] = abt["ABT"].shift(5)
abt["lag_15"] = abt["ABT"].shift(15)
abt["lag_25"] = abt["ABT"].shift(25)
abt["lag_30"] = abt["ABT"].shift(30)
abt["lag_60"] = abt["ABT"].shift(60)
abt["lag_90"] = abt["ABT"].shift(90)
##
# Los rezagos capturan el "efecto memoria" de la serie, muy útil para algoritmos de regresión y modelos no lineales.
##

# %% [markdown]
# #### Creación de medias móviles
# #### Objetivo: Las medias móviles suavizan el comportamiento de la serie, reduciendo el ruido y permitiendo identificar la tendencia subyacente. Se construyeron tanto medias móviles simples (SMA) como una exponencial (EMA):
# * SMA (Simple Moving Average): promedios de los últimos 10, 20, 30, 50 y 200 días.
# * EMA (Exponential Moving Average): media móvil ponderada, que otorga mayor importancia a los datos recientes.

# %%
# Media móvil exponencial (EMA)
abt["EMA_20"] = abt["ABT"].ewm(span=20, adjust=False).mean()

# Medias móviles simples (SMA)
abt["SMA_10"]  = abt["ABT"].rolling(window=10).mean().shift(1)
abt["SMA_20"]  = abt["ABT"].rolling(window=20).mean().shift(1)
abt["SMA_30"]  = abt["ABT"].rolling(window=30).mean().shift(1)
abt["SMA_50"]  = abt["ABT"].rolling(window=50).mean().shift(1)
abt["SMA_200"] = abt["ABT"].rolling(window=200).mean().shift(1)
##
# Las medias móviles son indicadores técnicos ampliamente usados en finanzas, ya que ayudan a detectar puntos de entrada/salida y tendencias alcistas o bajistas.
##

# %% [markdown]
# #### Creación de indicadores financieros
# 
# Además de los precios, es necesario generar indicadores que reflejen la dinámica del activo, su nivel de riesgo y las tendencias que siguen los inversores.
# Para este análisis se construyeron tres tipos de variables adicionales: retornos, volatilidad y momentum.

# %%
## =================
# Retornos
## =================
abt["retorno_diario"] = abt["ABT"].pct_change().shift(1)
abt["retorno_5d"] = abt["ABT"].pct_change(5).shift(1)
abt["retorno_20d"] = abt["ABT"].pct_change(20).shift(1)
abt["retorno_30d"] = abt["ABT"].pct_change(30).shift(1)
abt["retorno_200d"] = abt["ABT"].pct_change(200).shift(1)

## =========================
# iterpretación:
## =========================
# * retorno_diario: indica cuánto subió o bajó la acción en un día.
# * retorno_5d (o mayor): mide la variación acumulada en intervalos más largos, lo cual ayuda a detectar tendencias de mediano y largo plazo.

# %%
## =================
# Volatilidad
## =================
# La volatilidad mide la variabilidad de los retornos en un periodo de tiempo y es utilizada como indicador de riesgo en los mercados financieros.
# Se calculó la desviación estándar de los retornos en una ventana móvil de 20 días:

# Volatilidad (desviación estándar de los retornos en ventana de 20 días)

abt["volatilidad_20"] = abt["retorno_diario"].rolling(window=20).std().shift(1)

## =================
# Momentum
## =================
# El momentum evalúa la fuerza de la tendencia comparando el precio actual con el de varios días atrás.
# Este indicador se usa comúnmente para determinar si una acción mantiene un impulso alcista o bajista.

abt["momentum_10"] = (abt["ABT"] / abt["ABT"].shift(10) - 1).shift(1)

# %% [markdown]
# ## Creación de indicadores técnicos
# Además de los rezagos, medias móviles y retornos, se incorporaron indicadores técnicos ampliamente utilizados en el análisis de mercados financieros. Estos indicadores proporcionan señales de tendencia, volatilidad y fortaleza del mercado, útiles tanto para inversores como para modelos predictivos.

# %% [markdown]
# #### Bandas de Bollinger
# Las Bandas de Bollinger permiten medir la volatilidad alrededor de una media móvil.
# * BB_medio: media móvil de 20 días.
# * BB_sup: dos desviaciones estándar por encima de la media (zona de sobrecompra).
# * BB_inf: dos desviaciones estándar por debajo (zona de sobreventa).

# %%
abt["BB_medio"] = abt["ABT"].rolling(window=20).mean().shift(1)
abt["BB_sup"] = (abt["BB_medio"] + 2*abt["ABT"].rolling(window=20).std()).shift(1)
abt["BB_inf"] = (abt["BB_medio"] - 2*abt["ABT"].rolling(window=20).std()).shift(1)
##
# Este indicador combina tendencia y volatilidad, mostrando cuándo el precio se aleja demasiado de su media.
##

# %% [markdown]
# ##### RSI (Relative Strength Index)
# El RSI mide la velocidad y el cambio de los movimientos de precio para identificar si un activo está sobrecomprado (>70) o sobrevendido (<30).

# %%
delta = abt["ABT"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
abt["RSI_14"] = (100 - (100 / (1 + rs))).shift(1)

# %% [markdown]
# ##### MACD (Moving Average Convergence Divergence)
# El MACD es un indicador de tendencia que compara dos medias móviles exponenciales (12 y 26 días) y genera una señal al cruzarse con una tercera (9 días).

# %%
ema12 = abt["ABT"].ewm(span=12, adjust=False).mean()
ema26 = abt["ABT"].ewm(span=26, adjust=False).mean()

abt["MACD"] = (ema12 - ema26).shift(1)
abt["Signal"] = abt["MACD"].ewm(span=9, adjust=False).mean().shift(1)

##
# El cruce entre MACD y Signal se interpreta como confirmación de una tendencia alcista o bajista.
##

# %% [markdown]
# #### Definición de la variable objetivo
# Para entrenar los modelos de Machine Learning se define como variable objetivo el retorno a 20 días.
# Esto permite que los modelos aprendan a predecir la variación futura del precio de Abbott, en lugar del valor absoluto del precio.

# %%
abt["target_retorno_20d"] = abt["ABT"].pct_change(20).shift(-20)
## Con este diseño, el objetivo del modelo es anticipar si la acción de Abbott generará un retorno positivo o negativo en el horizonte de 20 días.

# %% [markdown]
# #### Visualización del dataset final con nuevas variables
# Tras la creación de todas las variables derivadas (rezagos, medias móviles, retornos, volatilidad, indicadores técnicos), el dataset abt contiene ahora múltiples columnas listas para el modelado:

# %%
## ==============================
# Visualización de las primeras filas con todas las columnas
## ==============================
abt.head(20)

# Este paso permite verificar que todas las variables fueron generadas correctamente y que están alineadas con el índice temporal.

# %% [markdown]
# ##### Eliminación de valores nulos (NaN) 
# La creación de nuevas características basadas en ventanas de tiempo (rolling, shift, etc.) genera automáticamente valores nulos en los primeros registros, ya que no existen suficientes datos previos para calcularlos.
# Para evitar problemas en el entrenamiento de los modelos, se eliminan todas las filas que contienen al menos un valor nulo:

# %%
# Eliminar todas las filas con al menos un NaN
abt = abt.dropna()

# Confirmar que ya no hay valores faltantes
abt.head(20)

##
# Este procedimiento asegura que todos los registros utilizados por los modelos estén completos y que no haya inconsistencias en las variables predictoras.
##

# %%
## =========================
# Verificación del dataset final
## =========================
abt.info()

# %% [markdown]
# #### Detencion de Valores nulos
# Antes de entrenar los modelos es importante verificar la calidad de los datos y asegurarse de que no existan valores faltantes.
# 

# %%
# Identificar columnas con valores nulos
col_con_nulos = abt.columns[abt.isnull().any()]
col_con_nulos

# %%
# Contar el número de nulos por columna
total_nulos_col = abt[col_con_nulos].isnull().sum()
total_nulos_col

# %%
# Calcular porcentaje de nulos en el dataset
porcentaje_nulos = abt.isnull().median() * 100
porcentaje_nulos

# %%
## =========================
# Resultados
## =========================
# No se detectaron columnas con valores nulos (Index([], dtype='object')).
# El conteo de valores faltantes es cero en todas las columnas.
# El pocentaje de nulos es 0% en todas las variables.

# %% [markdown]
# ## Visualizaciones Exploratorias (EDA)

# %% [markdown]
# #### Graficos de Distribuciones y detección de outliers
# 
# Un paso fundamental en el análisis exploratorio es evaluar cómo se distribuyen las variables y si existen valores atípicos (outliers) que puedan afectar el desempeño de los modelos.
# Para esto se creó una función que, para cualquier variable seleccionada:
# * Genera un histograma con curva de densidad para visualizar la distribución de los valores.
# * Genera un boxplot para detectar valores extremos.

# %%
# Creamos una funcion para graficar Distribucion de los datos y un Boxplot para detectar posibles Outliers

def graficos_1(df,col):
    plt.figure(figsize=(17,8))
    # Grafico de distribucion
    plt.subplot(1,2,1)
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribuccion de la variable {col}")
    plt.xlabel(f"{col}")
    plt.ylabel("Frecuencia")
    plt.grid()
    
    # Grafico boxplot para deteccion de outliers
    plt.subplot(1,2,2)
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot de la variable {col} para detenccion de Outliers")
    plt.xlabel(f"{col}")
    plt.tight_layout()
    plt.show()

# %%
# Aplicamos la funcion creada anteriormente
graficos_1(abt,"ABT")

# Kurtosis
kur = abt["ABT"].kurt()
kur # Curtosis con un valor de -1.4041419490658114
print(f"Curtosis: {kur}")

# Skew
sesgo = abt["ABT"].skew()
sesgo # 0.33987852019923925
print(f"Sesgo: {sesgo}")

## ==================================
# Análisis de distribución de la variable ABT
## ==================================
## Se generaron dos gráficos principales para analizar la variable ABT (precio ajustado de Abbott):
# * Histograma con curva de densidad (KDE) : permite observar cómo se distribuyen los precios.
# * Boxplot : facilita la detección de posibles valores atípicos.

## Resultados observados:
# * El histograma muestra una distribución asimétrica a la derecha, con mayor concentración de precios en rangos bajos (entre 20 y 60 USD), pero con presencia de valores más altos (hasta 140 USD).
# * El boxplot confirma que no existen outliers extremos significativos, ya que los precios se mantienen dentro de los límites de las bandas del gráfico.

# * Kurtosis = -1.40 : distribución platicúrtica, con colas más ligeras que la distribución normal. Esto significa que hay menos valores extremos de lo esperado.
# * Sesgo (Skew) = 0.33 : distribución con ligero sesgo positivo, es decir, la mayoría de los precios se concentran en valores bajos, pero existen algunos precios relativamente más altos que alargan la cola hacia la derecha.

# %%
graficos_1(abt,"retorno_diario")
# Kurtosis 
kur = abt["retorno_diario"].kurt()
kur # 6.992968614161424
print(f"Curtosis: {kur}")
# 
# Skew
sesgo = abt["retorno_diario"].skew()
sesgo # -0.19681772696453598
print(f"Sesgo: {sesgo}")

## ==================================
# Análisis de distribución de la variable "retorno_diario"
## ==================================

#* El histograma refleja una distribución con forma de campana, centrada en 0.
#* El boxplot muestra outliers frecuentes, tanto positivos como negativos, lo cual es típico en datos financieros por la volatilidad de los mercados.

#* Kurtosis = 6.99 : distribución leptocúrtica, con colas mucho más pesadas que una distribución normal.
#* Implica que existen eventos extremos (grandes variaciones diarias), aunque poco frecuentes.
#* Sesgo (Skew) = -0.19 : ligera asimetría negativa, lo que sugiere que los retornos negativos (pérdidas) tienden a ser un poco más frecuentes o pronunciados que los positivos.

#* Los retornos diarios no siguen una distribución normal, sino que presentan colas largas y outliers frecuentes, lo cual es consistente con la naturaleza de las series financieras.
#* Este comportamiento debe considerarse al momento de modelar, ya que algunos algoritmos lineales pueden verse afectados por la presencia de valores extremos.
#* La ligera asimetría negativa sugiere un sesgo hacia caídas más abruptas que los aumentos, lo cual es común en mercados financieros.

# %%
graficos_1(abt,"retorno_20d")

# Kurtosis
kur = abt["retorno_20d"].kurt()
kur # 3.698683102033043
print(f"Curtosis: {kur}")

# Skew
sesgo = abt["retorno_20d"].skew()
sesgo # 0.3443728526261572
print(f"Sesgo: {sesgo}")

## ==================================
# Análisis de distribución de la variable "retorno_20d"
## =================

#* El histograma muestra que la mayoría de los retornos a 20 días están concentrados cerca de 0, aunque se observan eventos con valores positivos y negativos más pronunciado
#* El boxplot revela la presencia de outliers en ambos extremos (ganancias y pérdidas muy fuertes), lo cual es consistente con la naturaleza de los mercados financieros.

#* Kurtosis = 3.69 : distribución leptocúrtica, con colas más pesadas que una distribución normal. Esto indica que existen eventos extremos de forma recurrente.
#* Sesgo (Skew) = 0.34 : ligera asimetría positiva, lo que significa que los retornos positivos tienden a ser más frecuentes o algo más extremos que los negativos.

#* La variable presenta una distribución concentrada en torno a 0, pero con colas largas (típico en retornos financieros).
#* Los outliers detectados representan cambios de precio poco frecuentes pero muy relevantes, que pueden afectar la predicción.
#* La asimetría positiva sugiere que, a 20 días, existen episodios en los que los rendimientos positivos son más marcados que las caídas.


# %%
graficos_1(abt,"retorno_200d")
# Kurtosis
kur = abt["retorno_200d"].kurt()
kur # 0.014583202695658048
print(f"Curtosis: {kur}")

# Skew
sesgo = abt["retorno_200d"].skew()
sesgo # -0.2593912709996365
print(f"Sesgo: {sesgo}")

## ==================================
# Análisis de distribución de la variable "retorno_200d"
## ==================================

#* El histograma muestra que la mayoría de los valores se concentran entre -0.2 y 0.4, con un pico alrededor del 0.1.
#* El boxplot evidencia algunos outliers positivos y negativos, aunque en menor cantidad respecto a horizontes más cortos como retorno_diario.

#* Kurtosis = 0.01 : la distribución es mesocúrtica, muy cercana a la normal. Esto significa que los eventos extremos no son tan frecuentes como en los retornos diarios.
#* Sesgo (Skew) = -0.26 : ligera asimetría negativa, lo cual indica que las caídas prolongadas tienden a ser algo más pronunciadas que las subidas en el largo plazo.

#* El retorno a 200 días muestra una distribución bastante estable y cercana a la normalidad, en comparación con retornos de corto plazo.
#* La presencia de algunos outliers es normal en periodos largos debido a shocks del mercado.
#* La ligera asimetría negativa indica que, en el largo plazo, las caídas suelen tener un mayor impacto que las subidas, lo cual es relevante para la gestión de riesgo en inversiones.

# %%
graficos_1(abt,"volatilidad_20")
# Kurtosis
kur = abt["volatilidad_20"].kurt()
kur # 21.495322632274927
print(f"Curtosis: {kur}")
# 
# Skew
sesgo = abt["volatilidad_20"].skew()
sesgo # 3.332938340141451
print(f"Sesgo: {sesgo}")

## ==================================
# Análisis de distribución de la variable volatilidad_20
## ==================================

#* La mayor parte de los valores se concentran en niveles bajos (entre 0.005 y 0.02).
#* El boxplot muestra numerosos outliers positivos, correspondientes a periodos de alta inestabilidad del mercado.
#* La distribución es marcadamente asimétrica, lo cual es consistente con cómo se comporta la volatilidad en mercados financieros (periodos prolongados de calma con picos abruptos de riesgo).

#* Kurtosis = 21.49 : distribución altamente leptocúrtica, con colas muy pesadas. Esto confirma la existencia de eventos extremos de volatilidad.
#* Sesgo (Skew) = 3.33 : fuerte asimetría positiva, lo que indica que en raras ocasiones se presentan picos de volatilidad muy superiores a la media.

#* La volatilidad de corto plazo suele ser baja y estable en la mayoría de los periodos, pero presenta eventos extremos significativos en situaciones de crisis o incertidumbre del mercado.
#* Esta variable es crucial en modelos predictivos financieros, ya que permite anticipar escenarios de alto riesgo.
#* Su naturaleza no normal y con outliers justifica el uso de modelos robustos y no lineales (Random Forest, XGBoost, CatBoost) que puedan manejar distribuciones con colas pesadas.

# %% [markdown]
# ##### Gráficos de evolución en el tiempo
# Se creó una función llamada serie_temporal() que permite visualizar la evolución de cualquier variable a lo largo del tiempo, usando como eje x la fecha y como eje y el valor de la variable expresado en porcentaje

# %%
# Creamos una funcion para aplicar estos graficos

def serie_temporal(df, col_fecha, col_valor):
    
    prom = df[col_valor].mean()

    plt.figure(figsize=(18,9))
    plt.plot(df[col_fecha], df[col_valor]*100, label=f"Retorno: {col_valor} (%)")
    plt.title(f"Evolución del retorno acumulado a {col_valor}")
    plt.xlabel("Fecha")
    plt.ylabel(f"{col_valor} (%)")
    plt.axhline(y=prom*100, color="red", linestyle="--", label=f"Promedio: {prom*100:.2f}%")
    plt.legend()
    plt.grid(True)
    plt.show()

# %%
serie_temporal(abt,"Date","retorno_diario")

## ==================================
# Análisis evolucion temporal de la variable "retorno_diario"
## ==================================

#* La serie presenta retornos diarios que oscilan en torno a 0%, lo que es típico en activos financieros.
#* La línea roja discontinua representa el promedio histórico del retorno diario: 0.06%, confirmando que en el largo plazo los rendimientos diarios tienden a ser casi nulos.
#* Se observan picos extremos, tanto positivos como negativos, que corresponden a eventos puntuales de alta volatilidad (crisis financieras, anuncios de resultados, noticias de mercado, etc.)

#* La serie muestra un patrón volátil y errático, sin tendencia clara a largo plazo, coherente con la naturaleza de los retornos diarios.
#* Los retornos negativos suelen ser más abruptos que los positivos (caídas rápidas), confirmando lo visto en el análisis de sesgo y kurtosis.
#* El promedio cercano a cero indica que, si bien hay movimientos diarios importantes, estos se compensan en el tiempo.

# %%
serie_temporal(abt,"Date","retorno_5d")
## ==================================
# Análisis evolucion temporal de la variable "retorno_5d"
## ==================================

#* La serie muestra mayor estabilidad en comparación con retorno_diario, aunque sigue presentando oscilaciones frecuentes.
#* La línea roja discontinua indica un retorno promedio de 0.30%, lo que refleja un ligero sesgo positivo en ventanas cortas de 5 días.
#* Se observan picos extremos en momentos puntuales, especialmente alrededor de 2020, donde la volatilidad fue mayor (coincidiendo con la pandemia y shocks de mercado).

#* retorno_5d es un indicador útil para capturar la dinámica de corto plazo con menor ruido que los retornos diarios.
#* Muestra un promedio positivo, lo que indica que en horizontes de una semana, el activo tiende a generar valor.
#* Sin embargo, la volatilidad sigue siendo significativa, lo que reafirma la necesidad de usar indicadores adicionales (volatilidad, momentum, medias móviles) para robustecer las predicciones.

# %%
# aplicamos la funcion para ver el retorno para 30 dias
serie_temporal(abt,"Date","retorno_30d")
## ==================================
# Análisis evolucion temporal de la variable "retorno_30d"
## ==================================

#* La serie muestra fluctuaciones amplias en horizontes mensuales, con valores que en ocasiones alcanzan ±30%.
#* La línea roja discontinua corresponde al retorno promedio de 1.84%, lo que indica una tendencia positiva en este plazo.
#* Los episodios más extremos se observan en torno a 2020, con retornos superiores al 40% y caídas cercanas al -30%, reflejando eventos de fuerte volatilidad global.

# A diferencia de los retornos diarios y de 5 días, los retornos a 30 días suavizan parte del ruido, permitiendo identificar ciclos de ganancia y pérdida más claros.
# El promedio positivo sugiere que mantener la inversión por al menos un mes tiende a ser rentable en el largo plazo.
# Sin embargo, los outliers extremos evidencian que aún existen riesgos significativos asociados a choques de mercado.

# %%
serie_temporal(abt,"Date","retorno_200d")
## ==================================
# Análisis evolucion temporal de la variable "retorno_200d"
## ==================================

#* La serie presenta ciclos prolongados de subidas y bajadas, lo que refleja la naturaleza de los retornos de largo plazo.
#* El retorno promedio es de 12.01%, lo que sugiere que en periodos anuales Abbott ha tendido a generar una rentabilidad positiva.
#* Se observan episodios de fuertes caídas, como en 2020, donde los retornos llegaron a estar por debajo de -20%, pero también picos positivos cercanos al 80%, que muestran fases de crecimiento sostenido.

#* retorno_200d es una variable fundamental para el análisis de inversión a largo plazo.
#* Refuerza la idea de que Abbott, como activo, ha tenido un desempeño globalmente positivo, aunque expuesto a shocks de mercado.
#* Es especialmente valioso en modelos predictivos porque captura la tendencia de fondo, reduciendo la volatilidad extrema presente en horizontes cortos.


# %%
serie_temporal(abt,"Date","volatilidad_20")
## ==================================
# Análisis evolucion temporal de la variable "volatilidad_20"
## ==================================

# La variable nombre_variable resulta clave para [análisis de corto/mediano/largo plazo]. 
# Confirma que el activo [ejemplo: Abbott] ha mostrado [rendimiento positivo/negativo/mixto] a lo largo del tiempo. 
# Es relevante para modelos predictivos porque [reduce ruido, captura tendencia, mide riesgo].

# %% [markdown]
# #### Grafico Precio con medias móviles (SMA y EMA)

# %%
plt.figure(figsize=(17,9))
plt.plot(abt["Date"], abt["ABT"], label="Precio ABT", color="blue")
plt.plot(abt["Date"], abt["SMA_50"], label="SMA 50", color="orange")
plt.plot(abt["Date"], abt["SMA_200"], label="SMA 200", color="red")
plt.title("Precio con medias móviles (SMA y EMA)")
plt.xlabel("Fecha")
plt.ylabel("Precio (USD)")
plt.legend()
plt.grid()
plt.show()

## ==================================
# Análisis media móviles (SMA y EMA)
## ==================================
#* El precio de Abbott (ABT) muestra una clara tendencia alcista de largo plazo, pasando de valores cercanos a 20 USD en 2010 a más de 130 USD en 2025.
#* La SMA 50 (línea naranja) reacciona con mayor rapidez a los cambios de tendencia, mientras que la SMA 200 (línea roja) actúa como un indicador más estable, reflejando la tendencia principal.
#* Se observan cruces de medias móviles: cuando la SMA 50 cruza por encima de la SMA 200 (golden cross), se anticipan fases de crecimiento; cuando ocurre lo contrario (death cross), se presentan caídas o correcciones.

# El uso combinado de SMA 50 y SMA 200 permite identificar momentos clave de compra/venta y validar la solidez de una tendencia.
# En el caso de Abbott, la tendencia alcista de largo plazo se mantiene, pero el análisis de los cruces de medias móviles resulta esencial para gestionar riesgos en periodos de alta volatilidad.

# %% [markdown]
# #### Grafico Análisis RSI (Relative Strength Index)

# %%
plt.figure(figsize=(18,9))
plt.plot(abt["Date"], abt["RSI_14"])
plt.axhline(70, color="red", linestyle="--")
plt.axhline(30, color="green", linestyle="--")
plt.title("RSI (14 días) de ABT")
plt.xlabel("Fecha")
plt.ylabel("RSI")
plt.show()

## ==================================
# Análisis RSI
## ==================================
#* El RSI oscila frecuentemente entre los valores de 30 y 70, lo que indica que la acción suele mantenerse en una zona neutral gran parte del tiempo.
#* Se identifican episodios en los que el RSI supera 70 (sobrecompra) y cae por debajo de 30 (sobreventa), lo que marca señales potenciales de corrección o rebote.
#* La volatilidad en el RSI refleja que Abbott ha tenido múltiples fases de presión compradora y vendedora a lo largo del periodo analizado.

#* El RSI es un indicador esencial para analizar momentos de sobrecompra y sobreventa en Abbott.
#* Confirma la importancia de usarlo como complemento a otros indicadores (medias móviles, volatilidad, retornos acumulados), ya que su sensibilidad puede generar falsos positivos.
#* Su incorporación al dataset enriquece el modelo predictivo porque añade una perspectiva de momentum en el corto plazo.

# %% [markdown]
# #### Análisis de Tendencia con el Indicador MACD

# %%
plt.figure(figsize=(18,9))
plt.plot(abt["Date"], abt["MACD"], label="MACD", color="blue")
plt.plot(abt["Date"], abt["Signal"], label="Signal", color="red")
plt.title("MACD vs Señal - ABT")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.legend()
plt.grid()
plt.show()

## ==================================
# Análisis MACD
## ==================================

#* La línea MACD (azul) oscila alrededor de 0, reflejando los cambios de tendencia en el precio de ABT.
#* La línea de señal (roja) suaviza el MACD y sirve como referencia para detectar cruces.
#* Se identifican varios episodios en los que la línea azul (MACD) cruza hacia arriba la línea roja (Señal), lo que suele asociarse con señales de compra.
#* En cambio, los cruces hacia abajo (MACD por debajo de Señal) son frecuentes en períodos de correcciones o caídas, indicando posibles ventas.
#* Los movimientos más amplios y volátiles se concentran en años de crisis o alta volatilidad (ej. 2020).

#* El MACD de ABT confirma que el activo presenta fases claras de impulso alcista y bajista, siendo un indicador valioso para estrategias de entrada y salida en el mercado.
#* Su utilidad principal radica en reforzar la detección de cambios de tendencia junto con otros indicadores (ej. RSI y medias móviles).
#* Para un modelo predictivo, el MACD aporta información complementaria sobre la fuerza y dirección de la tendencia, aumentando la robustez del análisis técnico.


# %% [markdown]
# ####  Gráfico de Bandas de Bollinger (Volatilidad y Tendencia)

# %%
plt.figure(figsize=(18,9))
plt.plot(abt["Date"], abt["ABT"], label="Precio ABT", color="blue")
plt.plot(abt["Date"], abt["BB_sup"], label="Banda Superior", color="red", linestyle="--")
plt.plot(abt["Date"], abt["BB_inf"], label="Banda Inferior", color="green", linestyle="--")
plt.plot(abt["Date"], abt["BB_medio"], label="Media", color="orange")
plt.fill_between(abt["Date"], abt["BB_inf"], abt["BB_sup"], color="gray", alpha=0.2)
plt.title("Bandas de Bollinger - ABT")
plt.xlabel("Fecha")
plt.ylabel("Precio (USD)")
plt.legend()
plt.grid()
plt.show()

## ==================================
# Análisis Bandas de Bollinger
## ==================================

#* El precio de ABT (azul) se mantiene en la mayoría del tiempo dentro de las bandas superior e inferior, lo cual refleja que las Bandas de Bollinger funcionan como un rango natural de volatilidad.
#* En los periodos de 2020 y 2022 se observan episodios donde el precio toca la banda inferior, asociados a caídas bruscas.
#* En las fases alcistas (2017-2019 y 2020-2021) el precio se acerca con frecuencia a la banda superior, indicando presión compradora.
#* La banda media (línea naranja, SMA20) actúa como nivel de soporte/resistencia dinámico.

#* Las Bandas de Bollinger son un indicador clave para complementar el análisis de volatilidad y tendencias.
#* En el caso de Abbott (ABT), muestran un activo con fases claras de sobrecompra/sobreventa, pero en general dentro de un marco de crecimiento sostenido.
#* Este indicador es útil para modelos predictivos porque captura la dinámica de volatilidad, lo que permite anticipar posibles rompimientos o consolidaciones.

# %%
# Exportamos el data para BI
abt.to_csv("data_snp_paraBI.csv", index=False)

# %% [markdown]
# ## Verificación de variables categóricas y numéricas
# En este paso se revisa el tipo de variables dentro del dataset con el fin de identificar cuáles requieren un tratamiento adicional antes de aplicar algoritmos de Machine Learning.

# %%
variables_categoricas = abt.select_dtypes(include=["object"]).columns
variables_categoricas 

## =========================
# Resultado
## =========================
# No se encontraron variables categóricas.

# %%
variables_numericas = abt.select_dtypes(include=np.number)
variables_numericas

## =========================
# Resultado
## =========================
# Todas las columnas son numéricas, principalmente de tipo float64.

# %% [markdown]
# ## Tratamiento de Valores Atípicos
# ### No serán tratados ni eliminados.
# En el contexto financiero y bursátil, los valores atípicos suelen representar información valiosa y no simples errores de medición. Su eliminación podría distorsionar los resultados del análisis.
# 
# * Medición de riesgo: Los outliers reflejan movimientos bruscos del mercado que son críticos para evaluar volatilidad.
# * Identificación de crisis: Caídas extremas o subidas inusuales suelen marcar eventos de crisis o de euforia en los mercados.
# * Estrategias de trading: Los algoritmos de trading suelen aprovechar esos puntos extremos.
# * Predicción del futuro: Los outliers ayudan a modelar escenarios de estrés y anticipar posibles comportamientos futuros.

# %% [markdown]
# ## Análisis de Correlación y Selección de Variables Importantes
# #### Objetivo:
# * Analizar las correlaciones entre todas las variables numéricas.
# * Identificar las que tienen mayor relación con la variable objetivo (retorno_20d).
# * No se eliminarán variables en esta etapa, únicamente se usará el análisis para ranking de importancia y mejor interpretación del dataset.

# %%
## =========================
# Matriz de correlacion
## =========================
mx_corr = abt.corr()
mx_corr

# %% [markdown]
# #### Selección de Variables Más Importantes según Correlación
# * Se calcularon las 20 variables con mayor correlación (en valor absoluto) respecto a la variable objetivo target_retorno_20d.
# * Estas variables corresponden a indicadores técnicos y series derivadas que muestran relación directa con el comportamiento de los retornos a 20 días.

# %%
## =========================
# Variables mas importantes segun nlargest con la variable objetivo "target_retorno_20d"
## =========================

variables_top_1 = mx_corr["target_retorno_20d"].abs().nlargest(20).index
variables_top_1

filtrado = variables_top_1.tolist()
filtrado

# %%
data_filtrado = abt[variables_top_1]
data_filtrado

# %%
## =================================================
# Eliminar la variable "ABT"
## =================================================

#* En el filtrado inicial, la variable ABT apareció como una de las más correlacionadas con la variable objetivo target_retorno_20d.
#* Sin embargo, ABT corresponde al precio absoluto de la acción y no a un indicador derivado.
#* Mantener esta variable generaría un sesgo, ya que el modelo estaría aprendiendo del precio en sí y no de la dinámica de los retornos.

data_f = data_filtrado.drop(columns=["ABT"])
data_f

#* El dataset final (data_f) queda conformado por 19 variables relevantes (indicadores técnicos y retardos), 
# todas ellas potencialmente útiles para explicar y predecir el target_retorno_20d.

# %% [markdown]
# #### Mapa de Calor - Correlaciones entre Predictores y Target

# %%
# Se crea nuvamente la matriz ya con el data_f
mz_corr = data_f.corr()

# Crear mapa de calor
plt.subplots(figsize=(21,10))
sns.heatmap(mz_corr, annot=True, fmt=".2f", vmax=0.8, linewidths=0.2)
plt.title("Mapa de Calor")

## =================================
# Resultados
## =================================

#* Se generó una matriz de correlación para las 19 variables filtradas.
#* El mapa de calor permite identificar relaciones lineales fuertes o débiles entre predictores y con la variable objetivo target_retorno_20d.
#* Se observan grupos de variables altamente correlacionadas entre sí (por ejemplo, entre medias móviles y bandas de Bollinger).

# %% [markdown]
# ## División de Datos – Definición de Features y Target
# 
# * Se ha definido el conjunto de predictores (X) eliminando la columna objetivo target_retorno_20d.
# * La variable objetivo (y) corresponde a los retornos a 20 días, que será el valor que los modelos intentarán predecir.
# * X contiene 18 variables independientes derivadas de indicadores técnicos y retardos, mientras que y es una serie unidimensional.

# %%
## ===================
# Definir features y target:
## ===================

# Features
X = data_f.drop(columns=["target_retorno_20d"])

# Target 
y = data_f["target_retorno_20d"]

# %%
# Se carga libreria
from sklearn.model_selection import train_test_split

# Se crean las dos variables, una de train y otra de test, y se separa en entrenamiento y prueba con 20 y 80
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# =====================
## Comentarios
# =====================
# test_size=0.2 : 80% entrenamiento, 20% prueba
# random_state=42 : asegura reproducibilidad.
# shuffle=False : muy importante en series temporales, porque no se debe mezclar el orden de los datos (si se pusiera shuffle=True, se perderia la secuencia cronológica).

# %% [markdown]
# ####  Escalado de variables (features)
# * Se aplicó RobustScaler sobre las variables predictoras (X) para los modelos lineales.
# * Para los modelos no lineales (árboles, boosting, random forest, etc.), se conservaron los valores originales sin escalado.
# * El target (y) no se modificó en ninguno de los casos.

# %%
from sklearn.preprocessing import RobustScaler

# Inicializamos el escalador robusto (menos sensible a outliers)
scaler = RobustScaler()

# ---- Modelos lineales (necesitan escalado de X) ----
X_train_lin = scaler.fit_transform(X_train)
X_test_lin  = scaler.transform(X_test)

# ---- Modelos no lineales (se usan X originales sin escalar) ----
X_train_nonlin = X_train.copy()
X_test_nonlin  = X_test.copy()

# ---- El target (y) se mantiene igual en ambos casos ----
y_train_lin    = y_train.copy()
y_test_lin     = y_test.copy()
y_train_nonlin = y_train.copy()
y_test_nonlin  = y_test.copy()


# %% [markdown]
# ## Creacion de modelos NO lineales
# ##### Capturan relaciones complejas entre las variables predictoras y la variable objetivo.
# * No requieren escalado de datos (a diferencia de los modelos lineales).
# * Son robustos frente a outliers y variables de distinta escala.

# %%
## =========================
# Modelos NO LINEALES
## =========================
#* RandomForestRegressor: ensamble basado en múltiples árboles de decisión.
#* CatBoostRegressor: boosting eficiente con manejo nativo de variables categóricas.
#* XGBRegressor: implementación optimizada de gradient boosting.
#* LGBMRegressor: boosting de alta velocidad diseñado para grandes volúmenes de datos.

from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor

# %%
# ==========================================
# Modelo 1: Random Forest Regressor
# ==========================================

# Crear la instancia del modelo
rfr_model = RandomForestRegressor(
    n_estimators=200,       # número de árboles (default=100, aquí lo aumentamos para mayor estabilidad)
    max_depth=None,         # profundidad máxima (None = sin límite)
    random_state=42,        # semilla para reproducibilidad
    n_jobs=-1               # uso de todos los núcleos disponibles
)

# Entrenar el modelo con los datos de entrenamiento
rfr_model.fit(X_train_nonlin, y_train)

# Generar predicciones para train y test
y_train_pred_rfr = rfr_model.predict(X_train_nonlin)
y_test_pred_rfr  = rfr_model.predict(X_test_nonlin)

# %%
# ==========================================
# Modelo 2: CatBoost Regressor
# ==========================================

from catboost import CatBoostRegressor

# Crear instancia del modelo
cat_model = CatBoostRegressor(
    iterations=500,       # número de iteraciones (default=1000)
    learning_rate=0.05,   # tasa de aprendizaje
    depth=8,              # profundidad de los árboles
    random_state=42,      # semilla para reproducibilidad
    verbose=0             # suprime la salida de logs
)

# Entrenar el modelo con los datos de entrenamiento
cat_model.fit(X_train_nonlin, y_train)

# Generar predicciones para train y test
y_train_pred_cat = cat_model.predict(X_train_nonlin)
y_test_pred_cat  = cat_model.predict(X_test_nonlin)

# %%
# ==========================================
# Modelo 3: XGBoost Regressor
# ==========================================

from xgboost import XGBRegressor

# Crear instancia del modelo
xgb_model = XGBRegressor(
    n_estimators=500,      # número de árboles
    learning_rate=0.05,    # tasa de aprendizaje
    max_depth=6,           # profundidad máxima de los árboles
    subsample=0.8,         # muestreo de filas
    colsample_bytree=0.8,  # muestreo de columnas
    random_state=42,
    n_jobs=-1              # usar todos los núcleos disponibles
)

# Entrenar el modelo con los datos de entrenamiento
xgb_model.fit(X_train_nonlin, y_train)

# Generar predicciones para train y test
y_train_pred_xgb = xgb_model.predict(X_train_nonlin)
y_test_pred_xgb  = xgb_model.predict(X_test_nonlin)


# %%
# ==========================================
# Modelo 4: LightGBM Regressor
# ==========================================

from lightgbm import LGBMRegressor

# Crear instancia del modelo con parámetros iniciales
lgbm_model = LGBMRegressor(
    n_estimators=1000,     # número de árboles
    learning_rate=0.05,    # tasa de aprendizaje
    max_depth=-1,          # sin límite de profundidad
    subsample=0.8,         # muestreo de filas
    colsample_bytree=0.8,  # muestreo de columnas
    random_state=42,
    n_jobs=-1              # usar todos los núcleos
)

# Entrenar el modelo
lgbm_model.fit(X_train_nonlin, y_train)

# Generar predicciones
y_train_pred_lgbm = lgbm_model.predict(X_train_nonlin)
y_test_pred_lgbm  = lgbm_model.predict(X_test_nonlin)

# %%
## ============================
# Modelo 1: MLP Regressor
## ============================

mlp_model = MLPRegressor(
    hidden_layer_sizes=(100,),   # 1 capa oculta con 100 neuronas
    activation='relu',           # función de activación
    solver='adam',               # optimizador
    max_iter=1000,               # iteraciones máximas
    early_stopping=True,         # parada temprana para evitar overfitting
    validation_fraction=0.1,     # fracción del train reservada para validar
    random_state=42
)

# Entrenamiento (con X escaladas)
mlp_model.fit(X_train_lin, y_train_lin)

# Predicciones (también con X escaladas)
y_train_pred_mlp = mlp_model.predict(X_train_lin)
y_test_pred_mlp  = mlp_model.predict(X_test_lin)

# %% [markdown]
# ## Creacion modelos Lineales

# %%
from sklearn.linear_model import ElasticNet, HuberRegressor

# %%
## ============================
# MODELO LINEAL 1
## ============================

mlp_model = MLPRegressor(
    hidden_layer_sizes=(100,),   # 1 capa oculta con 100 neuronas
    activation='relu',           # función de activación
    solver='adam',               # optimizador
    max_iter=500,                # número máximo de iteraciones
    random_state=42
)

# Entrenar el modelo (usando las features escaladas)
mlp_model.fit(X_train_lin, y_train_lin)

# Generar predicciones
y_train_pred_mlp = mlp_model.predict(X_train_lin)
y_test_pred_mlp  = mlp_model.predict(X_test_lin)


# %%
## ========================
## MODELO LINEAL 2
## ========================
# ElasticNet combina Lasso (L1) y Ridge (L2) en una sola penalización.
# alpha: controla la fuerza de la regularización (0.1 en este caso).
# l1_ratio: controla el balance entre L1 y L2 (0.5 significa mitad Lasso, mitad Ridge).

elas_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elas_model.fit(X_train_lin, y_train)

# Generamos predicciones
y_train_pred_elas = elas_model.predict(X_train_lin) 
y_test_pred_elas  = elas_model.predict(X_test_lin)  

# %%
## ========================
## MODELO LINEAL 3
## ========================
# Es una regresión robusta a outliers, combinando lo mejor de la regresión lineal 
# con una función de pérdida menos sensible a valores extremos.
# epsilon: determina qué tan sensible es a los outliers (1.35 es un valor común).
# alpha: regularización (penalización para evitar sobreajuste).

huber_model = HuberRegressor(epsilon=1.35, alpha=0.0001)
huber_model.fit(X_train_lin, y_train) 

# Generar predicciones
y_train_pred_huber = huber_model.predict(X_train_lin) 
y_test_pred_huber  = huber_model.predict(X_test_lin)  


# %% [markdown]
# ## Evaluación de modelos
# En esta sección se definen las métricas y la función para evaluar el rendimiento de los modelos.
# El objetivo es obtener una visión clara de la precisión, el error relativo y la capacidad de explicación de cada modelo entrenado.

# %%
## =========================
#  Evaluación de modelos
## =========================

# MAE (Mean Absolute Error): mide el error promedio absoluto entre los valores reales y los predichos
# Error relativo MAE: representa el MAE como porcentaje del valor real promedio.
# MSE (Mean Squared Error): error cuadrático medio, penaliza más los errores grandes.
# RMSE (Root Mean Squared Error): raíz del MSE, facilita la interpretación en las mismas unidades de la variable objetivo.
# Error relativo RMSE: compara el RMSE con el promedio real, en porcentaje.
# R² (Coeficiente de determinación): mide qué tan bien el modelo explica la variabilidad de los datos (1 = ajuste perfecto).


# 1. Importar metricas de evaluacion
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Creamos una funcion de evaluado de modelos

def evaluar(y_real, y_predicho, modelo=""):
    
    real = np.mean(y_real)
    
    mae = mean_absolute_error(y_real, y_predicho)
    error_relativo_mae = (mae/real)*100
    
    mse = mean_squared_error(y_real, y_predicho)
    rmse = np.sqrt(mean_squared_error(y_real,y_predicho))
    error_relativo_rmse = (rmse/real) * 100
    
    r2 = r2_score(y_real,y_predicho)
    
    print(f"==={modelo.upper()}===")
    print(f"MAE: {mae}")
    print(f"Error Relativo MAE: {error_relativo_mae:.2f} %")
    print(f"MSE: {mse}")
    print(f"Error Relativo RMSE: {error_relativo_rmse:.2f} %")
    print(f"R2 : {r2*100:.2f}")
    print(f"RMSE: {rmse}")

# %% [markdown]
# ## Resultados de los Modelos
# * En este bloque se ejecuta la función evaluar() sobre todos los modelos entrenados, tanto NO lineales como lineales.
# * El objetivo es comparar métricas de desempeño en el conjunto de prueba (y_test) y así identificar qué algoritmo se adapta mejor a la predicción de retornos financieros.

# %%
print("========== MODELOS NO LINEALES ==========\n")

evaluar(y_test, y_test_pred_rfr, "RandomForestRegressor")
evaluar(y_test, y_test_pred_cat, "CatBoostRegressor")
evaluar(y_test, y_test_pred_xgb, "XGBRegressor")
evaluar(y_test, y_test_pred_lgbm, "LGBMRegressor")

print("\n========== MODELOS LINEALES ==========\n")

evaluar(y_test, y_test_pred_mlp, "MLPRegressor")
evaluar(y_test, y_test_pred_elas, "ElasticNet")
evaluar(y_test, y_test_pred_huber, "HuberRegressor")


# %%
## =========================
#  Resultados de Modelos
## =========================
#* Los modelos NO lineales (RandomForest, CatBoost, XGB, LGBM) no lograron capturar bien la relación entre las variables predictoras y el objetivo.
#* Los modelos lineales con regularización (ElasticNet y Huber) tuvieron un desempeño relativamente mejor, aunque los errores siguen siendo altos y R² no es positivo.

### *** Esto indica que:
#* El conjunto de variables predictoras disponibles podría no ser suficiente para predecir retornos a 20 días.
#* El problema en sí puede ser demasiado ruidoso, algo común en datos financieros.

## **** Para la predicción de retornos financieros a 20 días, el problema es altamente ruidoso. 
# Los modelos lineales con regularización (ElasticNet, Huber) ofrecen un desempeño más razonable y deberían ser los candidatos principales 
# destacando que aún no hay un ajuste predictivo confiable y que la alta incertidumbre es inherente al dominio financiero.

# %% [markdown]
# ## Visualizaciones Gráficas

# %% [markdown]
# #### Visualización de Predicciones vs Valores Reales
# 
# * Para evaluar visualmente el desempeño de los modelos, se generaron gráficos de dispersión que muestran la relación entre los valores reales observados (en el eje X) y los valores predichos por cada modelo (en el eje Y). La línea diagonal representa la línea ideal, es decir, el caso en que las predicciones coinciden exactamente con los valores reales.

# %%
plt.figure(figsize=(18,8))

# ================================
# Gráfico 1 - RandomForestRegressor
# ================================
plt.subplot(1,2,1)
sns.scatterplot(x=y_test, y=y_test_pred_rfr, alpha=0.6, color="blue", edgecolor="k")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle="--") 
plt.title("RandomForestRegressor: Valores Reales vs Predichos")
plt.xlabel("Valores Reales (target_retorno_20d)")
plt.ylabel("Valores Predichos")
plt.grid(True)

# ================================
# Gráfico 2 - CatBoostRegressor
# ================================
plt.subplot(1,2,2)
sns.scatterplot(x=y_test, y=y_test_pred_cat, alpha=0.6, color="green", edgecolor="k")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle="--")
plt.title("CatBoostRegressor: Valores Reales vs Predichos")
plt.xlabel("Valores Reales (target_retorno_20d)")
plt.ylabel("Valores Predichos")
plt.grid(True)

# Ajustar diseño
plt.tight_layout()
plt.show()

# ================================
# Resultados
# ================================
#* Los gráficos muestran la relación entre los valores reales del target (retorno a 20 días) y los valores predichos por cada modelo.
#* Las lineas punteadas representa la predicción perfecta (valores reales = valores predichos).
#* Cuanto más cercanos estén los puntos a esta línea, mejor es la capacidad del modelo para aproximar los valores reales.
#* Al observar la dispersión:
#* RandomForest y CatBoost presentan un patrón con menor ajuste (alta dispersión).
#* En general, todos los modelos presentan dificultad en capturar la señal, confirmando lo ya observado en las métricas (R² negativo).

# %%
## ==============================
# Importancia de variables con CatBoost
## ==============================

# Obtenemos las importancias del modelo
importancias = cat_model.feature_importances_

# Creamos DataFrame ordenado
importancia = pd.DataFrame({
    "Variable": X_train.columns,
    "Importancia": importancias
}).sort_values(by="Importancia", ascending=False)

# Mostrar tabla
display(importancia)

# Gráfico de barras
plt.figure(figsize=(10,6))
sns.barplot(x="Importancia", y="Variable", data=importancia.head(15), palette="viridis")

plt.title("Importancia de Variables - CatBoost")
plt.xlabel("Importancia (%)")
plt.ylabel("Variable")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# %%
# ================================
# Resultados
# ================================

#*** El modelo CatBoost identifica que las variables más influyentes en la predicción del target_retorno_20d son principalmente indicadores técnicos como:
#* Volatilidad_20, que tiene la mayor relevancia con diferencia.
#* Signal, SMA_30, y BB_sup también aparecen con pesos altos.
#* Las variables de rezagos (lag_25, lag_15, lag_1) también contribuyen, aunque en menor medida.


# %% [markdown]
# #### Gráficos de Residuos - Modelos LGBM y CatBoost
# Objetivo: Analizar la distribución de los residuos (diferencia entre valores reales y predichos) para evaluar si el modelo presenta sesgos o errores sistemáticos.

# %%
plt.figure(figsize=(17,6))

# ================================================
# Gráfico 1 - Residuos LightGBMRegressor
# ================================================
plt.subplot(1,2,1)
residuos_lgbm = y_test - y_test_pred_lgbm

sns.histplot(residuos_lgbm, kde=True, color="orange", bins=30)
plt.title("Distribución de Residuos - Modelo LGBMRegressor")
plt.xlabel("Valor del Residuo")
plt.ylabel("Frecuencia")
plt.axvline(x=0, color="red", linestyle="--", linewidth=1.2, label="Línea de Referencia")
plt.legend()
plt.grid(alpha=0.5)

# ================================================
# Gráfico 2 - Residuos CatBoostRegressor
# ================================================
plt.subplot(1,2,2)
residuos_cat = y_test - y_test_pred_cat

sns.histplot(residuos_cat, kde=True, color="steelblue", bins=30)
plt.title("Distribución de Residuos - Modelo CatBoostRegressor")
plt.xlabel("Valor del Residuo")
plt.ylabel("Frecuencia")
plt.axvline(x=0, color="red", linestyle="--", linewidth=1.2, label="Línea de Referencia")
plt.legend()
plt.grid(alpha=0.5)

# Ajustar diseño
plt.tight_layout()
plt.show()

# ================================
# Resultados
# ================================
#*** Gráfico 1 - LightGBMRegressor:
# Los residuos se concentran alrededor de 0, con ligera dispersión hacia los extremos. 
# Esto indica que el modelo capta ciertas tendencias pero aún muestra errores importantes en los valores extremos.

# *** Gráfico 2 - CatBoostRegressor:
# La distribución es más simétrica que en LGBM, aunque todavía se observan colas largas. 
# Esto refleja que el modelo tiende a ajustarse mejor en valores medios, pero pierde precisión en casos atípicos.

# Conclusión: Ningún modelo logra un ajuste perfecto. 
# Ambos presentan dispersión significativa, confirmando lo observado en las métricas (MAE y R² negativos). 
# No obstante, la comparación visual permite ver que CatBoost ofrece una distribución más cercana a la simetría alrededor de 0.


# %% [markdown]
# #### Interpretabilidad del modelo con SHAP
# Objetivo: Evaluar la importancia e impacto de cada variable en las predicciones del modelo

# %%
import shap

# Crear el explicador para el modelo LGBM
explainer = shap.Explainer(lgbm_model)
shap_values = explainer(X_test)

# ================================================
# Gráfico tipo Beeswarm (impacto de cada variable)
# ================================================
shap.plots.beeswarm(shap_values, max_display=15, show=True)

# %%
# ================================
# Resultados
# ================================
#*** Cada punto representa una observación del dataset de test.
#* El eje X muestra cuánto impacta esa variable en la predicción final (positivo o negativo).
#* Los colores indican el valor de la variable: rojo = valores altos, azul = valores bajos.
#* Las variables se ordenan de mayor a menor importancia en el modelo.
#* EMA_20 y lag_1 aparecen como las variables más influyentes en el modelo, seguidas por lag_25 y Signal.
#* Variables como volatilidad_20 y retorno_30d también aportan de forma significativa, aunque con menor impacto.
#* El gráfico muestra que el modelo se apoya en indicadores técnicos de tendencia (EMA, lag, señales) para ajustar sus predicciones.


# %% [markdown]
# #### (Waterfall Plot)
# Interpretar cómo cada variable influyó en la predicción de una observación concreta del dataset de test.

# %%
# ================================================
# Gráfico SHAP tipo Waterfall para un caso puntual
# ================================================

# Seleccionamos la primera observación del conjunto de test
shap.plots.waterfall(shap_values[1])

# ================================
# Resultados
# ================================
# El gráfico permite una interpretación local del modelo, mostrando de forma transparente qué factores 
# llevaron a una predicción específica. Esto es muy útil para justificar decisiones o validar la coherencia del modelo con los supuestos financieros.


