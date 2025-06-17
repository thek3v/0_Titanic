# 1. Titanic

## 1.1 Cargar entorno y archivos
El dataset de entrenamiento (train.csv) contiene 891 registros y la variable objetivo Survived, que indica si el pasajero sobrevivió (1) o no (0).

El dataset de test (test.csv) contiene 418 registros y no incluye la variable Survived → mi modelo deberá predecir este valor.


```python
# Librerías para manejo de datos
import pandas as pd
import numpy as np 

# Librerías para visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de los gráficos
#%matplotlib inline
#sns.set_style('whitegrid')

# Librerías para el modelado
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

```


```python
# Definir la ruta donde tienes los CSV
data_path = r"C:\Users\kevin.vargas\Desktop\ds_kevin_vargas\0_titanic\data"

# Cargar los datasets
train_df = pd.read_csv(f"{data_path}\\train.csv")
test_df = pd.read_csv(f"{data_path}\\test.csv")

# Mostrar primeras filas del train
print("Train dataset:")
display(train_df)

# Mostrar primeras filas del test
print("Test dataset:")
display(test_df)
train_df

```

    Train dataset:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.0000</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.0000</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 12 columns</p>
</div>


    Test dataset:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>1305</td>
      <td>3</td>
      <td>Spector, Mr. Woolf</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>A.5. 3236</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>414</th>
      <td>1306</td>
      <td>1</td>
      <td>Oliva y Ocana, Dona. Fermina</td>
      <td>female</td>
      <td>39.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17758</td>
      <td>108.9000</td>
      <td>C105</td>
      <td>C</td>
    </tr>
    <tr>
      <th>415</th>
      <td>1307</td>
      <td>3</td>
      <td>Saether, Mr. Simon Sivertsen</td>
      <td>male</td>
      <td>38.5</td>
      <td>0</td>
      <td>0</td>
      <td>SOTON/O.Q. 3101262</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>416</th>
      <td>1308</td>
      <td>3</td>
      <td>Ware, Mr. Frederick</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>359309</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>417</th>
      <td>1309</td>
      <td>3</td>
      <td>Peter, Master. Michael J</td>
      <td>male</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>2668</td>
      <td>22.3583</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
<p>418 rows × 11 columns</p>
</div>


## 1.2 Análisis exploratorio

### Información general


```python
# Información general del dataset
print("Train dataset info:")
train_df.info()
```

    Train dataset info:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    


```python
print("\nTest dataset info:")
test_df.info()
```

    
    Test dataset info:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 11 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  418 non-null    int64  
     1   Pclass       418 non-null    int64  
     2   Name         418 non-null    object 
     3   Sex          418 non-null    object 
     4   Age          332 non-null    float64
     5   SibSp        418 non-null    int64  
     6   Parch        418 non-null    int64  
     7   Ticket       418 non-null    object 
     8   Fare         417 non-null    float64
     9   Cabin        91 non-null     object 
     10  Embarked     418 non-null    object 
    dtypes: float64(2), int64(4), object(5)
    memory usage: 36.0+ KB
    


```python
# Resumen estadístico de las variables numéricas
print("Train dataset - Statistical summary:")
display(train_df.describe())

print("\nTest dataset - Statistical summary:")
display(test_df.describe())

```

    Train dataset - Statistical summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>


    
    Test dataset - Statistical summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>332.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>417.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1100.500000</td>
      <td>2.265550</td>
      <td>30.272590</td>
      <td>0.447368</td>
      <td>0.392344</td>
      <td>35.627188</td>
    </tr>
    <tr>
      <th>std</th>
      <td>120.810458</td>
      <td>0.841838</td>
      <td>14.181209</td>
      <td>0.896760</td>
      <td>0.981429</td>
      <td>55.907576</td>
    </tr>
    <tr>
      <th>min</th>
      <td>892.000000</td>
      <td>1.000000</td>
      <td>0.170000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>996.250000</td>
      <td>1.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.895800</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1100.500000</td>
      <td>3.000000</td>
      <td>27.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1204.750000</td>
      <td>3.000000</td>
      <td>39.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1309.000000</td>
      <td>3.000000</td>
      <td>76.000000</td>
      <td>8.000000</td>
      <td>9.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>


### Distribucion de la variable objetivo


```python
# Distribución de la variable objetivo
sns.countplot(data=train_df, x='Survived')
plt.title("Distribución de la variable 'Survived'")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Cantidad de pasajeros")
plt.show()

# porcentajes
survived_rate = train_df['Survived'].value_counts(normalize=True) * 100
print(f"Porcentaje de supervivientes:\n{survived_rate}")

```


    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_10_0.png)
    


    Porcentaje de supervivientes:
    Survived
    0    61.616162
    1    38.383838
    Name: proportion, dtype: float64
    

### Distribucion de variables categoricas clave vs survived


```python
sns.countplot(data=train_df, x='Sex', hue='Survived')
plt.title("Supervivencia según sexo")
plt.xlabel("Sexo")
plt.ylabel("Cantidad de pasajeros")
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

```


    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_12_0.png)
    



```python
sns.countplot(data=train_df, x='Pclass', hue='Survived')
plt.title("Supervivencia según clase del billete (Pclass)")
plt.xlabel("Clase")
plt.ylabel("Cantidad de pasajeros")
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

```


    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_13_0.png)
    



```python
sns.countplot(data=train_df, x='Embarked', hue='Survived')
plt.title("Supervivencia según puerto de embarque (Embarked)")
plt.xlabel("Puerto de embarque")
plt.ylabel("Cantidad de pasajeros")
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

```


    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_14_0.png)
    


### Explicación

Primero analizamos la estructura general de los datasets (info()), identificando valores nulos y tipos de variables.
Luego, obtenemos un resumen estadístico de las variables numéricas (describe()).

A continuación, exploramos visualmente la distribución de la variable objetivo (Survived), y la relación de esta variable con algunas variables categóricas clave como:

- Sex (sexo del pasajero)

- Pclass (clase del billete)

- Embarked (puerto de embarque)

- Estos gráficos nos permitirán obtener los primeros insights sobre qué tipo de personas tenían más probabilidad de sobrevivir.




Valores nulos:

- Age → faltan ~177 valores en train y ~86 en test → Despues los trato

- Cabin → casi todos nulos → probablemente eliminare esta variable

- Embarked → faltan 2 valores en train, ninguno en test

- Fare → falta 1 valor en test




Relaciones clave:

- Sexo:

Las mujeres tienen una probabilidad de supervivencia claramente mayor.

- Clase (Pclass):

Los pasajeros de 1ª clase sobreviven mucho más que los de 3ª.

- Puerto de embarque (Embarked):

Los pasajeros embarcados en C parecen tener una mayor tasa de supervivencia que S o Q.

### Función general para información general
eda_utils.py y la funcion se llama basic_eda


```python
import sys
sys.path.append(r'C:\Users\kevin.vargas\Desktop\ds_kevin_vargas\Funciones')

from eda_utils import basic_eda

basic_eda(train_df)

```

    🔍 Info general:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 11 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          891 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Embarked     891 non-null    object 
    dtypes: float64(2), int64(5), object(4)
    memory usage: 76.7+ KB
    


    None


    
    📏 Filas: 891  |  Columnas: 11
    
    🗂️  Columnas duplicadas: 0
    
    📊 Tipos de columnas:
    Numéricas (7): ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    Categóricas (4): ['Name', 'Sex', 'Ticket', 'Embarked']
    Otras (0): []
    
    📈 Descripción estadística:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.112424</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>13.304424</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>21.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>26.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>36.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>


    
    ❓ Valores nulos por columna:
    


    PassengerId    0
    Survived       0
    Pclass         0
    Name           0
    Sex            0
    Age            0
    SibSp          0
    Parch          0
    Ticket         0
    Fare           0
    Embarked       0
    dtype: int64


    
    📊 Distribución de columnas numéricas:
    


    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_18_7.png)
    



    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_18_8.png)
    



    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_18_9.png)
    



    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_18_10.png)
    



    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_18_11.png)
    



    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_18_12.png)
    



    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_18_13.png)
    


    
    📊 Distribución de columnas categóricas:
    


    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_18_15.png)
    



    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_18_16.png)
    



    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_18_17.png)
    



    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_18_18.png)
    


## 1.3 Limpieza de datos

### Age


```python
# Histograma de la variable Age
plt.figure(figsize=(8,5))
sns.histplot(train_df['Age'], bins=30, kde=True)
plt.title('Distribución de Age')
plt.xlabel('Age')
plt.ylabel('Cantidad de pasajeros')
plt.show()

```


    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_21_0.png)
    



```python
# Distribución de Age segmentada por Sex y Pclass
plt.figure(figsize=(10,6))
sns.boxplot(x='Pclass', y='Age', hue='Sex', data=train_df)
plt.title('Distribución de Age según Pclass y Sex')
plt.xlabel('Clase del billete (Pclass)')
plt.ylabel('Age')
plt.show()

```


    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_22_0.png)
    



```python
# Crear una función que permita imputar Age por mediana de (Sex, Pclass)
def impute_age(row):
    if pd.isnull(row['Age']):
        return median_age_by_group.loc[row['Sex'], row['Pclass']]
    else:
        return row['Age']

# Calcular la mediana de Age por grupo (Sex, Pclass)
median_age_by_group = train_df.groupby(['Sex', 'Pclass'])['Age'].median()

# Mostrar la tabla de medianas para que veas los valores que se van a imputar
print("Mediana de Age por grupo (Sex, Pclass):")
display(median_age_by_group)

# Aplicar la imputación al train_df
train_df['Age'] = train_df.apply(impute_age, axis=1)

# Comprobar que ya no quedan valores nulos en Age
print(f"Valores nulos en Age después de imputación: {train_df['Age'].isnull().sum()}")

```

    Mediana de Age por grupo (Sex, Pclass):
    


    Sex     Pclass
    female  1         35.0
            2         28.0
            3         21.5
    male    1         40.0
            2         30.0
            3         25.0
    Name: Age, dtype: float64


    Valores nulos en Age después de imputación: 0
    

Para imputar los valores faltantes de Age, decidi no utilizar una mediana global, sino una mediana segmentada por Sex y Pclass.
Estas dos variables correlacionan con la edad de los pasajeros, como se ve en el boxplot.

Calculo la mediana de Age para cada combinación (Sex, Pclass) y uso esta información para imputar los valores nulos.
De este modo, la imputación es más realista y respeta las diferencias de edad entre grupos socio-demográficos.

### Fare


```python
# Histograma de la variable Fare (en test_df, que es donde falta un valor)
plt.figure(figsize=(8,5))
sns.histplot(test_df['Fare'], bins=30, kde=True)
plt.title('Distribución de Fare en test_df')
plt.xlabel('Fare')
plt.ylabel('Cantidad de pasajeros')
plt.show()

```


    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_26_0.png)
    



```python
# Calcular la mediana de Fare por Pclass
median_fare_by_pclass = test_df.groupby('Pclass')['Fare'].median()

# Mostrar la tabla de medianas
print("Mediana de Fare por Pclass:")
display(median_fare_by_pclass)

# Detectar qué fila tiene el Fare nulo
fare_nan_row = test_df[test_df['Fare'].isnull()]
print("Fila con Fare nulo:")
display(fare_nan_row)

# Imputar el valor faltante con la mediana correspondiente
pclass_of_nan = fare_nan_row['Pclass'].values[0]
median_fare_value = median_fare_by_pclass.loc[pclass_of_nan]

# Imputar el valor
test_df.loc[test_df['Fare'].isnull(), 'Fare'] = median_fare_value

# Verificar que ya no hay valores nulos en Fare
print(f"Valores nulos en Fare después de imputación: {test_df['Fare'].isnull().sum()}")
test_df

```

    Mediana de Fare por Pclass:
    


    Pclass
    1    60.0000
    2    15.7500
    3     7.8958
    Name: Fare, dtype: float64


    Fila con Fare nulo:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>152</th>
      <td>1044</td>
      <td>3</td>
      <td>Storey, Mr. Thomas</td>
      <td>male</td>
      <td>60.5</td>
      <td>0</td>
      <td>0</td>
      <td>3701</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>


    Valores nulos en Fare después de imputación: 0
    

En la variable Fare, detecto que solo había un valor nulo en el conjunto test_df.
Dado que Fare está altamente correlacionado con la clase del billete (Pclass), decido imputar este valor utilizando la mediana de Fare para la clase correspondiente.
Esta estrategia permite mantener la coherencia del valor imputado con el perfil socio-económico del pasajero.

### Embarked


```python
# Ver cuántos valores nulos hay en Embarked
print(f"Valores nulos en Embarked (train_df): {train_df['Embarked'].isnull().sum()}")

# Ver la distribución actual de Embarked
sns.countplot(x='Embarked', data=train_df)
plt.title('Distribución de Embarked')
plt.xlabel('Puerto de embarque')
plt.ylabel('Cantidad de pasajeros')
plt.show()

```

    Valores nulos en Embarked (train_df): 2
    


    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_30_1.png)
    



```python
# Calcular la moda de Embarked
mode_embarked = train_df['Embarked'].mode()[0]
print(f"Moda de Embarked: {mode_embarked}")

# Imputar valores nulos con la moda
train_df['Embarked'].fillna(mode_embarked, inplace=True)

# Verificar que ya no hay valores nulos
print(f"Valores nulos en Embarked después de imputación: {train_df['Embarked'].isnull().sum()}")

```

    Moda de Embarked: S
    Valores nulos en Embarked después de imputación: 0
    

La variable Embarked tenía 2 valores nulos en el conjunto train_df.
Dado que Embarked es una variable categórica con 3 posibles valores (S, C, Q), decido imputar los valores faltantes con la moda, es decir, el valor más frecuente.
En este caso, el puerto de embarque más frecuente es S (Southampton), lo que es consistente con la distribución global de la variable.

### Cabin


```python
# Eliminar la columna Cabin en ambos datasets
train_df.drop('Cabin', axis=1, inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)

# Verificar columnas restantes
print("Columnas actuales en train_df:")
print(train_df.columns.tolist())

print("\nColumnas actuales en test_df:")
print(test_df.columns.tolist())

```

    Columnas actuales en train_df:
    ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']
    
    Columnas actuales en test_df:
    ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']
    

La variable Cabin contenía más de un 75% de valores faltantes tanto en train_df como en test_df.
Debido a su alto grado de incompletitud y a la dificultad de extraer información útil sin un tratamiento adicional complejo, decidimos eliminar esta variable del dataset para evitar introducir ruido en el modelo.

## 1.4 Feature engineering

### Crear la columna title
Voy a extraer title de la columna nombre

Esto lo hacemos porque aparte de mr, ms...etc tambien hay titulos como dr que pueden indicar tambien la clase social de la persona y su relación con su supervivencia. 


```python
# Crear nueva columna 'Title' extrayendo el título del campo 'Name'
train_df['Title'] = train_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_df['Title'] = test_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Ver los títulos únicos encontrados
print("Títulos únicos en train_df:")
print(train_df['Title'].unique())

print("\nTítulos únicos en test_df:")
print(test_df['Title'].unique())

```

    Títulos únicos en train_df:
    ['Mr' 'Mrs' 'Miss' 'Master' 'Don' 'Rev' 'Dr' 'Mme' 'Ms' 'Major' 'Lady'
     'Sir' 'Mlle' 'Col' 'Capt' 'Countess' 'Jonkheer']
    
    Títulos únicos en test_df:
    ['Mr' 'Mrs' 'Miss' 'Master' 'Ms' 'Col' 'Rev' 'Dr' 'Dona']
    


```python
# Definir lista de títulos comunes
common_titles = ['Mr', 'Miss', 'Mrs', 'Master']

# Reemplazar los títulos raros por 'Rare'
train_df['Title'] = train_df['Title'].apply(lambda x: x if x in common_titles else 'Rare')
test_df['Title'] = test_df['Title'].apply(lambda x: x if x in common_titles else 'Rare')

# Verificar títulos únicos después de la agrupación
print("Títulos únicos en train_df después de agrupar:")
print(train_df['Title'].unique())

print("\nTítulos únicos en test_df después de agrupar:")
print(test_df['Title'].unique())
train_df

```

    Títulos únicos en train_df después de agrupar:
    ['Mr' 'Mrs' 'Miss' 'Master' 'Rare']
    
    Títulos únicos en test_df después de agrupar:
    ['Mr' 'Mrs' 'Miss' 'Master' 'Rare']
    

#### Explicación de title

En este paso de Feature Engineering, decido extraer el título (Title) de la columna Name, ya que este campo aporta información relevante sobre el perfil sociodemográfico de los pasajeros.
El título permite capturar indirectamente información sobre la edad, el estatus familiar y social, aspectos que pueden influir en la probabilidad de supervivencia. Además en el transcurso de este proceso, creare una columna llamada precisamente Title. 
Para extraer Title, utilizo una expresión regular que localiza el patrón habitual en el campo Name: una palabra seguida de un punto (.) después de la coma.

Tras extraer la variable Title, observo que algunos títulos son muy poco frecuentes y representan muy pocos registros.
Mantener demasiadas categorías poco representadas puede introducir ruido en el modelo que vaya a entrenar. Por lo que lo mejor es crear una nueva columna. 
Decido conservar los títulos más comunes (Mr, Mrs, Miss, Master) y agrupar el resto de títulos bajo una nueva categoría "Rare", mi nueva columna.
Este enfoque permite simplificar el modelo y mejorar su capacidad de generalización. Además en caso de que fuera a utilizar OneHotEncoding simplifico el proceso. 

### Crear la columna family size 
SibSp → número de hermanos/esposos a bordo

Parch → número de padres/hijos a bordo


```python
# Crear la variable FamilySize
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

# Verificar la nueva variable
print("Distribución de FamilySize en train_df:")
print(train_df['FamilySize'].value_counts().sort_index())

# Visualizar FamilySize vs. Survived para entender la relación
plt.figure(figsize=(8,5))
sns.barplot(x='FamilySize', y='Survived', data=train_df)
plt.title('Supervivencia según tamaño de la familia (FamilySize)')
plt.xlabel('FamilySize')
plt.ylabel('Tasa de Supervivencia')
plt.show()

```

    Distribución de FamilySize en train_df:
    FamilySize
    1     537
    2     161
    3     102
    4      29
    5      15
    6      22
    7      12
    8       6
    11      7
    Name: count, dtype: int64
    


    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_44_1.png)
    


#### Explicación

Como parte del Feature Engineering, creo la variable FamilySize, que representa el tamaño total de la familia a bordo.
Esta variable se calcula como la suma de SibSp (hermanos/esposos) y Parch (padres/hijos), más uno (el propio pasajero).
La variable FamilySize permite capturar patrones de comportamiento y probabilidad de supervivencia relacionados con si el pasajero viajaba solo, en pareja o en familia.Tuve que buscar que significaban esas dos columnas. 
De esta manera me aseguro de tener la información de dos columnas en una sola, mejor para el modelo a priori. 

## 1.5 Revisión y codificación de columnas

### Eliminamos columnas innecesarias

Vamos a eliminar columnas de las que hemos sacado información util necesaria y que ya no necesitamos. 


```python
# En train_df sí eliminamos PassengerId
columns_to_drop_train = ['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch']
train_df.drop(columns=columns_to_drop_train, axis=1, inplace=True)

# En test_df NO eliminamos PassengerId (lo necesitamos para la entrega)
columns_to_drop_test = ['Name', 'Ticket', 'SibSp', 'Parch']
test_df.drop(columns=columns_to_drop_test, axis=1, inplace=True)

# Verificar columnas finales
print("Columnas finales en train_df:")
print(train_df.columns.tolist())

print("\nColumnas finales en test_df:")
print(test_df.columns.tolist())


```

    Columnas finales en train_df:
    ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize']
    
    Columnas finales en test_df:
    ['PassengerId', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize']
    

#### Explicación

Antes de preparar los datos para el modelado, selecciono las variables que consideramos más relevantes y elimino aquellas que no aportan información útil o que pueden introducir ruido en el modelo.
En particular:

Elimino Name, Ticket, SibSp, Parch tras haber extraído Title y creado FamilySize. Ya no me hacen falta porque sería información redundante.

PassengerId la conservaré únicamente para la entrega final en el test_df, pero no preveo que no se usará en mis modelos.

### LabelEncoding y OneHotEncoding

#### LabelEncoding


```python
# Codificar Sex como 0/1
sex_mapping = {'male': 0, 'female': 1}
train_df['Sex'] = train_df['Sex'].map(sex_mapping)
test_df['Sex'] = test_df['Sex'].map(sex_mapping)

```

#### OneHotEncoding


```python
# One-Hot Encoding para Embarked
train_df = pd.get_dummies(train_df, columns=['Embarked'], prefix='Embarked')
test_df = pd.get_dummies(test_df, columns=['Embarked'], prefix='Embarked')

```


```python
# One-Hot Encoding para Title
train_df = pd.get_dummies(train_df, columns=['Title'], prefix='Title')
test_df = pd.get_dummies(test_df, columns=['Title'], prefix='Title')

```

### Explicación

Para preparar las variables categóricas para el modelado, apliqué las siguientes transformaciones:

Sex: codificación binaria (Label Encoding), mapeando 'male' a 0 y 'female' a 1. Unicamente tengo ese para labelencoding. 

Embarked y Title: codificación One-Hot Encoding, generando una columna binaria para cada categoría.
Este enfoque evita que el modelo interprete un orden ficticio en variables categóricas y mejora su capacidad de generalización.

## 1.6 Modelado

### Separar las variables en X e y


```python
# Separar target y features
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

# Verificar por si acaso los shapes
print(f"Shape de X: {X.shape}")
print(f"Shape de y: {y.shape}")

```

    Shape de X: (891, 13)
    Shape de y: (891,)
    

### Separar x e y en train/test split


```python
from sklearn.model_selection import train_test_split

# Dividir en 80% train y 20% test para validación
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificar shapes
print(f"Shape de X_train: {X_train.shape}")
print(f"Shape de X_valid: {X_valid.shape}")

```

    Shape de X_train: (712, 13)
    Shape de X_valid: (179, 13)
    

### Logistic regression
Tomaré este primer resultado como la base para compararlo con otros modelos y mejorar los indices.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Entrenar modelo Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Predecir en el conjunto de validación
y_pred = lr_model.predict(X_valid)

# Evaluar el modelo
print("Resultados en el conjunto de validación (Logistic Regression):")
print(f"Accuracy: {accuracy_score(y_valid, y_pred):.4f}")
print(f"Precision: {precision_score(y_valid, y_pred):.4f}")
print(f"Recall: {recall_score(y_valid, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_valid, y_pred):.4f}")

# Matriz de confusión
conf_matrix = confusion_matrix(y_valid, y_pred)
print("\nMatriz de confusión:")
print(conf_matrix)

# Reporte completo
print("\nClassification Report:")
print(classification_report(y_valid, y_pred))

```

    Resultados en el conjunto de validación (Logistic Regression):
    Accuracy: 0.8156
    Precision: 0.7808
    Recall: 0.7703
    F1 Score: 0.7755
    
    Matriz de confusión:
    [[89 16]
     [17 57]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.84      0.85      0.84       105
               1       0.78      0.77      0.78        74
    
        accuracy                           0.82       179
       macro avg       0.81      0.81      0.81       179
    weighted avg       0.82      0.82      0.82       179
    
    

#### Explicación

Como primer modelo baseline, utilizo una regresión logística (LogisticRegression), que es un modelo sencillo y ampliamente utilizado para problemas de clasificación binaria.
Mi objetivo es obtener una primera referencia de rendimiento, que servirá como base para comparar modelos más complejos posteriormente.
Evaluo el modelo utilizando las métricas de accuracy, precisión, recall y F1 score, así como la matriz de confusión.

El modelo baseline de regresión logística alcanzó un accuracy del 81.56 %, con valores equilibrados de precisión (78 %) y recall (77 %) en la clase Survived.
Estos resultados indican que el conjunto de features seleccionadas proporciona información relevante para la predicción.
No obstante, el número de falsos negativos sugiere que es posible mejorar el recall mediante modelos más complejos, como Random Forest o XGBoost.

### RandomForest

#### Sin gridsearch


```python
from sklearn.ensemble import RandomForestClassifier

# Entrenar modelo Random Forest (baseline)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predecir en el conjunto de validación
y_pred_rf = rf_model.predict(X_valid)

# Evaluar el modelo
print("Resultados en el conjunto de validación (Random Forest):")
print(f"Accuracy: {accuracy_score(y_valid, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_valid, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_valid, y_pred_rf):.4f}")
print(f"F1 Score: {f1_score(y_valid, y_pred_rf):.4f}")

# Matriz de confusión
conf_matrix_rf = confusion_matrix(y_valid, y_pred_rf)
print("\nMatriz de confusión:")
print(conf_matrix_rf)

# Reporte completo
print("\nClassification Report:")
print(classification_report(y_valid, y_pred_rf))

```

    Resultados en el conjunto de validación (Random Forest):
    Accuracy: 0.8324
    Precision: 0.7895
    Recall: 0.8108
    F1 Score: 0.8000
    
    Matriz de confusión:
    [[89 16]
     [14 60]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.86      0.85      0.86       105
               1       0.79      0.81      0.80        74
    
        accuracy                           0.83       179
       macro avg       0.83      0.83      0.83       179
    weighted avg       0.83      0.83      0.83       179
    
    

#### Con gridsearch


```python
from sklearn.model_selection import GridSearchCV

# Definir el espacio de búsqueda
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

```


```python
# Configurar el modelo base
rf_model_base = RandomForestClassifier(random_state=42)

# Configurar GridSearchCV
grid_search = GridSearchCV(estimator=rf_model_base,
                           param_grid=param_grid,
                           cv=5,            # 5-fold cross-validation
                           scoring='accuracy',
                           n_jobs=-1,       # usa todos los cores disponibles
                           verbose=2)       # para que veas el progreso

# Ejecutar GridSearchCV
grid_search.fit(X_train, y_train)

# Mostrar el mejor resultado
print(f"\nMejores hiperparámetros encontrados:")
print(grid_search.best_params_)

print(f"\nMejor accuracy en validación cruzada: {grid_search.best_score_:.4f}")

```

    Fitting 5 folds for each of 108 candidates, totalling 540 fits
    
    Mejores hiperparámetros encontrados:
    {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
    
    Mejor accuracy en validación cruzada: 0.8356
    


```python
# Evaluar el mejor modelo en X_valid
best_rf_model = grid_search.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_valid)

# Métricas
print("\nResultados del mejor modelo Random Forest en conjunto de validación:")
print(f"Accuracy: {accuracy_score(y_valid, y_pred_best_rf):.4f}")
print(f"Precision: {precision_score(y_valid, y_pred_best_rf):.4f}")
print(f"Recall: {recall_score(y_valid, y_pred_best_rf):.4f}")
print(f"F1 Score: {f1_score(y_valid, y_pred_best_rf):.4f}")

# Matriz de confusión
conf_matrix_best_rf = confusion_matrix(y_valid, y_pred_best_rf)
print("\nMatriz de confusión:")
print(conf_matrix_best_rf)

# Reporte completo
print("\nClassification Report:")
print(classification_report(y_valid, y_pred_best_rf))

```

    
    Resultados del mejor modelo Random Forest en conjunto de validación:
    Accuracy: 0.8324
    Precision: 0.8235
    Recall: 0.7568
    F1 Score: 0.7887
    
    Matriz de confusión:
    [[93 12]
     [18 56]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.84      0.89      0.86       105
               1       0.82      0.76      0.79        74
    
        accuracy                           0.83       179
       macro avg       0.83      0.82      0.82       179
    weighted avg       0.83      0.83      0.83       179
    
    

Tras optimizar el modelo de Random Forest mediante GridSearchCV, obtuve una combinación de hiperparámetros que proporciona un modelo más robusto y generalizable.
Aunque la mejora en Accuracy frente al modelo base fue pequeña, el uso de validación cruzada garantiza que el modelo optimizado evita sobreajuste y mantiene un buen equilibrio entre Precision y Recall.

#### Aplicar en df_test


```python
# Diccionario inverso
sex_reverse_mapping = {0: 'male', 1: 'female'}

# Función corregida para test_df
def impute_age_test(row):
    sex_label = sex_reverse_mapping[row['Sex']]
    return median_age_by_group.loc[sex_label, row['Pclass']] if pd.isnull(row['Age']) else row['Age']

# Aplicar la función corregida
test_df['Age'] = test_df.apply(impute_age_test, axis=1)

# Verificar que ya no hay NaN
print(f"Valores nulos en Age después de imputación en test_df: {test_df['Age'].isnull().sum()}")

```

    Valores nulos en Age después de imputación en test_df: 0
    


```python
# Predecir sobre test_df
# Ojo: en test_df no tengo 'Survived', pero sí PassengerId
# Uso solo las columnas que el modelo ha visto

# Cuidado: asegurarnos de que las columnas de test_df coinciden con X_train
# Si todo ha sido correcto, deberían coincidir después del mismo preprocesado.

# Hacemos la predicción
X_test = test_df.drop('PassengerId', axis=1)
y_test_pred = best_rf_model.predict(X_test)

# Crear dataframe para la submission
submission_df = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': y_test_pred
})

# Guardar el CSV en la ruta que me diste
output_path = r'C:\Users\kevin.vargas\Desktop\ds_kevin_vargas\0_titanic\resultados\submission_rf.csv'
submission_df.to_csv(output_path, index=False)

print(f"\nArchivo de resultados guardado en: {output_path}")

```

    
    Archivo de resultados guardado en: C:\Users\kevin.vargas\Desktop\ds_kevin_vargas\0_titanic\resultados\submission_rf.csv
    


```python
# Mostrar Feature Importance del modelo optimizado de Random Forest

import matplotlib.pyplot as plt
import seaborn as sns

# Obtener importancias
importances = best_rf_model.feature_importances_
features = X_train.columns

# Crear dataframe ordenado
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Visualizar
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance - Random Forest Optimizado')
plt.xlabel('Importancia')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

```

    C:\Users\kevin.vargas\AppData\Local\Temp\ipykernel_12872\3876516595.py:18: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    


    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_81_1.png)
    


Una vez optimizado el modelo de Random Forest, lo aplico al conjunto de test para generar las predicciones finales.
Antes de realizar la predicción, imputo nuevamente los valores faltantes en Age y Fare en test_df de manera consistente con el preprocesamiento realizado en train_df.
Posteriormente, verifico que no existieran valores nulos en el conjunto de test.
Finalmente, utilizo el modelo optimizado para predecir la variable Survived y genero el archivo CSV de resultados (PassengerId, Survived).

Tras entrenar un modelo base de Random Forest y evaluar su rendimiento, aplico una optimización de hiperparámetros mediante GridSearchCV.
Esta optimización me permitió obtener un modelo más robusto y generalizable, evitando el sobreajuste y mejorando ligeramente el rendimiento en el conjunto de validación.
Finalmente, utilizo el modelo optimizado para generar las predicciones sobre el conjunto de test, que exporté en el formato requerido.
El modelo de Random Forest optimizado constituye una solución sólida y competitiva para el problema de predicción de supervivencia en el Titanic.

Para interpretar el modelo de Random Forest optimizado, analizo la importancia de las diferentes variables utilizadas en la predicción.
La feature importance me permite identificar qué variables han contribuido en mayor medida a las decisiones del modelo.
Este análisis complementa la evaluación cuantitativa del modelo y aporta información valiosa sobre los factores que más influyen en la probabilidad de supervivencia de los pasajeros. Como era esperable el sexo y los titulos son features que han sido claves a la hora de predecir la supervivencia de los individuos. 

### XGBoost

#### Sin gridsearch


```python
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Entrenar modelo XGBoost (modelo base)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Predecir en el conjunto de validación
y_pred_xgb = xgb_model.predict(X_valid)

# Evaluar el modelo
print("Resultados en el conjunto de validación (XGBoost):")
print(f"Accuracy: {accuracy_score(y_valid, y_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_valid, y_pred_xgb):.4f}")
print(f"Recall: {recall_score(y_valid, y_pred_xgb):.4f}")
print(f"F1 Score: {f1_score(y_valid, y_pred_xgb):.4f}")

# Matriz de confusión
conf_matrix_xgb = confusion_matrix(y_valid, y_pred_xgb)
print("\nMatriz de confusión:")
print(conf_matrix_xgb)

# Reporte completo
print("\nClassification Report:")
print(classification_report(y_valid, y_pred_xgb))

```

    Resultados en el conjunto de validación (XGBoost):
    Accuracy: 0.8324
    Precision: 0.8143
    Recall: 0.7703
    F1 Score: 0.7917
    
    Matriz de confusión:
    [[92 13]
     [17 57]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.84      0.88      0.86       105
               1       0.81      0.77      0.79        74
    
        accuracy                           0.83       179
       macro avg       0.83      0.82      0.83       179
    weighted avg       0.83      0.83      0.83       179
    
    

    c:\Users\kevin.vargas\AppData\Local\miniconda3\envs\pit2\lib\site-packages\xgboost\core.py:158: UserWarning: [12:21:21] WARNING: C:\b\abs_90_bwj_86a\croot\xgboost-split_1724073762025\work\src\learner.cc:740: 
    Parameters: { "use_label_encoder" } are not used.
    
      warnings.warn(smsg, UserWarning)
    

#### Con gridsearch


```python
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Definir el espacio de búsqueda
param_grid_xgb = {
    'n_estimators': [100, 200, 300, 400, 500, 600],
    'max_depth': [2, 3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2, 0.5],
    'subsample': [0.8, 1.0, 0.7, 0.6],
    'colsample_bytree': [0.8, 0.9, 1.0, 0.7, 0.6]
}

# Configurar el modelo base
xgb_model_base = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Configurar GridSearchCV
grid_search_xgb = GridSearchCV(estimator=xgb_model_base,
                               param_grid=param_grid_xgb,
                               cv=5,
                               scoring='accuracy',
                               n_jobs=-1,
                               verbose=2)

# Ejecutar GridSearchCV
grid_search_xgb.fit(X_train, y_train)

# Mostrar el mejor resultado
print(f"\nMejores hiperparámetros encontrados para XGBoost:")
print(grid_search_xgb.best_params_)

print(f"\nMejor accuracy en validación cruzada: {grid_search_xgb.best_score_:.4f}")

```

    Fitting 5 folds for each of 1920 candidates, totalling 9600 fits
    

    c:\Users\kevin.vargas\AppData\Local\miniconda3\envs\pit2\lib\site-packages\xgboost\core.py:158: UserWarning: [12:37:51] WARNING: C:\b\abs_90_bwj_86a\croot\xgboost-split_1724073762025\work\src\learner.cc:740: 
    Parameters: { "use_label_encoder" } are not used.
    
      warnings.warn(smsg, UserWarning)
    

    
    Mejores hiperparámetros encontrados para XGBoost:
    {'colsample_bytree': 0.6, 'learning_rate': 0.01, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.6}
    
    Mejor accuracy en validación cruzada: 0.8399
    


```python
# Evaluar el mejor modelo en X_valid
best_xgb_model = grid_search_xgb.best_estimator_
y_pred_best_xgb = best_xgb_model.predict(X_valid)

# Métricas
print("\nResultados del mejor modelo XGBoost en conjunto de validación:")
print(f"Accuracy: {accuracy_score(y_valid, y_pred_best_xgb):.4f}")
print(f"Precision: {precision_score(y_valid, y_pred_best_xgb):.4f}")
print(f"Recall: {recall_score(y_valid, y_pred_best_xgb):.4f}")
print(f"F1 Score: {f1_score(y_valid, y_pred_best_xgb):.4f}")

# Matriz de confusión
conf_matrix_best_xgb = confusion_matrix(y_valid, y_pred_best_xgb)
print("\nMatriz de confusión:")
print(conf_matrix_best_xgb)

# Reporte completo
print("\nClassification Report:")
print(classification_report(y_valid, y_pred_best_xgb))

```

    
    Resultados del mejor modelo XGBoost en conjunto de validación:
    Accuracy: 0.8156
    Precision: 0.8154
    Recall: 0.7162
    F1 Score: 0.7626
    
    Matriz de confusión:
    [[93 12]
     [21 53]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.82      0.89      0.85       105
               1       0.82      0.72      0.76        74
    
        accuracy                           0.82       179
       macro avg       0.82      0.80      0.81       179
    weighted avg       0.82      0.82      0.81       179
    
    

Evaluación del modelo XGBoost (con y sin optimización)
Además del modelo de Random Forest, probé también un modelo de XGBoost (XGBClassifier), ampliamente reconocido por su buen rendimiento.

Inicialmente, entrené un modelo base de XGBoost con hiperparámetros por defecto, lo que me permitió obtener ya un rendimiento competitivo. El modelo base mostró un buen balance entre precisión y recall, comparable al modelo optimizado de Random Forest.

Posteriormente, apliqué una optimización de hiperparámetros mediante GridSearchCV, explorando combinaciones de parámetros clave como:

- número de árboles (n_estimators)

- profundidad máxima (max_depth)

- tasa de aprendizaje (learning_rate)

- proporción de muestras (subsample)

- proporción de features (colsample_bytree)

El objetivo era encontrar la configuración que maximizara la accuracy mediante validación cruzada (5-fold cross validation).

Resultados:

El modelo optimizado por GridSearch mostró un comportamiento más conservador (learning_rate bajo y subsample/colsample reducidos), lo que se tradujo en un modelo más robusto pero con un recall ligeramente inferior respecto a Random Forest.

Aunque el accuracy obtenida en validación cruzada fue buena (~0.8399), en el conjunto de validación (X_valid) el modelo de XGBoost optimizado no logró superar al modelo de Random Forest optimizado, que mantiene un mejor equilibrio general entre precisión, recall y F1 Score.

Conclusión:

El uso de XGBoost en este contexto ha permitido explorar una alternativa robusta y de alto rendimiento. Sin embargo, en este caso concreto, el modelo de Random Forest optimizado ofrece mejores resultados en términos de balance general y será el modelo seleccionado para la generación de las predicciones finales.

### HistBoosting

#### Sin gridsearch


```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Modelo base de HistGradientBoostingClassifier
histgbc_model = HistGradientBoostingClassifier(random_state=42)

# Entrenamiento
histgbc_model.fit(X_train, y_train)

# Predicciones
y_pred_histgbc = histgbc_model.predict(X_valid)

# Evaluación
print("Resultados en el conjunto de validación (HistGradientBoostingClassifier):")
print(f"Accuracy: {accuracy_score(y_valid, y_pred_histgbc):.4f}")
print(f"Precision: {precision_score(y_valid, y_pred_histgbc):.4f}")
print(f"Recall: {recall_score(y_valid, y_pred_histgbc):.4f}")
print(f"F1 Score: {f1_score(y_valid, y_pred_histgbc):.4f}")

# Matriz de confusión
conf_matrix_histgbc = confusion_matrix(y_valid, y_pred_histgbc)
print("\nMatriz de confusión:")
print(conf_matrix_histgbc)

# Reporte completo
print("\nClassification Report:")
print(classification_report(y_valid, y_pred_histgbc))

```

    Resultados en el conjunto de validación (HistGradientBoostingClassifier):
    Accuracy: 0.8547
    Precision: 0.8429
    Recall: 0.7973
    F1 Score: 0.8194
    
    Matriz de confusión:
    [[94 11]
     [15 59]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.86      0.90      0.88       105
               1       0.84      0.80      0.82        74
    
        accuracy                           0.85       179
       macro avg       0.85      0.85      0.85       179
    weighted avg       0.85      0.85      0.85       179
    
    

#### Con gridsearch


```python
from sklearn.model_selection import GridSearchCV

# Definir espacio de búsqueda
param_grid_histgbc = {
    'learning_rate': [0.01, 0.001, 0.005, 0.05, 0.1, 0.2],
    'max_iter': [100, 200, 300],
    'max_leaf_nodes': [15, 31, 63],
    'max_depth': [None, 5, 7, 9]
}

# Modelo base
histgbc_base = HistGradientBoostingClassifier(random_state=42)

# Configurar GridSearchCV
grid_search_histgbc = GridSearchCV(estimator=histgbc_base,
                                   param_grid=param_grid_histgbc,
                                   cv=5,
                                   scoring='accuracy',
                                   n_jobs=-1,
                                   verbose=2)

# Ejecutar GridSearchCV
grid_search_histgbc.fit(X_train, y_train)

# Mostrar mejor resultado
print(f"\nMejores hiperparámetros encontrados para HistGradientBoostingClassifier:")
print(grid_search_histgbc.best_params_)

print(f"\nMejor accuracy en validación cruzada: {grid_search_histgbc.best_score_:.4f}")

```

    Fitting 5 folds for each of 216 candidates, totalling 1080 fits
    
    Mejores hiperparámetros encontrados para HistGradientBoostingClassifier:
    {'learning_rate': 0.01, 'max_depth': 5, 'max_iter': 200, 'max_leaf_nodes': 31}
    
    Mejor accuracy en validación cruzada: 0.8258
    


```python
# Evaluar mejor modelo en X_valid
best_histgbc_model = grid_search_histgbc.best_estimator_
y_pred_best_histgbc = best_histgbc_model.predict(X_valid)

# Métricas
print("\nResultados del mejor modelo HistGradientBoostingClassifier en conjunto de validación:")
print(f"Accuracy: {accuracy_score(y_valid, y_pred_best_histgbc):.4f}")
print(f"Precision: {precision_score(y_valid, y_pred_best_histgbc):.4f}")
print(f"Recall: {recall_score(y_valid, y_pred_best_histgbc):.4f}")
print(f"F1 Score: {f1_score(y_valid, y_pred_best_histgbc):.4f}")

# Matriz de confusión
conf_matrix_best_histgbc = confusion_matrix(y_valid, y_pred_best_histgbc)
print("\nMatriz de confusión:")
print(conf_matrix_best_histgbc)

# Reporte completo
print("\nClassification Report:")
print(classification_report(y_valid, y_pred_best_histgbc))

```

    
    Resultados del mejor modelo HistGradientBoostingClassifier en conjunto de validación:
    Accuracy: 0.8212
    Precision: 0.8088
    Recall: 0.7432
    F1 Score: 0.7746
    
    Matriz de confusión:
    [[92 13]
     [19 55]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.83      0.88      0.85       105
               1       0.81      0.74      0.77        74
    
        accuracy                           0.82       179
       macro avg       0.82      0.81      0.81       179
    weighted avg       0.82      0.82      0.82       179
    
    

El modelo HistGradientBoostingClassifier mostró un rendimiento excelente en este problema.
Entrenado sin optimización, el modelo logró la mejor combinación de métricas entre todos los modelos probados:

- Accuracy 85.47%

- Recall 79.73% → muy importante, ya que interesa recuperar correctamente a los pasajeros supervivientes.

- F1 Score 81.94%, superior a los obtenidos con Random Forest y XGBoost.

Aplicando posteriormente una optimización con GridSearchCV, pero en este caso el modelo base sin optimización mostró un mejor equilibrio en el conjunto de validación.

#### Aplicar en df_test


```python
# Predecir sobre test_df con HistGradientBoostingClassifier (modelo sin GridSearch)

# Ojo: en test_df no tengo 'Survived', pero sí PassengerId
# Uso solo las columnas que el modelo ha visto

# Hacemos la predicción
X_test = test_df.drop('PassengerId', axis=1)
y_test_pred_histgbc = histgbc_model.predict(X_test)

# Crear dataframe para la submission
submission_df_histgbc = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': y_test_pred_histgbc
})

# Guardar el CSV en la ruta que me diste
output_path_histgbc = r'C:\Users\kevin.vargas\Desktop\ds_kevin_vargas\0_titanic\resultados\submission_histboost.csv'
submission_df_histgbc.to_csv(output_path_histgbc, index=False)

print(f"\nArchivo de resultados guardado en: {output_path_histgbc}")
submission_df_histgbc

```

    
    Archivo de resultados guardado en: C:\Users\kevin.vargas\Desktop\ds_kevin_vargas\0_titanic\resultados\submission_histboost.csv
    


```python
from sklearn.inspection import permutation_importance

# Calcular permutation importance en el conjunto de validación
perm_importance_histgbc = permutation_importance(histgbc_model, X_valid, y_valid, n_repeats=10, random_state=42, n_jobs=-1)

# Crear dataframe ordenado
perm_importance_df = pd.DataFrame({
    'Feature': X_valid.columns,
    'Importance': perm_importance_histgbc.importances_mean
}).sort_values(by='Importance', ascending=False)

# Visualizar
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=perm_importance_df, palette='magma')
plt.title('Permutation Importance - HistGradientBoostingClassifier')
plt.xlabel('Importancia media (Permutation)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


```

    C:\Users\kevin.vargas\AppData\Local\Temp\ipykernel_12872\114272416.py:14: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(x='Importance', y='Feature', data=perm_importance_df, palette='magma')
    


    
![png](0_titanic%20-%20copia_files/0_titanic%20-%20copia_101_1.png)
    


Elección del modelo final: HistGradientBoostingClassifier sin GridSearch
Tras comparar varios modelos de clasificación, incluyendo:

- Regresión Logística

- Random Forest (con y sin GridSearch)

- XGBoost (con y sin GridSearch)

- HistGradientBoostingClassifier (con y sin GridSearch)

el modelo que ha mostrado el mejor rendimiento global en el conjunto de validación ha sido: HistGradientBoostingClassifier sin optimización (modelo base).

Resultados obtenidos:
Modelo	Accuracy	Precision	Recall	F1 Score
HistGradientBoostingClassifier (base)	0.8547	0.8429	0.7973	0.8194

Este modelo supera claramente en todas las métricas relevantes al resto de modelos evaluados, incluidos los optimizados con GridSearch.

Razones para elegir este modelo:

- Uso de histogramas para una mayor eficiencia computacional.

- Soporte nativo para valores faltantes.

- Menor riesgo de overfitting frente a Random Forest.

En este caso, el uso de GridSearch con métricas de optimización basadas únicamente en accuracy provocó que los modelos optimizados se volvieran más conservadores, sacrificando Recall y F1 Score.

El modelo base de HistGradientBoostingClassifier, sin GridSearch, mostró un excelente equilibrio entre:

- Precision (84%) → pocas falsas alarmas.

- Recall (80%) → excelente capacidad de identificar correctamente a los pasajeros que sobrevivieron.

- F1 Score (81.94%) → mejor combinación Precision/Recall de todos los modelos probados.

Por este motivo, se selecciona HistGradientBoostingClassifier sin optimización como modelo final para generar la submission.

Análisis de Feature Importance

Para evaluar las variables más relevantes para el modelo, se aplicó la técnica de Permutation Importance sobre el conjunto de validación.

Esta técnica mide el impacto real de cada feature en la métrica de rendimiento del modelo.

Principales variables identificadas:
- Title_Mr → el título del pasajero (Mr, Mrs, Miss, etc.) ha resultado ser la variable más influyente en la predicción de supervivencia.

- Sex → el sexo del pasajero es otra variable clave, como cabía esperar.

- Pclass → la clase del billete (indicador indirecto de status socioeconómico).

- Age → la edad también influye de manera notable en la supervivencia.

- Fare → el precio del billete aporta información adicional relacionada con el perfil del pasajero.

Este análisis corrobora los resultados del análisis exploratorio inicial y proporciona una mayor interpretabilidad al modelo final.

## 1.7 Elección del modelo


```python
import pandas as pd

# Definir los modelos y las métricas que has obtenido (rellena con tus valores reales)
model_names = [
    'LogisticRegression',
    'RandomForestClassifier',
    'RandomForest (GridSearch)',
    'XGBoost',
    'XGBoost (GridSearch)',
    'HistGradientBoostingClassifier',
    'HistGradientBoosting (GridSearch)'
]

# Los valores de las métricas que has obtenido (puedes copiar los de las imágenes y outputs)
accuracy = [0.8156, 0.8324, 0.8324, 0.8324, 0.8101, 0.8547, 0.8212]
precision = [0.7808, 0.7895, 0.8235, 0.8143, 0.8030, 0.8429, 0.8088]
recall =    [0.7703, 0.8108, 0.7568, 0.7703, 0.7162, 0.7973, 0.7432]
f1_score =  [0.7755, 0.8000, 0.7887, 0.7917, 0.7571, 0.8194, 0.7746]

# Crear dataframe
results_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1_score
})

# Definir el modelo seleccionado (por ejemplo, el HistGradientBoostingClassifier base)
selected_model = 'HistGradientBoostingClassifier'

# Función para resaltar el modelo seleccionado en rojo
def highlight_selected(row):
    color = 'background-color: lightcoral' if row['Model'] == selected_model else ''
    return [color] * len(row)

# Mostrar la tabla con el estilo
results_df.style.apply(highlight_selected, axis=1).format({'Accuracy': "{:.4f}", 'Precision': "{:.4f}", 'Recall': "{:.4f}", 'F1 Score': "{:.4f}"})

```




<style type="text/css">
#T_20f26_row5_col0, #T_20f26_row5_col1, #T_20f26_row5_col2, #T_20f26_row5_col3, #T_20f26_row5_col4 {
  background-color: lightcoral;
}
</style>
<table id="T_20f26">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_20f26_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_20f26_level0_col1" class="col_heading level0 col1" >Accuracy</th>
      <th id="T_20f26_level0_col2" class="col_heading level0 col2" >Precision</th>
      <th id="T_20f26_level0_col3" class="col_heading level0 col3" >Recall</th>
      <th id="T_20f26_level0_col4" class="col_heading level0 col4" >F1 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_20f26_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_20f26_row0_col0" class="data row0 col0" >LogisticRegression</td>
      <td id="T_20f26_row0_col1" class="data row0 col1" >0.8156</td>
      <td id="T_20f26_row0_col2" class="data row0 col2" >0.7808</td>
      <td id="T_20f26_row0_col3" class="data row0 col3" >0.7703</td>
      <td id="T_20f26_row0_col4" class="data row0 col4" >0.7755</td>
    </tr>
    <tr>
      <th id="T_20f26_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_20f26_row1_col0" class="data row1 col0" >RandomForestClassifier</td>
      <td id="T_20f26_row1_col1" class="data row1 col1" >0.8324</td>
      <td id="T_20f26_row1_col2" class="data row1 col2" >0.7895</td>
      <td id="T_20f26_row1_col3" class="data row1 col3" >0.8108</td>
      <td id="T_20f26_row1_col4" class="data row1 col4" >0.8000</td>
    </tr>
    <tr>
      <th id="T_20f26_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_20f26_row2_col0" class="data row2 col0" >RandomForest (GridSearch)</td>
      <td id="T_20f26_row2_col1" class="data row2 col1" >0.8324</td>
      <td id="T_20f26_row2_col2" class="data row2 col2" >0.8235</td>
      <td id="T_20f26_row2_col3" class="data row2 col3" >0.7568</td>
      <td id="T_20f26_row2_col4" class="data row2 col4" >0.7887</td>
    </tr>
    <tr>
      <th id="T_20f26_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_20f26_row3_col0" class="data row3 col0" >XGBoost</td>
      <td id="T_20f26_row3_col1" class="data row3 col1" >0.8324</td>
      <td id="T_20f26_row3_col2" class="data row3 col2" >0.8143</td>
      <td id="T_20f26_row3_col3" class="data row3 col3" >0.7703</td>
      <td id="T_20f26_row3_col4" class="data row3 col4" >0.7917</td>
    </tr>
    <tr>
      <th id="T_20f26_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_20f26_row4_col0" class="data row4 col0" >XGBoost (GridSearch)</td>
      <td id="T_20f26_row4_col1" class="data row4 col1" >0.8101</td>
      <td id="T_20f26_row4_col2" class="data row4 col2" >0.8030</td>
      <td id="T_20f26_row4_col3" class="data row4 col3" >0.7162</td>
      <td id="T_20f26_row4_col4" class="data row4 col4" >0.7571</td>
    </tr>
    <tr>
      <th id="T_20f26_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_20f26_row5_col0" class="data row5 col0" >HistGradientBoostingClassifier</td>
      <td id="T_20f26_row5_col1" class="data row5 col1" >0.8547</td>
      <td id="T_20f26_row5_col2" class="data row5 col2" >0.8429</td>
      <td id="T_20f26_row5_col3" class="data row5 col3" >0.7973</td>
      <td id="T_20f26_row5_col4" class="data row5 col4" >0.8194</td>
    </tr>
    <tr>
      <th id="T_20f26_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_20f26_row6_col0" class="data row6 col0" >HistGradientBoosting (GridSearch)</td>
      <td id="T_20f26_row6_col1" class="data row6 col1" >0.8212</td>
      <td id="T_20f26_row6_col2" class="data row6 col2" >0.8088</td>
      <td id="T_20f26_row6_col3" class="data row6 col3" >0.7432</td>
      <td id="T_20f26_row6_col4" class="data row6 col4" >0.7746</td>
    </tr>
  </tbody>
</table>




Al visualizar esta tabla podemos ver claramente que los mejores parametros son los que se obtiene del modelo HistGradienteBoostingClassifier tanto en accuracy, precision, recall (el segundo mas alto) y F1 score. Por lo que elijo ese modelo para realizar las predicciones y obtener el archivo submission_histboost.

## 1.8 Conclusiones

#  Conclusiones del análisis y del proyecto

A lo largo de este proyecto se ha desarrollado un pipeline completo para resolver el clásico problema de clasificación del dataset del Titanic:

 **Objetivo:** predecir la supervivencia de los pasajeros en función de distintas variables disponibles.

---

##  Metodología seguida

1. **Análisis exploratorio (EDA)**

    - Exploración de valores faltantes.
    - Análisis univariado y bivariado.
    - Identificación de outliers.
    - Extracción de nuevas features relevantes (`Title`, `FamilySize`).

2. **Preprocesado y Feature Engineering**

    - Imputación de valores faltantes basada en la distribución de los datos.
    - One-hot encoding para variables categóricas.
    - Normalización del pipeline para asegurar reproducibilidad.

3. **Evaluación de modelos**

    Se entrenaron y evaluaron los siguientes modelos:

    - `LogisticRegression`
    - `RandomForestClassifier` (con y sin GridSearch)
    - `XGBoost` (con y sin GridSearch)
    - `HistGradientBoostingClassifier` (con y sin GridSearch)

4. **Validación**

    - Se realizó validación cruzada y análisis del conjunto de validación.
    - Se evaluaron múltiples métricas: **Accuracy**, **Precision**, **Recall**, **F1 Score**.

---

##  Resultados

Como se refleja en la tabla resumen de métricas obtenidas:

| Modelo                           | Accuracy | Precision | Recall | F1 Score |
|----------------------------------|----------|-----------|--------|----------|
| **HistGradientBoostingClassifier**   | **0.8547**   | **0.8429**    | **0.7973** | **0.8194**   |

El modelo **HistGradientBoostingClassifier sin GridSearch** ha sido seleccionado como el modelo final por mostrar:

- El mejor equilibrio global en las métricas clave.  
- La mayor capacidad de generalización.  
- Robustez frente a overfitting.  
- Una excelente interpretación de Feature Importance (Permutation Importance).

---

##  Insights adicionales

El análisis de **Permutation Importance** ha revelado que las variables más influyentes para predecir la supervivencia son:

- **Title** del pasajero (`Title_Mr`, `Title_Miss`, etc.).
- **Sexo** (`Sex`).
- **Clase del billete** (`Pclass`).
- **Edad** (`Age`).
- **Tarifa del billete** (`Fare`).

Este resultado es coherente con el conocimiento histórico del Titanic, donde factores socioeconómicos y demográficos tuvieron un impacto relevante en las probabilidades de supervivencia.

---
