# Filtro de Spam con Interfaz Gráfica

Este proyecto implementa un clasificador de spam utilizando aprendizaje automático con una interfaz gráfica para facilitar su uso.

## Integrantes

### Chaparro Castillo Christopher
### Peñuelas López Luis Antonio

## Descripción

El sistema analiza correos electrónicos y determina si son spam o no spam (ham) utilizando técnicas de procesamiento de lenguaje natural y aprendizaje automático. 

### Características principales:

- Interfaz gráfica para el análisis de correos
- Procesamiento de texto utilizando TF-IDF
- Implementación de dos modelos:
  - Clasificador Bayesiano
  - Regresión Logística
- Métricas de rendimiento del modelo (precisión y recuperación)

## Funcionamiento del Código

### 1. Preprocesamiento de Datos
```python
# Carga y limpieza de datos
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')
mail_data = raw_mail_data.drop_duplicates(subset=['text'], keep='first')
```
- Carga los datos del dataset 'spam_assassin.csv'
- Elimina valores nulos y duplicados
- Limpia el texto removiendo caracteres especiales
- Elimina las stop words (palabras comunes que no aportan significado)

### 2. Vectorización del Texto
```python
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)
```
- Convierte el texto en vectores numéricos usando TF-IDF
- Permite que el modelo procese el texto de manera eficiente

### 3. Modelos de Clasificación

#### Modelo Bayesiano
```python
# Calcula probabilidades previas y condicionales
P_spam = Y_train.mean()
P_no_spam = 1 - P_spam
```
- Implementa un clasificador Naive Bayes
- Calcula probabilidades de spam/no spam
- Usa el teorema de Bayes para la clasificación

#### Regresión Logística
```python
model = LogisticRegression()
model.fit(X_train_features, Y_train)
```
- Entrena un modelo de regresión logística
- Proporciona predicciones binarias (spam/no spam)

### 4. Interfaz Gráfica

La interfaz incluye campos para:
- From (remitente)
- To (destinatario)
- Subject (asunto)
- Date (fecha)
- Message Body (cuerpo del mensaje)
- Additional Headers (cabeceras adicionales)

Funcionalidades:
- Botón "Analizar Correo": Realiza la predicción
- Botón "Limpiar Campos": Reinicia todos los campos
- Visualización de resultados con código de colores
- Métricas del modelo visibles

## Requisitos
- numpy
- scikit-learn
- nltk
- tkinter

## Uso

1. Ejecute el script principal:
```bash
python3 SpamFilter.py
```

2. En la interfaz gráfica:
   - Complete los campos del correo
   - Presione "Analizar Correo"
   - El resultado se mostrará en color:
     - Verde: No es spam
     - Rojo: Es spam

## Métricas de Rendimiento

El sistema muestra dos métricas principales:
- Precisión: Porcentaje de predicciones correctas
- Recuperación: Capacidad para identificar todos los casos positivos

## Estructura del Proyecto

```
├── SpamFilter.py          # Código principal
├── Datasets/
│   └── spam_assassin.csv  # Dataset de entrenamiento
└── README.md             # Documentación
```

## Notas Adicionales

- El modelo se entrena con cada ejecución del programa
- Los correos se procesan de manera similar al conjunto de entrenamiento
- La interfaz permite analizar múltiples correos sin necesidad de reiniciar

## Ejecución

### No-Spam
![image](https://github.com/user-attachments/assets/2c1fc96e-a473-48f7-818e-73db1129980b)

### Spam
![image](https://github.com/user-attachments/assets/9471a78b-f515-4227-a7f3-af2d504a90a7)

