# In[]:
import numpy as np
import pandas as pd
from sklearn import feature_extraction
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# In[]:
# Cargar datos
raw_mail_data = pd.read_csv('Datasets/spam_assassin.csv')
# In[]:
# Imprimiendo dataframe antes del preprocesamiento
print(raw_mail_data)

# In[]:
# Preprocesamiento
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')
mail_data = raw_mail_data.drop_duplicates(subset=['text'], keep='first')
mail_data['text'] = mail_data['text'].str.lower().str.replace(r"[^a-zA-Z0-9 ]", " ", regex=True)
mail_data['text'] = mail_data['text'].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))

# In[]:
# Imprimiendo dataframe despues del preprocesamiento
print(mail_data.head())

# In[]:
# Tamaño del dataframe
print(mail_data.shape)

# In[]:
# Dividir datos
X = mail_data['text']
Y = mail_data['target'].astype(int)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=3)

# In[]:
# Imprimimos el dataframe de X(Input)
print(X)

# In[]:
# Imprimimos el dataframe de Y(Output)
print(Y)

# In[]:
# Vectorización
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)
# In[]:
# Imprimir dataframe de X(Input) vectorizados
print(X_train_features)
# In[]:
# Modelo por el teorema de Bayes
# Probabilidades previas
P_spam = Y_train.mean()
P_no_spam = 1 - P_spam

# Probabilidades condicionales (usando matrices dispersas)
X_train_spam = X_train_features[Y_train == 1]
X_train_no_spam = X_train_features[Y_train == 0]

P_caracteristicas_spam = (X_train_spam.sum(axis=0) + 1) / (X_train_spam.sum() + X_train_features.shape[1])
P_caracteristicas_no_spam = (X_train_no_spam.sum(axis=0) + 1) / (X_train_no_spam.sum() + X_train_features.shape[1])

# Probabilidad de spam dado características
log_P_spam_caracteristicas = np.log(P_spam) + X_test_features @ np.log(P_caracteristicas_spam.T)
log_P_no_spam_caracteristicas = np.log(P_no_spam) + X_test_features @ np.log(P_caracteristicas_no_spam.T)

# In[]:
# Clasificación
clasificaciones = (log_P_spam_caracteristicas > log_P_no_spam_caracteristicas).astype(int)
clasificaciones = clasificaciones.ravel()
Y_test = Y_test.to_numpy().ravel()

# In[]:
# Evaluación
precision = np.mean(clasificaciones == Y_test)
recuperacion = np.sum((clasificaciones == 1) & (Y_test == 1)) / Y_test.sum()

# In[]:
# Entrenando el modelo
model = LogisticRegression()

# Entrenando el modelo de regresión logística con los datos de entrenamiento
model.fit(X_train_features, Y_train)

# In[]:
# Construyendo un sistema de predicción
mail = input('Email: ')
input_mail = [mail]

# Convertirlo en un vector de caracteristicas
input_mail_features = vectorizer.transform(input_mail)

# Haciendo la predicción
prediction = model.predict(input_mail_features)

# In[]:
# Imprimir resultados
print(prediction)

if(prediction[0]==0):
    print('Correo No Spam')

else:
    print('Correo Spam')

print('Precisión:', precision)
print('Recuperación:', recuperacion)