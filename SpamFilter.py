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
X_train_spam = X_train_features[Y_train.to_numpy() == 1]
X_train_no_spam = X_train_features[Y_train.to_numpy() == 0]

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

# Importar tkinter
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext

class SpamFilterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Filtro de Spam")
        self.root.geometry("700x800")
        
        # Frame principal con scroll
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Campos del correo
        self.create_email_fields(main_frame)
        
        # Botones
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.analyze_button = ttk.Button(button_frame, text="Analizar Correo", command=self.analyze_email)
        self.analyze_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = ttk.Button(button_frame, text="Limpiar Campos", command=self.clear_fields)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Área de resultados
        result_frame = ttk.LabelFrame(main_frame, text="Resultado del Análisis")
        result_frame.pack(fill=tk.X, pady=10)
        
        self.result_text = ttk.Label(result_frame, text="", font=('Arial', 12, 'bold'))
        self.result_text.pack(pady=10)
        
        # Métricas
        metrics_frame = ttk.LabelFrame(main_frame, text="Métricas del modelo")
        metrics_frame.pack(fill=tk.X, pady=10)
        
        self.precision_label = ttk.Label(metrics_frame, text=f"Precisión: {precision:.2%}")
        self.precision_label.pack(pady=5)
        
        self.recall_label = ttk.Label(metrics_frame, text=f"Recuperación: {recuperacion:.2%}")
        self.recall_label.pack(pady=5)

    def create_email_fields(self, parent):
        # Frame para los campos del correo
        email_frame = ttk.LabelFrame(parent, text="Datos del Correo")
        email_frame.pack(fill=tk.X, pady=10)
        
        # From
        ttk.Label(email_frame, text="From:").pack(anchor=tk.W, padx=5, pady=2)
        self.from_entry = ttk.Entry(email_frame, width=60)
        self.from_entry.pack(fill=tk.X, padx=5, pady=2)
        
        # To
        ttk.Label(email_frame, text="To:").pack(anchor=tk.W, padx=5, pady=2)
        self.to_entry = ttk.Entry(email_frame, width=60)
        self.to_entry.pack(fill=tk.X, padx=5, pady=2)
        
        # Subject
        ttk.Label(email_frame, text="Subject:").pack(anchor=tk.W, padx=5, pady=2)
        self.subject_entry = ttk.Entry(email_frame, width=60)
        self.subject_entry.pack(fill=tk.X, padx=5, pady=2)
        
        # Date
        ttk.Label(email_frame, text="Date:").pack(anchor=tk.W, padx=5, pady=2)
        self.date_entry = ttk.Entry(email_frame, width=60)
        self.date_entry.pack(fill=tk.X, padx=5, pady=2)
        
        # Message Body
        ttk.Label(email_frame, text="Message Body:").pack(anchor=tk.W, padx=5, pady=2)
        self.body_text = scrolledtext.ScrolledText(email_frame, width=60, height=10)
        self.body_text.pack(fill=tk.BOTH, padx=5, pady=2)
        
        # Headers adicionales
        ttk.Label(email_frame, text="Additional Headers:").pack(anchor=tk.W, padx=5, pady=2)
        self.headers_text = scrolledtext.ScrolledText(email_frame, width=60, height=5)
        self.headers_text.pack(fill=tk.BOTH, padx=5, pady=2)

    def clear_fields(self):
        self.from_entry.delete(0, tk.END)
        self.to_entry.delete(0, tk.END)
        self.subject_entry.delete(0, tk.END)
        self.date_entry.delete(0, tk.END)
        self.body_text.delete('1.0', tk.END)
        self.headers_text.delete('1.0', tk.END)
        self.result_text.config(text="")

    def analyze_email(self):
        email_parts = [
            f"From: {self.from_entry.get()}",
            f"To: {self.to_entry.get()}",
            f"Subject: {self.subject_entry.get()}",
            f"Date: {self.date_entry.get()}",
            self.headers_text.get('1.0', tk.END).strip(),
            "",
            self.body_text.get('1.0', tk.END).strip()
        ]
        
        complete_email = "\n".join(email_parts)
        
        input_mail_features = vectorizer.transform([complete_email])
        prediction = model.predict(input_mail_features)
        
        if prediction[0] == 0:
            result = "Este correo NO es Spam"
            color = "green"
        else:
            result = "Este correo es SPAM"
            color = "red"
            
        self.result_text.config(text=result, foreground=color)

if __name__ == "__main__":
    root = tk.Tk()
    app = SpamFilterGUI(root)
    root.mainloop()
