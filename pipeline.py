import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Coleta de Dados
df = sns.load_dataset('iris')

# 2. Pré-processamento
X = df.drop('species', axis=1)  # Variáveis independentes
y = df['species']                # Variável dependente

# Dividindo os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizando os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Treinamento do Modelo
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Avaliação do Modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")

# 5. Salvar o Modelo
joblib.dump(model, 'iris_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
