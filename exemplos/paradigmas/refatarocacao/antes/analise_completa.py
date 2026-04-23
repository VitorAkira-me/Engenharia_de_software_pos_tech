# exemplos/refatoracao/antes/analise_completa.py
"""
# Código de análise
# Autor: Desconhecido
# Última modificação: 6 meses atrás
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Variáveis globais
df = None
model = None
trained = False
features = []

def load_data():
    global df
    # Caminho hardcoded
    df = pd.read_csv('C:/Users/antigo/dados/customer_data.csv')
    print(len(df))

def clean_data():
    global df
    df = df.dropna()
    df = df[df['age'] > 0]
    df = df[df['age'] < 120]
    df = df[df['income'] > 0]
    df['income'] = df['income'].fillna(df['income'].mean())
    df['education'] = df['education'].fillna('unknown')
    
def create_features():
    global df, features
    df['income_age_ratio'] = df['income'] / df['age']
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 45, 65, 120], labels=['young', 'adult', 'middle', 'senior'])
    df = pd.get_dummies(df, columns=['education', 'age_group'])
    df['high_value'] = (df['income'] > 100000).astype(int)
    features = [c for c in df.columns if c not in ['customer_id', 'churn']]

def split_data():
    global df, features
    X = df[features]
    y = df['churn']
    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_model(X_train, y_train):
    global model, trained
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5)
    model.fit(X_train, y_train)
    trained = True
    print('model trained')

def evaluate(X_test, y_test):
    global model, trained
    if not trained:
        print('ERROR: model not trained')
        return
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'accuracy: {acc}')
    print(classification_report(y_test, y_pred))
    
def plot_feature_importance():
    global model, features
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print('plot saved')

def predict_new_customer(age, income, education):
    global model, trained, df
    if not trained:
        print('ERROR: train first')
        return None
    
    # Precisa recriar TODAS as features
    new_data = pd.DataFrame({
        'age': [age],
        'income': [income],
        'education': [education],
        'customer_id': [0]
    })
    
    new_data['income_age_ratio'] = new_data['income'] / new_data['age']
    new_data['age_group'] = pd.cut(new_data['age'], bins=[0, 25, 45, 65, 120], labels=['young', 'adult', 'middle', 'senior'])
    new_data = pd.get_dummies(new_data, columns=['education', 'age_group'])
    new_data['high_value'] = (new_data['income'] > 100000).astype(int)
    
    # Problema: precisa ter EXATAMENTE as mesmas colunas
    for col in features:
        if col not in new_data.columns:
            new_data[col] = 0
    
    new_data = new_data[features]
    pred = model.predict(new_data)[0]
    return 'Churn' if pred == 1 else 'No Churn'

def generate_report():
    global df
    print('=== REPORT ===')
    print(f'Total customers: {len(df)}')
    print(f'Churn rate: {df["churn"].mean():.2%}')
    print(f'Avg income: ${df["income"].mean():,.2f}')
    print(f'Avg age: {df["age"].mean():.1f}')

def main():
    load_data()
    clean_data()
    create_features()
    X_train, X_test, y_train, y_test = split_data()
    train_model(X_train, y_train)
    evaluate(X_test, y_test)
    plot_feature_importance()
    generate_report()
    
    # Teste predição
    result = predict_new_customer(35, 75000, 'bachelor')
    print(f'Prediction: {result}')

if __name__ == '__main__':
    main()