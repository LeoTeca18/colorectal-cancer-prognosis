"""
Script de Treinamento e Serialização de Modelos Clínicos para Prognóstico de Câncer Colorretal.
Gera:
1. model_xgboost.pkl: Classificador de Risco de Recorrência (XGBClassifier com restrições monotônicas clínicas).
2. model_survival.pkl: Regressor de Sobrevida Livre de Doença - DFS em meses (XGBRegressor com restrições monotônicas clínicas).
3. preprocessor_params.json: Parâmetros exatos de normalização estatística (StandardScaler para Idade e Dukes Stage).
"""

import os
import json
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib

def train_and_export():
    # 1. Localizar o dataset oficial
    possible_paths = [
        r"C:\Users\Lenovo\.cache\kagglehub\datasets\amandam1\colorectal-cancer-patients\versions\3\Colorectal Cancer Patient Data.csv",
        os.path.join(os.path.expanduser("~"), ".cache", "kagglehub", "datasets", "amandam1", "colorectal-cancer-patients", "**", "Colorectal Cancer Patient Data.csv")
    ]
    
    dataset_path = None
    for p in possible_paths:
        matches = glob.glob(p, recursive=True)
        if matches:
            dataset_path = matches[0]
            break
            
    if not dataset_path or not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Arquivo 'Colorectal Cancer Patient Data.csv' não encontrado em: {possible_paths}")
        
    print(f"Carregando dataset de: {dataset_path}")
    df = pd.read_csv(dataset_path).dropna().copy()
    print(f"Total de registros válidos: {len(df)}")
    
    # 2. Engenharia de Variáveis
    def calcular_estrategia(row):
        if row['Adj_Radio'] == 1.0 and row['Adj_Chem'] == 1.0:
            return 'Both_Therapies'
        elif row['Adj_Radio'] == 1.0:
            return 'Radio_Only'
        elif row['Adj_Chem'] == 1.0:
            return 'Chem_Only'
        else:
            return 'No_Treatment'

    df['Adjuvant_Strategy'] = df.apply(calcular_estrategia, axis=1)
    
    # Codificação Ordinal Clínica Real: A=0, B=1, C=2, D=3
    stage_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    df['Dukes_Num'] = df['Dukes Stage'].map(stage_map)
    
    # 3. Normalização Estatística (Z-Score)
    scaler_age = StandardScaler()
    age_scaled = scaler_age.fit_transform(df[['Age (in years)']]).ravel()
    
    scaler_stage = StandardScaler()
    stage_scaled = scaler_stage.fit_transform(df[['Dukes_Num']]).ravel()
    
    feature_names = [
        'Age (in years)', 
        'Dukes Stage', 
        'Gender_Male', 
        'Location_Left', 
        'Location_Rectum', 
        'Location_Right', 
        'Adjuvant_Strategy_Chem_Only', 
        'Adjuvant_Strategy_No_Treatment', 
        'Adjuvant_Strategy_Radio_Only'
    ]
    
    X = pd.DataFrame({
        'Age (in years)': age_scaled,
        'Dukes Stage': stage_scaled,
        'Gender_Male': (df['Gender'] == 'Male').astype(float),
        'Location_Left': (df['Location'] == 'Left').astype(float),
        'Location_Rectum': (df['Location'] == 'Rectum').astype(float),
        'Location_Right': (df['Location'] == 'Right').astype(float),
        'Adjuvant_Strategy_Chem_Only': (df['Adjuvant_Strategy'] == 'Chem_Only').astype(float),
        'Adjuvant_Strategy_No_Treatment': (df['Adjuvant_Strategy'] == 'No_Treatment').astype(float),
        'Adjuvant_Strategy_Radio_Only': (df['Adjuvant_Strategy'] == 'Radio_Only').astype(float)
    })
    
    y_event = df['DFS event'].astype(int)
    y_time = df['DFS (in months)'].astype(float)
    
    # 4. Treinamento do Classificador com Restrições Monotônicas Clínicas
    # Idade (+1), Dukes (+1), Reto (+1), Direita (-1), Quimio (-1), Sem Tratamento (+1)
    monotone_clf = (1, 1, 0, 0, 1, -1, -1, 1, 0)
    clf = xgb.XGBClassifier(
        n_estimators=45,
        max_depth=3,
        learning_rate=0.07,
        monotone_constraints=monotone_clf,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric='logloss'
    )
    clf.fit(X, y_event)
    print("Classificador XGBoost treinado com sucesso.")
    
    # 5. Treinamento do Regressor de Sobrevida (DFS em meses) com Restrições Monotônicas
    # Idade (-1), Dukes (-1), Reto (-1), Direita (+1), Quimio (+1), Sem Tratamento (-1)
    monotone_surv = (-1, -1, 0, 0, -1, 1, 1, -1, 0)
    surv_reg = xgb.XGBRegressor(
        n_estimators=45,
        max_depth=3,
        learning_rate=0.07,
        monotone_constraints=monotone_surv,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric='rmse'
    )
    surv_reg.fit(X, y_time)
    print("Regressor de Sobrevida XGBoost treinado com sucesso.")
    
    # 6. Serialização dos Artefatos
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir)
    os.makedirs(models_dir, exist_ok=True)
    
    model_clf_path = os.path.join(models_dir, "model_xgboost.pkl")
    model_surv_path = os.path.join(models_dir, "model_survival.pkl")
    params_path = os.path.join(models_dir, "preprocessor_params.json")
    
    joblib.dump(clf, model_clf_path)
    joblib.dump(surv_reg, model_surv_path)
    
    params = {
        "age_mean": float(scaler_age.mean_[0]),
        "age_scale": float(scaler_age.scale_[0]),
        "stage_mean": float(scaler_stage.mean_[0]),
        "stage_scale": float(scaler_stage.scale_[0]),
        "feature_names": feature_names
    }
    
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4)
        
    print(f"Artefatos salvos em: {models_dir}")
    print(f" - {model_clf_path}")
    print(f" - {model_surv_path}")
    print(f" - {params_path}")

if __name__ == "__main__":
    train_and_export()
