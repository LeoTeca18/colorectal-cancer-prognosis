import os
import joblib
import pandas as pd
from typing import List
from src.domain.patient import Patient, PrognosisResult
from src.interfaces.inference import IInferenceEngine
from src.infrastructure.preprocessor import ClinicalDataPreprocessor
from src.infrastructure.explainer import ModelExplainer

class XGBoostInferenceEngine(IInferenceEngine):
    """
    Motor de Inferência Real utilizando o classificador XGBoost treinado.
    Consome o modelo como uma caixa-preta isolada, utilizando as classes de pré-processamento e explicabilidade (LIME).
    """
    
    def __init__(self, model_path: str = None):
        if model_path is None:
            # Caminho padrão relativo ao arquivo actual
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "models", "model_xgboost.pkl")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Ficheiro do modelo XGBoost não encontrado em: {model_path}")
            
        self._model = joblib.load(model_path)
        self._preprocessor = ClinicalDataPreprocessor()
        self._explainer = ModelExplainer(self._model, self._preprocessor.feature_names)

    def predict(self, patient: Patient) -> PrognosisResult:
        # 1. Módulo de Entrada e Pré-processamento (IQR Capping, Imputação, Engenharia e Padronização)
        X = self._preprocessor.preprocess_patient(patient)
        
        # 2. Módulo Preditivo: Consumindo o Modelo como Caixa-Preta
        # predict_proba() para extrair probabilidade de risco do evento DFS (Classe 0 = Recorrência)
        proba_recurrence = self._model.predict_proba(X)[0][0]
        
        # predict() genérico para obter a predição da classe (0.0 ou 1.0)
        predicted_class = self._model.predict(X)[0]
        
        # 3. Módulo de Interpretabilidade Local (LIME XAI)
        explanation_weights = self._explainer.explain_patient(patient)
        
        # Determinar classe de risco baseada no limiar de recorrência clínico
        if proba_recurrence < 0.30:
            risk = "Baixo"
        elif proba_recurrence < 0.65:
            risk = "Médio"
        else:
            risk = "Alto"
            
        # Estimativa de DFS (Disease-Free Survival em meses)
        stage = patient.dukes_stage.strip().upper()
        if stage == 'A':
            base_survival = 96.0
        elif stage == 'B':
            base_survival = 72.0
        elif stage == 'C':
            base_survival = 48.0
        else:  # 'D'
            base_survival = 18.0
            
        age_factor = (patient.age - 60) / 100.0
        
        treatment_benefit = 0.0
        if patient.adj_chem:
            treatment_benefit += 0.08
        if patient.adj_radio:
            treatment_benefit += 0.05
            
        # Modulação dinâmica do tempo livre de doença baseando-se no predict() e predict_proba()
        # Se predicted_class for 0.0 (Recorrência), reduzimos o DFS
        class_factor = 0.45 if predicted_class == 0.0 else 1.0
        predicted_survival = class_factor * ((1.2 - proba_recurrence) * base_survival) - (age_factor * 12.0) + (treatment_benefit * 60.0)
        predicted_survival = max(1.0, min(120.0, predicted_survival))
        
        class_str = "Recorrência" if predicted_class == 0.0 else "Sem Recorrência"
        details = (
            f"Previsão gerada pelo motor de inferência real (XGBoost Classifier).\n"
            f"Probabilidade calculada de recorrência: {proba_recurrence * 100:.1f}%. Classe prevista: {int(predicted_class)} ({class_str}).\n"
            f"Factores clínicos de entrada: Dukes Stage {stage}, Localização {patient.location}, "
            f"Idade {patient.age} anos, Terapias Adjuvantes (Quimioterapia: {patient.adj_chem}, Radioterapia: {patient.adj_radio})."
        )
        
        return PrognosisResult(
            patient_id=patient.patient_id,
            recurrence_probability=round(float(proba_recurrence), 3),
            risk_level=risk,
            predicted_survival_months=round(float(predicted_survival), 1),
            details=details,
            explanation=explanation_weights
        )
