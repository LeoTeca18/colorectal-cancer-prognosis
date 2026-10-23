import os
import joblib
import pandas as pd
from typing import List, Optional
from src.domain.patient import Patient, PrognosisResult
from src.interfaces.inference import IInferenceEngine
from src.infrastructure.preprocessor import ClinicalDataPreprocessor
from src.infrastructure.explainer import ModelExplainer

class XGBoostInferenceEngine(IInferenceEngine):
    """
    Motor de Inferência Baseado em Machine Learning (XGBoost).
    Combina:
    1. Modelo de Classificação (XGBClassifier): Estima a probabilidade clínica de recorrência/falha (DFS Event).
    2. Modelo de Sobrevida (XGBRegressor): Estima a expectativa de tempo livre de doença (DFS em meses).
    3. Interpretabilidade Local (LIME): Identifica a contribuição de cada fator para o paciente.
    """
    
    def __init__(self, model_path: Optional[str] = None, survival_model_path: Optional[str] = None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "models")
        
        if model_path is None:
            model_path = os.path.join(models_dir, "model_xgboost.pkl")
        if survival_model_path is None:
            survival_model_path = os.path.join(models_dir, "model_survival.pkl")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo XGBoost de classificação não encontrado em: {model_path}")
        if not os.path.exists(survival_model_path):
            raise FileNotFoundError(f"Modelo XGBoost de sobrevida não encontrado em: {survival_model_path}")
            
        self._model = joblib.load(model_path)
        self._survival_model = joblib.load(survival_model_path)
        self._preprocessor = ClinicalDataPreprocessor()
        self._explainer = ModelExplainer(self._model, self._preprocessor)

    def predict(self, patient: Patient) -> PrognosisResult:
        # 1. Pré-processamento padronizado (Matriz de 9 atributos com normalização e dummies)
        X = self._preprocessor.preprocess_patient(patient)
        
        # 2. Inferência Pura do Modelo Preditivo (XGBClassifier)
        # Extrai diretamente a probabilidade da classe 1 (Recorrência de Doença)
        proba_recurrence = float(self._model.predict_proba(X)[0][1])
        proba_recurrence = max(0.01, min(0.99, proba_recurrence))
        
        predicted_class = 1.0 if proba_recurrence >= 0.50 else 0.0
        
        # Estratificação de Risco Clínico Baseada em Limiares Calibrados
        if proba_recurrence < 0.40:
            risk = "Baixo"
        elif proba_recurrence < 0.70:
            risk = "Médio"
        else:
            risk = "Alto"
            
        # 3. Inferência do Modelo Real de Sobrevida (XGBRegressor)
        # Estima a duração esperada de Disease-Free Survival (DFS em meses) a partir dos pesos treinados
        predicted_survival = float(self._survival_model.predict(X)[0])
        predicted_survival = max(1.0, min(120.0, predicted_survival))
        
        # 4. Explicabilidade Clínica Local (LIME)
        explanation_weights = self._explainer.explain_patient(patient)
        
        stage = patient.dukes_stage.strip().upper()
        class_str = "Recorrência Prevista" if predicted_class == 1.0 else "Livre de Recorrência"
        
        adj_desc = []
        if patient.adj_chem:
            adj_desc.append("Quimioterapia")
        if patient.adj_radio:
            adj_desc.append("Radioterapia")
        treatment_str = ", ".join(adj_desc) if adj_desc else "Sem Tratamento Adjuvante"
        
        details = (
            f"Previsão gerada pelo pipeline de Machine Learning (XGBoost Dual Engine).\n"
            f"• Probabilidade de Recorrência: {proba_recurrence * 100:.1f}% ({class_str} - Risco {risk}).\n"
            f"• Expectativa de Sobrevida Livre de Doença (DFS): {predicted_survival:.1f} meses.\n"
            f"• Parâmetros Clínicos: Estágio de Dukes {stage}, Idade {patient.age} anos, "
            f"Localização {patient.location}, Terapias: {treatment_str}."
        )
        
        return PrognosisResult(
            patient_id=patient.patient_id,
            recurrence_probability=round(proba_recurrence, 3),
            risk_level=risk,
            predicted_survival_months=round(predicted_survival, 1),
            details=details,
            explanation=explanation_weights
        )

    def predict_batch(self, patients: List[Patient]) -> List[PrognosisResult]:
        """
        Processamento vetorizado de inferência em lote para múltiplos pacientes.
        """
        if not patients:
            return []
            
        X_batch = self._preprocessor.preprocess_batch(patients)
        probas = self._model.predict_proba(X_batch)[:, 1]
        survivals = self._survival_model.predict(X_batch)
        
        results = []
        for i, patient in enumerate(patients):
            prob = float(max(0.01, min(0.99, probas[i])))
            surv = float(max(1.0, min(120.0, survivals[i])))
            
            if prob < 0.40:
                risk = "Baixo"
            elif prob < 0.70:
                risk = "Médio"
            else:
                risk = "Alto"
                
            details = (
                f"Triagem em Lote: Paciente {patient.patient_id} classificado com Risco {risk} "
                f"(Probabilidade de Recorrência: {prob * 100:.1f}%, DFS estimado: {surv:.1f} meses)."
            )
            
            results.append(PrognosisResult(
                patient_id=patient.patient_id,
                recurrence_probability=round(prob, 3),
                risk_level=risk,
                predicted_survival_months=round(surv, 1),
                details=details,
                explanation=[]
            ))
            
        return results
