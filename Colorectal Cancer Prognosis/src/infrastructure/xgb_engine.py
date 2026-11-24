import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
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

    def get_dashboard_analytics(self) -> Dict[str, Any]:
        """
        Constrói dinamicamente todos os indicadores do Painel Clínico a partir dos artefatos
        exportados pelo Notebook (model_xgboost.pkl, model_survival.pkl e preprocessor_params.json).
        """
        params = getattr(self._preprocessor, "raw_params", {}) or {}
        cohort_summary = params.get("cohort_summary", {})
        model_metrics = params.get("model_metrics", {})
        location_epidemiology = params.get("location_epidemiology", [])
        
        # 1. Inferência dinâmica por Estágio de Dukes (A, B, C, D) diretamente nos modelos .pkl
        mean_age = int(round(self._preprocessor.age_mean))
        stages = ["A", "B", "C", "D"]
        stage_patients = [
            Patient(
                patient_id=f"STAGE-{s}",
                age=mean_age,
                gender="Male",
                dukes_stage=s,
                location="Colon",
                adj_radio=False,
                adj_chem=True
            )
            for s in stages
        ]
        X_stages = self._preprocessor.preprocess_batch(stage_patients)
        stage_probs = np.clip(self._model.predict_proba(X_stages)[:, 1], 0.01, 0.99)
        stage_dfs = np.clip(self._survival_model.predict(X_stages), 1.0, 120.0)
        
        # 2. Geração dinâmica das curvas de sobrevida S(t) = exp(-(lambda * t)^gamma)
        # calibradas diretamente pela mediana/média de meses DFS e probabilidade prevista pelo modelo para cada estágio
        times = np.linspace(0, 100, 50)
        survival_curves = {"times": times.tolist(), "stages": {}}
        stage_profiles = []
        gammas = {"A": 1.10, "B": 1.15, "C": 1.20, "D": 1.25}
        stage_Meta = params.get("stage_epidemiology", {})
        
        for idx, s in enumerate(stages):
            pred_prob = float(stage_probs[idx])
            pred_months = float(stage_dfs[idx])
            gamma = gammas[s]
            # Taxa de risco (hazard rate lambda) derivada diretamente da predição dos modelos treinados
            hazard_lambda = (-np.log(max(0.05, 1.0 - pred_prob))) ** (1.0 / gamma) / max(12.0, pred_months)
            curve_y = (np.exp(-((hazard_lambda * times) ** gamma)) * 100.0).tolist()
            survival_curves["stages"][s] = curve_y
            
            emp_info = stage_Meta.get(s, {})
            stage_profiles.append({
                "Estágio": f"Dukes {s}",
                "Casos na Coorte": emp_info.get("cases", "-"),
                "Prob. Recorrência Modelo (%)": round(pred_prob * 100.0, 1),
                "DFS Previsto Modelo (Meses)": round(pred_months, 1),
                "DFS Médio Real Coorte (Meses)": emp_info.get("mean_dfs_months", round(pred_months, 1))
            })
            
        # 3. Impacto dinâmico da Quimioterapia Adjuvante calculado pelo Classificador .pkl
        p_no_treat = Patient("SIM-NT", mean_age, "Male", "C", "Left", False, False)
        p_chemo = Patient("SIM-CH", mean_age, "Male", "C", "Left", False, True)
        X_comp = self._preprocessor.preprocess_batch([p_no_treat, p_chemo])
        probs_comp = self._model.predict_proba(X_comp)[:, 1]
        dfs_comp = self._survival_model.predict(X_comp)
        rel_risk_diff_pct = round(((float(probs_comp[1]) - float(probs_comp[0])) / max(0.01, float(probs_comp[0]))) * 100.0, 1)
        dfs_gain_months = round(float(dfs_comp[1]) - float(dfs_comp[0]), 1)
        
        # 4. Importância Global das Variáveis extraída diretamente dos pesos dos modelos .pkl
        feature_labels = {
            "Age (in years)": "Idade (Anos)",
            "Dukes Stage": "Estágio de Dukes",
            "Gender_Male": "Género (Masculino)",
            "Location_Left": "Localização: Cólon Esquerdo",
            "Location_Rectum": "Localização: Reto",
            "Location_Right": "Localização: Cólon Direito",
            "Adjuvant_Strategy_Chem_Only": "Terapia: Apenas Quimio",
            "Adjuvant_Strategy_No_Treatment": "Terapia: Sem Tratamento",
            "Adjuvant_Strategy_Radio_Only": "Terapia: Apenas Radio"
        }
        clf_imp = getattr(self._model, "feature_importances_", np.zeros(len(self._preprocessor.feature_names)))
        surv_imp = getattr(self._survival_model, "feature_importances_", np.zeros(len(self._preprocessor.feature_names)))
        feature_importances = []
        for fname, c_val, s_val in zip(self._preprocessor.feature_names, clf_imp, surv_imp):
            feature_importances.append({
                "Atributo": feature_labels.get(fname, fname),
                "Importância Recorrência (%)": round(float(c_val) * 100.0, 1),
                "Importância Sobrevida DFS (%)": round(float(s_val) * 100.0, 1),
                "Importância Média (%)": round((float(c_val) + float(s_val)) * 50.0, 1)
            })
        feature_importances.sort(key=lambda x: x["Importância Média (%)"], reverse=True)
        
        return {
            "cohort_summary": cohort_summary,
            "model_metrics": model_metrics,
            "location_epidemiology": location_epidemiology,
            "survival_curves": survival_curves,
            "stage_profiles": stage_profiles,
            "chemo_impact": {
                "rel_risk_diff_pct": rel_risk_diff_pct,
                "dfs_gain_months": dfs_gain_months
            },
            "feature_importances": feature_importances
        }
