import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from lime.lime_tabular import LimeTabularExplainer
from src.domain.patient import Patient
from src.infrastructure.preprocessor import ClinicalDataPreprocessor

def generate_background_data(preprocessor: ClinicalDataPreprocessor, n_samples: int = 600) -> np.ndarray:
    """
    Gera uma base de calibração para o LIME alinhada com as 5 variáveis clínicas estruturadas:
    0: Idade (normal em torno da média clínica)
    1: Género (0=Feminino, 1=Masculino)
    2: Localização (0=Cólon, 1=Esquerdo, 2=Reto, 3=Direito)
    3: Estratégia Adjuvante (0=Ambas, 1=Apenas Quimio, 2=Sem Tratamento, 3=Apenas Radio)
    4: Estágio de Dukes (0=A, 1=B, 2=C, 3=D)
    """
    np.random.seed(42)
    ages = np.random.normal(preprocessor.age_mean, preprocessor.age_scale, n_samples).clip(20, 95)
    genders = np.random.binomial(1, 0.55, n_samples)
    locs = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.05, 0.35, 0.30, 0.30])
    adjs = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.20, 0.30, 0.20, 0.30])
    stages = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.25, 0.25, 0.30, 0.20])
    
    return np.column_stack([ages, genders, locs, adjs, stages])

class ModelExplainer:
    """
    Componente de Explicabilidade (XAI) baseado em LIME (SOLID - Single Responsibility Principle).
    Explica previsões individuais sem distorções ou compensações artificiais,
    refletindo fielmente as decisões aprendidas pelo modelo de Machine Learning.
    """
    
    def __init__(self, model, preprocessor: Optional[ClinicalDataPreprocessor] = None):
        self._model = model
        self._preprocessor = preprocessor or ClinicalDataPreprocessor()
        self.feature_names = ['Idade', 'Género', 'Localização do Tumor', 'Estratégia Terapêutica', 'Estágio de Dukes']
        
        # Gerar amostras representativas para calibrar o explicador tabular LIME
        self._background_data = generate_background_data(self._preprocessor, 600)
        
        categorical_names = {
            1: ['Feminino', 'Masculino'],
            2: ['Cólon (Geral)', 'Cólon Esquerdo', 'Reto', 'Cólon Direito'],
            3: ['Ambas Terapias', 'Apenas Quimioterapia', 'Sem Tratamento', 'Apenas Radioterapia'],
            4: ['Estágio A', 'Estágio B', 'Estágio C', 'Estágio D']
        }
        
        self._explainer = LimeTabularExplainer(
            training_data=self._background_data,
            feature_names=self.feature_names,
            categorical_features=[1, 2, 3, 4],
            categorical_names=categorical_names,
            class_names=['Sem Recorrência', 'Recorrência'],
            mode='classification',
            random_state=42
        )
        
    def _map_categorical_to_model_features(self, X_numpy: np.ndarray) -> pd.DataFrame:
        """
        Converte as 5 variáveis do LIME para o formato exato das 9 colunas esperadas pelo modelo.
        """
        N = X_numpy.shape[0]
        df_out = pd.DataFrame(index=range(N))
        
        # 0. Idade: Padronização Z-score
        df_out['Age (in years)'] = (X_numpy[:, 0] - self._preprocessor.age_mean) / self._preprocessor.age_scale
        
        # 1. Estágio Dukes: Codificação Ordinal Direta (0=A, 1=B, 2=C, 3=D) e Padronização
        stage_vals = X_numpy[:, 4].astype(float)
        df_out['Dukes Stage'] = (stage_vals - self._preprocessor.stage_mean) / self._preprocessor.stage_scale
        
        # 2. Género
        df_out['Gender_Male'] = (X_numpy[:, 1] == 1).astype(float)
        
        # 3. Localização (0=Colon como base)
        df_out['Location_Left'] = (X_numpy[:, 2] == 1).astype(float)
        df_out['Location_Rectum'] = (X_numpy[:, 2] == 2).astype(float)
        df_out['Location_Right'] = (X_numpy[:, 2] == 3).astype(float)
        
        # 4. Estratégia Adjuvante (0=Ambas como base)
        df_out['Adjuvant_Strategy_Chem_Only'] = (X_numpy[:, 3] == 1).astype(float)
        df_out['Adjuvant_Strategy_No_Treatment'] = (X_numpy[:, 3] == 2).astype(float)
        df_out['Adjuvant_Strategy_Radio_Only'] = (X_numpy[:, 3] == 3).astype(float)
        
        return df_out[self._preprocessor.feature_names]

    def _map_patient_to_vector(self, patient: Patient) -> np.ndarray:
        """
        Converte a entidade Patient num vetor clínico numérico de 5 dimensões.
        """
        # Idade com capping
        age = float(patient.age) if patient.age is not None else self._preprocessor.age_mean
        age = max(20.0, min(100.0, age))
        
        # Género (0=Feminino, 1=Masculino)
        gender_norm = str(patient.gender).strip().capitalize()
        gender_int = 1 if gender_norm in {"Male", "M", "Masculino"} else 0
        
        # Localização (0=Colon, 1=Left, 2=Rectum, 3=Right)
        loc_norm = str(patient.location).strip().capitalize()
        if loc_norm in {"Left", "Esquerdo"}:
            loc_int = 1
        elif loc_norm in {"Rectum", "Reto"}:
            loc_int = 2
        elif loc_norm in {"Right", "Direito"}:
            loc_int = 3
        else:
            loc_int = 0
            
        # Estratégia (0=Both, 1=Chem_Only, 2=No_Treatment, 3=Radio_Only)
        adj_chem = bool(patient.adj_chem)
        adj_radio = bool(patient.adj_radio)
        if adj_chem and not adj_radio:
            adj_int = 1
        elif not adj_chem and adj_radio:
            adj_int = 3
        elif not adj_chem and not adj_radio:
            adj_int = 2
        else:
            adj_int = 0
            
        # Estágio de Dukes (0=A, 1=B, 2=C, 3=D)
        stage_norm = str(patient.dukes_stage).strip().upper()
        stage_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        stage_int = stage_map.get(stage_norm, 1)
        
        return np.array([age, gender_int, loc_int, adj_int, stage_int])

    def explain_patient(self, patient: Patient) -> List[Tuple[str, float]]:
        """
        Gera as atribuições de impacto de cada característica clínica para a predição do paciente.
        Retorna uma lista de tuplos: (característica, peso_impacto).
        """
        patient_row = self._map_patient_to_vector(patient)
        
        def predict_proba_fn(X_numpy):
            df_features = self._map_categorical_to_model_features(X_numpy)
            # Predição direta sem adulterações
            return self._model.predict_proba(df_features)
            
        exp = self._explainer.explain_instance(
            data_row=patient_row,
            predict_fn=predict_proba_fn,
            num_features=5,
            labels=(1,)
        )
        
        return exp.as_list(label=1)
