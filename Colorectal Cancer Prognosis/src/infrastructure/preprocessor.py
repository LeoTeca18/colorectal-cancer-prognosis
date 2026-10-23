import os
import json
import pandas as pd
from typing import List, Optional
from src.domain.patient import Patient

class ClinicalDataPreprocessor:
    """
    Classe modular para pré-processamento de dados de pacientes (SOLID - Single Responsibility Principle).
    Realiza o tratamento estatístico alinhado com o treino do modelo:
    - Imputação de valores em branco.
    - Capping estatístico baseado em IQR para Idade (Outliers).
    - Codificação Ordinal direta do Estágio de Dukes (A=0, B=1, C=2, D=3).
    - Padronização estatística (Z-score) de Idade e Estágio de Dukes.
    - Codificação Dummy (One-Hot) de variáveis categóricas (Género, Localização, Estratégia Adjuvante).
    """
    def __init__(self, params_path: Optional[str] = None):
        # Tenta carregar os parâmetros calculados no treino
        if params_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            params_path = os.path.join(current_dir, "models", "preprocessor_params.json")
            
        if os.path.exists(params_path):
            with open(params_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            self.age_mean = params.get("age_mean", 61.1129)
            self.age_scale = params.get("age_scale", 9.5040)
            self.stage_mean = params.get("stage_mean", 1.4516)
            self.stage_scale = params.get("stage_scale", 1.0728)
            self.feature_names = params.get("feature_names", [
                'Age (in years)', 
                'Dukes Stage', 
                'Gender_Male', 
                'Location_Left', 
                'Location_Rectum', 
                'Location_Right', 
                'Adjuvant_Strategy_Chem_Only', 
                'Adjuvant_Strategy_No_Treatment', 
                'Adjuvant_Strategy_Radio_Only'
            ])
        else:
            # Fallback para os valores empíricos calculados na base
            self.age_mean = 61.1129
            self.age_scale = 9.5040
            self.stage_mean = 1.4516
            self.stage_scale = 1.0728
            self.feature_names = [
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

    def _preprocess_single_values(self, patient: Patient) -> List[float]:
        # 1. Imputação, Capping Estatístico (IQR) e Padronização (Z-score) da Idade
        age_raw = patient.age
        if age_raw is None or pd.isna(age_raw):
            age_raw = self.age_mean
            
        # Limites IQR clínicos: Q1=50, Q3=70, IQR=20 -> [20, 100]
        q1, q3 = 50.0, 70.0
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        age = max(lower_bound, min(upper_bound, float(age_raw)))
        age_scaled = (age - self.age_mean) / self.age_scale
        
        # 2. Codificação Ordinal Direta e Padronização (Z-score) do Dukes Stage
        # Mapeamento ordinal anatômico real: A=0, B=1, C=2, D=3
        stage_raw = patient.dukes_stage
        if stage_raw is None or (isinstance(stage_raw, str) and not stage_raw.strip()):
            stage_norm = "B"
        else:
            stage_norm = str(stage_raw).strip().upper()
            
        stage_map = {"A": 0.0, "B": 1.0, "C": 2.0, "D": 3.0}
        stage_val = stage_map.get(stage_norm, 1.0)
        stage_scaled = (stage_val - self.stage_mean) / self.stage_scale
        
        # 3. Codificação de Género (Gender_Male: 1=Masculino, 0=Feminino)
        gender_raw = patient.gender
        if gender_raw is None or (isinstance(gender_raw, str) and not gender_raw.strip()):
            gender_norm = "Female"
        else:
            gender_norm = str(gender_raw).strip().capitalize()
            
        gender_male = 1.0 if gender_norm in {"Male", "M", "Masculino"} else 0.0
        
        # 4. Codificação de Localização (Base de referência: Colon)
        loc_raw = patient.location
        if loc_raw is None or (isinstance(loc_raw, str) and not loc_raw.strip()):
            loc_norm = "Colon"
        else:
            loc_norm = str(loc_raw).strip().capitalize()
            if loc_norm in {"Cólon", "Colon"}:
                loc_norm = "Colon"
            elif loc_norm in {"Reto", "Rectum"}:
                loc_norm = "Rectum"
            elif loc_norm in {"Esquerdo", "Left"}:
                loc_norm = "Left"
            elif loc_norm in {"Direito", "Right"}:
                loc_norm = "Right"
                
        loc_left = 1.0 if loc_norm == "Left" else 0.0
        loc_rectum = 1.0 if loc_norm == "Rectum" else 0.0
        loc_right = 1.0 if loc_norm == "Right" else 0.0
        
        # 5. Engenharia de Atributos: Adjuvant Strategy (Base de referência: Both_Therapies)
        adj_chem = bool(patient.adj_chem) if patient.adj_chem is not None else False
        adj_radio = bool(patient.adj_radio) if patient.adj_radio is not None else False
        
        if adj_chem and not adj_radio:
            adj_strategy = "Chem_Only"
        elif not adj_chem and adj_radio:
            adj_strategy = "Radio_Only"
        elif not adj_chem and not adj_radio:
            adj_strategy = "No_Treatment"
        else:
            adj_strategy = "Both_Therapies"
            
        chem_only = 1.0 if adj_strategy == "Chem_Only" else 0.0
        no_treatment = 1.0 if adj_strategy == "No_Treatment" else 0.0
        radio_only = 1.0 if adj_strategy == "Radio_Only" else 0.0
        
        return [
            age_scaled,
            stage_scaled,
            gender_male,
            loc_left, 
            loc_rectum, 
            loc_right,
            chem_only, 
            no_treatment, 
            radio_only
        ]

    def preprocess_patient(self, patient: Patient) -> pd.DataFrame:
        """
        Pré-processa dados de um único paciente e retorna um DataFrame estruturado de 9 colunas.
        """
        features = self._preprocess_single_values(patient)
        return pd.DataFrame([features], columns=self.feature_names)

    def preprocess_batch(self, patients: List[Patient]) -> pd.DataFrame:
        """
        Pré-processa dados de uma lista de pacientes em lote e retorna um DataFrame consolidado.
        """
        features_list = [self._preprocess_single_values(p) for p in patients]
        return pd.DataFrame(features_list, columns=self.feature_names)
