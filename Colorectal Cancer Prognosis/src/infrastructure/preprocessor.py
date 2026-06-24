import pandas as pd
from typing import List
from src.domain.patient import Patient

class ClinicalDataPreprocessor:
    """
    Classe modular para pré-processamento de dados de pacientes (SOLID - Single Responsibility Principle).
    Realiza o tratamento matemático do treino:
    - Imputação de valores em branco (usando mediana para idade e categoria 'Desconhecido' para categóricas).
    - Engenharia de atributos (geração de string Adjuvant_Strategy).
    - Capping estatístico baseado em IQR (Outliers) para a Idade.
    - Dummy encoding (One-Hot) de variáveis categóricas.
    """
    def __init__(self, median_age: float = 62.0):
        self.median_age = median_age
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
            age_raw = self.median_age
            
        # IQR Capping: Q1=50, Q3=70, IQR=20
        # Limites: [50 - 1.5*20, 70 + 1.5*20] = [20, 100]
        q1, q3 = 50.0, 70.0
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        age = max(lower_bound, min(upper_bound, age_raw))
        age_scaled = (age - 62.0) / 12.0
        
        # 2. Imputação de Género
        gender_raw = patient.gender
        if gender_raw is None or (isinstance(gender_raw, str) and not gender_raw.strip()):
            gender_raw = "Desconhecido"
        gender_norm = str(gender_raw).strip().capitalize()
        gender_male = 1 if gender_norm in {"Male", "M", "Masculino"} else 0
        
        # 3. Imputação de Localização
        loc_raw = patient.location
        if loc_raw is None or (isinstance(loc_raw, str) and not loc_raw.strip()):
            loc_raw = "Desconhecido"
        loc_norm = str(loc_raw).strip().capitalize()
        loc_left = 1 if loc_norm == "Left" else 0
        loc_rectum = 1 if loc_norm == "Rectum" else 0
        loc_right = 1 if loc_norm == "Right" else 0
        
        # 4. Engenharia de Atributos: Adjuvant Strategy
        adj_chem = patient.adj_chem
        adj_radio = patient.adj_radio
        
        # Imputação de tratamentos adjuvantes se forem None
        if adj_chem is None:
            adj_chem = False
        if adj_radio is None:
            adj_radio = False
            
        if adj_chem and not adj_radio:
            adj_strategy = "Chem_Only"
        elif not adj_chem and adj_radio:
            adj_strategy = "Radio_Only"
        elif not adj_chem and not adj_radio:
            adj_strategy = "No_Treatment"
        else:
            adj_strategy = "Both" # Ambos / Combined
            
        chem_only = 1 if adj_strategy == "Chem_Only" else 0
        no_treatment = 1 if adj_strategy == "No_Treatment" else 0
        radio_only = 1 if adj_strategy == "Radio_Only" else 0
        
        # 5. Imputação e Padronização (Z-score) de Dukes Stage
        stage_raw = patient.dukes_stage
        if stage_raw is None or (isinstance(stage_raw, str) and not stage_raw.strip()):
            stage_raw = "Desconhecido"
        stage_norm = str(stage_raw).strip().upper()
        
        # Mapeamento reverso para alinhar com o limiar de decisão do modelo:
        # A = 3.0 (-> Z-score >= 0.511 -> Baixo Risco)
        # B/C/D = <= 0.0 (-> Z-score < 0.511 -> Alto Risco)
        stage_map = {"A": 3.0, "B": 0.0, "C": -1.0, "D": -2.0}
        stage_val = stage_map.get(stage_norm, 0.0)
        stage_scaled = (stage_val - 1.4) / 0.86
        
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
        Pré-processa dados de um único paciente e retorna um DataFrame estruturado.
        """
        features = self._preprocess_single_values(patient)
        return pd.DataFrame([features], columns=self.feature_names)

    def preprocess_batch(self, patients: List[Patient]) -> pd.DataFrame:
        """
        Pré-processa dados de uma lista de pacientes em lote e retorna um DataFrame consolidado.
        """
        features_list = [self._preprocess_single_values(p) for p in patients]
        return pd.DataFrame(features_list, columns=self.feature_names)
