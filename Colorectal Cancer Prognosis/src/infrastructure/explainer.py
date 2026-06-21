import numpy as np
import pandas as pd
from typing import List, Tuple
from lime.lime_tabular import LimeTabularExplainer
from src.domain.patient import Patient

def generate_background_data(n_samples: int = 1000) -> np.ndarray:
    """
    Gera um dataset de base (background) contendo 5 variáveis clínicas estruturadas
    para calibração e treinamento do explicador LIME.
    """
    np.random.seed(42)
    # 0. Idade (Idades normais clínicas de 18 a 90)
    ages = np.random.normal(62, 12, n_samples).clip(18, 90)
    
    # 1. Género (0=Feminino, 1=Masculino)
    genders = np.random.binomial(1, 0.55, n_samples)
    
    # 2. Localização (0=Cólon (Geral), 1=Cólon Esquerdo, 2=Reto, 3=Cólon Direito)
    locs = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.30, 0.25, 0.25, 0.20])
    
    # 3. Tratamento (0=Ambas Terapias, 1=Apenas Quimioterapia, 2=Sem Tratamento, 3=Apenas Radioterapia)
    adjs = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.10, 0.35, 0.40, 0.15])
    
    # 4. Estágio Dukes (0=A, 1=B, 2=C, 3=D)
    stages = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.15, 0.40, 0.35, 0.10])
    
    return np.column_stack([ages, genders, locs, adjs, stages])

def map_categorical_to_dummies(X_numpy: np.ndarray) -> pd.DataFrame:
    """
    Função de mapeamento interno (One-Hot Encoder do modelo de treino).
    Converte as 5 variáveis categóricas representadas por inteiros para as 11 colunas binárias que o modelo requer.
    """
    N = X_numpy.shape[0]
    df_out = pd.DataFrame(index=range(N))
    df_out['Age (in years)'] = X_numpy[:, 0]
    df_out['Gender_Male'] = (X_numpy[:, 1] == 1).astype(float)
    df_out['Location_Left'] = (X_numpy[:, 2] == 1).astype(float)
    df_out['Location_Rectum'] = (X_numpy[:, 2] == 2).astype(float)
    df_out['Location_Right'] = (X_numpy[:, 2] == 3).astype(float)
    df_out['Adjuvant_Strategy_Chem_Only'] = (X_numpy[:, 3] == 1).astype(float)
    df_out['Adjuvant_Strategy_No_Treatment'] = (X_numpy[:, 3] == 2).astype(float)
    df_out['Adjuvant_Strategy_Radio_Only'] = (X_numpy[:, 3] == 3).astype(float)
    df_out['Dukes Stage_B'] = (X_numpy[:, 4] == 1).astype(float)
    df_out['Dukes Stage_C'] = (X_numpy[:, 4] == 2).astype(float)
    df_out['Dukes Stage_D'] = (X_numpy[:, 4] == 3).astype(float)
    return df_out

class ModelExplainer:
    """
    Componente de Explicabilidade (XAI) baseado em LIME (SOLID - Single Responsibility Principle).
    Explica previsões individuais indicando a contribuição positiva ou negativa de cada variável clínica consolidada.
    """
    
    def __init__(self, model, feature_names: List[str] = None):
        self._model = model
        self.feature_names = ['Idade', 'Género', 'Localização do Tumor', 'Estratégia Terapêutica', 'Estágio de Dukes']
        
        # Gerar amostras representativas para calibrar a vizinhança do LIME
        self._background_data = generate_background_data(1000)
        
        # Dicionários de nomes para as variáveis categóricas
        categorical_names = {
            1: ['Feminino', 'Masculino'],
            2: ['Cólon (Geral)', 'Cólon Esquerdo', 'Reto', 'Cólon Direito'],
            3: ['Ambas Terapias', 'Apenas Quimioterapia', 'Sem Tratamento', 'Apenas Radioterapia'],
            4: ['Estágio A', 'Estágio B', 'Estágio C', 'Estágio D']
        }
        
        # Inicializar o explicador tabular
        self._explainer = LimeTabularExplainer(
            training_data=self._background_data,
            feature_names=self.feature_names,
            categorical_features=[1, 2, 3, 4],
            categorical_names=categorical_names,
            class_names=['Sem Recorrência', 'Recorrência'],
            mode='classification',
            random_state=42
        )
        
    def _map_patient_to_categorical_vector(self, patient: Patient) -> np.ndarray:
        """
        Converte as propriedades do Patient num vetor numérico de 5 posições de categorias.
        """
        # 0. Idade
        age = patient.age if patient.age is not None else 62.0
        # Capping Estatístico (IQR) da idade
        q1, q3 = 50.0, 70.0
        iqr = q3 - q1
        age = max(q1 - 1.5 * iqr, min(q3 + 1.5 * iqr, age))
        
        # 1. Género (0=Feminino, 1=Masculino)
        gender_norm = patient.gender.strip().capitalize()
        gender_int = 1 if gender_norm in {"Male", "M", "Masculino"} else 0
        
        # 2. Localização (0=Colon, 1=Left, 2=Rectum, 3=Right)
        loc_norm = patient.location.strip().capitalize()
        if loc_norm == "Left":
            loc_int = 1
        elif loc_norm == "Rectum":
            loc_int = 2
        elif loc_norm == "Right":
            loc_int = 3
        else:
            loc_int = 0  # Cólon (Geral)
            
        # 3. Adjuvant Strategy (0=Both, 1=Chem_Only, 2=No_Treatment, 3=Radio_Only)
        adj_chem = patient.adj_chem
        adj_radio = patient.adj_radio
        if adj_chem and not adj_radio:
            adj_int = 1
        elif not adj_chem and adj_radio:
            adj_int = 3
        elif not adj_chem and not adj_radio:
            adj_int = 2
        else:
            adj_int = 0  # Ambas Terapias / Both
            
        # 4. Dukes Stage (0=A, 1=B, 2=C, 3=D)
        stage_norm = patient.dukes_stage.strip().upper()
        if stage_norm == "B":
            stage_int = 1
        elif stage_norm == "C":
            stage_int = 2
        elif stage_norm == "D":
            stage_int = 3
        else:
            stage_int = 0  # Estágio A
            
        return np.array([age, gender_int, loc_int, adj_int, stage_int])

    def explain_patient(self, patient: Patient) -> List[Tuple[str, float]]:
        """
        Calcula as atribuições de impacto de cada característica clínica para a previsão do paciente.
        Retorna uma lista de tuplos contendo (caracteristica, peso_impacto).
        """
        # Obter vetor clínico simplificado
        patient_row = self._map_patient_to_categorical_vector(patient)
        
        # Wrapper compatível que aceita o array 2D gerado pelas perturbações LIME
        def predict_proba_wrapper(X_numpy):
            df_dummies = map_categorical_to_dummies(X_numpy)
            return self._model.predict_proba(df_dummies)
            
        # Gerar a explicação local para a classe de Recorrência (índice 1)
        exp = self._explainer.explain_instance(
            data_row=patient_row,
            predict_fn=predict_proba_wrapper,
            num_features=5,
            labels=(1,)
        )
        
        # Retorna a lista mapeada
        return exp.as_list(label=1)
