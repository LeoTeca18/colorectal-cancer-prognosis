from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass(frozen=True)
class Patient:
    patient_id: str
    age: int
    gender: str
    dukes_stage: str
    location: str
    adj_radio: bool
    adj_chem: bool
    gene_expression: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # Validações de integridade do domínio clínico (Clean Code / DDD)
        if self.age < 0 or self.age > 120:
            raise ValueError(f"A idade do paciente deve ser um valor entre 0 e 120 anos (recebido: {self.age}).")
        
        normalized_gender = str(self.gender).strip().capitalize()
        valid_genders = {"Male", "Female", "M", "F", "Masculino", "Feminino"}
        if normalized_gender not in valid_genders:
            raise ValueError(f"Género inválido: '{self.gender}'. Opções válidas: {valid_genders}")
            
        normalized_stage = str(self.dukes_stage).strip().upper()
        valid_stages = {"A", "B", "C", "D"}
        if normalized_stage not in valid_stages:
            raise ValueError(f"Estágio de Dukes inválido: '{self.dukes_stage}'. Deve ser A, B, C ou D.")
            
        normalized_loc = str(self.location).strip().capitalize()
        valid_locations = {"Colon", "Rectum", "Left", "Right", "Cólon", "Reto", "Esquerdo", "Direito"}
        if normalized_loc not in valid_locations:
            raise ValueError(f"Localização do tumor inválida: '{self.location}'. Opções válidas: {valid_locations}")

@dataclass(frozen=True)
class PrognosisResult:
    patient_id: str
    recurrence_probability: float  # Valor de 0.0 a 1.0 (Probabilidade de recorrência)
    risk_level: str               # "Baixo", "Médio", "Alto"
    predicted_survival_months: float
    details: str                  # Explicação textual com fundamentação clínica
    explanation: list = field(default_factory=list)
