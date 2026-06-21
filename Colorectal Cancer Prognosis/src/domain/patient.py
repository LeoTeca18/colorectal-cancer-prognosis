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
        # Validações básicas para manter integridade dos dados (Clean Code)
        if self.age < 0 or self.age > 120:
            raise ValueError("A idade do paciente deve ser um valor entre 0 e 120.")
        
        normalized_gender = self.gender.strip().capitalize()
        valid_genders = {"Male", "Female", "M", "F", "Masculino", "Feminino"}
        if normalized_gender not in valid_genders:
            raise ValueError(f"Género inválido: '{self.gender}'. Deve ser um dos seguintes: {valid_genders}")
            
        normalized_stage = self.dukes_stage.strip().upper()
        valid_stages = {"A", "B", "C", "D"}
        if normalized_stage not in valid_stages:
            raise ValueError(f"Estágio de Dukes inválido: '{self.dukes_stage}'. Deve ser A, B, C ou D.")

@dataclass(frozen=True)
class PrognosisResult:
    patient_id: str
    recurrence_probability: float  # Valor de 0.0 a 1.0 (Probabilidade de recorrência)
    risk_level: str               # "Baixo", "Médio", "Alto"
    predicted_survival_months: float
    details: str                  # Explicação textual ou observações adicionais
    explanation: list = field(default_factory=list)

