from typing import List
from src.domain.patient import Patient, PrognosisResult
from src.interfaces.inference import IInferenceEngine

class MockInferenceEngine(IInferenceEngine):
    """
    Motor de Inferência Simulado (Mock) para demonstração (SOLID - Single Responsibility Principle).
    Retorna resultados realistas baseados em regras clínicas heurísticas simples.
    Será futuramente substituído pelo modelo real do utilizador.
    """
    
    def predict(self, patient: Patient) -> PrognosisResult:
        # Probabilidades base e meses de sobrevivência baseados no Estágio Dukes
        stage = patient.dukes_stage.strip().upper()
        if stage == 'A':
            base_prob = 0.10
            base_survival = 96.0
        elif stage == 'B':
            base_prob = 0.25
            base_survival = 72.0
        elif stage == 'C':
            base_prob = 0.55
            base_survival = 48.0
        else:  # 'D'
            base_prob = 0.85
            base_survival = 18.0

        # Influência da Idade (idade mais avançada aumenta ligeiramente a probabilidade de recorrência e reduz sobrevivência)
        age_factor = (patient.age - 60) / 100.0
        recurrence_prob = base_prob + (age_factor * 0.15)
        
        # Benefício terapêutico simulado (quimioterapia/radioterapia reduzem probabilidade de recorrência)
        treatment_benefit = 0.0
        if patient.adj_chem:
            treatment_benefit += 0.08
        if patient.adj_radio:
            treatment_benefit += 0.05
            
        recurrence_prob = max(0.02, min(0.98, recurrence_prob - treatment_benefit))
        
        # Cálculo simulado de meses de sobrevivência livre de doença
        predicted_survival = base_survival - (age_factor * 12.0) + (treatment_benefit * 60.0)
        predicted_survival = max(1.0, min(120.0, predicted_survival))
        
        # Classificação do nível de risco
        if recurrence_prob < 0.30:
            risk = "Baixo"
        elif recurrence_prob < 0.65:
            risk = "Médio"
        else:
            risk = "Alto"
            
        details = (
            f"Previsão gerada pelo motor simulado (MockEngine).\n"
            f"Variáveis chave: Estágio Dukes '{stage}', Idade {patient.age} anos.\n"
            f"Ajuste terapêutico efetuado com base em tratamentos ativos (Quimioterapia: {patient.adj_chem}, Radioterapia: {patient.adj_radio})."
        )
        
        return PrognosisResult(
            patient_id=patient.patient_id,
            recurrence_probability=round(recurrence_prob, 3),
            risk_level=risk,
            predicted_survival_months=round(predicted_survival, 1),
            details=details
        )
        
    def predict_batch(self, patients: List[Patient]) -> List[PrognosisResult]:
        return [self.predict(p) for p in patients]
