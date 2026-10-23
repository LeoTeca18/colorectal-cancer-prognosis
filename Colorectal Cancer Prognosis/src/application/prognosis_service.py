from typing import List
from src.domain.patient import Patient, PrognosisResult
from src.interfaces.inference import IInferenceEngine

class PrognosisService:
    """
    Serviço de Aplicação que orquestra a lógica de negócio (SOLID - Single Responsibility Principle).
    Usa Injeção de Dependência para acoplar o motor de inferência.
    """
    
    def __init__(self, inference_engine: IInferenceEngine):
        self._inference_engine = inference_engine

    def predict_single(self, patient: Patient) -> PrognosisResult:
        """
        Executa a predição para um único paciente.
        """
        return self._inference_engine.predict(patient)

    def predict_batch(self, patients: List[Patient]) -> List[PrognosisResult]:
        """
        Executa a predição em lote para múltiplos prontuários clínicos.
        """
        return self._inference_engine.predict_batch(patients)
