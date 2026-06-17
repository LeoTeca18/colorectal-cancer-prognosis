from abc import ABC, abstractmethod
from typing import List
from src.domain.patient import Patient, PrognosisResult

class IInferenceEngine(ABC):
    """
    Interface abstrata para o Motor de Inferência de Prognóstico (SOLID - Dependency Inversion Principle).
    Permite acoplar diferentes modelos de Machine Learning sem modificar a interface do utilizador.
    """
    
    @abstractmethod
    def predict(self, patient: Patient) -> PrognosisResult:
        """
        Gera uma previsão de prognóstico individual para um paciente.
        """
        pass
    
    @abstractmethod
    def predict_batch(self, patients: List[Patient]) -> List[PrognosisResult]:
        """
        Gera previsões de prognóstico em lote para uma lista de pacientes.
        """
        pass
