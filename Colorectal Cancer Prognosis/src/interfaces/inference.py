from abc import ABC, abstractmethod
from typing import List, Dict, Any
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
        Gera previsões de prognóstico em lote para múltiplos pacientes.
        """
        pass

    @abstractmethod
    def get_dashboard_analytics(self) -> Dict[str, Any]:
        """
        Extrai métricas, estatísticas da coorte, curvas de sobrevida e importância de variáveis
        diretamente dos artefatos gerados pelo Notebook (modelos .pkl e preprocessor_params.json).
        """
        pass
