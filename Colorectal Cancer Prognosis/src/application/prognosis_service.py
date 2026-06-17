from typing import List, Any
from src.domain.patient import Patient, PrognosisResult
from src.interfaces.inference import IInferenceEngine
from src.interfaces.data_loader import IDataLoader

class PrognosisService:
    """
    Serviço de Aplicação que orquestra a lógica de negócio (SOLID - Single Responsibility Principle).
    Usa Injeção de Dependência para acoplar o carregador de dados e o motor de inferência.
    """
    
    def __init__(self, inference_engine: IInferenceEngine, data_loader: IDataLoader):
        self._inference_engine = inference_engine
        self._data_loader = data_loader

    def predict_single(self, patient: Patient) -> PrognosisResult:
        """
        Executa a predição para um único paciente.
        """
        return self._inference_engine.predict(patient)

    def predict_batch(self, source: Any) -> List[PrognosisResult]:
        """
        Carrega dados de uma fonte externa (ex: arquivo carregado) e realiza a inferência em lote.
        """
        patients = self._data_loader.load_patients(source)
        return self._inference_engine.predict_batch(patients)
