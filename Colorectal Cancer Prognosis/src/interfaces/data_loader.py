from abc import ABC, abstractmethod
from typing import List, Any
from src.domain.patient import Patient

class IDataLoader(ABC):
    """
    Interface abstrata para o Carregador de Dados de Pacientes (SOLID - Dependency Inversion Principle).
    Permite ler de diferentes fontes (ex: CSV, Excel, Base de Dados SQL) de forma transparente para a aplicação.
    """
    
    @abstractmethod
    def load_patients(self, source: Any) -> List[Patient]:
        """
        Carrega dados de pacientes a partir de uma fonte (caminho de arquivo, stream, buffer).
        """
        pass
