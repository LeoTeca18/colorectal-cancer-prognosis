import io
import pandas as pd
from typing import List, Union, Any
from src.domain.patient import Patient
from src.interfaces.data_loader import IDataLoader

class CSVDataLoader(IDataLoader):
    """
    Implementação concreta do carregador de dados a partir de ficheiros CSV (SOLID - Single Responsibility Principle).
    Mapeia dados tabulares para a entidade de domínio Patient de forma flexível e tolerante a falhas.
    """
    
    def load_patients(self, source: Any) -> List[Patient]:
        """
        Carrega dados de pacientes a partir de um ficheiro CSV ou buffer (como o UploadedFile do Streamlit).
        """
        try:
            # Ler o CSV com pandas
            df = pd.read_csv(source)
            
            # Normalizar os nomes das colunas para busca flexível e insensível a maiúsculas/espaços/sublinhados
            col_mapping = {col.lower().replace(" ", "").replace("_", ""): col for col in df.columns}
            
            patients = []
            for index, row in df.iterrows():
                # Chaves e fallbacks seguros para mapear as propriedades clínicas do Kaggle
                patient_id = str(self._get_value_by_keys(row, col_mapping, ["idref", "patientid", "id", "patient_id"], f"PAT_{index+1}"))
                age = int(self._get_value_by_keys(row, col_mapping, ["age", "idade"], 60))
                gender = str(self._get_value_by_keys(row, col_mapping, ["gender", "sex", "sexo", "genero"], "Male"))
                dukes_stage = str(self._get_value_by_keys(row, col_mapping, ["dukesstage", "stage", "estagio", "estagiodukes"], "B"))
                location = str(self._get_value_by_keys(row, col_mapping, ["location", "site", "localizacao"], "Colon"))
                
                # Mapeamento robusto de tratamentos adjuvantes
                adj_radio_val = self._get_value_by_keys(row, col_mapping, ["adjradio", "radioterapia", "adj_radio", "adj_radio_event"], False)
                adj_radio = self._parse_bool(adj_radio_val)
                
                adj_chem_val = self._get_value_by_keys(row, col_mapping, ["adjchem", "quimioterapia", "adj_chem", "adj_chem_event"], False)
                adj_chem = self._parse_bool(adj_chem_val)
                
                # Extração opcional de expressão genética (quaisquer outras colunas numéricas que não sejam metadados clínicos ou de sobrevida)
                gene_expression = {}
                clinical_columns_clean = {
                    "idref", "patientid", "id", "patient_id", "age", "idade", "gender", "sex", "sexo", "genero", 
                    "dukesstage", "stage", "estagio", "estagiodukes", "location", "site", "localizacao", 
                    "adjradio", "radioterapia", "adj_radio", "adj_radio_event", "adjchem", "quimioterapia", "adj_chem", "adj_chem_event",
                    "dfs", "dfsevent", "dfs_event", "diseasefreesurvival"
                }
                
                for original_col in df.columns:
                    col_clean = original_col.lower().replace(" ", "").replace("_", "")
                    if col_clean not in clinical_columns_clean:
                        # Se for um valor numérico, considera-se dado genómico
                        val = row[original_col]
                        try:
                            gene_expression[original_col] = float(val)
                        except (ValueError, TypeError):
                            pass  # Ignorar colunas não numéricas que não mapearam para metadados
                
                patient = Patient(
                    patient_id=patient_id,
                    age=age,
                    gender=gender,
                    dukes_stage=dukes_stage,
                    location=location,
                    adj_radio=adj_radio,
                    adj_chem=adj_chem,
                    gene_expression=gene_expression
                )
                patients.append(patient)
                
            return patients
            
        except Exception as e:
            raise ValueError(f"Erro ao ler e processar o ficheiro CSV: {str(e)}")

    def _get_value_by_keys(self, row: Any, col_mapping: dict, keys: List[str], default: Any) -> Any:
        """Busca no row pelas chaves mapeadas normalizadas."""
        for key in keys:
            if key in col_mapping:
                return row[col_mapping[key]]
        return default

    def _parse_bool(self, val: Any) -> bool:
        """Converte de forma tolerante vários formatos para boolean."""
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            cleaned = val.strip().lower()
            return cleaned in {"yes", "true", "1", "y", "sim", "s"}
        return False
