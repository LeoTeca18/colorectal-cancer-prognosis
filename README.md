# Colorectal Cancer Prognosis — Clinical Decision Support System (CDSS)

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Machine_Learning-green?style=for-the-badge)
![Architecture](https://img.shields.io/badge/Architecture-Clean_Architecture_%2F_DDD-orange?style=for-the-badge)
![Domain](https://img.shields.io/badge/Sector-Agribusiness_%26_Healthcare-red?style=for-the-badge)

Este repositório contém um **Sistema de Suporte à Decisão Clínica (CDSS)** focado no prognóstico de pacientes diagnosticados com cancro colorretal. O sistema utiliza pipelines avançados de ciência de dados e algoritmos de Machine Learning (XGBoost) para auxiliar profissionais de saúde na avaliação de riscos e na estimativa de sobrevivência com base em variáveis clínicas estruturadas.

Desenvolvido com um foco rigoroso em engenharia de software de alta qualidade, o projeto serve como um modelo prático de aplicação dos princípios **SOLID**, **Clean Code** e **Domain-Driven Design (DDD)** no contexto de sistemas médicos críticos.

---

## 🏗️ Arquitetura do Sistema (Clean Architecture / DDD)

Diferente de abordagens tradicionais onde notebooks de Data Science são convertidos diretamente em scripts monolíticos, este projeto isola estritamente as regras de negócio das preocupações tecnológicas e de infraestrutura. A estrutura segue uma divisão em camadas modulares:

```
colorectal-cancer-prognosis/
└── Colorectal Cancer Prognosis/
    ├── app.py                         # Ponto de entrada da aplicação (Interface/Streamlit)
    ├── requirements.txt               # Dependências do ecossistema Python
    └── src/
        ├── domain/                    # Camada do Domínio (Regras de negócio puras)
        │   └── patient.py             # Entidade de Domínio "Patient" encapsulando atributos clínicos
        ├── application/               # Camada de Aplicação (Casos de uso e fluxos)
        │   └── prognosis_service.py   # Orquestração do caso de uso de inferência prognóstica
        ├── infrastructure/            # Camada de Infraestrutura (Detalhes técnicos)
        │   ├── models/
        │   │   └── model_xgboost.pkl  # Artefato serializado do modelo XGBoost pré-treinado
        │   ├── preprocessor.py        # Pipelines de tratamento de dados e normalização
        │   ├── xgb_engine.py          # Motor de execução e inferência do XGBoost
        │   └── explainer.py           # Módulo de explicabilidade clínica (SHAP/Feature Importance)
        ├── interfaces/                # Adaptadores de Interface
        │   └── inference.py           # Contratos e DTOs para comunicação externa
        └── ui/                        # Camada de Apresentação Gráfica
            ├── patient_form.py        # Formulários de input clínico interativo
            ├── dashboard.py           # Painéis visuais para análise de métricas
            └── batch_processor.py     # Processador assíncrono para triagem em lote
```

### Detalhes das Camadas:
1. **Domain (`src/domain`)**: Contém o núcleo da aplicação. A entidade `Patient` define a representação lógica de um paciente e as validações clínicas obrigatórias, permanecendo 100% independente de frameworks, bancos de dados ou bibliotecas de ML.
2. **Application (`src/application`)**: O `PrognosisService` orquestra a execução. Ele recebe dados brutos da interface, utiliza os contratos de domínio e delega a inferência para a infraestrutura.
3. **Infrastructure (`src/infrastructure`)**: Onde a matemática e a tecnologia operam. Gerencia o pipeline de dados (limpeza, engenharia de atributos e PCA), carrega o arquivo de pesos binários (`model_xgboost.pkl`) e expõe ferramentas de explicabilidade através do `explainer.py` para garantir que o modelo não seja uma "caixa preta".
4. **Interfaces & UI (`src/ui`)**: Fornece os pontos de contacto do utilizador através de formulários clínicos intuitivos, dashboards de monitoramento e processamento batch de prontuários.

---

## 🛠️ Tecnologias e Ferramentas Utilizadas

- **Linguagem Principal**: Python 3.12+
- **Machine Learning & Inferência**: XGBoost, Scikit-Learn
- **Manipulação de Dados**: Pandas, NumPy
- **Explicação do Modelo**: SHAP / Algoritmos de interpretabilidade local
- **Interface Gráfica (UI)**: Framework baseado em componentes interativos (App.py)
- **Serialização**: Pickle / Joblib para o pipeline de machine learning

---

## 🚀 Funcionalidades Principais

- **Avaliação de Prognóstico em Tempo Real**: Formulário clínico otimizado para introdução rápida de parâmetros laboratoriais e demográficos de pacientes.
- **Pipeline de Dados Padronizado**: Garantia de consistência através de pré-processamento estruturado de variáveis e tratamento automatizado de dados nulos.
- **Explicabilidade Clínica (XAI)**: Integração com mecanismos que indicam quais variáveis (ex: marcadores tumorais, idade, estadiamento TNM) mais impactaram a predição de risco do paciente individual.
- **Processamento em Lote (Batch Processing)**: Capacidade de submeter ficheiros CSV contendo dados de múltiplos pacientes para triagem automatizada em larga escala em hospitais ou centros de pesquisa.

---

## 📦 Instalação e Execução

### Pré-requisitos
Certifique-se de que tem o Python 3.12 (ou superior) e o `pip` instalados na sua máquina.

### Passos para Configuração

1. **Clonar o Repositório:**
   ```bash
   git clone https://github.com/seu-usuario/colorectal-cancer-prognosis.git
   cd colorectal-cancer-prognosis
   ```

2. **Criar um Ambiente Virtual:**
   ```bash
   python -m venv venv
   # No Windows (CMD/PowerShell):
   .\venv\Scripts\activate
   # No Linux/macOS:
   source venv/bin/activate
   ```

3. **Instalar as Dependências:**
   ```bash
   cd "Colorectal Cancer Prognosis"
   pip install -r requirements.txt
   ```

4. **Executar a Aplicação:**
   ```bash
   python app.py
   streamlit run app.py
   ```

---

## 🩺 Relevância Médica e Engenharia de Qualidade

Este sistema foi projetado para mitigar os problemas clássicos de transição de modelos experimentais para ambientes de produção médica. Ao adotar uma arquitetura baseada em **Monolito Modular**, o código torna-se altamente testável, manutenível e extensível. Caso surja a necessidade de substituir o motor do XGBoost por um modelo baseado em LightGBM ou Redes Neuronais, as alterações ficam restritas exclusivamente à camada de *Infrastructure*, mantendo o domínio clínico e a interface completamente intactos.

---
Desenvolvido como parte de uma iniciativa de investigação e excelência em Engenharia de Software Aplicada à Saúde.
