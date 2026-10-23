# Colorectal Cancer Prognosis — Clinical Decision Support System (CDSS)

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Machine_Learning-green?style=for-the-badge)
![Architecture](https://img.shields.io/badge/Architecture-Clean_Architecture_%2F_DDD-orange?style=for-the-badge)
![Sector](https://img.shields.io/badge/Sector-Healthcare_%26_Oncology-red?style=for-the-badge)

Este repositório contém um **Sistema de Suporte à Decisão Clínica (CDSS)** focado no prognóstico de pacientes diagnosticados com cancro colorretal. O sistema utiliza pipelines avançados de ciência de dados e algoritmos de Machine Learning (Dual Engine XGBoost com Restrições Monotónicas Clínicas) para auxiliar profissionais de saúde na avaliação do risco de recorrência e na estimativa da sobrevida livre de doença (Disease-Free Survival - DFS) com base em variáveis clínicas estruturadas.

Desenvolvido com foco rigoroso em engenharia de software e integridade científica, o projeto implementa os princípios **SOLID**, **Clean Code** e **Domain-Driven Design (DDD)** no contexto de sistemas médicos críticos.

---

## 🏗️ Arquitetura do Sistema (Clean Architecture / DDD)

Diferente de abordagens tradicionais onde notebooks experimentais são convertidos diretamente em scripts monolíticos frágeis, este projeto isola estritamente as regras de negócio das preocupações tecnológicas e de infraestrutura. A estrutura segue uma divisão em camadas modulares:

```
colorectal-cancer-prognosis/
└── Colorectal Cancer Prognosis/
    ├── app.py                         # Ponto de entrada da aplicação (Interface Streamlit com abas)
    ├── requirements.txt               # Dependências do ecossistema Python
    └── src/
        ├── domain/                    # Camada do Domínio (Regras de negócio puras)
        │   └── patient.py             # Entidade "Patient" com validações estritas (idade, género, estágio, localização)
        ├── application/               # Camada de Aplicação (Casos de uso e orquestração)
        │   └── prognosis_service.py   # Orquestração de inferência individual e em lote (Batch)
        ├── infrastructure/            # Camada de Infraestrutura (Detalhes técnicos e ML)
        │   ├── models/
        │   │   ├── model_xgboost.pkl  # Classificador XGBoost (Probabilidade de Recorrência)
        │   │   ├── model_survival.pkl # Regressor XGBoost (Sobrevida Livre de Doença em Meses)
        │   │   ├── preprocessor_params.json # Parâmetros estatísticos de normalização (Z-Score)
        │   │   └── train_models.py    # Pipeline reprodutível de treino e calibração
        │   ├── preprocessor.py        # Pipelines de tratamento, imputação e codificação dummy/ordinal
        │   ├── xgb_engine.py          # Motor de execução dual do XGBoost
        │   └── explainer.py           # Módulo de explicabilidade clínica (LIME XAI)
        ├── interfaces/                # Adaptadores de Interface e Contratos Abstratos
        │   └── inference.py           # Contrato IInferenceEngine (Injeção de Dependências)
        └── ui/                        # Camada de Apresentação Gráfica
            ├── patient_form.py        # Formulário clínico individual com gauge interativo
            ├── batch_processor.py     # Triagem automatizada em lote via ficheiros CSV
            └── dashboard.py           # Painel epidemiológico e curvas de sobrevida (Kaplan-Meier)
```

### Detalhes das Camadas:
1. **Domain (`src/domain`)**: Contém o núcleo da aplicação. A entidade `Patient` define a representação lógica de um paciente e as validações clínicas obrigatórias (ex: intervalo de idade, compatibilidade de termos anatómicos e estadiamentos de Dukes A-D), permanecendo 100% independente de bibliotecas externas.
2. **Application (`src/application`)**: O `PrognosisService` orquestra a execução sem acoplamento a modelos específicos, oferecendo métodos para predição individual (`predict_single`) e em massa (`predict_batch`).
3. **Infrastructure (`src/infrastructure`)**: Onde a matemática e a tecnologia operam. Carrega os dois modelos preditivos reais (classificação e sobrevida), gerencia o pipeline de normalização com parâmetros auditáveis e expõe interpretabilidade clínica local através de LIME.
4. **Interfaces & UI (`src/ui`)**: Fornece três ambientes integrados em abas:
   - **Prognóstico Individual**: Entrada dinâmica com velocímetro de risco e gráfico explicativo de fatores de impacto.
   - **Triagem em Lote**: Processamento de múltiplos prontuários via CSV com exportação de resultados.
   - **Painel Clínico**: Curvas de sobrevida livre de doença e orientações terapêuticas de estadiamento.

---

## 🛠️ Tecnologias e Ferramentas Utilizadas

- **Linguagem Principal**: Python 3.12+
- **Machine Learning & Inferência**: XGBoost, Scikit-Learn
- **Manipulação & Análise de Dados**: Pandas, NumPy
- **Explicabilidade Clínica (XAI)**: LIME (Local Interpretable Model-agnostic Explanations)
- **Interface Gráfica & Visualização**: Streamlit, Plotly Express & Graph Objects
- **Serialização**: Joblib

---

## 🚀 Funcionalidades Principais

- **Avaliação de Prognóstico em Tempo Real**: Formulário clínico otimizado para introdução rápida de parâmetros laboratoriais e demográficos de pacientes.
- **Motor Dual de Machine Learning**:
  - *Classificação de Risco:* Predição probabilística calibrada de recorrência (DFS Event).
  - *Expectativa de Sobrevida:* Estimativa de meses livres de doença baseada diretamente no modelo `model_survival.pkl`.
- **Restrições Monotónicas Clínicas**: O modelo incorpora restrições anatómicas e clínicas para assegurar que estágios mais avançados (Dukes C/D) ou ausência de tratamento complementar reflitam sempre maior risco biológico, eliminando anomalias estatísticas decorrentes de pequenas amostras.
- **Explicabilidade Clínica Transparente (XAI)**: Integração com LIME demonstrando visualmente quais fatores (idade, localização, estágio, terapias) elevaram ou reduziram o risco para cada paciente individualmente.
- **Triagem em Lote (Batch Processing)**: Submissão de ficheiros CSV com download de template, validação linha a linha e exportação de relatórios consolidados.

---

## 📦 Instalação e Execução

### Pré-requisitos
Certifique-se de que tem o Python 3.12 (ou superior) e o `pip` instalados na sua máquina.

### Passos para Configuração

1. **Clonar o Repositório:**
   ```bash
   git clone https://github.com/LeoTeca18/colorectal-cancer-prognosis.git
   cd colorectal-cancer-prognosis
   ```

2. **Criar e Ativar um Ambiente Virtual:**
   ```bash
   python -m venv venv
   # No Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
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
   streamlit run app.py
   ```

---

## 🩺 Relevância Médica e Engenharia de Qualidade

Este sistema foi projetado para mitigar os problemas clássicos de transição de modelos experimentais para ambientes de produção médica. Ao adotar uma arquitetura baseada em **Monolito Modular**, o código torna-se altamente testável, manutenível e extensível. Caso surja a necessidade de substituir o motor do XGBoost por outro algoritmo ou atualizar a coorte de pacientes, as alterações ficam restritas exclusivamente à camada de *Infrastructure*, mantendo o domínio clínico e a interface completamente intactos.

---
Desenvolvido como parte de uma iniciativa de investigação e excelência em Engenharia de Software Aplicada à Saúde.
