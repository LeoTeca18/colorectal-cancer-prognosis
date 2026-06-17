import streamlit as st

# Configuração de página do Streamlit (deve ser a primeira chamada da página)
st.set_page_config(
    page_title="Colorectal Cancer Prognosis Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injeção de dependências e importação dos módulos locais (arquitetura SOLID)
from src.infrastructure.mock_engine import MockInferenceEngine
from src.infrastructure.csv_loader import CSVDataLoader
from src.application.prognosis_service import PrognosisService
from src.ui.patient_form import render_patient_form
from src.ui.batch_processor import render_batch_processor

# Inicialização dos serviços (Injeção de Dependências - DIP)
@st.cache_resource
def init_prognosis_service():
    engine = MockInferenceEngine()
    data_loader = CSVDataLoader()
    return PrognosisService(inference_engine=engine, data_loader=data_loader)

service = init_prognosis_service()

# Estilos CSS Personalizados Premium com Paleta Médica (Cian/Azul/Verde-água)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Configuração global de fontes */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Cabeçalho principal com gradiente médico e científico */
    .header-container {
        padding: 2.2rem 1.5rem;
        background: linear-gradient(135deg, #0b1f3b 0%, #0d324d 50%, #0f4c5c 100%);
        border-radius: 16px;
        border: 1px solid rgba(0, 180, 219, 0.25);
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(13, 50, 77, 0.35);
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        padding-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .sub-title {
        font-size: 1.15rem;
        color: #e0f2fe !important;
        font-weight: 400;
        margin-top: 0.5rem;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Customização dos botões (Azul Clínico) */
    div.stButton > button {
        background: linear-gradient(90deg, #0083B0 0%, #00B4DB 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.3s ease-in-out !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0, 180, 219, 0.4) !important;
    }
    
    /* Cartões informativos de Glassmorphism */
    .glass-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(22, 129, 122, 0.15);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

# Renderização do cabeçalho da aplicação
st.markdown(
    """
    <div class="header-container">
        <h1 class="main-title">🧬 Colorectal Cancer Prognosis</h1>
        <div class="sub-title">Motor de Inferência e Análise Predictiva de Recorrência e Sobrevida (DFS)</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Painel Lateral para navegação estilizada
st.sidebar.image("https://img.icons8.com/ios-filled/128/00f2fe/artificial-intelligence.png", width=70)
st.sidebar.title("Navegação")
st.sidebar.markdown("Selecione a funcionalidade da interface:")

menu_option = st.sidebar.radio(
    label="Menu de Opções",
    options=[
        "🎯 Prognóstico Individual", 
        "📁Processamento em Lote (CSV)"
    ],
    index=0,
    label_visibility="collapsed"
)


# Fluxo de roteamento das vistas baseado no menu selecionado
if "🎯 Prognóstico Individual" in menu_option:
    render_patient_form(service)
    
elif "📁Processamento em Lote (CSV)" in menu_option:
    render_batch_processor(service)
