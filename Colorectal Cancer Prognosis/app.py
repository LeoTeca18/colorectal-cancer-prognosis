import streamlit as st

# Configuração de página do Streamlit (deve ser a primeira chamada da página)
st.set_page_config(
    page_title="Colorectal Cancer Prognosis Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injeção de dependências e importação dos módulos locais (arquitetura SOLID / DDD)
from src.infrastructure.xgb_engine import XGBoostInferenceEngine
from src.application.prognosis_service import PrognosisService
from src.ui.patient_form import render_patient_form
from src.ui.batch_processor import render_batch_processor
from src.ui.dashboard import render_dashboard

# Inicialização dos serviços (Injeção de Dependências - DIP)
@st.cache_resource
def init_prognosis_service():
    engine = XGBoostInferenceEngine()
    return PrognosisService(inference_engine=engine)

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
        margin-bottom: 1.5rem;
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
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(0, 180, 219, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
    }
    
    /* Abas personalizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        font-weight: 600;
        border-radius: 8px 8px 0px 0px;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

# Renderização do cabeçalho da aplicação
st.markdown(
    """
    <div class="header-container">
        <h1 class="main-title">🧬 Colorectal Cancer Prognosis Engine</h1>
        <div class="sub-title">Sistema de Suporte à Decisão Clínica (CDSS) • Recorrência e Sobrevida Livre de Doença (DFS)</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Abas de navegação da aplicação
tab_individual, tab_batch, tab_dashboard = st.tabs([
    "📋 Prognóstico Individual", 
    "📁 Triagem em Lote (Batch)", 
    "📊 Painel Clínico & Métricas"
])

with tab_individual:
    render_patient_form(service)

with tab_batch:
    render_batch_processor(service)

with tab_dashboard:
    render_dashboard()
