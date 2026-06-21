import streamlit as st
from src.domain.patient import Patient
from src.application.prognosis_service import PrognosisService
import plotly.graph_objects as go

def render_patient_form(service: PrognosisService):
    st.subheader("📋 Formulário de Prognóstico Individual")
    st.markdown("Insira os dados clínicos do paciente para calcular a probabilidade de recorrência do tumor e a expectativa de sobrevivência livre de doença.")
    
    with st.form("patient_data_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            patient_id = st.text_input("Identificador do Paciente", value="PAC-063", help="Código de identificação do paciente.")
            age = st.slider("Idade do Paciente (anos)", min_value=1, max_value=120, value=65, help="Idade no momento do diagnóstico secundário ou cirurgia.")
            gender = st.selectbox("Género", ["Male", "Female"], index=0, format_func=lambda x: "Masculino" if x == "Male" else "Feminino")
            
        with col2:
            dukes_stage = st.selectbox(
                "Estágio de Dukes", 
                ["A", "B", "C", "D"], 
                index=1,
                help=(
                    "Grau de disseminação do cancro:\n"
                    "A: Limitado à mucosa ou submucosa intestinal.\n"
                    "B: Infiltra a camada muscular sem afetar gânglios.\n"
                    "C: Afeta os gânglios linfáticos regionais.\n"
                    "D: Metástases à distância (ex: fígado, pulmão)."
                )
            )
            location = st.selectbox(
                "Localização do Tumor", 
                ["Colon", "Rectum", "Left", "Right"], 
                index=0,
                format_func=lambda x: {
                    "Colon": "Cólon (Geral)", 
                    "Rectum": "Reto", 
                    "Left": "Cólon Esquerdo (Descendente)", 
                    "Right": "Cólon Direito (Ascendente)"
                }[x]
            )
            
            st.markdown("**Terapias Adjuvantes Recebidas**")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                adj_chem = st.checkbox("Quimioterapia", value=False, help="O paciente recebeu quimioterapia complementar pós-cirúrgica.")
            with col_t2:
                adj_radio = st.checkbox("Radioterapia", value=False, help="O paciente recebeu radioterapia complementar pós-cirúrgica.")
                
        submitted = st.form_submit_button("Gerar Prognóstico", width="stretch")
        
    if submitted:
        try:
            # Sem dados de expressão genética informados via formulário individual
            gene_expression = {}
            
            # Instanciar Paciente (efetua as validações definidas no domínio)
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
            
            # Executar previsão através do serviço
            with st.spinner("A processar prognóstico..."):
                result = service.predict_single(patient)
                
            st.toast("Prognóstico gerado com sucesso!", icon="✅")
            
            # Apresentação do Resultado
            st.markdown("---")
            st.subheader(f"📊 Resultados de Prognóstico: {result.patient_id}")
            
            res_col1, res_col2 = st.columns([1.2, 1])
            
            with res_col1:
                # Gauge de probabilidade de recorrência
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result.recurrence_probability * 100,
                    number={
                        'suffix': "%",
                        'font': {'size': 54, 'color': '#000000', 'family': 'Arial Black, Arial, sans-serif'}
                    },
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Probabilidade de Recorrência", 'font': {'size': 18, 'color': '#FAF9F6'}},
                    gauge={
                        'axis': {'range': [0, 100], 'ticksuffix': "%", 'tickcolor': '#FAF9F6'},
                        'bar': {'color': "#4F46E5"},  # Indigo moderno
                        'bgcolor': "rgba(0,0,0,0)",
                        'steps': [
                            {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.2)"},  # Green
                            {'range': [30, 65], 'color': "rgba(245, 158, 11, 0.2)"}, # Yellow
                            {'range': [65, 100], 'color': "rgba(239, 68, 68, 0.2)"}  # Red
                        ],
                        'threshold': {
                            'line': {'color': "#EF4444", 'width': 3},
                            'thickness': 0.75,
                            'value': result.recurrence_probability * 100
                        }
                    }
                ))
                fig.update_layout(
                    height=250, 
                    margin=dict(l=30, r=30, t=40, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': "#FAF9F6"}
                )
                st.plotly_chart(fig, width="stretch")
                
            with res_col2:
                # Mapeamento estético para cada nível de risco
                risk_styles = {
                    "Baixo": {"color": "#10B981", "bg": "rgba(16, 185, 129, 0.1)"},
                    "Médio": {"color": "#F59E0B", "bg": "rgba(245, 158, 11, 0.1)"},
                    "Alto": {"color": "#EF4444", "bg": "rgba(239, 68, 68, 0.1)"}
                }
                style = risk_styles.get(result.risk_level, {"color": "#FAF9F6", "bg": "#333333"})
                
                st.markdown(
                    f"<div style='"
                    f"background-color: {style['bg']}; "
                    f"border: 1px solid {style['color']}; "
                    f"border-radius: 8px; "
                    f"padding: 15px; "
                    f"margin-top: 10px; "
                    f"text-align: center;'"
                    f">"
                    f"<span style='font-size: 14px; text-transform: uppercase; letter-spacing: 1.5px; color: #9CA3AF;'>Classificação de Risco</span><br/>"
                    f"<span style='font-size: 28px; font-weight: 800; color: {style['color']};'>{result.risk_level}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                
                # Card de meses de sobrevida
                st.metric(
                    label="Expectativa de Sobrevivência Livre de Recorrência (DFS)", 
                    value=f"{result.predicted_survival_months} Meses",
                    help="Estimativa do tempo livre da doença após a cirurgia primária."
                )
                
            st.markdown("#### 🔍 Explicação e Justificação do Modelo")
            st.info(result.details)
            
            if hasattr(result, 'explanation') and result.explanation:
                st.markdown("#### 🧠 Interpretabilidade Local (XAI - Explicação por LIME)")
                st.markdown(
                    "O gráfico abaixo exibe os fatores com maior impacto na predição de recorrência deste paciente. "
                    "<span style='color:#EF4444; font-weight:bold;'>Barras vermelhas (positivas)</span> indicam fatores que elevam o risco, "
                    "enquanto <span style='color:#10B981; font-weight:bold;'>barras verdes (negativas)</span> indicam fatores protetores ou redutores de risco.",
                    unsafe_allow_html=True
                )
                
                # Separar os nomes das variáveis e os pesos e formatar
                features_names = [item[0].replace('=', ': ') for item in result.explanation]
                weights = [item[1] for item in result.explanation]
                
                # Cores estéticas: Vermelho para impacto positivo, Verde para impacto negativo
                colors = ["#EF4444" if w > 0 else "#10B981" for w in weights]
                
                fig_lime = go.Figure(go.Bar(
                    x=weights,
                    y=features_names,
                    orientation='h',
                    marker_color=colors,
                    hovertemplate="Factor: %{y}<br>Impacto: %{x:.4f}<extra></extra>"
                ))
                
                fig_lime.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=20, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': "#FAF9F6"},
                    xaxis={
                        'gridcolor': 'rgba(255,255,255,0.05)', 
                        'tickcolor': '#FAF9F6', 
                        'title': 'Contribuição/Impacto na Probabilidade'
                    },
                    yaxis={'autorange': 'reversed', 'tickcolor': '#FAF9F6'}
                )
                st.plotly_chart(fig_lime, width="stretch")
            
        except ValueError as ve:
            st.error(f"⚠️ Erro de Validação: {ve}")
        except Exception as e:
            st.error(f"🔥 Erro inesperado: {e}")
