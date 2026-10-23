import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def render_dashboard():
    st.subheader("📊 Painel Clínico & Análise Epidemiológica de Sobrevida")
    st.markdown(
        "Monitoramento populacional das dinâmicas prognósticas de cancro colorretal. "
        "Compara curvas estimadas de sobrevida livre de doença (DFS) e a sensibilidade dos fatores tumorais."
    )
    
    # Cartões informativos de Glassmorphism
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            """
            <div class="glass-card">
                <span style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Incidência Populacional</span>
                <h3 style="margin: 0.2rem 0; color: #38bdf8;">3º Cancro Mais Comum</h3>
                <span style="font-size: 0.8rem; color: #cbd5e1;">Globalmente em homens e mulheres</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            """
            <div class="glass-card">
                <span style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Sobrevida Média em 5 Anos</span>
                <h3 style="margin: 0.2rem 0; color: #34d399;">~65% Geral</h3>
                <span style="font-size: 0.8rem; color: #cbd5e1;">Varia de 91% (A) a 14% (D)</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            """
            <div class="glass-card">
                <span style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Impacto da Quimioterapia</span>
                <h3 style="margin: 0.2rem 0; color: #a78bfa;">-30% Risco Relativo</h3>
                <span style="font-size: 0.8rem; color: #cbd5e1;">Em estágios II de alto risco e III</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            """
            <div class="glass-card">
                <span style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Predição do Modelo</span>
                <h3 style="margin: 0.2rem 0; color: #f472b6;">Dual XGBoost</h3>
                <span style="font-size: 0.8rem; color: #cbd5e1;">Risco + Curva paramétrica DFS</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("##### 📈 Curvas de Sobrevida Livre de Doença (DFS) por Estágio de Dukes")
        
        # Simulação paramétrica de curvas de Kaplan-Meier baseadas nos dados clínicos
        times = np.linspace(0, 100, 50)
        
        # Modelagem de sobrevivência decrescente S(t) = exp(-(lambda * t)^gamma)
        s_a = np.exp(-((0.007 * times) ** 1.1)) * 100
        s_b = np.exp(-((0.015 * times) ** 1.15)) * 100
        s_c = np.exp(-((0.025 * times) ** 1.2)) * 100
        s_d = np.exp(-((0.050 * times) ** 1.25)) * 100
        
        fig_km = go.Figure()
        fig_km.add_trace(go.Scatter(x=times, y=s_a, mode='lines', name='Estágio A (Localizado)', line=dict(color='#10B981', width=3)))
        fig_km.add_trace(go.Scatter(x=times, y=s_b, mode='lines', name='Estágio B (Muscular)', line=dict(color='#3B82F6', width=3)))
        fig_km.add_trace(go.Scatter(x=times, y=s_c, mode='lines', name='Estágio C (Linfonodal)', line=dict(color='#F59E0B', width=3)))
        fig_km.add_trace(go.Scatter(x=times, y=s_d, mode='lines', name='Estágio D (Metastático)', line=dict(color='#EF4444', width=3)))
        
        fig_km.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAF9F6"),
            xaxis=dict(title="Meses Pós-Intervenção Cirúrgica", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="% Livre de Recorrência", range=[0, 105], gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_km, width="stretch")
        
    with col_chart2:
        st.markdown("##### 🎯 Frequência e Risco Médio por Localização Anatômica")
        
        loc_data = pd.DataFrame({
            "Localização": ["Cólon Direito", "Cólon Esquerdo", "Reto", "Cólon Geral"],
            "Risco Médio de Recorrência (%)": [45.5, 60.0, 77.8, 50.0],
            "Casos na Coorte": [22, 20, 18, 2],
            "Impacto": ["Menor taxa local", "Perfil intermediário", "Maior risco local (anatomia pélvica)", "Referência"]
        })
        
        fig_loc = px.bar(
            loc_data,
            x="Localização",
            y="Risco Médio de Recorrência (%)",
            color="Risco Médio de Recorrência (%)",
            color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
            text="Risco Médio de Recorrência (%)",
            hover_data=["Casos na Coorte", "Impacto"]
        )
        fig_loc.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_loc.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAF9F6"),
            yaxis=dict(range=[0, 95], gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_loc, width="stretch")
        
    st.markdown("---")
    st.markdown("#### 🩺 Diretrizes de Estadiamento e Tomada de Decisão Clínica")
    
    t1, t2, t3, t4 = st.tabs(["Estágio Dukes A", "Estágio Dukes B", "Estágio Dukes C", "Estágio Dukes D"])
    with t1:
        st.markdown(
            "**Dukes A (T1-T2 N0 M0):** Tumor confinado à submucosa ou muscular própria do cólon/reto.\n"
            "* **Conduta Terapêutica:** Ressecção cirúrgica com margens livres (colectomia/proctectomia). Não requer rotineiramente quimioterapia adjuvante.\n"
            "* **Prognóstico:** Excelente (> 90% de sobrevida em 5 anos)."
        )
    with t2:
        st.markdown(
            "**Dukes B (T3-T4 N0 M0):** Invasão além da camada muscular, atingindo a subserosa ou tecidos perirretocolônicos, sem invasão linfonodal.\n"
            "* **Conduta Terapêutica:** Cirurgia primária. Quimioterapia adjuvante considerada em casos de alto risco (ex: < 12 linfonodos ressecados, perfuração tumoral, tumores T4).\n"
            "* **Prognóstico:** Intermediário (~70-80% de sobrevida em 5 anos)."
        )
    with t3:
        st.markdown(
            "**Dukes C (Qualquer T, N1-N2, M0):** Disseminação confirmada para gânglios linfáticos regionais.\n"
            "* **Conduta Terapêutica:** Colectomia oncológica com linfadenectomia alargada + Quimioterapia adjuvante sistêmica recomendada (ex: regime FOLFOX ou CAPOX).\n"
            "* **Prognóstico:** Risco moderado a alto de recorrência sistêmica (~50-60% sobrevida)."
        )
    with t4:
        st.markdown(
            "**Dukes D (Qualquer T, Qualquer N, M1):** Presença de metástases à distância (fígado, peritônio, pulmões).\n"
            "* **Conduta Terapêutica:** Abordagem multidisciplinar combinando quimioterapia sistêmica, terapias-alvo (anti-VEGF, anti-EGFR) e possível ressecção de metástases hepáticas oligometastáticas.\n"
            "* **Prognóstico:** Risco crítico, foco em prolongamento de sobrevida livre de progressão e qualidade de vida."
        )
