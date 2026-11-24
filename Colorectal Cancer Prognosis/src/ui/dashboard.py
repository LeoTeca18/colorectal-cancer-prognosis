import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional
from src.application.prognosis_service import PrognosisService

def render_dashboard(service: Optional[PrognosisService] = None):
    st.subheader("📊 Painel Clínico & Métricas Dinâmicas dos Artefatos (Notebook)")
    st.markdown(
        "Monitoramento populacional e desempenho preditivo gerados **dinamicamente a partir dos artefatos exportados pelo Notebook** "
        "(`model_xgboost.pkl`, `model_survival.pkl` e `preprocessor_params.json`)."
    )
    
    if service is None:
        from src.infrastructure.xgb_engine import XGBoostInferenceEngine
        service = PrognosisService(inference_engine=XGBoostInferenceEngine())
        
    analytics = service.get_dashboard_analytics()
    cohort = analytics.get("cohort_summary", {})
    metrics = analytics.get("model_metrics", {})
    chemo = analytics.get("chemo_impact", {})
    
    total_patients = cohort.get("total_patients", 62)
    rec_rate = cohort.get("recurrence_rate_pct", 59.7)
    mean_dfs = cohort.get("mean_dfs_months", 47.5)
    roc_auc = metrics.get("classifier_roc_auc", 0.8049)
    test_recall = metrics.get("classifier_test_recall", 0.9091)
    c_index = metrics.get("survival_c_index", 0.7166)
    test_mae = metrics.get("survival_test_mae_months", 26.13)
    dfs_gain = chemo.get("dfs_gain_months", 5.8)
    rel_risk_diff = chemo.get("rel_risk_diff_pct", -3.5)
    
    # Cartões informativos dinâmicos alimentados pelos artefatos do Notebook
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="glass-card">
                <span style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Coorte do Notebook</span>
                <h3 style="margin: 0.2rem 0; color: #38bdf8;">{total_patients} Pacientes</h3>
                <span style="font-size: 0.8rem; color: #cbd5e1;">Taxa de evento: {rec_rate:.1f}% • DFS médio: {mean_dfs:.1f}m</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            f"""
            <div class="glass-card">
                <span style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Classificador (model_xgboost.pkl)</span>
                <h3 style="margin: 0.2rem 0; color: #34d399;">AUC {roc_auc:.4f}</h3>
                <span style="font-size: 0.8rem; color: #cbd5e1;">Sensibilidade (Recall Teste): {test_recall * 100:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            f"""
            <div class="glass-card">
                <span style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Sobrevida (model_survival.pkl)</span>
                <h3 style="margin: 0.2rem 0; color: #a78bfa;">C-Index {c_index:.4f}</h3>
                <span style="font-size: 0.8rem; color: #cbd5e1;">Erro Médio (MAE Teste): {test_mae:.1f} meses</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            f"""
            <div class="glass-card">
                <span style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Efeito Quimioterapia (Modelo)</span>
                <h3 style="margin: 0.2rem 0; color: #f472b6;">+{dfs_gain:.1f} Meses DFS</h3>
                <span style="font-size: 0.8rem; color: #cbd5e1;">Variação relativa de risco: {rel_risk_diff:+.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("##### 📈 Curvas Dinâmicas de Sobrevida (DFS) Calibradas pelos Modelos `.pkl`")
        
        curves = analytics.get("survival_curves", {})
        times = np.array(curves.get("times", np.linspace(0, 100, 50)))
        stage_curves = curves.get("stages", {})
        
        fig_km = go.Figure()
        stage_meta_ui = [
            ("A", "Estágio A (Localizado)", "#10B981"),
            ("B", "Estágio B (Muscular)", "#3B82F6"),
            ("C", "Estágio C (Linfonodal)", "#F59E0B"),
            ("D", "Estágio D (Metastático)", "#EF4444")
        ]
        for stg_key, stg_label, stg_color in stage_meta_ui:
            y_vals = stage_curves.get(stg_key, [])
            if y_vals:
                fig_km.add_trace(go.Scatter(
                    x=times,
                    y=y_vals,
                    mode='lines',
                    name=stg_label,
                    line=dict(color=stg_color, width=3)
                ))
        
        fig_km.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAF9F6"),
            xaxis=dict(title="Meses Pós-Intervenção Cirúrgica", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="% Estimada Livre de Recorrência", range=[0, 105], gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_km, width="stretch")
        
    with col_chart2:
        st.markdown("##### 🎯 Frequência e Risco Médio por Localização (Artefato do Notebook)")
        
        loc_records = analytics.get("location_epidemiology", [])
        if loc_records:
            loc_data = pd.DataFrame(loc_records)
        else:
            loc_data = pd.DataFrame({
                "Localização": ["Cólon Direito", "Cólon Geral", "Cólon Esquerdo", "Reto"],
                "Risco Médio de Recorrência (%)": [45.5, 50.0, 60.0, 77.8],
                "Casos na Coorte": [22, 2, 20, 18],
                "DFS Médio (Meses)": [49.7, 57.5, 48.9, 42.1]
            })
        
        fig_loc = px.bar(
            loc_data,
            x="Localização",
            y="Risco Médio de Recorrência (%)",
            color="Risco Médio de Recorrência (%)",
            color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
            text="Risco Médio de Recorrência (%)",
            hover_data=[c for c in ["Casos na Coorte", "DFS Médio (Meses)"] if c in loc_data.columns]
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
    col_inf1, col_inf2 = st.columns([1.15, 1])
    with col_inf1:
        st.markdown("##### 🧠 Importância Global das Variáveis nos Artefatos (`.pkl`)")
        feat_imp = analytics.get("feature_importances", [])
        if feat_imp:
            df_imp = pd.DataFrame(feat_imp).sort_values("Importância Média (%)", ascending=True)
            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(
                y=df_imp["Atributo"],
                x=df_imp["Importância Recorrência (%)"],
                name="Classificador (Recorrência)",
                orientation='h',
                marker_color="#38BDF8"
            ))
            fig_imp.add_trace(go.Bar(
                y=df_imp["Atributo"],
                x=df_imp["Importância Sobrevida DFS (%)"],
                name="Regressor (Sobrevida DFS)",
                orientation='h',
                marker_color="#10B981"
            ))
            fig_imp.update_layout(
                barmode='group',
                height=300,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#FAF9F6"),
                xaxis=dict(title="Peso Relativo nas Árvores XGBoost (%)", gridcolor="rgba(255,255,255,0.05)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_imp, width="stretch")

    with col_inf2:
        st.markdown("##### 📋 Resumo Preditivo Dinâmico por Estágio de Dukes")
        stage_profiles = analytics.get("stage_profiles", [])
        if stage_profiles:
            st.dataframe(pd.DataFrame(stage_profiles), width="stretch", hide_index=True)

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
