import streamlit as st
import pandas as pd
import io
import plotly.express as px
from src.application.prognosis_service import PrognosisService

def render_batch_processor(service: PrognosisService):
    st.subheader("📁 Processamento em Lote (Datasets)")

    
    uploaded_file = st.file_uploader("Escolha o ficheiro CSV contendo os pacientes", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Ler e processar o arquivo usando o serviço
            with st.spinner("A carregar ficheiro e a executar o motor de inferência..."):
                # Carregar e prever em lote
                patients = service._data_loader.load_patients(uploaded_file)
                
                if not patients:
                    st.warning("⚠️ Nenhum paciente válido foi encontrado no ficheiro submetido.")
                    return
                
                results = service._inference_engine.predict_batch(patients)
            
            st.success(f"✅ Sucesso: {len(patients)} pacientes processados e analisados!")
            
            # Construir DataFrame para a tabela de visualização
            table_data = []
            for p, r in zip(patients, results):
                # Calcular número de genes informados
                num_genes = len(p.gene_expression)
                
                table_data.append({
                    "ID Paciente": p.patient_id,
                    "Idade": p.age,
                    "Género": "Masculino" if p.gender.strip().capitalize() in {"Male", "M", "Masculino"} else "Feminino",
                    "Estágio Dukes": p.dukes_stage,
                    "Localização": p.location,
                    "Radioterapia": "Sim" if p.adj_radio else "Não",
                    "Quimioterapia": "Sim" if p.adj_chem else "Não",
                    "Genes Informados": num_genes,
                    "Prob. Recorrência": f"{r.recurrence_probability * 100:.1f}%",
                    "Risco": r.risk_level,
                    "DFS Previsto (meses)": r.predicted_survival_months
                })
                
            df_results = pd.DataFrame(table_data)
            
            # Exibir resultados na tabela interativa do Streamlit
            st.markdown("### 📋 Resultados Detalhados por Paciente")
            st.dataframe(
                df_results,
                use_container_width=True,
                column_config={
                    "Prob. Recorrência": st.column_config.TextColumn(
                        "Prob. Recorrência (5 anos)",
                        help="Probabilidade calculada de o tumor recorrer.",
                    ),
                    "Risco": st.column_config.TextColumn(
                        "Nível de Risco",
                        help="Classificação baseada nos limiares de recorrência.",
                    )
                }
            )
            
            # Criar um CSV contendo os resultados previstos para download
            export_data = []
            for p, r in zip(patients, results):
                row = {
                    "patient_id": p.patient_id,
                    "age": p.age,
                    "gender": p.gender,
                    "dukes_stage": p.dukes_stage,
                    "location": p.location,
                    "adj_radio": int(p.adj_radio),
                    "adj_chem": int(p.adj_chem),
                    "recurrence_probability": r.recurrence_probability,
                    "risk_level": r.risk_level,
                    "predicted_survival_months": r.predicted_survival_months
                }
                # Adicionar genes extraídos ao CSV
                for g_id, g_val in p.gene_expression.items():
                    row[g_id] = g_val
                export_data.append(row)
                
            df_export = pd.DataFrame(export_data)
            csv_buffer = io.StringIO()
            df_export.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="💾 Descarregar Resultados Completos com Predições (CSV)",
                data=csv_buffer.getvalue(),
                file_name="prognostico_resultados_lote.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.markdown("---")
            st.subheader("📊 Métricas Consolidadas do Lote")
            
            # Métricas em KPI cards
            m_col1, m_col2, m_col3 = st.columns(3)
            
            # Converter a percentagem de recorrência de string para float para calcular a média
            recurrence_floats = [r.recurrence_probability * 100 for r in results]
            avg_recurrence = sum(recurrence_floats) / len(recurrence_floats)
            avg_survival = sum(r.predicted_survival_months for r in results) / len(results)
            
            high_risk_pct = (sum(1 for r in results if r.risk_level == "Alto") / len(results)) * 100
            
            m_col1.metric("Probabilidade Média de Recorrência", f"{avg_recurrence:.1f}%")
            m_col2.metric("Média de DFS Previsto", f"{avg_survival:.1f} meses")
            m_col3.metric("Pacientes em Alto Risco", f"{high_risk_pct:.1f}%")
            
            # Gráficos analíticos do lote
            g_col1, g_col2 = st.columns(2)
            
            with g_col1:
                # Distribuição por classes de risco
                risk_counts = df_results["Risco"].value_counts().reset_index()
                risk_counts.columns = ["Risco", "Contagem"]
                
                color_map = {"Baixo": "#10B981", "Médio": "#F59E0B", "Alto": "#EF4444"}
                
                fig_risk = px.pie(
                    risk_counts, 
                    values="Contagem", 
                    names="Risco", 
                    title="Distribuição das Classes de Risco",
                    color="Risco",
                    color_discrete_map=color_map,
                    hole=0.4
                )
                fig_risk.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", 
                    font={'color': "#FAF9F6"},
                    margin=dict(t=40, b=20, l=10, r=10),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_risk, use_container_width=True)
                
            with g_col2:
                # Histograma do tempo previsto livre de doença (DFS)
                df_results["DFS Previsto (meses)"] = df_results["DFS Previsto (meses)"].astype(float)
                
                fig_dfs = px.histogram(
                    df_results, 
                    x="DFS Previsto (meses)", 
                    title="Distribuição do DFS Previsto (Sobrevivência)",
                    color_discrete_sequence=["#4F46E5"], # Violeta/Indigo moderno
                    nbins=10
                )
                fig_dfs.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", 
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': "#FAF9F6"},
                    xaxis={'gridcolor': 'rgba(255,255,255,0.05)', 'tickcolor': '#FAF9F6'},
                    yaxis={'gridcolor': 'rgba(255,255,255,0.05)', 'tickcolor': '#FAF9F6'},
                    margin=dict(t=40, b=20, l=10, r=10)
                )
                st.plotly_chart(fig_dfs, use_container_width=True)
                
        except ValueError as ve:
            st.error(f"⚠️ Erro ao processar ficheiro CSV: {ve}")
        except Exception as e:
            st.error(f"🔥 Erro interno durante o processamento do lote: {e}")
