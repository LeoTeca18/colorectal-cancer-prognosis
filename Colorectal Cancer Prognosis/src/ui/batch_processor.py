import streamlit as st
import pandas as pd
import io
from typing import List
from src.domain.patient import Patient
from src.application.prognosis_service import PrognosisService

def render_batch_processor(service: PrognosisService):
    st.subheader("📁 Triagem Clínica em Lote (Batch Processing)")
    st.markdown(
        "Submeta ficheiros CSV contendo múltiplos prontuários para realizar triagem automatizada em larga escala. "
        "O sistema validará os dados através do domínio clínico e executará a inferência preditiva vetorizada."
    )
    
    # Modelo / Template de Exemplo para Download
    sample_df = pd.DataFrame([
        {"patient_id": "PAC-101", "age": 62, "gender": "Male", "dukes_stage": "A", "location": "Colon", "adj_chem": False, "adj_radio": False},
        {"patient_id": "PAC-102", "age": 68, "gender": "Female", "dukes_stage": "B", "location": "Left", "adj_chem": True, "adj_radio": False},
        {"patient_id": "PAC-103", "age": 72, "gender": "Male", "dukes_stage": "C", "location": "Rectum", "adj_chem": True, "adj_radio": True},
        {"patient_id": "PAC-104", "age": 75, "gender": "Male", "dukes_stage": "D", "location": "Right", "adj_chem": False, "adj_radio": False},
    ])
    
    col_dl, col_space = st.columns([1, 2])
    with col_dl:
        csv_buffer = io.StringIO()
        sample_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Descarregar Modelo CSV de Exemplo",
            data=csv_buffer.getvalue(),
            file_name="template_pacientes_oncologia.csv",
            mime="text/csv",
            width="stretch"
        )
        
    uploaded_file = st.file_uploader("Carregar Ficheiro CSV com Pacientes", type=["csv"], help="O ficheiro deve conter as colunas: patient_id, age, gender, dukes_stage, location, adj_chem, adj_radio")
    
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.markdown("##### Dados Carregados:")
            st.dataframe(df_input.head(10), width="stretch")
            
            required_cols = {"patient_id", "age", "gender", "dukes_stage", "location"}
            missing_cols = required_cols - set(df_input.columns)
            if missing_cols:
                st.error(f"⚠️ O ficheiro CSV não possui as colunas obrigatórias: {missing_cols}")
                return
                
            if st.button("Executar Triagem em Lote", type="primary", width="stretch"):
                patients: List[Patient] = []
                errors = []
                
                for idx, row in df_input.iterrows():
                    try:
                        p = Patient(
                            patient_id=str(row.get("patient_id", f"PAC-{idx:03d}")),
                            age=int(row.get("age", 62)),
                            gender=str(row.get("gender", "Male")),
                            dukes_stage=str(row.get("dukes_stage", "B")),
                            location=str(row.get("location", "Colon")),
                            adj_radio=bool(row.get("adj_radio", False)),
                            adj_chem=bool(row.get("adj_chem", False))
                        )
                        patients.append(p)
                    except Exception as err:
                        errors.append(f"Linha {idx + 1} (Paciente {row.get('patient_id', '?')}): {err}")
                        
                if errors:
                    st.warning(f"Foram encontrados {len(errors)} erros de validação nos registos:")
                    for e in errors[:5]:
                        st.caption(f"• {e}")
                    if len(errors) > 5:
                        st.caption(f"... e mais {len(errors) - 5} erros.")
                        
                if not patients:
                    st.error("Nenhum paciente válido para processamento.")
                    return
                    
                with st.spinner(f"A processar {len(patients)} prontuários..."):
                    results = service.predict_batch(patients)
                    
                st.success(f"Triagem concluída com sucesso para {len(results)} pacientes!")
                
                # Consolidar em DataFrame
                res_data = []
                for p, r in zip(patients, results):
                    res_data.append({
                        "ID": r.patient_id,
                        "Idade": p.age,
                        "Género": p.gender,
                        "Estágio": p.dukes_stage,
                        "Localização": p.location,
                        "Prob. Recorrência (%)": round(r.recurrence_probability * 100, 1),
                        "Nível de Risco": r.risk_level,
                        "DFS Estimado (Meses)": r.predicted_survival_months
                    })
                    
                res_df = pd.DataFrame(res_data)
                
                # Exibir KPIs do lote
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                with kpi1:
                    st.metric("Total de Pacientes", len(res_df))
                with kpi2:
                    high_risk = (res_df["Nível de Risco"] == "Alto").sum()
                    st.metric("Casos de Alto Risco", f"{high_risk} ({high_risk / len(res_df) * 100:.0f}%)")
                with kpi3:
                    avg_prob = res_df["Prob. Recorrência (%)"].mean()
                    st.metric("Média de Recorrência", f"{avg_prob:.1f}%")
                with kpi4:
                    avg_dfs = res_df["DFS Estimado (Meses)"].mean()
                    st.metric("DFS Médio", f"{avg_dfs:.1f} meses")
                    
                st.markdown("##### Tabela Consolidada de Triagem:")
                st.dataframe(res_df, width="stretch")
                
                # Botão para download dos resultados
                out_buffer = io.StringIO()
                res_df.to_csv(out_buffer, index=False)
                st.download_button(
                    label="📥 Descarregar Relatório da Triagem (CSV)",
                    data=out_buffer.getvalue(),
                    file_name="resultado_triagem_prognostico.csv",
                    mime="text/csv",
                    width="stretch"
                )
                
        except Exception as e:
            st.error(f"Erro ao processar ficheiro: {e}")
