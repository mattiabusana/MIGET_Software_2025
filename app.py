import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="MIGET Modern", layout="wide")

st.title("MIGET: Multiple Inert Gas Elimination Technique")

st.sidebar.header("Data Input")
input_method = st.sidebar.radio("Input Method", ["Manual Entry", "File Upload"])

from miget.io import MigetIO
import pandas as pd

if input_method == "File Upload":
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["txt", "csv", "for"])
    if uploaded_file is not None:
        content = uploaded_file.getvalue().decode("utf-8")
        try:
            if uploaded_file.name.endswith(".csv"):
                data = MigetIO.parse_csv(content)
            else:
                data = MigetIO.parse_legacy_file(content)
            
            st.session_state['miget_data'] = data
            st.success(f"Loaded {len(data)} runs.")
            
            # Display Summary
            summary_data = []
            for run in data:
                summary_data.append({
                    "Name": run.name,
                    "VE": run.ve_measured,
                    "QT": run.qt_measured,
                    "Temp (Body)": run.temp_body
                })
            st.dataframe(pd.DataFrame(summary_data))
            
            # 2. V/Q Recovery
            if st.button("Process Data (Recover V/Q Distribution)"):
                from miget.core import VQModel
                import matplotlib.pyplot as plt
                
                st.subheader("V/Q Distributions")
                
                tabs = st.tabs([run.name for run in data])
                
                for i, run in enumerate(data):
                    with tabs[i]:
                        # Run Model
                        run.calculate_metrics() # Ensure derived values
                        model = VQModel(run)
                        dist = model.solve()
                        
                        # Plot
                        fig, ax = plt.subplots(figsize=(10, 4))
                        # VQ Ratios are log spaced. Plot on log scale.
                        # Axes: X = Log(VA/Q)? Or just VQ with log scale?
                        # Usually plotted against Log(V/Q).
                        
                        x = dist.vaq_ratios
                        # Mask shunt/deadspace for log plot?
                        mask = x > 0
                        
                        ax.plot(x[mask], dist.blood_flow[mask], 'r-o', label='Blood Flow (Q)')
                        ax.plot(x[mask], dist.ventilation[mask], 'b-o', label='Ventilation (V)')
                        ax.set_xscale('log')
                        ax.set_xlabel('VA/Q Ratio')
                        ax.set_ylabel('Fractional Distribution')
                        ax.legend()
                        ax.grid(True, which="both", ls="-")
                        
                        st.pyplot(fig)
                        
                        # Summary Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Shunt (%)", f"{dist.shunt*100:.2f}")
                        col2.metric("Deadspace (%)", f"{dist.deadspace*100:.2f}") # Logic needed if dist.deadspace is populated
                        col3.metric("LogSD Q", f"{dist.sd_q:.2f}")
                        col4.metric("LogSD V", f"{dist.sd_v:.2f}")
                        
                        # Data Table
                        # st.write(dist.blood_flow)
            
        except Exception as e:
            st.error(f"Error parsing file: {e}")
            raise e
else:
    st.write("Manual entry not yet implemented.")

st.header("Results")
st.write("No data loaded.")
