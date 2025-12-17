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
            
        except Exception as e:
            st.error(f"Error parsing file: {e}")
else:
    st.write("Manual entry not yet implemented.")

st.header("Results")
st.write("No data loaded.")
