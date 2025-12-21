import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from scipy.interpolate import make_interp_spline
from miget.io import MigetIO
from miget.core import VQModel

def render_inverse_problem():
    st.header("Reverse Problem")
    
    # Sidebar for Data and Config
    with st.sidebar:
        st.header("1. Input Data")
        uploaded_file = st.file_uploader("Upload .RAW or .CSV", type=["raw", "RAW", "csv", "CSV"])
        
        st.divider()
        st.header("2. Configuration")
        
        # We can't determine default index until data is loaded, but we can set defaults
        species = st.selectbox("Subject Species", ["Human", "Dog", "Horse"])
        
        st.subheader("Model Settings")
        weight_mode = st.selectbox("Weighting Mode", ["Retention", "Excretion"], index=0)
        
        # JAX removed per user request
        backend_key = "scipy"
        
        z_factor_override = st.slider("Smoothing (Z)", 1.0, 100.0, 40.0, help="Higher = Smoother")


    # Main Area Logic
    if not uploaded_file:
         # Empty State
         st.info("👋 **Welcome to the Reverse Problem Solver!**\n\nPlease upload a MIGET data file (.RAW or .CSV) in the sidebar to begin analysis.")
         return

    # Process File
    try:
        try:
             content = uploaded_file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
             content = uploaded_file.getvalue().decode("latin-1")
            
        if uploaded_file.name.endswith(".csv"):
            data = MigetIO.parse_csv(content)
        else:
            data = MigetIO.parse_legacy_file(content)
            
        st.session_state['miget_data'] = data
        
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return

    # Update Data Objects
    for run in data:
        run.species = species.lower()
        run.calculate_metrics()

    # --- Analysis Pipeline ---
    
    # 1. Run Selection
    if len(data) > 1:
        st.markdown("### Run Selection")
        run_names = [r.name for r in data]
        selected_idx = st.selectbox("Select Measurement", range(len(data)), format_func=lambda x: run_names[x], label_visibility="collapsed")
    else:
        selected_idx = 0
        
    current_run = data[selected_idx]
    
    # 2. Solver Execution
    model = VQModel(current_run, z_factor=z_factor_override, backend=backend_key)
    dist = model.solve(weight_mode=weight_mode.lower())
    
    # Comparison logic removed


    # --- UI Layout ---
    
    # Removed divider per user request
    
    tab_overview, tab_details, tab_export = st.tabs(["📈 Analysis & Plots", "📋 Detailed Metrics", "📥 Export"])
    
    with tab_overview:
        # Top Metrics (Updated Request: Shunt, Deadspace, Mean Q, Mean V, SD Q, SD V)
        met_cols = st.columns(6)
        
        shunt_val = dist.shunt * 100
        met_cols[0].metric("Shunt", f"{shunt_val:.1f} %")
        
        ds_val = dist.deadspace * 100
        met_cols[1].metric("Deadspace", f"{ds_val:.1f} %")
        
        # Mean Q / SD Q
        met_cols[2].metric("Mean Q (Log)", f"{dist.mean_q:.2f}")
        met_cols[3].metric("SD Q (Log)", f"{dist.sd_q:.2f}")
        
        # Mean V / SD V
        met_cols[4].metric("Mean V (Log)", f"{dist.mean_v:.2f}")
        met_cols[5].metric("SD V (Log)", f"{dist.sd_v:.2f}")


        # Main Plot
        st.subheader("V/Q Distribution")
        
        plt.rcParams.update({'font.size': 9, 'grid.alpha': 0.5})
        # Higher DPI for better resolution
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        
        # Plot Logic
        x = dist.vaq_ratios
        mask = (x > 0.005) & (x < 100.0)
        
        q_abs = dist.blood_flow * current_run.qt_measured
        v_abs = dist.ventilation * dist.va_total
        
        x_plot = x[mask]
        q_plot = q_abs[mask]
        v_plot = v_abs[mask]
        
        if len(x_plot) > 3:
            x_log = np.log10(x_plot)
            x_new = np.linspace(x_log.min(), x_log.max(), 300)
            
            spl_q = make_interp_spline(x_log, q_plot, k=3)
            spl_v = make_interp_spline(x_log, v_plot, k=3)
            q_smooth = spl_q(x_new)
            v_smooth = spl_v(x_new)
            q_smooth[q_smooth < 0] = 0
            v_smooth[v_smooth < 0] = 0
            
            x_smooth = 10**x_new
            
            
            # Filled area for better visuals
            ax.fill_between(x_smooth, q_smooth, color='tab:red', alpha=0.1)
            ax.plot(x_smooth, q_smooth, 'tab:red', linewidth=2, label='Perfusion')
            
            ax.fill_between(x_smooth, v_smooth, color='tab:blue', alpha=0.1)
            ax.plot(x_smooth, v_smooth, 'tab:blue', linewidth=2, label='Ventilation')




        else:
            ax.plot(x_plot, q_plot, 'tab:red', marker='o', label='Perfusion')
            ax.plot(x_plot, v_plot, 'tab:blue', marker='o', label='Ventilation')
            
        # Shunt Plot
        if shunt_val > 0.1:
            s_flow = dist.shunt * current_run.qt_measured
            ax.scatter([0.002], [s_flow], color='black', s=80, label=f'Shunt ({s_flow:.2f}L)')
            
        ax.set_xscale('log')
        ax.set_xlim(0.001, 100.0)
        ax.set_xlabel(r'$V_A/Q_T$ Ratio')
        ax.set_ylabel('Compartmental Flow (L/min)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False)
        ax.grid(True, which="major", ls="-", color="#e0e0e0")
        
        st.pyplot(fig, use_container_width=True)
        
        # Fit Plots (R/E)
        st.subheader("Goodness of Fit")
        col_fit1, col_fit2 = st.columns(2)
        
        # We need the continuous curves data
        sweep_sol, sweep_r, sweep_e, sweep_homo_r, sweep_homo_e = model.get_continuous_curves(dist)
        
        with col_fit1:
            fig_r, ax_r = plt.subplots(figsize=(5, 4))
            ax_r.plot(current_run.solubilities, current_run.retention, 'o', color='tab:orange', label='Measured')
            ax_r.plot(sweep_sol, sweep_r, '-', color='tab:orange', alpha=0.8, label='Predicted')
            ax_r.set_xscale('log')
            ax_r.set_title("Retention")
            ax_r.spines['top'].set_visible(False)
            ax_r.spines['right'].set_visible(False)
            ax_r.grid(True, ls="--", alpha=0.5)
            st.pyplot(fig_r, use_container_width=True)
            
        with col_fit2:
            fig_e, ax_e = plt.subplots(figsize=(5, 4))
            ax_e.plot(current_run.solubilities, current_run.excretion, 'o', color='tab:purple', label='Measured')
            ax_e.plot(sweep_sol, sweep_e, '-', color='tab:purple', alpha=0.8, label='Predicted')
            ax_e.set_xscale('log')
            ax_e.set_title("Excretion")
            ax_e.spines['top'].set_visible(False)
            ax_e.spines['right'].set_visible(False)
            ax_e.grid(True, ls="--", alpha=0.5)
            st.pyplot(fig_e, use_container_width=True)

    with tab_details:
        st.markdown("#### Recovered Parameters")
        
        params = {
            "Parameter": ["RSS", "Shunt (%)", "Deadspace (%)", "Mean Q (Log)", "SD Q (Log)", "Mean V (Log)", "SD V (Log)"],
            "Value": [
                dist.rss,
                dist.shunt * 100,
                dist.deadspace * 100,
                dist.mean_q,
                dist.sd_q,
                dist.mean_v,
                dist.sd_v
            ]
        }
        
        st.table(pd.DataFrame(params).style.format({
            "Value": "{:.4f}"
        }))
        
        st.markdown("#### Raw Gas Data")
        st.dataframe(pd.DataFrame({
            "Gas": current_run.gas_names,
            "Solubility": current_run.solubilities,
            "Retention (Meas)": current_run.retention,
            "Excretion (Meas)": current_run.excretion
        }), use_container_width=True)
        
        st.divider()
        st.markdown("#### Compartmental Distribution (Ranges)")
        
        # Calculate Interval Sums
        # Ranges: 0-0.01, 0.01-0.1, 0.1-1.0, 1.0-10.0, 10.0-100.0
        # Plus Shunt (Zero) and Deadspace (Infinity)
        
        x_ratios = dist.vaq_ratios
        q_vals = dist.blood_flow # fractional, sums to (1-shunt)
        v_vals = dist.ventilation # fractional, sums to (1-deadspace)
        
        # Indices for ranges (excluding shunt/deadspace logic which is handled separately)
        # Note: x_ratios[0] is Shunt, handled by dist.shunt
        # But x_ratios has 50 bins.
        
        def sum_range(low, high):
            mask = (x_ratios >= low) & (x_ratios < high)
            q_sum = np.sum(q_vals[mask])
            v_sum = np.sum(v_vals[mask])
            return q_sum, v_sum

        r1_q, r1_v = sum_range(0.0, 0.01)
        r2_q, r2_v = sum_range(0.01, 0.1)
        r3_q, r3_v = sum_range(0.1, 1.0)
        r4_q, r4_v = sum_range(1.0, 10.0)
        r5_q, r5_v = sum_range(10.0, 100.1) # Include 100
        
        # Shunt and Deadspace
        # Shunt is Q at V/Q=0. dist.shunt is fractional of TOTAL Q.
        # Deadspace is V at V/Q=Inf. dist.deadspace is fractional of TOTAL V.
        
        rows = [
            {"RANGE": "VA/Q OF ZERO",      "BLOOD FLOW": dist.shunt, "VENTILATION": 0.0},
            {"RANGE": "VA/Q RANGE 0-0.01",   "BLOOD FLOW": r1_q,       "VENTILATION": r1_v},
            {"RANGE": "VA/Q RANGE 0.01-0.1", "BLOOD FLOW": r2_q,       "VENTILATION": r2_v},
            {"RANGE": "VA/Q RANGE 0.1-1.0",  "BLOOD FLOW": r3_q,       "VENTILATION": r3_v},
            {"RANGE": "VA/Q RANGE 1.0-10.",  "BLOOD FLOW": r4_q,       "VENTILATION": r4_v},
            {"RANGE": "VA/Q RANGE 10.-100.", "BLOOD FLOW": r5_q,       "VENTILATION": r5_v},
            {"RANGE": "VA/Q OF INFINITY",    "BLOOD FLOW": 0.0,        "VENTILATION": dist.deadspace},
        ]
        
        df_ranges = pd.DataFrame(rows)
        
        # Display as a clean table
        st.table(df_ranges.style.format({
            "BLOOD FLOW": "{:.3f}",
            "VENTILATION": "{:.3f}"
        }))
        
        st.markdown("---")
    
    with tab_export:
        st.info("Batch Export for all runs in the file")
        
        if st.button("Generate Excel Report"):
             # Logic to process ALL runs again for the report 
             # (Simple implementation or reusing logic)
             results = []
             progress_bar = st.progress(0)
             
             for i, r in enumerate(data):
                 m = VQModel(r, z_factor=z_factor_override, backend=backend_key)
                 d = m.solve(weight_mode=weight_mode.lower())
                 
                 results.append({
                     "Run": r.name,
                     "RSS": d.rss,
                     "Shunt": d.shunt,
                     "Deadspace": d.deadspace,
                     "Mean Q": d.mean_q,
                     "SD Q": d.sd_q
                 })
                 progress_bar.progress((i+1)/len(data))
             
             df_res = pd.DataFrame(results)
             
             buffer = io.BytesIO()
             with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                 df_res.to_excel(writer, sheet_name='Summary', index=False)
                 
             st.download_button(
                 "📥 Download Excel",
                 data=buffer.getvalue(),
                 file_name=f"miget_results_{len(data)}_runs.xlsx",
                 mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
             )

