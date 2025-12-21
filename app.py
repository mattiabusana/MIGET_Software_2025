import streamlit as st
import numpy as np
import pandas as pd
from miget.io import MigetIO
from miget.core import VQModel
import matplotlib.pyplot as plt
import io
from scipy.interpolate import make_interp_spline

# Page Config
st.set_page_config(page_title="MIGET Modern", layout="wide")

def render_inverse_problem():
    st.header("Reverse Problem")
    
    st.sidebar.header("Data Input")
    input_method = "File Upload"
    
    if input_method == "File Upload":
        # File Uploader
        uploaded_file = st.sidebar.file_uploader("Upload .RAW or .CSV", type=["raw", "RAW", "csv", "CSV"])
        
        if uploaded_file is not None:
            # Robust decoding for legacy files
            try:
                 content = uploaded_file.getvalue().decode("utf-8")
            except UnicodeDecodeError:
                 content = uploaded_file.getvalue().decode("latin-1")
            try:
                if uploaded_file.name.endswith(".csv"):
                    data = MigetIO.parse_csv(content)
                else:
                    data = MigetIO.parse_legacy_file(content)
                
                st.session_state['miget_data'] = data
                st.success(f"Loaded {len(data)} runs.")
                
                # Species Selection
                st.sidebar.subheader("Configuration")
                
                # Auto-detect default
                default_idx = 0
                if data and "dog" in data[0].name.lower():
                    default_idx = 1
                elif data and "horse" in data[0].name.lower():
                    default_idx = 2
                    
                species = st.sidebar.selectbox("Subject Species", ["Human", "Dog", "Horse"], index=default_idx)
                
                # Update Species for all runs
                for run in data:
                    run.species = species.lower()
                
                # Display Summary (Collapsible)
                with st.expander(f"Summary of Runs ({len(data)})", expanded=False):
                    summary_data = []
                    for run in data:
                        summary_data.append({
                            "Name": run.name,
                            "VE": run.ve_measured,
                            "QT": run.qt_measured
                        })
                    st.dataframe(pd.DataFrame(summary_data))
                
                # Configuration Controls
                st.sidebar.subheader("Model Configuration")
                weight_mode = st.sidebar.selectbox("Weighting Mode", ["Retention", "Excretion"], index=0)
                bohr_int = st.sidebar.selectbox("Bohr Integration", ["Off", "On"], index=0) # Placeholder for now, logic likely requires VQBOHR core Update
                
                with st.expander("Inspect Raw Data (Peaks & PCs)"):
                    for run in data:
                        st.write(f"**{run.name}**")
                        # Create a dataframe for the gas data
                        df_run = pd.DataFrame({
                            "Gas": run.gas_names,
                            "Partition Coeff": run.partition_coeffs,
                            "Arterial (PA)": run.pa_raw,
                            "Expired (PE)": run.pe_raw,
                            "Venous (PV)": run.pv_raw
                        })
                        st.dataframe(df_run)
                
                # Recalculate metrics for display
                # Note: We do this before the model button so user can see verification
                for run in data:
                    run.calculate_metrics()
                    
    
                with st.expander("Retentions and Excretions"):
                    for run in data:
                        st.write(f"**{run.name}** - Corrections Applied")
                        st.write(f"VGA: {run.vga}, VBA: {run.vba}")
                        df_metrics = pd.DataFrame({
                            "Gas": run.gas_names,
                            "Solubility": run.solubilities,
                            "Retention (R)": run.retention,
                            "Excretion (E)": run.excretion,
                            "Ret Weights": run.retention_weights
                        })
                        st.dataframe(df_metrics.style.format({
                            "Solubility": "{:.5f}",
                            "Retention (R)": "{:.5f}",
                            "Excretion (E)": "{:.5f}",
                            "Ret Weights": "{:.5f}"
                        }))
    
                if not data:
                    st.error("No valid runs found in file.")
                    return

                # --- 1. Batch Process All Runs ---
                results_cache = []
                summary_rows = []
                dfs_dist = []
                
                z_factor = 40.0
                
                # Progress bar if many runs
                if len(data) > 2:
                    progress_bar = st.progress(0)
                
                for i, run in enumerate(data):
                    run.calculate_metrics()
                    # Use run.z if available, else default
                    z_use = run.z if hasattr(run, 'z') else 40.0
                    model = VQModel(run, z_factor=z_use)
                    dist = model.solve(weight_mode=weight_mode.lower())
                    
                    results_cache.append({
                        "run": run,
                        "model": model,
                        "dist": dist
                    })
                    
                    # Summary Row
                    summary_rows.append({
                        "Run": run.name,
                        "RSS": dist.rss,
                        "Shunt (%)": dist.shunt * 100,
                        "Deadspace (%)": dist.deadspace * 100,
                        "Mean Q (Log)": dist.mean_q,
                        "SD Q (Log)": dist.sd_q,
                        "Mean V (Log)": dist.mean_v,
                        "SD V (Log)": dist.sd_v,
                        "QT": run.qt_measured,
                        "VE": run.ve_measured,
                        "Z": z_use
                    })
                    
                    # Distribution DataFrame for Export
                    df_d = pd.DataFrame({
                        "Run": run.name,
                        "VA/Q": dist.vaq_ratios,
                        "Blood Flow (Q)": dist.blood_flow,
                        "Ventilation (V)": dist.ventilation
                    })
                    dfs_dist.append(df_d)
                    
                    if len(data) > 2:
                        progress_bar.progress((i + 1) / len(data))
                
                if len(data) > 2:
                    progress_bar.empty()

                # --- 2. Navigation & Selection ---
                if len(data) > 1:
                    run_names = [r["run"].name for r in results_cache]
                    selected_idx = st.sidebar.selectbox("Select Measurement", range(len(data)), format_func=lambda x: run_names[x])
                else:
                    selected_idx = 0
                
                current_res = results_cache[selected_idx]
                run = current_res["run"]
                model = current_res["model"]
                dist = current_res["dist"]
                
                # --- 3. Top Section: Header ---
                st.subheader(f"Analysis: {run.name}")
                
                # --- 4. Main UI Layout (Reorganized) ---
                
                # Row 1: Large V/Q Plot
                # Row 1: Large V/Q Plot
                st.write("#### V/Q Distribution")
                fig, ax = plt.subplots(figsize=(10, 4)) # Wider
                
                # Plot absolute flows (L/min)
                x = dist.vaq_ratios
                # Exclude Shunt (idx 0) and Deadspace (idx 49) for the main curve
                mask = (x > 0.005) & (x < 100.0) # Filter suitable range for log plot
                
                # Absolute values
                q_abs = dist.blood_flow * run.qt_measured
                v_abs = dist.ventilation * dist.va_total
                
                x_plot = x[mask]
                q_plot = q_abs[mask]
                v_plot = v_abs[mask]
                
                # Smooth Spline Interpolation
                if len(x_plot) > 3:
                    # Interpolate in Log X domain
                    x_log = np.log10(x_plot)
                    x_new = np.linspace(x_log.min(), x_log.max(), 300)
                    
                    spl_q = make_interp_spline(x_log, q_plot, k=3)
                    spl_v = make_interp_spline(x_log, v_plot, k=3)
                    
                    q_smooth = spl_q(x_new)
                    v_smooth = spl_v(x_new)
                    
                    # Clip negatives from spline oscillation
                    q_smooth[q_smooth < 0] = 0
                    v_smooth[v_smooth < 0] = 0
                    
                    x_smooth = 10**x_new
                    
                    ax.plot(x_smooth, q_smooth, 'tab:red', label='Perfusion (L/min)')
                    ax.plot(x_smooth, v_smooth, 'tab:blue', label='Ventilation (L/min)')
                else:
                    # Fallback if too few points
                    ax.plot(x_plot, q_plot, 'tab:red', marker='o', markersize=10, label='Perfusion (L/min)')
                    ax.plot(x_plot, v_plot, 'tab:blue', marker='o', markersize=10, label='Ventilation (L/min)')

                # Add Points for visual reference (optional, or just use smooth line)
                ax.plot(x_plot, q_plot, '.', color='tab:red', markersize=10, alpha=1)
                ax.plot(x_plot, v_plot, '.', color='tab:blue', markersize=10, alpha=1)
                
                # Plot Shunt as a separate dot on the left axis
                shunt_val = dist.shunt * run.qt_measured
                if shunt_val > 0.01:
                    ax.plot(0.002, shunt_val, 'ko', markersize=8, label=f'Shunt ({shunt_val:.2f} L/min)')
                
                ax.set_xscale('log')
                ax.set_xlim(0.001, 100.0)
                ax.set_xlabel(r'$V_A/Q_T$ Ratio')
                ax.set_ylabel('Compartimental flow (L/min)')
                ax.legend()
                ax.grid(True, which="both", ls="-", color="#f0f0f0", alpha=0.9)
                st.pyplot(fig)
                
                # Row 2: Two Columns (R/E Plot and Parameters Table)
                col_re, col_tbl = st.columns([1, 1])
                
                with col_re:
                    st.write("#### Retention & Excretion Fit")
                    fig2, ax2 = plt.subplots(figsize=(6, 5)) # Slightly taller/narrower for column
                    
                    sol = run.solubilities
                    meas_r = run.retention
                    meas_e = run.excretion
                    
                    sweep_sol, sweep_r, sweep_e, sweep_homo_r, sweep_homo_e = model.get_continuous_curves(dist)
                    
                    ax2.plot(sol, meas_r, 'o', color='tab:orange', markersize=10, label='Measured R')
                    ax2.plot(sweep_sol, sweep_r, '-', color='tab:orange', alpha=0.8, label='Predicted R')
                    ax2.plot(sweep_sol, sweep_homo_r, '--', color='tab:orange', alpha=0.5, label='Homogeneous R')

                    ax2.plot(sol, meas_e, 'o', color='tab:purple', markersize=10, label='Measured E')
                    ax2.plot(sweep_sol, sweep_e, '-', color='tab:purple', alpha=0.8, label='Predicted E')
                    ax2.plot(sweep_sol, sweep_homo_e, '--', color='tab:purple', alpha=0.5, label='Homogeneous E')
                    
                    ax2.set_xscale('log')
                    ax2.set_xlabel('Solubility')
                    ax2.set_ylabel('R / E')
                    ax2.legend()
                    ax2.grid(True, which="both", ls="-", color="#f0f0f0", alpha=0.9)
                    st.pyplot(fig2)
                    
                with col_tbl:
                    st.write("#### Parameters")
                    row = summary_rows[selected_idx]
                    disp_keys = ["RSS", "Shunt (%)", "Deadspace (%)", "Mean Q (Log)", "SD Q (Log)", "Mean V (Log)", "SD V (Log)"]
                    
                    df_run_params = pd.DataFrame({
                        "Parameter": disp_keys,
                        "Value": [row[k] for k in disp_keys]
                    })
                    st.table(df_run_params.style.format({"Value": "{:.4f}"}))
                    
                    st.markdown("---")
                    
                    # Create Consolidated Excel
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        # Sheet 1: Summary Parameters
                        df_summary = pd.DataFrame(summary_rows)
                        df_summary.to_excel(writer, sheet_name='Summary Parameters', index=False)
                        
                        # Sheet 2: All Distributions
                        df_all_dist = pd.concat(dfs_dist, ignore_index=True)
                        df_all_dist.to_excel(writer, sheet_name='All Distributions', index=False)
                        
                    st.download_button(
                        label="📥 Download All Results (Excel)",
                        data=buffer.getvalue(),
                        file_name=f"miget_results_all_{len(data)}_runs.xlsx",
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        use_container_width=True
                    )
            
            except Exception as e:
                st.error(f"Error parsing file: {e}")
                raise e
    else:
        st.write("Please upload a file to begin.")

def render_direct_problem():
    st.header("Forward Problem")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution Parameters")
        shunt = st.number_input("Shunt (%)", 0.0, 100.0, 5.0, step=0.1)
        deadspace = st.number_input("Deadspace (%)", 0.0, 100.0, 30.0, step=0.1)
        
        qt_sim = st.number_input("Cardiac Output (QT L/min)", 0.1, 20.0, 5.0, step=0.5)
        ve_sim = st.number_input("Minute Ventilation (VE L/min)", 0.1, 50.0, 6.0, step=0.5)
        
        st.markdown("---")
        st.markdown("### Perfusion (Q) Modes")
        use_secondary = st.checkbox("Enable Secondary Mode")
        
        col_q1, col_q2 = st.columns(2)
        
        with col_q1:
             st.caption("Primary Mode")
             q1_mean = st.number_input("Q1 Mean", 0.01, 100.0, 1.0, step=0.1, help="Center (Linear Scale)")
             q1_sd = st.number_input("Q1 LogSD", 0.0, 3.0, 0.4, step=0.05, help="Dispersion (Log Scale)")
             
        q2_mean = None
        q2_sd = None
        q2_flow_pct = 0.0
        
        with col_q2:
            if use_secondary:
                st.caption("Secondary Mode")
                q2_mean = st.number_input("Q2 Mean", 0.01, 100.0, 10.0, step=0.1)
                q2_sd = st.number_input("Q2 LogSD", 0.0, 3.0, 0.4, step=0.05)
                q2_flow_pct = st.number_input("Q2 Flow %", 0.0, 100.0, 5.0, step=0.1)
            else:
                st.caption("Secondary Mode disabled")

    with col2:
        st.subheader("Gas Solubilities")
        use_custom = st.checkbox("Use Custom Solubilities", value=False)
        
        # Standard Human 37C
        # SF6, Ethane, Cyclo, Enflurane, Ether, Acetone
        std_names = ["SF6", "Ethane", "Cyclopropane", "Enflurane", "Ether", "Acetone"]
        std_sols = [0.005, 0.09, 0.42, 1.8, 12.0, 333.0] # Approximate
        
        if use_custom:
            custom_sols = []
            for i, name in enumerate(std_names):
                val = st.number_input(f"Solubility {name}", value=std_sols[i], format="%.4f")
                custom_sols.append(val)
            solubilities = np.array(custom_sols)
        else:
            st.info("Using Standard Human Solubilities (37°C)")
            st.table(pd.DataFrame({"Gas": std_names, "Solubility": std_sols}))
            solubilities = np.array(std_sols)

    # Calculate
    from miget.core import VQForwardModel, VQModel
    
    # Always calculate if inputs change
    fw_model = VQForwardModel()
    
    # Check consistency
    if shunt + deadspace > 100:
        st.error("Shunt + Deadspace cannot exceed 100%")
        return

    q_dist, v_dist, ds_frac = fw_model.generate_distribution(
        shunt_pct=shunt, deadspace_pct=deadspace,
        q1_mean=q1_mean, q1_sd=q1_sd,
        q2_mean=q2_mean, q2_sd=q2_sd, q2_flow_pct=q2_flow_pct
    )
    
    meas_r, meas_e = fw_model.calculate_re(q_dist, v_dist, ds_frac, solubilities)
    
    # Calculate absolute flows
    q_abs_total = q_dist * qt_sim
    va_total_sim = ve_sim * (1.0 - deadspace / 100.0)
    v_abs_total = v_dist * va_total_sim
    
    # --- Visualization ---
    st.subheader("1. Constructed V/Q Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot absolute flows (L/min)
    x = fw_model.vq_ratios
    # Exclude Shunt (idx 0) and Deadspace (idx 49) for the main curve
    mask = (x >= 0.005) & (x <= 100.0)
    
    x_plot = x[mask]
    q_plot = q_abs_total[mask]
    v_plot = v_abs_total[mask]
    
    # Smooth Spline Interpolation
    if len(x_plot) > 3:
        # Interpolate in Log X domain
        x_log = np.log10(x_plot)
        x_new = np.linspace(x_log.min(), x_log.max(), 300)
        
        spl_q = make_interp_spline(x_log, q_plot, k=3)
        spl_v = make_interp_spline(x_log, v_plot, k=3)
        
        q_smooth = spl_q(x_new)
        v_smooth = spl_v(x_new)
        
        ax.plot(10**x_new, q_smooth, 'tab:red', label='Perfusion (L/min)')
        ax.plot(10**x_new, v_smooth, 'tab:blue', label='Ventilation (L/min)')
    else:
        # Fallback if too few points
        ax.plot(x_plot, q_plot, 'tab:red', marker='o', markersize=10, label='Perfusion (L/min)')
        ax.plot(x_plot, v_plot, 'tab:blue', marker='o', markersize=10, label='Ventilation (L/min)')

    # Add Points
    ax.plot(x_plot, q_plot, '.', color='tab:red', markersize=10, alpha=1)
    ax.plot(x_plot, v_plot, '.', color='tab:blue', markersize=10, alpha=1)
    
    # Plot Shunt
    shunt_flow = (shunt / 100.0) * qt_sim
    if shunt_flow > 0.001:
        ax.plot(0.002, shunt_flow, 'ko', markersize=8, label=f'Shunt ({shunt_flow:.2f} L/min)')
    
    ax.set_xscale('log')
    ax.set_xlim(0.001, 100.0)
    ax.set_xlabel(r'$V_A/Q_T$ Ratio')
    ax.set_ylabel('Compartmental flow (L/min)')
    ax.legend()
    ax.grid(True, which="both", ls="-", color="#f0f0f0", alpha=0.9)
    st.pyplot(fig)
    
    # Results Row: Plot left, Summary Table right (Matching Inverse Problem)
    # Define df_out early for download button
    df_out = pd.DataFrame({
        "Gas": std_names,
        "Solubility": solubilities,
        "Retention (R)": meas_r,
        "Excretion (E)": meas_e
    })

    col_re_plot, col_sum = st.columns([1, 1])
    
    with col_re_plot:
        st.write("#### Retention & Excretion Fit")
        fig2, ax2 = plt.subplots(figsize=(6, 5)) # Matching Inverse Problem size
        
        # Plot Continuous curves?
        # We only have discrete points for the gases.
        # To show continuous curve, we'd need to sweep solubility.
        
        sweep_sols = np.logspace(-3, 3, 50)
        sweep_r, sweep_e = fw_model.calculate_re(q_dist, v_dist, ds_frac, sweep_sols)
        
        ax2.plot(sweep_sols, sweep_r, '-', color='tab:orange', alpha=0.8, label='Simulated R')
        ax2.plot(sweep_sols, sweep_e, '-', color='tab:purple', alpha=0.8, label='Simulated E')
        
        # Plot Points
        ax2.plot(solubilities, meas_r, 'o', color='tab:orange', markersize=10, label='Gas R')
        ax2.plot(solubilities, meas_e, 'o', color='tab:purple', markersize=10, label='Gas E')
        
        ax2.set_xscale('log')
        ax2.set_xlabel('Solubility (ml/dl/torr)')
        ax2.set_ylabel('R / E')
        ax2.legend()
        ax2.grid(True, which="both", ls="-", color="#f0f0f0", alpha=0.9)
        st.pyplot(fig2)
        
    with col_sum:
        st.write("#### Parameters")
        df_sum_dict = {
            "Shunt (%)": shunt,
            "Deadspace (%)": deadspace,
            "QT (L/min)": qt_sim,
            "VE (L/min)": ve_sim,
            "VA (L/min)": va_total_sim
        }
        
        df_disp = pd.DataFrame({
            "Parameter": list(df_sum_dict.keys()),
            "Value": list(df_sum_dict.values())
        })
        st.table(df_disp.style.format({"Value": "{:.1f}"}))
        
        st.markdown("---")
        # Download Button (Moved here to match Inverse Problem)
        st.download_button(
             label="Download Simulated Data (CSV)",
             data=df_out.to_csv(index=False).encode('utf-8'),
             file_name="simulated_miget_data.csv",
             mime='text/csv'
        )
    
    # Data Table
    
    # Data Table
    # Data Table removed to match Inverse Problem layout
    
    # Download logic moved to summary column

def main():
    if 'mode' not in st.session_state:
        st.session_state['mode'] = None

    if st.session_state['mode'] is None:
        st.title("MIGET Software 2025")
        st.write("Select the analysis mode:")
        
        # Custom CSS for Big Buttons
        st.markdown("""
        <style>
        div.stButton > button {
            width: 100%;
            height: 120px; /* Increased height */
            font-size: 32px !important; /* Increased font */
            font-weight: 700 !important;
            border-radius: 16px !important;
            border: none !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.2s ease-in-out;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        /* Specific coloring for the two buttons */
        /* Note: With new columns layout, we need to target specific children */
        div[data-testid="stColumn"]:nth-of-type(2) div.stButton > button {
             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
             color: white !important;
             border: none !important;
        }
        div[data-testid="stColumn"]:nth-of-type(3) div.stButton > button {
             background: linear-gradient(135deg, #ff9966 0%, #ff5e62 100%) !important;
             color: white !important;
             border: none !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Centered layout with reduced gap effect by using middle columns
        _, col1, col2, _ = st.columns([0.5, 3, 3, 0.5], gap="small")
        
        with col1:
             # Regular button but styled by the above CSS targeting this column
            if st.button("Forward Problem ➡️", key="dir_btn"):
                st.session_state['mode'] = 'direct'
                st.rerun()
                
        with col2:
            # Regular button but styled by the above CSS targeting this column
            if st.button("⬅️ Reverse Problem", key="inv_btn"):
                st.session_state['mode'] = 'inverse'
                st.rerun()
                
    elif st.session_state['mode'] == 'inverse':
        if st.sidebar.button("← Back to Home"):
            st.session_state['mode'] = None
            st.rerun()
        render_inverse_problem()
        
    elif st.session_state['mode'] == 'direct':
        if st.sidebar.button("← Back to Home"):
            st.session_state['mode'] = None
            st.rerun()
        render_direct_problem()

if __name__ == "__main__":
    main()
