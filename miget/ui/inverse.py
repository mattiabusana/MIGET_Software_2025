import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from scipy.interpolate import make_interp_spline
from miget.io import MigetIO
from miget.core import VQModel
from miget.bohr import BohrIntegrator
from miget.physiology import BloodGasParams

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
    
    # Render Physiology sidebar (Now that we have current_run)
    with st.sidebar:
        st.divider()
        with st.expander("Physiology & Diffusion", expanded=True):
            st.caption("Parameters for Blood Gas Prediction")
            
            # Defaults from Data (if available)
            def_vo2 = current_run.vo2_measured if current_run.vo2_measured > 0 else 250.0
            def_vco2 = current_run.vco2_measured if current_run.vco2_measured > 0 else 200.0
            def_hb = current_run.hb if current_run.hb > 0 else 15.0
            def_temp = current_run.temp_body if current_run.temp_body > 0 else 37.0
            
            vo2_sim = st.number_input("VO2 (ml/min)", value=float(def_vo2), step=10.0, disabled=True)
            vco2_sim = st.number_input("VCO2 (ml/min)", value=float(def_vco2), step=10.0, disabled=True)
            hb_sim = st.number_input("Hemoglobin (g/dL)", value=float(def_hb), step=0.5, disabled=True)
            temp_sim = st.number_input("Temperature (C)", value=float(def_temp), step=0.1, disabled=True)
            
            c1, c2 = st.columns(2)
            with c1:
                # FiO2 might not be in file? Leave enabled for now unless sure?
                # The user said "parameters" generally. 
                # But without FiO2 in file, user is stuck.
                # Assuming FiO2 is experimental setting allowed to change?
                # Or assume 21%?
                fio2_sim = st.number_input("FiO2 (%)", value=21.0, step=1.0) 
            with c2:
                # Default to measured PB if available, else 760
                default_pb = current_run.pb_sea if current_run.pb_sea > 0 else 760.0
                pb_sim = st.number_input("PB (Torr)", value=float(default_pb), step=10.0, disabled=True)
            
            dlo2_sim = st.number_input("DLO2 (ml/min/Torr)", value=25.0, step=1.0, help="Diffusion Capacity")
            
            # Derived PIO2
            pio2_sim = (fio2_sim / 100.0) * (pb_sim - 47.0)
            st.caption(f"Calculated PIO2: {pio2_sim:.1f} mmHg")

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
        
        # --- PREDICTED BLOOD GASES (BOHR) ---
        st.markdown("#### Predicted Blood Gases (Bohr Integration)")
        
        # 1. Run Bohr Integration
        # We need to construct a BohrIntegrator with the recovered distribution.
        # Ensure we have the necessary params from Sidebar
        
        # We need to compute BloodGasParams object
        bg_params = BloodGasParams(
            hb=hb_sim,
            temp=temp_sim,
            # pio2 removed - it is an environmental param pass to Integrator
            # pico2 removed
            # Constants
            sol_o2=0.003, # Default
            # sol_co2=0.06, # Default
            dp50=26.8,     # Default (Note param name is dp50 not p50)
            # acid_base_excess=0.0 # Not in class
        )
        
        # Create Integrator
        # Note: BohrIntegrator expects vq_dist to be VQDistribution object.
        # dist IS a VQDistribution object.
        
        # Update Data object with VO2/VCO2 for the integrator
        # (It uses them for Mixed Venous iteration)
        # We don't want to mutate the persistent data object if we can avoid it, 
        # but BohrIntegrator might read from it.
        # Actually BohrIntegrator takes `data` in init.
        
        # Create a proxy data object or update current
        # Let's clone relevant fields
        from copy import copy
        bohr_data = copy(current_run)
        bohr_data.vo2_measured = vo2_sim
        bohr_data.vco2_measured = vco2_sim
        bohr_data.qt_measured = qt_sim if 'qt_sim' in locals() else current_run.qt_measured # Use measured or override?
        # Note: current_run.qt_measured is measured.
        # For prediction we should use the same QT as the inert gas solution unless overriden?
        # The V/Q fit used qt_measured. So we should use it.
        
        integrator = BohrIntegrator(dist, bohr_data, bg_params, dlo2=dlo2_sim, pio2=pio2_sim)
        
        try:
            with st.spinner("Calculating Blood Gases..."):
                # 1. Find Mixed Venous
                pvo2_pred, pvco2_pred = integrator.find_mixed_venous(vo2_sim, vco2_sim)
                
                # 2. Calculate Arterial Content (Accounting for DLO2)
                # We use the JIT solver pipeline
                from miget.physiology import blood_gas_calc, inverse_blood_gas_calc
                
                # Need Cvo2 for solver
                cvo2_pred, cvco2_pred = blood_gas_calc(pvo2_pred, pvco2_pred, bg_params)
                
                # Solve Forward
                pa_o2_all, pa_co2_all, pc_o2_all, pc_co2_all = integrator.solve_all_compartments_gas_lines(
                    dist.vaq_ratios, pvo2_pred, pvco2_pred, cvo2_pred, cvco2_pred
                )
                
                # Calculate Mixed Arterial Content
                # Sum (Flow_frac * Content)
                # dist.blood_flow is normalized?
                # VQDistribution.blood_flow sums to (1-shunt) usually in core?
                # No, look at `solve`: `q_dist_norm` sums to 1.
                # `shunt` is q_dist_norm[0].
                # So `dist.blood_flow` includes shunt at index 0.
                # Does `pc_o2_all` include shunt?
                # Yes, `solve_all` returns sizes N.
                # Shunt index 0 has `Pc = Pv` (after my fix).
                
                cc_o2_all, cc_co2_all = blood_gas_calc(pc_o2_all, pc_co2_all, bg_params)
                
                art_o2_c = np.sum(dist.blood_flow * cc_o2_all)
                art_co2_c = np.sum(dist.blood_flow * cc_co2_all)
                
                # Inverse to get PaO2
                pao2_pred, paco2_pred = inverse_blood_gas_calc(art_o2_c, art_co2_c, bg_params)
                
                # AaDO2
                # Calculate Ideal PAO2 (Alveolar Gas Equation)
                # PAO2 = PIO2 - PCO2/R + ...
                # Use simple approx or full?
                # R = VCO2/VO2
                rq = vco2_sim / vo2_sim if vo2_sim > 0 else 0.8
                
                # Use True Mean Alveolar PO2 for robust AaDO2
                # Mean PAO2 = Sum(Vent * PA_i) / Total Vent
                # Check normalized ventilation
                
                if np.sum(dist.ventilation) > 0:
                     mean_pa_o2 = np.sum(dist.ventilation * pa_o2_all) / np.sum(dist.ventilation)
                else:
                     mean_pa_o2 = pio2_sim - paco2_pred/rq # Fallback
                     
                aado2 = mean_pa_o2 - pao2_pred
                
                # Display Results
                bg_cols = st.columns(3)
                bg_cols[0].metric("Predicted PaO2", f"{pao2_pred:.1f} mmHg")
                bg_cols[1].metric("Predicted PaCO2", f"{paco2_pred:.1f} mmHg")
                bg_cols[2].metric("AaDO2", f"{aado2:.1f} mmHg")
                
                st.caption(f"Mixed Venous: PvO2={pvo2_pred:.1f}, PvCO2={pvco2_pred:.1f}")
                
        except Exception as e:
            st.error(f"Blood Gas Calculation Failed: {e}")

        
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

