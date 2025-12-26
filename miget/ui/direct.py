import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from miget.core import VQForwardModel, VQModel

def render_direct_problem():
    st.header("Forward Problem")
    
    # Layout: Side-by-side (Left Control Panel, Right Results)
    col_ctrl, col_res = st.columns([1, 2], gap="large")

    try:
        # --- LEFT COLUMN: CONTROLS ---
        with col_ctrl:
            with st.container():
                st.subheader("⚙️ Parameters")
                
                # --- Solubilities Section (Moved to Top) ---
                with st.expander("Gas Solubilities Configuration", expanded=False):
                     use_custom = st.checkbox("Use Custom Solubilities", value=False)
                     
                     # Standard Human 37C
                     std_names = ["SF6", "Ethane", "Cyclopropane", "Enflurane", "Ether", "Acetone"]
                     std_sols = [0.005, 0.09, 0.42, 1.8, 12.0, 333.0] 
                     
                     if use_custom:
                         custom_sols = []
                         for i, name in enumerate(std_names):
                             val = st.number_input(f"Solubility {name}", value=std_sols[i], format="%.4f")
                             custom_sols.append(val)
                         solubilities = np.array(custom_sols)
                     else:
                         st.info("Standard Human Solubilities (37°C)")
                         solubilities = np.array(std_sols)
                
                st.divider()

                st.markdown("##### Global Settings")
                shunt = st.slider("Shunt (%)", 0.0, 100.0, 1.0, step=0.1)
                deadspace = st.slider("Deadspace (%)", 0.0, 100.0, 25.0, step=0.1)
                qt_sim = st.slider("Cardiac Output (L/min)", 0.1, 20.0, 5.0, step=0.1)
                ve_sim = st.slider("Minute Ventilation (L/min)", 0.1, 50.0, 6.0, step=0.5)
                
                st.divider()
                
                # Create Log-Spaced Options for Q Mean Sliders
                # 0.01 to 100.0
                log_options = np.logspace(np.log10(0.01), np.log10(100.0), 101)
                
                st.markdown("##### Perfusion (Q) Modes")
                # Primary Mode
                st.caption("**Primary Mode**")
                
                # Logarithmic Slider using select_slider
                q1_mean = st.select_slider(
                    "Q1 Mean (Log Scale)", 
                    options=log_options,
                    value=log_options[50], # ~1.0
                    format_func=lambda x: f"{x:.2f}"
                )
                q1_sd = st.slider("Q1 LogSD (Dispersion)", 0.0, 3.0, 0.4, step=0.05)
                
                # Secondary Mode
                st.markdown("---")
                use_secondary = st.checkbox("Enable Secondary Mode")
                
                q2_mean = None
                q2_sd = None
                q2_flow_pct = 0.0
                
                if use_secondary:
                    st.caption("**Secondary Mode**")
                    q2_mean = st.select_slider(
                        "Q2 Mean (Log Scale)", 
                        options=log_options,
                        value=log_options[75], # ~10.0
                        format_func=lambda x: f"{x:.2f}"
                    )
                    q2_sd = st.slider("Q2 LogSD", 0.0, 3.0, 0.4, step=0.05)
                    q2_flow_pct = st.slider("Q2 Flow %", 0.0, 100.0, 5.0, step=0.1)


                # --- Blood Gas & Diffusion Section ---
                with st.expander("Blood Gas & Diffusion", expanded=False):
                    st.caption("Physiological Parameters")
                    vo2_sim = st.number_input("VO2 (ml/min)", value=250.0, step=10.0)
                    vco2_sim = st.number_input("VCO2 (ml/min)", value=200.0, step=10.0)
                    hb_sim = st.number_input("Hemoglobin (g/dL)", value=15.0, step=0.5)
                    temp_sim = st.number_input("Temperature (C)", value=37.0, step=0.1)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        fio2_sim = st.number_input("FiO2 (%)", value=21.0, step=1.0)
                    with c2:
                        pb_sim = st.number_input("PB (Torr)", value=760.0, step=10.0)
                        
                    # Calculate PIO2 (Assuming PH2O = 47 at 37C)
                    pio2_sim = (fio2_sim / 100.0) * (pb_sim - 47.0)
                    st.caption(f"Calculated PIO2: {pio2_sim:.1f} Torr")
                    
                    st.divider()
                    enable_diffusion = st.checkbox("Enable Diffusion Limitation (Bohr)", value=False)
                    dlo2_sim = 9000.0
                    if enable_diffusion:
                        dlo2_sim = st.number_input("DLO2 (ml/min/Torr)", value=25.0, step=1.0, help="Lung Diffusion Capacity for Oxygen")


        # Calculation Logic
        fw_model = VQForwardModel()
        
        if shunt + deadspace > 100:
            st.error("Error: Shunt + Deadspace cannot exceed 100%")
            return

        q_dist, v_dist, ds_frac = fw_model.generate_distribution(
            shunt_pct=shunt, deadspace_pct=deadspace,
            q1_mean=q1_mean, q1_sd=q1_sd,
            q2_mean=q2_mean, q2_sd=q2_sd, q2_flow_pct=q2_flow_pct
        )
        
        meas_r, meas_e = fw_model.calculate_re(q_dist, v_dist, ds_frac, solubilities)
        
        q_abs_total = q_dist * qt_sim
        va_total_sim = ve_sim * (1.0 - deadspace / 100.0)
        
        # v_dist sums to (1 - deadspace_pct/100).
        # So multipying by VE gives correct Comportmental Ventilation summing to VA.
        v_abs_total = v_dist * ve_sim
        
        # --- Bohr / Blood Gas Calculation ---
        bohr_res = None
        if True: # Always run blood gas if inputs exist
            from miget.core import VQDistribution, InertGasData
            from miget.physiology import BloodGasParams, blood_gas_calc, inverse_blood_gas_calc
            from miget.bohr import BohrIntegrator
            
            # Calculate Effective VA/Q Ratios based on current VE/QT/Deadspace
            # Because Deadspace changes VA, the actual VA/Q of a compartment receiving X% of V and Y% of Q changes.
            if qt_sim > 1e-4:
                 # vaq_i = (v_dist[i] * va_total_sim) / (q_dist[i] * qt_sim)
                 # Handle cases where q_dist is 0.
                 # If q_dist[i] is 0, VA/Q is infinite (Deadspace).
                 # If v_dist[i] is 0, VA/Q is 0 (Shunt).
                 
                 # Initialize with bin centers as fallback/scale
                 vaq_effective = np.zeros_like(fw_model.vq_ratios)
                 
                 for k in range(len(vaq_effective)):
                     q_k = q_dist[k]
                     v_k = v_dist[k]
                     
                     if q_k > 1e-9:
                         # v_k is fraction of VE * (1-DS). 
                         # So to get Absolute VA, we multiply v_k * VE? 
                         # Wait. v_dist sums to (1-DS).
                         # Absolute Compartment Ventilation = v_k * VE.
                         # (Sum of Abs Comp Vent = (1-DS) * VE = VA Total). Correct.
                         
                         vaq_effective[k] = (v_k * ve_sim) / (q_k * qt_sim)
                     else:
                         vaq_effective[k] = fw_model.vq_ratios[k] # Fallback or Infinite
            else:
                 vaq_effective = fw_model.vq_ratios
            
            # Construct VQ object
            vq_obj = VQDistribution(
                compartments=50,
                blood_flow=q_dist,
                ventilation=v_dist,
                vaq_ratios=vaq_effective
            )
            
            # Data object (mocking what Bohr needs)
            # Bohr needs qt_measured.
            data_obj = InertGasData(qt_measured=qt_sim, ve_measured=ve_sim)
            
            # Params
            bg_params = BloodGasParams(hb=hb_sim, temp=temp_sim)
            # Note: We need PIO2 passed? BohrIntegrator in bohr.py currently hardcodes 150.0 inside solve_compartment_gas_lines
            # START_BUG_FIX: Pass PIO2 to BohrIntegrator or Params? 
            # Plan: Modify bohr.py to accept PIO2 in data or params, OR just hack it here by monkeypatching?
            # Better: bohr.py assumed standard air. I should fix bohr.py to use data.pio2 ideally. 
            # But let's check bohr.py. Line 117: `pio2 = 150.0 # Placeholder`
            # I will need to update bohr.py to allow dynamic PIO2. 
            # For now, let's assume 150.0 or if the user changed it, wait... 
            # I will do a quick patch injection or update bohr.py later. 
            # Actually, `InertGasData` doesn't strictly field pio2.
            # I'll update bohr.py in next step. Direct.py continues...
            
            integrator = BohrIntegrator(vq_obj, data_obj, bg_params, dlo2=dlo2_sim)
            # Inject PIO2 (Hack for now until bohr.py update)
            integrator.pio2 = pio2_sim 
            
            # Find Pv
            pvo2_calc, pvco2_calc = integrator.find_mixed_venous(vo2_sim, vco2_sim)
            
            # Calculate Arterial P
            # We need to sum up arterial content -> P
            qt = qt_sim
            total_o2_content = 0.0
            total_co2_content = 0.0
            
            # Shunt contribution
            # q_dist includes shunt at index 0 (if shunt > 0).
            # We iterate through all compartments, so we don't need to add shunt manually.
            # We just need to handle vaq=0 correctly in the loop.
            
            cvo2_val, cvco2_val = blood_gas_calc(pvo2_calc, pvco2_calc, bg_params)
            
            # ----------------------------------------------------------------
            # VECTORIZED SOLUTION (High Performance)
            # ----------------------------------------------------------------
            
            # 1. Solve all compartments at once for P
            # Returns arrays for PA and Pc' (End Capillary)
            pa_o2_all, pa_co2_all, pc_o2_all, pc_co2_all = integrator.solve_all_compartments_gas_lines(
                vaq_effective, pvo2_calc, pvco2_calc, cvo2_val, cvco2_val
            )
            
            # 2. Convert Pc' (End Capillary Pressure) to Content
            cc_o2_all, cc_co2_all = blood_gas_calc(pc_o2_all, pc_co2_all, bg_params)
            
            # 3. Calculate Weighting Factors (Flow and Ventilation)
            # q_dist is fractional flow distribution (sums to 1.0)
            # absolute flow = q_dist * qt
            
            flow_abs = q_dist * qt
            vent_abs = flow_abs * vaq_effective # V = Q * VA/Q
            
            # 4. Calculate Arterial Content (Flow Weighted Sum)
            total_o2_content = np.sum(flow_abs * cc_o2_all)
            total_co2_content = np.sum(flow_abs * cc_co2_all)
            
            art_o2_content = total_o2_content / qt
            art_co2_content = total_co2_content / qt

            from miget.physiology import inverse_blood_gas_calc
            pa_art_o2, pa_art_co2 = inverse_blood_gas_calc(art_o2_content, art_co2_content, bg_params)

            # 5. Calculate Mean Alveolar PO2 (Ventilation Weighted Sum)
            # Mask out shunt/undef ventilation to be safe (though V=0 there)
            mean_palv_o2 = pio2_sim # Default if no ventilation
            
            total_vent = np.sum(vent_abs)
            if total_vent > 1e-6:
                mean_palv_o2 = np.sum(vent_abs * pa_o2_all) / total_vent
            else:
                mean_palv_o2 = pio2_sim
            
            bohr_res = {
                "PaO2": pa_art_o2,
                "PaCO2": pa_art_co2,
                "PvO2": pvo2_calc,
                "PvCO2": pvco2_calc,
                "AaDO2": mean_palv_o2 - pa_art_o2,
                "DLO2": dlo2_sim
            }

        
        # --- RIGHT COLUMN: RESULTS ---
        with col_res:
            st.subheader("📊 Simulation Results")
            
            # Metrics Row
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("QT", f"{qt_sim:.1f}")
            m2.metric("VE", f"{ve_sim:.1f}")
            m3.metric("Shunt", f"{shunt:.1f}%")
            m4.metric("DeadSp", f"{deadspace:.1f}%")
            m5.metric("VA", f"{va_total_sim:.1f}")
            
            # Blood Gas Prediction
            if bohr_res:
                st.markdown("#### 0. Blood Gas Prediction")
                bg1, bg2, bg3, bg4 = st.columns(4)
                bg1.metric("PaO2", f"{bohr_res['PaO2']:.1f}", help="Arterial PO2 (Torr)")
                bg2.metric("PaCO2", f"{bohr_res['PaCO2']:.1f}", help="Arterial PCO2 (Torr)")
                bg3.metric("A-a DO2", f"{bohr_res['AaDO2']:.1f}", help="Alveolar-Arterial Gradient")
                
                if enable_diffusion and dlo2_sim < 100:
                    bg4.metric("DLO2", f"{dlo2_sim:.1f}", delta="-Diffusion Limited" if dlo2_sim < 25 else "Normal")
                else:
                    bg4.metric("PvO2", f"{bohr_res['PvO2']:.1f}")
                
            # Row 1: V/Q Plot

            st.markdown("#### 1. Constructed V/Q Distribution")
            
            # Plotting styling
            plt.rcParams.update({'font.size': 9, 'grid.alpha': 0.5})
            fig, ax = plt.subplots(figsize=(10, 4))
            
            x = fw_model.vq_ratios
            mask = (x >= 0.005) & (x <= 100.0)
            x_plot = x[mask]
            q_plot = q_abs_total[mask]
            v_plot = v_abs_total[mask]
            
            if len(x_plot) > 3:
                x_log = np.log10(x_plot)
                x_new = np.linspace(x_log.min(), x_log.max(), 300)
                
                spl_q = make_interp_spline(x_log, q_plot, k=3)
                spl_v = make_interp_spline(x_log, v_plot, k=3)
                q_smooth = spl_q(x_new)
                v_smooth = spl_v(x_new)
                q_smooth[q_smooth < 0] = 0
                v_smooth[v_smooth < 0] = 0
                
                ax.fill_between(10**x_new, q_smooth, color='tab:red', alpha=0.1)
                ax.plot(10**x_new, q_smooth, 'tab:red', linewidth=2, label='Perfusion (Q)')
                
                ax.fill_between(10**x_new, v_smooth, color='tab:blue', alpha=0.1)
                ax.plot(10**x_new, v_smooth, 'tab:blue', linewidth=2, label='Ventilation (V)')
                
            else:
                 ax.plot(x_plot, q_plot, 'tab:red', marker='o', label='Perfusion (Q)')
                 ax.plot(x_plot, v_plot, 'tab:blue', marker='o', label='Ventilation (V)')

            # Shunt Indicator
            shunt_flow = (shunt / 100.0) * qt_sim
            if shunt_flow > 0.001:
                ax.scatter([0.002], [shunt_flow], color='black', s=100, zorder=10, label=f'Shunt ({shunt_flow:.2f} L)')
            
            ax.set_xscale('log')
            ax.set_xlim(0.001, 100.0)
            ax.set_xlabel(r'$V_A/Q_T$ Ratio')
            ax.set_ylabel('Compartmental Flow (L/min)')
            
            # Remove top/right spines for cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax.legend(frameon=False)
            ax.grid(True, which="major", ls="-", color="#e0e0e0")
            
            st.pyplot(fig, use_container_width=True)
            
            # Row 2: R/E Curves
            st.markdown("#### 2. Retention & Excretion Prediction")
            
            # Giving more space to the table (Ratio 3:2 instead of 2:1)
            c_plot, c_data = st.columns([3, 2])
            
            with c_plot:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                
                sweep_sols = np.logspace(-3, 3, 50)
                sweep_r, sweep_e = fw_model.calculate_re(q_dist, v_dist, ds_frac, sweep_sols)
                
                ax2.plot(sweep_sols, sweep_r, '-', color='tab:orange', linewidth=2, alpha=0.8, label='Simulated R')
                ax2.plot(sweep_sols, sweep_e, '-', color='tab:purple', linewidth=2, alpha=0.8, label='Simulated E')
                
                # Points
                ax2.plot(solubilities, meas_r, 'o', color='white', markeredgecolor='tab:orange', markeredgewidth=2, markersize=8, label='Inert Gases R')
                ax2.plot(solubilities, meas_e, 'o', color='white', markeredgecolor='tab:purple', markeredgewidth=2, markersize=8, label='Inert Gases E')
                
                ax2.set_xscale('log')
                ax2.set_xlabel('Solubility')
                ax2.set_ylabel('Fraction')
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.grid(True, ls="--", alpha=0.5)
                
                st.pyplot(fig2, use_container_width=True)
                
            with c_data:
                 df_out = pd.DataFrame({
                    "Gas": std_names,
                    "Solubility": solubilities,
                    "Retention (R)": meas_r,
                    "Excretion (E)": meas_e
                })
                 
                 # Display Table (Hidden Solubility Column for Space)
                 # Create a copy for display dropping Solubility
                 df_display = df_out.drop(columns=["Solubility"])
                 
                 st.dataframe(df_display.style.format({
                     "Retention (R)": "{:.4f}", 
                     "Excretion (E)": "{:.4f}"
                 }), use_container_width=True, hide_index=True)
                 
                 # Download Button gets FULL data including Solubility
                 st.download_button(
                     label="📥 CSV",
                     data=df_out.to_csv(index=False).encode('utf-8'),
                     file_name="miget_simulated_data.csv",
                     mime='text/csv',
                     use_container_width=True
                 )

    except Exception as e:
        st.error(f"An error occurred during simulation: {str(e)}")
        st.info("Please check the input parameters and try again.")
