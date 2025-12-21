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
                shunt = st.slider("Shunt (%)", 0.0, 100.0, 5.0, step=0.1)
                deadspace = st.slider("Deadspace (%)", 0.0, 100.0, 30.0, step=0.1)
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
        v_abs_total = v_dist * va_total_sim
        
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
