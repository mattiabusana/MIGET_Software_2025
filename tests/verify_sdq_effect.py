
import sys
import os
import numpy as np

# Ensure path is set to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from miget.core import VQForwardModel, VQDistribution, InertGasData
from miget.physiology import BloodGasParams, blood_gas_calc, inverse_blood_gas_calc
from miget.bohr import BohrIntegrator

def run_simulation(sdq_val):
    # Fixed Parameters
    shunt = 3.0
    deadspace = 25.0
    qt_sim = 5.0
    ve_sim = 6.0
    fio2 = 21.0
    pb = 760.0
    hb = 15.0
    temp = 37.0
    
    # Derived
    pio2 = (fio2 / 100.0) * (pb - 47.0)
    bg_params = BloodGasParams(hb=hb, temp=temp)
    
    # Logic from direct.py
    fw_model = VQForwardModel()
    
    # Generate Distribution
    q_dist, v_dist, ds_frac = fw_model.generate_distribution(
        shunt_pct=shunt, deadspace_pct=deadspace,
        q1_mean=1.0, q1_sd=sdq_val
    )
    
    # Calculate Effective VA/Q
    va_total_sim = ve_sim * (1.0 - deadspace / 100.0)
    
    vaq_effective = np.zeros_like(fw_model.vq_ratios)
    for k in range(len(vaq_effective)):
        q_k = q_dist[k]
        v_k = v_dist[k]
        if q_k > 1e-9:
            vaq_effective[k] = (v_k * ve_sim) / (q_k * qt_sim)
        else:
             vaq_effective[k] = fw_model.vq_ratios[k]

    # Run Bohr
    vq_obj = VQDistribution(50, q_dist, v_dist, vaq_effective)
    data_obj = InertGasData(qt_measured=qt_sim, ve_measured=ve_sim)
    
    integrator = BohrIntegrator(vq_obj, data_obj, bg_params, dlo2=9000.0, pio2=pio2) # Infinite Diffusion - Fast
    
    pvo2, pvco2 = integrator.find_mixed_venous(250.0, 200.0) # VO2=250, VCO2=200
    
    # Arterial Calc
    cvo2, cvco2 = blood_gas_calc(pvo2, pvco2, bg_params)
    total_o2 = 0.0
    total_co2 = 0.0
    
    sum_v_pao2 = 0.0
    sum_v = 0.0
    
    for i in range(50):
        q_i = q_dist[i] * qt_sim
        if q_i <= 1e-6: continue
        vaq = vaq_effective[i]
        
        # Calculate Ventilation for this compartment
        v_i = q_i * vaq
        
        if vaq <= 1e-5:
            cc_o2, cc_co2 = cvo2, cvco2
            # Shunt PAO2? Effectively Mixed Venous for Gas, but V=0.
            # So contributes 0 to Mean PA.
        else:
            pa, pac, pc, pcc = integrator.solve_compartment_gas_lines(vaq, pvo2, pvco2, cvo2, cvco2)
            cc_o2, cc_co2 = blood_gas_calc(pc, pcc, bg_params)
            
            sum_v_pao2 += v_i * pa
            sum_v += v_i
            
        total_o2 += q_i * cc_o2
        total_co2 += q_i * cc_co2
        
    art_o2 = total_o2 / qt_sim
    art_co2 = total_co2 / qt_sim
    
    # Mean PAO2
    if sum_v > 0:
        mean_pao2 = sum_v_pao2 / sum_v
    else:
        mean_pao2 = pio2
    
    pao2, paco2 = inverse_blood_gas_calc(art_o2, art_co2, bg_params)
    
    # AaDO2
    r_ratio = 200.0 / 250.0
    fio2_frac = fio2 / 100.0
    palv_ideal = pio2 - (paco2 / r_ratio) * (1.0 - fio2_frac * (1.0 - r_ratio))
    
    return pao2, paco2, palv_ideal, palv_ideal - pao2, mean_pao2, mean_pao2 - pao2, np.mean(vaq_effective)

print(f"{'SDQ':<6} | {'PaO2':<8} | {'PaCO2':<8} | {'IdealPA':<8} | {'ClinAa':<8} | {'MeanPA':<8} | {'TrueAa':<8}")
print("-" * 80)
for sd in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
    po2, pco2, palv, clin_aa, mean_pa, true_aa, mvaq = run_simulation(sd)
    print(f"{sd:<6.2f} | {po2:<8.2f} | {pco2:<8.2f} | {palv:<8.2f} | {clin_aa:<8.2f} | {mean_pa:<8.2f} | {true_aa:<8.2f}")
