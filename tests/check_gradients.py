
import sys
import os
import numpy as np

# Ensure path is set to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from miget.core import VQDistribution, InertGasData
from miget.physiology import BloodGasParams
from miget.bohr import BohrIntegrator

def check_gradients():
    params = BloodGasParams()
    data = InertGasData(qt_measured=5.0, ve_measured=5.0)
    data.vo2_measured = 250.0
    data.vco2_measured = 200.0
                       
    # Create LogNormal distribution (Mean 1.0, SD 0.4)
    n = 50
    # Log-space
    # In miget.core, VQDistribution takes Q and V distributions.
    # We can use helper or just construct manually using logpdf.
    
    # Actually, simpler to use random for quick check or just known profile
    # Let's use numpy to generate smooth distributions
    x = np.linspace(-3, 3, n) # log10 vaq
    vaq_ratios = 10**x
    
    # Gaussian on log scale
    mu = 0.0 # log10(1)
    sigma = 0.4
    dist = np.exp(-(x - mu)**2 / (2 * sigma**2))
    dist /= np.sum(dist)
    
    q_dist = dist.copy()
    # Add 1% Shunt
    # Normalize q_dist to 0.99
    q_dist = q_dist * 0.99
    q_dist[0] += 0.01 # Put shunt in first bin (assuming vaq[0] is small enough)
    vaq_ratios[0] = 0.0 # Force pure shunt
    
    # Ventilation? V = Q * VAQ roughly?
    # v_dist = q_dist * vaq_ratios ... normalized
    v_dist = q_dist * vaq_ratios
    v_dist /= np.sum(v_dist)
    
    vq_obj = VQDistribution(n, q_dist, v_dist, vaq_ratios)
    
    integrator = BohrIntegrator(vq_obj, data, params, dlo2=30.0, pio2=150.0)
    
    # Solve Mixed Venous
    pvo2, pvco2 = integrator.find_mixed_venous(data.vo2_measured, data.vco2_measured)
    
    # Solve Forward
    pa_o2, pa_co2, pc_o2, pc_co2 = integrator.solve_all_compartments_gas_lines(
        vaq_ratios, pvo2, pvco2, integrator.params.hb, integrator.params.temp # Dummy args, unused by new wrapper actually?
        # Wait, wrapper arguments changed? 
        # BohrIntegrator.solve_all_compartments_gas_lines signature in bohr.py:
        # (self, vaq_array, pvo2, pvco2, cvo2, cvco2)
    )
    
    # Need Cv for call
    from miget.physiology import blood_gas_calc, inverse_blood_gas_calc
    cvo2, cvco2 = blood_gas_calc(pvo2, pvco2, params)
    
    pa_o2, pa_co2, pc_o2, pc_co2 = integrator.solve_all_compartments_gas_lines(
        vaq_ratios, pvo2, pvco2, cvo2, cvco2
    )

    # Calculate Arterial
    qt = 5.0
    cc_o2_all, cc_co2_all = blood_gas_calc(pc_o2, pc_co2, params)
    
    art_o2_content = np.sum(q_dist * cc_o2_all) # sums reference unit flow
    art_co2_content = np.sum(q_dist * cc_co2_all)
    
    pao2, paco2 = inverse_blood_gas_calc(art_o2_content, art_co2_content, params)

    print(f"PvO2: {pvo2:.2f}, PbO2 (Arterial): {pao2:.2f}")
    print(f"PvCO2: {pvco2:.2f}, PaCO2 (Arterial): {paco2:.2f}")
    
    if pao2 < pvo2:
        print("ERROR: Arterial PO2 < Venous PO2! (Physiologically Impossible)")
    else:
        print("OK: Arterial PO2 > Venous PO2")
        
    if paco2 > pvco2:
        print("ERROR: Arterial PCO2 > Venous PCO2! (Physiologically Impossible)")
    else:
        print("OK: Arterial PCO2 < Venous PCO2")

if __name__ == "__main__":
    check_gradients()
