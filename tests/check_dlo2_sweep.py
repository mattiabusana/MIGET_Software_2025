
import sys
import os
import numpy as np

# Ensure path is set to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from miget.core import VQDistribution, InertGasData
from miget.physiology import BloodGasParams, inverse_blood_gas_calc, blood_gas_calc
from miget.bohr import BohrIntegrator

def check_dlo2_sweep():
    print("Starting DLO2 Sweep...")
    params = BloodGasParams()
    data = InertGasData(qt_measured=5.0, ve_measured=6.0)
    data.vo2_measured = 250.0
    data.vco2_measured = 200.0

    # Create Normalized LogNormal Distribution with 1% Shunt
    n = 50
    x = np.linspace(-3, 3, n) # log10 vaq
    vaq_ratios = 10**x
    vaq_ratios[0] = 0.0 # Force pure shunt
    
    mu = 0.0
    sigma = 0.4
    dist = np.exp(-(x - mu)**2 / (2 * sigma**2))
    dist /= np.sum(dist)
    
    q_dist = dist.copy()
    q_dist = q_dist * 0.99
    q_dist[0] += 0.01 
    
    v_dist = q_dist * vaq_ratios
    v_dist /= np.sum(v_dist)
    
    vq_obj = VQDistribution(n, q_dist, v_dist, vaq_ratios)
    
    # Sweep DLO2
    dlo2_values = np.linspace(60.0, 5.0, 12) # 60, 55, ..., 5
    
    prev_pao2 = 999.0
    
    for dlo2 in dlo2_values:
        integrator = BohrIntegrator(vq_obj, data, params, dlo2=dlo2, pio2=150.0)
        
        # Helper to run full cycle
        # 1. Solve Mixed Venous
        print(f"  Solving Mixed Venous for DLO2={dlo2}...")
        pvo2, pvco2 = integrator.find_mixed_venous(data.vo2_measured, data.vco2_measured)
        print(f"  > PvO2={pvo2}, PvCO2={pvco2}")
        
        # 2. Get Arterial P
        from miget.physiology import blood_gas_calc
        cvo2, cvco2 = blood_gas_calc(pvo2, pvco2, params)
        
        pa_o2, pa_co2, pc_o2, pc_co2 = integrator.solve_all_compartments_gas_lines(
            vaq_ratios, pvo2, pvco2, cvo2, cvco2
        )
        
        # Recalc cc
        # Note: we need cvo2
        cc_o2, cc_co2 = blood_gas_calc(pc_o2, pc_co2, params)
        
        art_o2_c = np.sum(q_dist * cc_o2)
        art_co2_c = np.sum(q_dist * cc_co2)
        
        print(f"  > Art C O2={art_o2_c}, CO2={art_co2_c}")
        
        pao2, paco2 = inverse_blood_gas_calc(art_o2_c, art_co2_c, params)
        
        print(f"DLO2: {dlo2:.1f} -> PaO2: {pao2:.2f}, PvO2: {pvo2:.2f}")
        
        if pao2 > prev_pao2 + 0.5: # Tolerance
             print(f"  WARNING: Non-monotonic increase! ({prev_pao2:.2f} -> {pao2:.2f})")
             
        prev_pao2 = pao2

if __name__ == "__main__":
    check_dlo2_sweep()
