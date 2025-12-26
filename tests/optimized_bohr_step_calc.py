
import sys
import os
import numpy as np
import time

# Ensure path is set to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from miget.physiology import BloodGasParams, blood_gas_calc

def calculate_slope(po2, pco2, params, delta=1.0):
    """
    Calculates dC/dP (slope of dissociation curve) using finite differences.
    Returns (dO2/dPO2, dCO2/dPCO2) - Simplified diagonal approximation.
    Ideally should be Jacobian, but usually independent enough.
    """
    
    # Central difference for better accuracy
    # dO2 / dPO2
    c_o2_plus, _ = blood_gas_calc(po2 + delta, pco2, params)
    c_o2_minus, _ = blood_gas_calc(po2 - delta, pco2, params)
    dco2_dpo2 = (c_o2_plus - c_o2_minus) / (2 * delta)
    
    # dCO2 / dPCO2
    _, c_co2_plus = blood_gas_calc(po2, pco2 + delta, params)
    _, c_co2_minus = blood_gas_calc(po2, pco2 - delta, params)
    dcco2_dpco2 = (c_co2_plus - c_co2_minus) / (2 * delta)
    
    return dco2_dpo2, dcco2_dpco2


def integrate_pressure(pa_o2, pa_co2, pvo2, pvco2, k_o2, k_co2, params, n_steps=20):
    """
    Integrates Partial Pressures directly.
    dC/dt = K * (PA - P)
    dP/dt = (dP/dC) * (dC/dt) = (1 / Slope) * K * (PA - P)
    """
    
    dt = 1.0 / n_steps
    
    curr_po2 = pvo2
    curr_pco2 = pvco2
    
    # RK4 needs derivatives at intermediate points.
    
    def derivatives_p(p_o2, p_co2):
        # Calculate Slope (dC/dP) at current P
        # Using analytical or finite diff
        # Bounds check
        if p_o2 < 0.1: p_o2 = 0.1
        if p_co2 < 0.1: p_co2 = 0.1
        
        slope_o2, slope_co2 = calculate_slope(p_o2, p_co2, params)
        
        if slope_o2 < 1e-6: slope_o2 = 1e-6
        if slope_co2 < 1e-6: slope_co2 = 1e-6
        
        # dC/dt
        dc_o2 = k_o2 * (pa_o2 - p_o2)
        dc_co2 = k_co2 * (pa_co2 - p_co2)
        
        # dP/dt
        dp_o2_dt = dc_o2 / slope_o2
        dp_co2_dt = dc_co2 / slope_co2
        
        return dp_o2_dt, dp_co2_dt
    
    for _ in range(n_steps):
        k1_o2, k1_co2 = derivatives_p(curr_po2, curr_pco2)
        
        k2_o2, k2_co2 = derivatives_p(curr_po2 + 0.5 * k1_o2 * dt, 
                                      curr_pco2 + 0.5 * k1_co2 * dt)
                                      
        k3_o2, k3_co2 = derivatives_p(curr_po2 + 0.5 * k2_o2 * dt, 
                                      curr_pco2 + 0.5 * k2_co2 * dt)
                                      
        k4_o2, k4_co2 = derivatives_p(curr_po2 + k3_o2 * dt, 
                                      curr_pco2 + k3_co2 * dt)
                                      
        curr_po2 += (dt / 6.0) * (k1_o2 + 2*k2_o2 + 2*k3_o2 + k4_o2)
        curr_pco2 += (dt / 6.0) * (k1_co2 + 2*k2_co2 + 2*k3_co2 + k4_co2)
        
        # Sanity clamping strictly not exceeding alveolar (if diffusion limited)
        # But overshoot is possible with RK4 if step too large?
        # Usually fine.
        
    return curr_po2, curr_pco2

# Benchmarking
params = BloodGasParams()
pvo2 = 40.0
pvco2 = 45.0
pa_o2 = 100.0
pa_co2 = 40.0

# Lower K (Severe Limitation)
k_o2 = 0.05
k_co2 = 0.1

start = time.time()
res_po2, res_pco2 = integrate_pressure(pa_o2, pa_co2, pvo2, pvco2, k_o2, k_co2, params, n_steps=20)
end = time.time()

print(f"Time taken (20 steps): {(end-start)*1000:.4f} ms")
print(f"Result PO2: {res_po2:.2f} (Target approx < 100), PCO2: {res_pco2:.2f}")
