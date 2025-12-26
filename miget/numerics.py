
import numpy as np
from numba import jit

# ---------------------------------------------------------------------------
# JIT-Compilable Physiology Functions (No objects, scalars/arrays only)
# ---------------------------------------------------------------------------

@jit(nopython=True)
def calculate_ph_jit(pco2, temp):
    # Simplified legacy PH = 7.59 - 0.2741 * log(PCO2/20) + Y
    # But this function only calculates base PH without Y
    # Y is added outside.
    return 7.59 - 0.2741 * np.log(np.maximum(pco2, 0.1) / 20.0)

@jit(nopython=True)
def calculate_saturation_jit(po2, pco2, ph, temp, dp50):
    po2 = np.maximum(po2, 0.0)
    
    # Kelman Coefficients
    a1 = -8532.229
    a2 = 2121.401
    a3 = -67.07399
    a4 = 935960.9
    a5 = -31346.26
    a6 = 2396.167
    a7 = -67.10441

    temp_diff = 37.0 - temp
    ph_diff = ph - 7.4
    
    pco2_safe = np.maximum(pco2, 1e-6)
    pco2_term = 0.06 * np.log10(40.0 / pco2_safe)
    
    x = po2 * (10.0 ** (0.024 * temp_diff + 0.4 * ph_diff + pco2_term))
    
    # P50 shift
    x = 26.8 * x / (26.8 + dp50)
    
    # Low saturation approx
    sat_low = 0.003683 * x + 0.000584 * x * x
    
    # High saturation Kelman
    num = x * (x * (x * (x + a3) + a2) + a1)
    den = x * (x * (x * (x + a7) + a6) + a5) + a4
    
    # Check den != 0 not strictly needed for physiological range
    sat_high = num / den
    
    sat = np.where(x < 10.0, sat_low, sat_high)
    return sat * 100.0

@jit(nopython=True)
def calculate_co2_content_jit(pco2, ph, saturation, temp, hcrit):
    p = 7.4 - ph
    temp_diff = 37.0 - temp
    
    pk = 6.086 + 0.042 * p + (38.0 - temp) * (0.00472 + 0.00139 * p)
    
    sol = 0.0307 + 0.00057 * temp_diff + 0.00002 * temp_diff**2
    
    dox = 0.59 + 0.2913 * p - 0.0844 * p * p
    dr = 0.664 + 0.2275 * p - 0.0938 * p * p
    
    ddd = dox + (dr - dox) * (1.0 - saturation / 100.0)
    
    cp = sol * pco2 * (1.0 + 10.0 ** (ph - pk))
    
    ccc = ddd * cp
    hct_frac = hcrit / 100.0
    
    content = (hct_frac * ccc + (1.0 - hct_frac) * cp) * 2.22
    return content

@jit(nopython=True)
def blood_gas_calc_jit(po2, pco2, hb, hcrit, temp, dp50, sol_o2):
    # 1. PH 1
    ph1 = calculate_ph_jit(pco2, temp)
    
    # 2. Sat 1
    sat1 = calculate_saturation_jit(po2, pco2, ph1, temp, dp50)
    
    # 3. Y Correction
    y = 0.003 * hb * (1.0 - sat1 / 100.0)
    
    # 4. PH 2
    ph2 = calculate_ph_jit(pco2, temp) + y
    
    # 5. Final Sat
    sat_final = calculate_saturation_jit(po2, pco2, ph2, temp, dp50)
    
    # 6. O2 Content
    o2_content = 1.39 * hb * (sat_final / 100.0) + sol_o2 * po2
    
    # 7. CO2 Content
    co2_content = calculate_co2_content_jit(pco2, ph2, sat_final, temp, hcrit)
    
    return o2_content, co2_content, ph2

# ---------------------------------------------------------------------------
# JIT-Compilable Integrator
# ---------------------------------------------------------------------------

@jit(nopython=True)
def integrate_all_jit(pa_o2, pa_co2, pvo2, pvco2, vaq_array, valid_mask, 
                      dlo2, dlco2, qt_perfused, n_steps,
                      hb, hcrit, temp, dp50, sol_o2):
    
    n = len(pa_o2)
    
    # K factors
    # DLO2 ml/min/Torr. Q L/min. -> Factor 10 for units?
    if qt_perfused <= 0: qt_perfused = 1.0
    
    k_val_o2 = dlo2 / qt_perfused / 10.0
    k_val_co2 = dlco2 / qt_perfused / 10.0
    
    current_p_o2 = np.full(n, pvo2)
    current_p_co2 = np.full(n, pvco2)
    
    dt = 1.0 / n_steps

    # Integrate Loop
    for _ in range(n_steps):
        # We need derivatives at current state
        # But derivatives depend on Slope which depends on P
        
        # --- RK4 Step ---
        # State: current_p_o2, current_p_co2
        
        # k1
        kp1_o2, kp1_co2 = calculate_derivatives_jit(current_p_o2, current_p_co2, pa_o2, pa_co2, k_val_o2, k_val_co2, hb, hcrit, temp, dp50, sol_o2)
        
        # k2
        kp2_o2, kp2_co2 = calculate_derivatives_jit(current_p_o2 + 0.5*kp1_o2*dt, current_p_co2 + 0.5*kp1_co2*dt, pa_o2, pa_co2, k_val_o2, k_val_co2, hb, hcrit, temp, dp50, sol_o2)
        
        # k3
        kp3_o2, kp3_co2 = calculate_derivatives_jit(current_p_o2 + 0.5*kp2_o2*dt, current_p_co2 + 0.5*kp2_co2*dt, pa_o2, pa_co2, k_val_o2, k_val_co2, hb, hcrit, temp, dp50, sol_o2)
        
        # k4
        kp4_o2, kp4_co2 = calculate_derivatives_jit(current_p_o2 + kp3_o2*dt, current_p_co2 + kp3_co2*dt, pa_o2, pa_co2, k_val_o2, k_val_co2, hb, hcrit, temp, dp50, sol_o2)
        
        current_p_o2 += (dt / 6.0) * (kp1_o2 + 2*kp2_o2 + 2*kp3_o2 + kp4_o2)
        current_p_co2 += (dt / 6.0) * (kp1_co2 + 2*kp2_co2 + 2*kp3_co2 + kp4_co2)
        
        # Stability Clamping to prevent NaN/Inf divergence on flat transformation portions
        current_p_o2 = np.maximum(0.0, np.minimum(current_p_o2, 2000.0))
        current_p_co2 = np.maximum(0.0, np.minimum(current_p_co2, 500.0))
        
    # Clamping
    # Only if finite diffusion (implicit by calling this function)
    # AND mask
    # If pvo2 < pa and final > pa -> clamp (Overshoot)
    for i in range(n):
        if not valid_mask[i]: continue
        
        # O2
        if pvo2 < pa_o2[i] and current_p_o2[i] > pa_o2[i]:
            current_p_o2[i] = pa_o2[i]
        elif pvo2 > pa_o2[i] and current_p_o2[i] < pa_o2[i]:
            current_p_o2[i] = pa_o2[i]
            
        # CO2 (usually diffuses out, pv > pa)
        if pvco2 > pa_co2[i] and current_p_co2[i] < pa_co2[i]:
            current_p_co2[i] = pa_co2[i]
        elif pvco2 < pa_co2[i] and current_p_co2[i] > pa_co2[i]:
            current_p_co2[i] = pa_co2[i]
            
    # Calculate Final Content
    cc_o2, cc_co2, _ = blood_gas_calc_jit(current_p_o2, current_p_co2, hb, hcrit, temp, dp50, sol_o2)
    
    return cc_o2, cc_co2, current_p_o2, current_p_co2

@jit(nopython=True)
def calculate_derivatives_jit(p_o2, p_co2, pa_o2, pa_co2, k_o2, k_co2, hb, hcrit, temp, dp50, sol_o2):
    delta = 0.5
    p_o2_safe = np.maximum(p_o2, 0.1)
    p_co2_safe = np.maximum(p_co2, 0.1)
    
    # Calculate Slope dC/dP
    co2_plus, _, _ = blood_gas_calc_jit(p_o2_safe + delta, p_co2_safe, hb, hcrit, temp, dp50, sol_o2)
    co2_minus, _, _ = blood_gas_calc_jit(p_o2_safe - delta, p_co2_safe, hb, hcrit, temp, dp50, sol_o2)
    slope_o2 = (co2_plus - co2_minus) / (2 * delta)
    slope_o2 = np.maximum(slope_o2, 1e-6)
    
    _, cco2_plus, _ = blood_gas_calc_jit(p_o2_safe, p_co2_safe + delta, hb, hcrit, temp, dp50, sol_o2)
    _, cco2_minus, _ = blood_gas_calc_jit(p_o2_safe, p_co2_safe - delta, hb, hcrit, temp, dp50, sol_o2)
    slope_co2 = (cco2_plus - cco2_minus) / (2 * delta)
    slope_co2 = np.maximum(slope_co2, 1e-6)
    
    # Gradients
    grad_o2 = pa_o2 - p_o2
    grad_co2 = pa_co2 - p_co2
    
    # dP/dt = K * Grad / Slope
    dp_o2 = (k_o2 * grad_o2) / slope_o2
    dp_co2 = (k_co2 * grad_co2) / slope_co2
    
    return dp_o2, dp_co2

@jit(nopython=True)
def solve_system_jit(vaq_array, pvo2, pvco2, cvo2, cvco2, 
                     dlo2, dlco2, qt_perfused, pio2, pico2,
                     hb, hcrit, temp, dp50, sol_o2):
    
    n = len(vaq_array)
    pa_o2 = np.full(n, 100.0)
    pa_co2 = np.full(n, 40.0)
    valid_mask = (vaq_array > 1e-4)
    
    inv_vaq = np.zeros(n)
    for i in range(n):
        if valid_mask[i]:
            inv_vaq[i] = 1.0 / vaq_array[i]
            
    # Implicit Solver Loop
    # Increased iterations and damping for stability
    for _ in range(100):
        # 1. Integrate Capillaries
        if dlo2 > 8000.0:
            # Equilibrated: Pc = PA
            cc_o2, cc_co2, _ = blood_gas_calc_jit(pa_o2, pa_co2, hb, hcrit, temp, dp50, sol_o2)
        else:
            # Finite
            cc_o2, cc_co2, _, _ = integrate_all_jit(pa_o2, pa_co2, pvo2, pvco2, vaq_array, valid_mask,
                                                    dlo2, dlco2, qt_perfused, 20,
                                                    hb, hcrit, temp, dp50, sol_o2)
                                                    
        # 2. Update PA
        new_pa_o2 = pio2 - (8.63 * inv_vaq) * (cc_o2 - cvo2)
        new_pa_o2 = np.maximum(0.0, np.minimum(new_pa_o2, 5000.0))
        
        new_pa_co2 = pico2 + (8.63 * inv_vaq) * (cvco2 - cc_co2)
        new_pa_co2 = np.maximum(0.1, np.minimum(new_pa_co2, 2000.0))
        
        # Check diff
        # Simple max diff logic on valid
        max_diff = 0.0
        for i in range(n):
            if valid_mask[i]:
                d1 = abs(new_pa_o2[i] - pa_o2[i])
                d2 = abs(new_pa_co2[i] - pa_co2[i])
                if d1 > max_diff: max_diff = d1
                if d2 > max_diff: max_diff = d2
        
        if max_diff < 0.05: # Stricter tolerance
            pa_o2 = new_pa_o2
            pa_co2 = new_pa_co2
            break
            
        # Heavy damping to prevent oscillation
        pa_o2 = 0.8 * pa_o2 + 0.2 * new_pa_o2
        pa_co2 = 0.8 * pa_co2 + 0.2 * new_pa_co2
        
    # Final Integration to return Pc
    if dlo2 > 8000.0:
        pc_o2 = pa_o2.copy()
        pc_co2 = pa_co2.copy()
        cc_o2, cc_co2, _ = blood_gas_calc_jit(pc_o2, pc_co2, hb, hcrit, temp, dp50, sol_o2)
    else:
        cc_o2, cc_co2, pc_o2, pc_co2 = integrate_all_jit(pa_o2, pa_co2, pvo2, pvco2, vaq_array, valid_mask,
                                                        dlo2, dlco2, qt_perfused, 20,
                                                        hb, hcrit, temp, dp50, sol_o2)
                                                        
    # Explicitly fix Shunt / zero VAQ compartments to be Mixed Venous
    for i in range(n):
        if not valid_mask[i]:
            pc_o2[i] = pvo2
            pc_co2[i] = pvco2
            pa_o2[i] = pvo2 # Alveolar P is undefined, but for safety set to venous
            pa_co2[i] = pvco2

    return pa_o2, pa_co2, pc_o2, pc_co2

