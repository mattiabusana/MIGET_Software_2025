
import numpy as np

from miget.physiology import BloodGasParams, blood_gas_calc, inverse_blood_gas_calc
from miget.core import VQDistribution, InertGasData
from miget.bohr import BohrIntegrator

def test_inverse_blood_gas():
    params = BloodGasParams()
    po2_target = 90.0
    pco2_target = 40.0
    
    # Forward
    o2_c, co2_c = blood_gas_calc(po2_target, pco2_target, params)
    
    # Inverse
    po2_calc, pco2_calc = inverse_blood_gas_calc(o2_c, co2_c, params)
    
    print(f"Target P: {po2_target}, {pco2_target}")
    print(f"Calculated P: {po2_calc}, {pco2_calc}")
    
    assert abs(po2_calc - po2_target) < 0.1
    assert abs(pco2_calc - pco2_target) < 0.1

def test_bohr_infinite_diffusion():
    """High diffusion should mean Pc' ~= PA"""
    params = BloodGasParams()
    # Mock data
    vq_dist = VQDistribution(
        compartments=50,
        blood_flow=np.zeros(50),
        ventilation=np.zeros(50),
        vaq_ratios=np.logspace(-2, 2, 50)
    )
    # Set one compartment to have flow
    vq_dist.blood_flow[25] = 1.0 # 100% flow to middle (VA/Q ~ 1.0)
    
    data = InertGasData(qt_measured=5.0, ve_measured=5.0) 
    
    # Infinite Diffusion
    bohr = BohrIntegrator(vq_dist, data, params, dlo2=9000.0)
    
    # Standard venous
    pvo2, pvco2 = 40.0, 45.0
    cvo2, cvco2 = blood_gas_calc(pvo2, pvco2, params)
    
    vaq = 1.0
    
    pa_o2, pa_co2, pc_o2, pc_co2 = bohr.solve_compartment_gas_lines(vaq, pvo2, pvco2, cvo2, cvco2)
    
    print(f"\nInfinite Diffusion (VA/Q=1): PAO2={pa_o2:.2f}, Pc'O2={pc_o2:.2f}")
    
    # Should be equal
    assert abs(pa_o2 - pc_o2) < 0.2

def test_bohr_finite_diffusion():
    """Low diffusion should mean Pc' < PA for O2"""
    params = BloodGasParams()
    # Mock data
    vq_dist = VQDistribution(
        compartments=50,
        blood_flow=np.zeros(50),
        ventilation=np.zeros(50),
        vaq_ratios=np.logspace(-2, 2, 50)
    )
    vq_dist.blood_flow[25] = 1.0
    
    data = InertGasData(qt_measured=5.0, ve_measured=5.0)
    
    # Low Diffusion (e.g. 3 ml/min/Torr)
    bohr = BohrIntegrator(vq_dist, data, params, dlo2=3.0)
    
    pvo2, pvco2 = 40.0, 45.0
    cvo2, cvco2 = blood_gas_calc(pvo2, pvco2, params)
    
    vaq = 1.0
    
    pa_o2, pa_co2, pc_o2, pc_co2 = bohr.solve_compartment_gas_lines(vaq, pvo2, pvco2, cvo2, cvco2)
    
    print(f"\nFinite Diffusion (DLO2=15): PAO2={pa_o2:.2f}, Pc'O2={pc_o2:.2f}")
    
    # O2 should not equilibrate fully
    # Expect Pc'O2 < PAO2
    assert pc_o2 < pa_o2 - 1.0 # At least 1 Torr difference
    
    # Check if calculation is physically sound (Pc > Pv)
    assert pc_o2 > pvo2
    assert pc_o2 < 150.0 # PIO2

if __name__ == "__main__":
    test_inverse_blood_gas()
    test_bohr_infinite_diffusion()
    test_bohr_finite_diffusion()
