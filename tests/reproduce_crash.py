
import numpy as np
from miget.physiology import BloodGasParams, blood_gas_calc, inverse_blood_gas_calc
from miget.core import VQDistribution, InertGasData
from miget.bohr import BohrIntegrator

def test_shunt_crash():
    params = BloodGasParams()
    vq_dist = VQDistribution(compartments=50) # Defaults
    data = InertGasData(qt_measured=5.0)
    
    bohr = BohrIntegrator(vq_dist, data, params)
    
    # Mock Venous
    pvo2, pvco2 = 40.0, 45.0
    cvo2, cvco2 = blood_gas_calc(pvo2, pvco2, params)
    
    print("Attempting solve with VAQ=0 (Shunt)...")
    try:
        # This is what happened in the loop for i=0
        bohr.solve_compartment_gas_lines(0.0, pvo2, pvco2, cvo2, cvco2)
        print("Did not crash? Unexpected.")
    except Exception as e:
        print(f"Caught expected crash: {e}")

if __name__ == "__main__":
    test_shunt_crash()
