
import sys
import os
import time
import numpy as np

# Ensure path is set to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from miget.core import VQForwardModel, VQDistribution, InertGasData
from miget.physiology import BloodGasParams
from miget.bohr import BohrIntegrator

def benchmark_full_vector():
    params = BloodGasParams()
    data = InertGasData(qt_measured=5.0, ve_measured=6.0)
    
    # Mock VQ Dist
    n = 50
    q_dist = np.random.lognormal(0, 0.5, n)
    q_dist /= np.sum(q_dist)
    v_dist = np.random.lognormal(0, 0.5, n)
    v_dist /= np.sum(v_dist)
    vaq_ratios = np.linspace(0.01, 10.0, n)
    
    vq_obj = VQDistribution(n, q_dist, v_dist, vaq_ratios)
    
    integrator = BohrIntegrator(vq_obj, data, params, dlo2=20.0, pio2=150.0)
    
    pvo2 = 40.0
    pvco2 = 45.0
    cvo2 = 15.0
    cvco2 = 50.0
    
    print("\nBenchmarking `solve_all_compartments_gas_lines`...")
    start = time.time()
    
    # This runs the 20 * 20 * RK4 loop
    integrator.solve_all_compartments_gas_lines(vaq_ratios, pvo2, pvco2, cvo2, cvco2)
    
    end = time.time()
    print(f"Total time: {end-start:.4f} seconds")

if __name__ == "__main__":
    benchmark_full_vector()
