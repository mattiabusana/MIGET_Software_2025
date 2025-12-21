
import numpy as np
import jax
import sys
import os

# Set path to allow import
sys.path.append(os.getcwd())

try:
    from miget import jax_core
    print("MIGET JAX Core imported successfully.")
except ImportError as e:
    print(f"Failed to import jax_core: {e}")
    sys.exit(1)

def test_jax_solver():
    print("Testing JAX Solver...")
    
    # Mock data: A simple mix of 2 components
    n_gases = 6
    n_compartments = 50
    A = np.random.rand(n_gases, 50)
    true_x = np.zeros(50)
    true_x[10] = 1.0 # Peak 1
    true_x[30] = 0.5 # Peak 2
    
    b = A @ true_x
    
    # Weighting and Regularization
    weights = np.ones(n_gases)
    reg_matrix = np.eye(50) * 0.1
    
    A_w = A * weights[:, None]
    b_w = b * weights
    
    # Run Solver
    try:
        x_est = jax_core.solve_nnls_jax(A_w, b_w, reg_matrix)
        print(f"Solver executed. X sum: {np.sum(x_est):.4f}")
        
        # Test Covariance
        cov = jax_core.compute_covariance(A_w, b_w, reg_matrix, x_est, np.ones(n_gases))
        print(f"Covariance computed. Trace: {np.trace(cov):.4f}")
        
    except Exception as e:
        print(f"Solver failed: {e}")
        raise e

if __name__ == "__main__":
    test_jax_solver()
