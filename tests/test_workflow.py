import numpy as np
import unittest
from miget.core import InertGasData, VQModel

class TestMigetWorkflow(unittest.TestCase):
    def test_synthetic_recovery(self):
        # 1. Create Synthetic 'True' V/Q Distribution
        # Log-normal distribution centered at VA/Q = 1.0 (Log10 = 0)
        n_compartments = 50
        vq_ratios = np.logspace(np.log10(0.005), np.log10(100.0), n_compartments)
        vq_ratios[0] = 0.0 # Shunt
        
        # True Q distribution: Gaussian in log-domain
        log_vq = np.log10(vq_ratios)
        log_mean = 0.0 # VA/Q = 1.0
        log_sd = 0.5 
        
        true_q = np.exp(-(log_vq - log_mean)**2 / (2 * log_sd**2))
        true_q[0] = 0.05 # Add 5% Shunt
        true_q = true_q / np.sum(true_q) # Normalize
        
        # 2. Forward Simulate Retention (R) for 6 gases
        # R_i = Sum_j ( Q_j * PC_i / (PC_i + VAQ_j) )
        # Standard PCs: SF6, Ethane, Cyclo, Enflurane, Ether, Acetone
        pcs = np.array([0.005, 0.1, 0.5, 2.0, 10.0, 300.0]) # Approx values
        
        retentions = []
        for pc in pcs:
            vals = pc / (pc + vq_ratios)
            vals[0] = 0.0 # Shunt component? R_shunt = 1.0? No, Pv/Pv = 1.0? 
            # Wait, Retention R = Pa/Pv.
            # Pa = Sum(Q_j * Cc_j). Cc_j = Pv * PC/(PC+VAQ).
            # So R = Sum(Q_j * PC / (PC+VAQ)).
            # For Shunt (VAQ=0): PC/(PC+0) = 1.0. Correct.
            vals[0] = 1.0
            
            r = np.sum(true_q * vals)
            retentions.append(r)
            
        retentions = np.array(retentions)
        
        # 3. Setup InertGasData
        data = InertGasData()
        data.partition_coeffs = pcs
        data.retention = retentions
        data.retention_weights = np.ones(6) # Uniform weights
        data.qt_measured = 5.0
        data.ve_measured = 5.0
        
        # 4. Recover
        model = VQModel(data)
        result = model.solve()
        
        # 5. Check
        # Re-calc retentions from recovered distribution
        recovered_q = result.blood_flow
        recovered_retentions = []
        for pc in pcs:
            vals = pc / (pc + vq_ratios)
            vals[0] = 1.0
            r = np.sum(recovered_q * vals)
            recovered_retentions.append(r)
            
        recovered_retentions = np.array(recovered_retentions)
        
        residual = np.sum((retentions - recovered_retentions)**2)
        print(f"True Retentions: {retentions}")
        print(f"Recovered Retentions: {recovered_retentions}")
        print(f"Residual Sum Sq: {residual}")
        print(f"Recovered Shunt: {recovered_q[0]} vs True: {true_q[0]}")
        
        self.assertLess(residual, 1e-4, "Recovered retentions should match input")
        
if __name__ == '__main__':
    unittest.main()
