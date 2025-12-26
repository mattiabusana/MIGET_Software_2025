import numpy as np
from typing import Tuple, List
from miget.core import VQDistribution, InertGasData
from miget.physiology import BloodGasParams, blood_gas_calc, calculate_saturation, calculate_co2_content

class BohrIntegrator:
    """
    Implements the Bohr Integration for Oxygen Diffusion Capacity (DLO2).
    Ref: VQBOHR.FOR BOHRI / CALC / FNDTEN logic.
    """

    def __init__(self, vq_dist: VQDistribution, data: InertGasData, params: BloodGasParams, dlo2: float = 9000.0, pio2: float = 150.0):
        self.vq_dist = vq_dist
        self.data = data 
        self.params = params
        self.dlo2 = dlo2 
        self.dlco2 = dlo2 * 5.0
        self.pio2 = pio2
        
        # Integration settings
        self.n_steps = 100 
        
    def find_mixed_venous(self, vo2_target, vco2_target, tolerance=0.1) -> Tuple[float, float]:
        """
        Iteratively finds PvO2 and PvCO2 that satisfy the Fick principle.
        """
        pvo2 = 40.0
        pvco2 = 45.0
        
        for _ in range(50):
            calc_vo2, calc_vco2 = self.calculate_total_exchange(pvo2, pvco2)
            
            diff_vo2 = calc_vo2 - vo2_target
            diff_vco2 = calc_vco2 - vco2_target
            
            if abs(diff_vo2) < tolerance and abs(diff_vco2) < tolerance:
                break
                
            # Adjustment factors (Heuristic)
            pvo2 += 0.5 * (calc_vo2 - vo2_target) / 100.0 
            pvco2 -= 0.5 * (calc_vco2 - vco2_target) / 100.0
            
            # Constraints
            pvo2 = max(1.0, min(100.0, pvo2))
            pvco2 = max(1.0, min(100.0, pvco2))
            
        return pvo2, pvco2
        
        return final_c_o2, final_c_co2

    # ------------------------------------------------------------------------
    # Numba JIT Accelerated Solvers
    # ------------------------------------------------------------------------

    def solve_all_compartments_gas_lines(self, vaq_array, pvo2, pvco2, cvo2, cvco2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates Steady-State Gas Levels for all compartments.
        Uses Numba JIT compiled kernel for extreme speed.
        """
        from miget.numerics import solve_system_jit
        
        qt_perfused = self.data.qt_measured * (1 - self.vq_dist.blood_flow[0])
        if qt_perfused <= 0: qt_perfused = 1.0
        
        pa_o2, pa_co2, pc_o2, pc_co2 = solve_system_jit(
            vaq_array, pvo2, pvco2, cvo2, cvco2,
            self.dlo2, self.dlco2, qt_perfused, self.pio2, 0.0,
            self.params.hb, self.params.hcrit, self.params.temp, self.params.dp50, self.params.sol_o2
        )
        
        return pa_o2, pa_co2, pc_o2, pc_co2
        
    def calculate_total_exchange(self, pvo2, pvco2) -> Tuple[float, float]:
        """
        Sum uptake across compartments using vectorized solver.
        """
        # Get Mixed Venous Content
        from miget.physiology import blood_gas_calc
        cvo2, cvco2 = blood_gas_calc(pvo2, pvco2, self.params)

        # vaq array from vq_dist? 
        # vq_dist.vaq_ratios might not be effective ones (if deadspace adjusted).
        # We should use effective VA/Q if possible.
        # However, for `find_mixed_venous`, the inputs are usually the raw distribution or effective?
        # Typically the wrapper calls this.
        # Let's use `vq_dist.vaq_ratios` for now.
        # Note: If `calculate_total_exchange` is called internally by `find_mixed_venous`,
        # it relies on `self.vq_dist`.
        
        vaq_ratios = self.vq_dist.vaq_ratios
        
        # JIT Solve all
        _, _, pc_o2_all, pc_co2_all = self.solve_all_compartments_gas_lines(vaq_ratios, pvo2, pvco2, cvo2, cvco2)
        
        # Convert Pc to Content using JIT (or fast blood_gas_calc)
        # Note: solve_all returns Pc. We need Cc.
        # Use vectorized blood_gas_calc from physiology (it accepts arrays)
        cc_o2_all, cc_co2_all = blood_gas_calc(pc_o2_all, pc_co2_all, self.params)
        
        # Sum fluxes
        qt = self.data.qt_measured
        q_fracs = self.vq_dist.blood_flow
        
        # Flux = Q_i * (Cc_i - Cv) * 10
        # Iterate or Vector sum
        
        # Handle Shunt? (vaq=0). 
        # solve_system_jit handles vaq=0 gracefully by returning venous values?
        # Or masking. Let's ensure shunt is handled.
        # In solve_system_jit: if valid_mask is false (vaq < 1e-4), it skips update.
        # It assumes PA/Pc stay at initial?
        # Wait, inside `solve_system_jit`, invalid indices are skipped. 
        # So `pc_o2` for shunt indices might be 0 or garbage?
        # NO. `solve_system_jit` initializes PA/Pc arrays but doesn't set shunt specifically unless valid_mask matches.
        # I should check `numerics.py` logic.
        # `integrate_all_jit`: skips invalid indices. Current P initialized to Pv. So it stays Pvo2. Correct.
        # So, for Shunt, Pc = Pv. Cc = Cv.
        # Flux = Q * (Cv - Cv) = 0. Correct.
        
        flux_o2_i = q_fracs * qt * (cc_o2_all - cvo2) * 10.0
        flux_co2_i = q_fracs * qt * (cvco2 - cc_co2_all) * 10.0
        
        total_vo2 = np.sum(flux_o2_i)
        total_vco2 = np.sum(flux_co2_i)
        
        return total_vo2, total_vco2


