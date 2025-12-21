from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional

@dataclass
class InertGasData:
    """
    Stores data for a single MIGET run (one subject/condition).
    Corresponds to the data processed in SHORT.FOR.
    """
    name: str = "Run 1"
    
    # Experimental Metadata
    pb_sea: float = 760.0 # Barometric pressure at sea level (Torr)
    temp_blood: float = 37.0 # Electrode temp (typically 37)
    temp_bath: float = 37.0 # Water bath temp for solubility
    temp_body: float = 37.0 # Patient body temp
    
    # Gas Properties
    # Arrays size = N_GASES (usually 6)
    # Partition Coefficients
    # Arrays size = N_GASES
    gas_names: List[str] = field(default_factory=lambda: ["SF6", "Ethane", "Cyclopropane", "Enflurane", "Ether", "Acetone"])
    partition_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(6))
    solubilities: np.ndarray = field(default_factory=lambda: np.zeros(6))
    z: float = 40.0 # Smoothing factor
    
    # Species for Temp Correction
    species: str = "human" # human, dog, horse
    
    # Solubility Temp Slopes (SHORT.FOR data)
    # SF6, Ethane, Cyclo, Enflurane, Ether, Acetone
    SLOPES: dict = field(default_factory=lambda: {
        "human": np.array([2950., 1374., 2025., 3016., 4066., 805.]),
        "dog":   np.array([2263., 2396., 2909., 4435., 3281., 1391.]),
        "horse": np.array([1251., 2466., 3262., 2696., 3877., 2412.])
    })
    
    # Measured Data (Pressures in mm/peak height)
    # These might be 0 if derived.
    pa_raw: np.ndarray = field(default_factory=lambda: np.zeros(6)) # Arterial Peak (mm)
    pe_raw: np.ndarray = field(default_factory=lambda: np.zeros(6)) # Mixed Expired Peak (mm)
    pv_raw: np.ndarray = field(default_factory=lambda: np.zeros(6)) # Mixed Venous Peak (mm)
    
    # Calibration/Gain Factors
    ga: np.ndarray = field(default_factory=lambda: np.ones(6))
    ge: np.ndarray = field(default_factory=lambda: np.ones(6))
    gv: np.ndarray = field(default_factory=lambda: np.ones(6))
    
    # Calculated R and E
    retention: np.ndarray = field(default_factory=lambda: np.zeros(6))
    excretion: np.ndarray = field(default_factory=lambda: np.zeros(6))
    retention_weights: np.ndarray = field(default_factory=lambda: np.zeros(6))
    excretion_weights: np.ndarray = field(default_factory=lambda: np.zeros(6))
    
    # Ventilation / Perfusion
    ve_measured: float = 0.0 # Minute Ventilation (L/min)
    qt_measured: float = 0.0 # Cardiac Output (L/min)
    
    # Sampling Volumes (Headspace Correction)
    vga: float = 0.0 # Vol Gas Arterial
    vba: float = 0.0 # Vol Blood Arterial
    vha: float = 0.0 # Vol ? (VHA usually Vol Headspace?)
    vgv: float = 0.0 # Vol Gas Venous
    vbv: float = 0.0 # Vol Blood Venous
    vhv: float = 0.0 # Vol ?
    
    def calculate_metrics(self, temp_error_check=False):
        """
        Calculate R, E, and Weights from raw data.
        Logic ported from SHORT.FOR.
        """
        n_gases = len(self.gas_names)
        
        # FACT = (PBSEA - SVPB) / 100.0 (Line 313)
        # But here logic is a bit spread out.
        # Let's implement the core derived values.
        
        # 1. Temperature Corrections (Lines 353-363)
        # Calculates PC at body temperature.
        # Select slopes based on species
        species_key = self.species.lower()
        if species_key not in self.SLOPES:
            species_key = "human"
            
        slopes = self.SLOPES[species_key]
        
        # PCFACT(I) = EXP(SLP * (1/(TEMPB+273) - 1/(TBATH+273)))
        tt1 = 1.0 / (self.temp_body + 273.0)
        tt2 = 1.0 / (self.temp_bath + 273.0)
        
        # Assuming gases are sorted as in legacy: SF6, Ethane, Cyclo, Enflurane, Ether, Acetone
        pcfact = np.exp(slopes[:n_gases] * (tt1 - tt2))
        pc_body = self.partition_coeffs * pcfact
        
        # 2. Solubility (Line 364)
        # FACT = (PBSEA - SVPB)/100 ? No wait, SHORT loops over runs.
        # Line 313: FACT=(PBSEA-SVPB)/100.0  where SVPB is SVP at Body Temp.
        svpb = 47.0 # Approximation or calc? SHORT calculates it:
        # BEXP=9.0578-2290.5/(273.15+TEMPB); SVPB=10.0**BEXP
        bexp = 9.0578 - 2290.5 / (273.15 + self.temp_body)
        svpb = 10.0 ** bexp
        fact = (self.pb_sea - svpb) / 100.0
        
        # DEBUG
        # print(f"DEBUG: PB={self.pb_sea}, SVPB={svpb}, FACT={fact}, Raw PC={self.partition_coeffs[0]}")
        
        self.mb_solubilities = pc_body # Use Raw (Temp Corrected) for Mass Balance
        self.solubilities = pc_body # Use Raw (Ostwald) for Model S(I). No FACT scaling needed if Ostwald.
        # print(f"DEBUG: Corrected S[0]={self.solubilities[0]}")
        
        # 3. Corrected Pressures
        # PAC, PEC, PVC calculation (Lines 365-372)
        # PAC(I) = GA(I)*PA(I)*(1.0 + VHA/VBA + VGA/(VBA*PC(I)))/(PCFACT(I)*PAPV)
        # PEC(I)=GE(I)*PE(I)*EXPFAC
        # PVC(I) = GV(I)*PV(I)*(1.0 + VHV/VBV + VGV/(VBV*PC(I)))/PCFACT(I)
        
        
        # 2. PAPV Correction (Machine params)
        papv = 1.0 # Default (Arterial)
        # TODO: Add logic/field for Venous check if needed.
        
        # 3. Expansion Factor (EXPFAC)
        # EXPFAC=(PB-SVPB)/(PBSEA-SVPB)
        # Simplified:
        exp_fac = 1.0 
        
        # 4. Headspace Correction (THE BIG ONE)
        # (1.0 + VHA/VBA + VGA/(VBA*PC(I)))
        
        # Arterial
        pac_factor = np.ones(n_gases)
        if self.vba > 0 and self.vga > 0:
             # VGA/(VBA*PC(I)) - Uses PC Bath?
             # SHORT.FOR uses PC(I) (bath temp).
             term1 = self.vha / self.vba
             term2 = self.vga / (self.vba * self.partition_coeffs)
             pac_factor = 1.0 + term1 + term2
        
        # PAC Calculation - DIVIDE by PCFACT
        # PAC(I) = GA(I)*PA(I)*PAC_FACTOR / (PCFACT(I)*PAPV)
        pac = self.ga * self.pa_raw * pac_factor / (pcfact * papv)
        
        # Expired (Gas Phase)
        # PEC(I)=GE(I)*PE(I)*EXPFAC
        pec = self.ge * self.pe_raw * exp_fac
        
        # Venous
        pvc_factor = np.ones(n_gases)
        if self.vbv > 0 and self.vgv > 0:
             term1 = self.vhv / self.vbv
             term2 = self.vgv / (self.vbv * self.partition_coeffs)
             pvc_factor = 1.0 + term1 + term2
             
        # PVC Calculation - DIVIDE by PCFACT
        pvc = self.gv * self.pv_raw * pvc_factor / pcfact
        
        # --- ACETONE CORRECTION (SHORT.FOR Lines 386-398) ---
        # Logic: Acetone (Gas 6) loss prevention.
        # "ACETONE BOHR VDVT MUST BE NO GREATER THAN THAT OF ETHER"
        # Compare Ether (Gas 5, index 4) and Acetone (Gas 6, index 5)
        # Only if we have at least 6 gases
        if n_gases >= 6:
            idx_ether = 4
            idx_acetone = 5
            
            # Use Corrected Pressures? SHORT.FOR uses "PAC" and "PEC".
            # Yes, lines 388 check PAC/PEC.
            pa_eth = pac[idx_ether]
            pe_eth = pec[idx_ether]
            pa_ac = pac[idx_acetone]
            pe_ac = pec[idx_acetone] # Current calc
            
            if pa_eth > 0 and pe_eth > 0 and pa_ac > 0:
                # ETHER = PEC/PAC (Ratio)
                ether_ratio = pe_eth / pa_eth
                
                # Predicted PE Acetone
                # SHORT.FOR Line 393: IF(PV(NGASES).EQ.0.0) ACEXP=ETHER*PAC(NGASES)
                # If PV exists: ACEXP=0.5*ETHER*(PAC+PVC). (Line 392)
                
                # Check raw PV to know correct formula
                pv_raw_ac = self.pv_raw[idx_acetone]
                
                if pv_raw_ac > 0:
                    pv_ac = pvc[idx_acetone] # Corrected
                    ac_target = 0.5 * ether_ratio * (pa_ac + pv_ac)
                else:
                    ac_target = ether_ratio * pa_ac
                    
                # Replace if Target is LARGER (i.e. we measured too little PE, meaning we lost Acetone)
                if ac_target > pe_ac:
                    factor = ac_target / pe_ac
                    # Update PEC
                    pec[idx_acetone] = ac_target
                    # print(f"DEBUG: Correcting Acetone Loss. Factor: {factor:.3f}")
                    
        # Handle "Derived" values (Lines 940+)
        # If PA is missing but PV and PE exist...
        # Also, check if PV is effectively zero (MISSISSIPPI Case)
        
        # Check if PV is zero (sum of raw values is very small)
        if np.sum(self.pv_raw) < 1e-6:
            # PV Not Measured. Calculate from Mass Balance.
            # Conservation of Mass: Input = Output
            # QT * PV_conc = QT * PA_conc + VE * PE_conc
            # PV_conc = PA_conc + (VE/QT) * PE_conc
            # Note: Concentrations = Solubility * Partial Pressure?
            # Or just Pressure if normalized?
            # Standard Mass Balance uses CONCENTRATIONS (ml gas / ml liquid).
            # C_venous = C_arterial * (1 - E) + E * C_venous? No.
            # C_venous = C_arterial + (VE/QT) * C_mixed_expired_gas
            
            # C_art = Solubility * PAC
            # C_ven = Solubility * PVC
            # C_exp_gas = PEC / (PB - 47)? No, PEC is partial pressure.
            # C_exp_gas = PEC / 760? Or just PEC proportional to amount.
            # VQBOHR usually works with Normalized retentions, but here we work with Pressures.
            
            # Legacy Code implies: 
            # R = PA/PV = PC / (PC + VAQ).
            # E = PE/PV = VAQ / (PC + VAQ) * (PC_gas / PC_blood)?
            
            # Simple Mass Balance in terms of Pressures:
            # S * PV = S * PA + (VE/QT) * PE_gas_conc?
            # PE_gas_conc (ml/ml) = PE / (PB - 47).
            # S (ml/ml/atm).
            
            # Wait, let's look at SHORT.FOR or standard logic.
            # If we assume standard units for PAC and PEC are compatible pressures:
            # PV = PA + PE * (VE/QT) / S_eff?
            
            # Alternatively use the Ratio:
            # Ratio = VE/QT.
            # PV = PA + PE * Ratio.
            # BUT PE is gas phase, PA is liquid phase.
            # They are related by Solubility S.
            # Amount in Gas = V_gas * PE / (PB).
            # Amount in Liq = V_liq * PA * S / (PB).
            
            # Mass Balance Rate:
            # M_ven = QT * S * PV
            # M_art = QT * S * PA
            # M_exp = VE * PE (if PE is fraction. If PE is Torr: VE * PE / (PB-47)).
            
            # QT * S * PV = QT * S * PA + VE * PE_frac
            # PV = PA + (VE / (QT * S)) * PE_frac
            
            # How is PE stored? Torr? Or Chrome area?
            # Usually Chrome Area is proportional to Amount.
            # If PE and PA are areas from same detector:
            # PE_area proportional to Amount in Sample Loop.
            # PA_area proportional to Amount in Sample Loop (Liquid -> Gas extracted).
            # Headspace correction (PAC, PEC) converts Area to "Partial Pressure equivalent in Liquid".
            # So PAC and PEC are comparable Partial Pressures.
            
            # If PEC is "Partial Pressure in Gas Phase that would be in equilibrium",
            # PEC is actual gas pressure.
            # PAC is gas pressure in equilibrium with blood.
            
            # Rate Leaving via Lungs = VE * (PEC / (PB - 47)).  (Volumetric flow of gas).
            # Rate Leaving via Blood = QT * S * (PAC / 760).
            # Rate Entering = QT * S * (PVC / 760).
            
            # QT * S * PVC = QT * S * PAC + VE * PEC * (760 / (PB-47))?
            # PVC = PAC + PEC * (VE/QT) * (1/S) * (760/(PB-47)).
            
            # Factor = (VE/QT) * (1/S) * (Factor_P).
            
            # Let's simplify.
            # If we use Ret/Exc definitions:
            # E_meas = (PE_frac * VE) / (PV_Content * QT) ?
            # No. E = (PE/PV).
            
            # Let's try the formula: PV = PA + PE * (VE/QT).
            # This assumes S=1?
            # If S is low (SF6), E is high. PE is high.
            # PV should be dominated by PE term.
            # PV ~ PE * (VE/QT) / S.
            
            # Correct Formula:
            # PVC = PAC + PEC * (self.ve_measured / self.qt_measured) / self.solubilities
            
            # BUT we need to check if PEC is already normalized?
            # We calculated PEC = GE * PE * EXPFAC.
            # This is Partial Pressure (Torr).
            # Solubilities are Ostwald (ml gas / ml blood).
            
            # Wait. VE is btps? QT is L/min.
            # S is ml/ml.
            
            # Term = PEC * (VE/QT) / S.
            # Units: Torr * (L/min / L/min) / (1) = Torr.
            # This seems dimensionally correct for Pressure.
            
            # Let's check magnitude for SF6.
            # SF6 S=0.005. VE/QT ~ 2.
            # Term ~ PEC * 2 / 0.005 = 400 * PEC.
            # If PEC ~ 3500 (Raw), is it huge?
            # If Raw is 3500, Gain ~ 1? PEC ~ 3500 Torr??
            # Impossible. Pressure < 760.
            # So Raw Data is NOT Torr. It's Area.
            
            # If Raw Data is Area:
            # We need to know calibration factor to get Pressure?
            # OR we assume linearity.
            # Area_gas_sample = k * Amount_gas_sample.
            # Amount_gas ~ P_gas * Vol_loop / T.
            # So Area ~ P_gas.
            
            # The proportionality constant k cancels out if we are consistent.
            # So we can treat Area as Pressure (Arbitrary Units).
            # But the Constant in Mass Balance depends on Units.
            # PV = PA + PE * (VE/QT) / S.
            # This works if PA and PE are in same Pressure units.
            # BUT PA comes from Liquid Headspace (V_gas + V_liq/S ...).
            # PE comes from Gas Sample.
            # The Headspace Corrections (done above in lines 130-156) convert Raw Area to "Corrected Area equal to Pressure".
            # If so, PAC and PEC are consistent Pressures.
            
            # So the formula `PVC = PAC + PEC * (self.ve_measured / self.qt_measured) / self.solubilities` SHOULD work.
            
            if self.qt_measured > 0:
                ratio = self.ve_measured / self.qt_measured
                # Element-wise calculation
                pvc = pac + (pec * ratio) / self.solubilities
            
            # print(f"DEBUG: Calculated PV (Mass Balance): {pvc}")
        
        # 4. Handle Derived Values (Mass Balance)
        # Check available data (using first gas as sentinel, valid if > 0)
        has_pa = np.any(pac > 0)
        has_pe = np.any(pec > 0)
        has_pv = np.any(pvc > 0)
        
        # We need VE, QT for derivation
        # Note: SHORT.FOR uses VESTAR (Corrected VE) for these calcs?
        # Line 322: VE=VEO*CORR.
        # But for Mass Balance P_v = P_a + V_e*P_e/(Q_t*S).
        # We should use BTPS values presumably.
        ve = self.ve_measured
        qt = self.qt_measured
        
        # Case A: PV Missing (Calculate from PA, PE)
        # SHORT.FOR Line 1010
        if has_pa and has_pe and not has_pv:
            # Pv = Pa + VE * Pe / (QT * S)
            if qt > 0:
                pvc = pac + (ve * pec) / (qt * self.mb_solubilities)
            else:
                pvc = np.zeros_like(pac) # Error state
                
        # Case B: PA Missing (Calculate from PV, PE)
        # SHORT.FOR Line 940
        elif has_pv and has_pe and not has_pa:
            # Pa = Pv - VE * Pe / (QT * S)
            if qt > 0:
                pac = pvc - (ve * pec) / (qt * self.mb_solubilities)
            else:
                pac = np.zeros_like(pvc)
                
        # Case C: PE Missing (Calculate from PA, PV)
        # SHORT.FOR Line 980
        elif has_pv and has_pa and not has_pe:
            # Pe = QT * S * (Pv - Pa) / VE
            if ve > 0:
                pec = (qt * self.mb_solubilities * (pvc - pac)) / ve
            else:
                pec = np.zeros_like(pvc)
        else:
             # print(f"DEBUG: Logic Fallthrough? PA={has_pa}, PE={has_pe}, PV={has_pv}")
             # If all present, do nothing (pac, pec, pvc already set)
             # If multiple missing, we can't solve.
             pass
                
        # 5. R and E (Lines 436-437)
        # R = Pa / Pv
        # E = Pe / Pv
        
        with np.errstate(divide='ignore', invalid='ignore'):
            self.retention = np.divide(pac, pvc)
            self.excretion = np.divide(pec, pvc)
        
        self.retention = np.nan_to_num(self.retention)
        self.excretion = np.nan_to_num(self.excretion)
        
        # 6. Weights (Lines 451-472)
        # Logic from SHORT.FOR (approximate)
        # WT(I) = 1.0/SQRT(VRCE) where VRCE is error variance.
        # Simplified model: Relative error + Min error
        # Line 327: ERRA = (2*RER)^2 + (0.212/PA)^2
        # RER is Relative Error Rate (0.03 typ).
        rer = 0.03
        
        # Weights for Retention (Default 1/Variance)
        # Using a simpler weight for now to avoid complexity: 
        # Weight ~ 1 / (R * 0.03)^2
        # But if R is small, error is dominated by noise floor.
        
        # Let's use uniform weights for now unless precision requires it, 
        # or implement a robust weighting if solution is unstable.
        # SHORT.FOR weighting is quite specific.
        # Let's try to match it roughly:
        # If we use NNLS, we minimize ||W*(Ax-b)||.
        # W = 1/sigma.
        # Sigma ~ 3% of Value + constant noise.
        
        sigma_r = 0.03 * self.retention + 0.001
        self.retention_weights = 1.0 / sigma_r
        
        self.retention_weights = np.nan_to_num(self.retention_weights, nan=0.0, posinf=0.0)

@dataclass
class VQDistribution:
    """
    Stores the output V/Q distribution.
    """
    compartments: int = 50
    blood_flow: np.ndarray = field(default_factory=lambda: np.zeros(50))
    ventilation: np.ndarray = field(default_factory=lambda: np.zeros(50))
    vaq_ratios: np.ndarray = field(default_factory=lambda: np.zeros(50))
    shunt: float = 0.0
    deadspace: float = 0.0
    va_total: float = 0.0 # Alveolar Ventilation
    
    mean_q: float = 0.0
    sd_q: float = 0.0
    mean_v: float = 0.0
    sd_v: float = 0.0
    
    predicted_retention: np.ndarray = field(default_factory=lambda: np.zeros(6))
    predicted_excretion: np.ndarray = field(default_factory=lambda: np.zeros(6))
    
    rss: float = 0.0

from scipy.optimize import nnls
# ... (rest of imports)

class VQModel:
    """
    Implements the multiple inert gas elimination technique (MIGET)
    recovery of V/Q distributions.
    """
    def __init__(self, data: InertGasData, n_compartments=50, z_factor=40.0):
        self.data = data
        self.n = n_compartments
        self.z = z_factor
        
        # Legacy Binning Logic (VQBOHR Lines 314-321)
        # N=50
        # VAQ(1) = 0.0 (Shunt)
        # VAQ(2..49) = Log Spaced from VQLO (0.005) to VQHI (100.0)
        # VAQ(50) = 10000.0 (Deadspace-like)
        
        self.vq_ratios = np.zeros(n_compartments)
        
        # Log spacing for middle compartments
        # Legacy: DVQ=(ALOG(VQHI/VQLO))/RNV where RNV = NVAQS-1 = 49?
        # Wait, Loop J=1,NVAQS: VAQ(J) = VQLO*EXP(DVQ*(TJ-1.0))
        # Then override 1 and 50.
        # This means bins 2..49 follow the curve.
        # Let's replicate this curve first then override.
        
        vq_lo = 0.005 # From screenshot "VQLO= 0.005"
        vq_hi = 100.0 # From screenshot "VQHI= 100.0"
        
        # Calculation from Line 316: RNV = NVAQS - 1 = 49.0
        # DVQ = Log(100/0.005) / 49
        rnv = float(n_compartments - 1)
        dvq = np.log(vq_hi / vq_lo) / rnv
        
        for j in range(n_compartments):
            tj = float(j + 1) # 1-based index
            self.vq_ratios[j] = vq_lo * np.exp(dvq * (tj - 1.0))
            
        # Overrides
        self.vq_ratios[0] = 0.0 # Shunt
        # self.vq_ratios[-1] = 10000.0 # Deadspace (Removed as per user request)
        
    def solve(self, weight_mode: str = "retention") -> VQDistribution:
        """
        Recover the V/Q distribution using Ridge Regression with non-negativity constraint.
        Legacy: SMOOTH subroutine.
        
        Args:
            weight_mode (str): "retention" (default) or "excretion".
        """
        fit_r = (weight_mode.lower() == "retention")
        
        # 1. Setup Kernel Matrix A
        # If fit_r: Solve for Q (indices 0..48). Shunt (0) included. Deadspace (49) excluded.
        # If fit_e: Solve for V (indices 1..49). Shunt (0) excluded. Deadspace (49) included.
        
        n_solve_start = 0
        n_solve_end = self.n - 1 # Default 0..48
        
        if not fit_r:
            n_solve_start = 1
            n_solve_end = self.n # 1..49
            
        n_opt = n_solve_end - n_solve_start
        n_gases = len(self.data.gas_names)
        A = np.zeros((n_gases, n_opt))
        
        pcs = self.data.solubilities
        
        for i in range(n_gases):
            pc = pcs[i]
            for idx_opt in range(n_opt):
                j = n_solve_start + idx_opt
                vaq = self.vq_ratios[j]
                
                if pc + vaq == 0:
                     val = 0
                else:
                    val = pc / (pc + vaq)
                A[i, idx_opt] = val
                
        # 2. Setup Measurement Vector b and Weights
        if fit_r:
            b = self.data.retention
            weights = self.data.retention_weights
        else:
            b = self.data.excretion
            weights = self.data.excretion_weights
        
        # 3. Weighting
        A_w = A * weights[:, np.newaxis]
        b_w = b * weights
        
        # 4. Regularization (Smoothing Z)
        # Apply smoothing to Exchanging Compartments (1..48).
        # We need to filter indices in our unknown vector x that map to 1..48.
        
        exch_indices = []
        for idx_opt in range(n_opt):
            j = n_solve_start + idx_opt
            if j > 0 and j < self.n - 1:
                exch_indices.append(idx_opt)
                
        reg_size = len(exch_indices)
        if reg_size > 2:
            # Create D for exchanging block
            D_exch = np.zeros((reg_size - 2, n_opt))
            for k in range(reg_size - 2):
                 idx1 = exch_indices[k]
                 idx2 = exch_indices[k+1]
                 idx3 = exch_indices[k+2]
                 D_exch[k, idx1] = 1.0
                 D_exch[k, idx2] = -2.0
                 D_exch[k, idx3] = 1.0
                 
            reg_matrix = np.sqrt(self.z) * D_exch
            
            # Augment
            A_aug = np.vstack([A_w, reg_matrix])
            b_aug = np.concatenate([b_w, np.zeros(reg_size - 2)])
        else:
             A_aug = A_w
             b_aug = b_w
        
        # 5. Solve NNLS
        x_sol, residual = nnls(A_aug, b_aug)
        
        # 6. Reconstruct Full Vector (Q or V)
        full_dist = np.zeros(self.n)
        # Place x_sol into correct indices
        for idx_opt in range(n_opt):
            j = n_solve_start + idx_opt
            full_dist[j] = x_sol[idx_opt]
            
        # 7. Derive Q and V distributions
        if fit_r:
            # Solved for Q
            q_dist = full_dist
            # V = Q * VAQ
            v_abs = q_dist * self.vq_ratios
            
            # Normalize
            if np.sum(q_dist) > 0: q_dist_norm = q_dist / np.sum(q_dist)
            else: q_dist_norm = q_dist
            
            if np.sum(v_abs) > 0: v_dist_norm = v_abs / np.sum(v_abs)
            else: v_dist_norm = v_abs
            
        else:
            # Solved for V
            v_dist = full_dist
            # Q = V / VAQ (Handle /0)
            q_abs = np.zeros_like(v_dist)
            for j in range(self.n):
                if self.vq_ratios[j] > 0:
                    q_abs[j] = v_dist[j] / self.vq_ratios[j]
                elif j == 0 and fit_r == False:
                     q_abs[j] = 0.0
            
            # Normalize
            if np.sum(v_dist) > 0: v_dist_norm = v_dist / np.sum(v_dist)
            else: v_dist_norm = v_dist
            
            if np.sum(q_abs) > 0: q_dist_norm = q_abs / np.sum(q_abs)
            else: q_dist_norm = q_abs

        # 8. Deadspace / Shunt Calculation (Post-hoc for display)
        
        deadspace_est = 0.0
        va_total = 0.0
        qt = self.data.qt_measured
        ve = self.data.ve_measured
        
        if fit_r:
             # Calculate derived Deadspace
             if qt > 0 and ve > 0:
                va_total = qt * np.sum(q_dist_norm * self.vq_ratios) # Derived VA
                if ve > va_total: deadspace_est = (ve - va_total) / ve
        else:
             # Fit E: Deadspace is explicitly in v_dist_norm[49].
             deadspace_est = v_dist_norm[-1] # Index 49
             if ve > 0:
                 va_total = ve * (1.0 - deadspace_est)
        
        # 9. Moments
        self.q_mean, self.q_sd = self._calc_moments(q_dist_norm, self.vq_ratios)
        self.v_mean, self.v_sd = self._calc_moments(v_dist_norm, self.vq_ratios)
        
        # 10. Predicted R and E (For plotting fit)
        pred_r = np.zeros(n_gases)
        pred_e = np.zeros(n_gases)
        
        for i in range(n_gases):
            pc = pcs[i]
            # Kernel vector for this gas
            kernel_r = np.zeros(self.n)
            kernel_e = np.zeros(self.n)
            
            for j in range(self.n):
                vaq = self.vq_ratios[j]
                if pc + vaq == 0:
                    r_val = 0
                else:
                    r_val = pc / (pc + vaq)
                
                kernel_r[j] = r_val
                kernel_e[j] = r_val * vaq # Partial Excretion Kernel (Before QT/VE scaling)
            
            # Pred R = Sum (Q_i * Kernel_i)
            pred_r[i] = np.sum(kernel_r * q_dist_norm)
            
            # Pred E = Sum (V_i * Kernel_i)
            if ve > 0:
                pred_e[i] = np.sum(kernel_e * q_dist_norm) * (qt / ve)
        
        # 11. RSS
        if fit_r:
             residuals = (self.data.retention - pred_r) * self.data.retention_weights
        else:
             residuals = (self.data.excretion - pred_e) * self.data.excretion_weights
             
        rss_val = np.sum(residuals**2)
        
        return VQDistribution(
            compartments=self.n,
            blood_flow=q_dist_norm,
            ventilation=v_dist_norm,
            vaq_ratios=self.vq_ratios,
            shunt=q_dist_norm[0],
            deadspace=deadspace_est,
            va_total=va_total,
            mean_q=self.q_mean, 
            sd_q=self.q_sd,
            mean_v=self.v_mean,
            sd_v=self.v_sd,
            predicted_retention=pred_r,
            predicted_excretion=pred_e,
            rss=rss_val
        )

    def _calc_moments(self, dist, ratios):
        mask = (ratios > 0) & (dist > 0)
        if not np.any(mask):
            return 0.0, 0.0
            
        valid_dist = dist[mask]
        valid_log_ratios = np.log(ratios[mask]) # Natural Log
        
        sum_w = np.sum(valid_dist)
        if sum_w == 0: return 0.0, 0.0
        
        # Mean
        mean_log = np.sum(valid_dist * valid_log_ratios) / sum_w
        
        # SD (2nd Moment around mean)
        var_log = np.sum(valid_dist * (valid_log_ratios - mean_log)**2) / sum_w
        sd_log = np.sqrt(var_log)
        
        return np.exp(mean_log), sd_log


    def get_continuous_curves(self, dist, n_points=100):
        """
        Calculates smooth R and E curves for plotting based on the recovered Q distribution.
        Also calculates Homogeneous lung curves.
        """
        solubilities = np.logspace(-3, 3, n_points)
        pred_r_curve = np.zeros(n_points)
        pred_e_curve = np.zeros(n_points)
        
        qt = self.data.qt_measured
        ve = self.data.ve_measured
        
        q_dist_norm = dist.blood_flow
        
        # Calculate Homogeneous Parameters
        # Overall VA/Q of the exchanging lung (excluding shunt/deadspace behavior from ratio, but accounting for their loss)
        # VA_total = VE * (1 - deadspace using inert gas data)
        # Q_perf = QT * (1 - shunt)
        # Ratio_homo = VA_total / Q_perf
        
        # Note: dist.deadspace is fractional. dist.shunt is fractional.
        # But wait, Q_dist includes shunt at index 0. 
        # If we say 'Homogeneous', do we mean the 'ideal' curve for the *perfusion* part?
        # Usually Homogeneous R curve includes shunt? 
        # "Retentions Homogeneous Lung": Dashed blue line. 
        # If Shunt was high, would Homogeneous line start high? 
        # Usually it's the "Ideal" line, implying Shunt=0, Deadspace=0? 
        # OR Shunt=0, Deadspace=Measured?
        # Given E plateau, Deadspace is definitely included in E.
        # If R starts at 0 in screenshot despite small shunt, maybe Shunt is excluded?
        # Let's assume Shunt=0, Deadspace=dist.deadspace for Homogeneous.
        
        shunt = dist.shunt
        deadspace = dist.deadspace
        
        if qt > 0 and (1 - shunt) > 0:
             q_perf = qt * (1 - shunt)
             # VA = VE - VD_absolute. 
             # VD_absolute = VE * deadspace. 
             va = ve * (1 - deadspace)
             
             if q_perf > 0:
                 homo_ratio = va / q_perf
             else:
                 homo_ratio = 1.0 # Fallback
        else:
             homo_ratio = 1.0

        homo_r_curve = np.zeros(n_points)
        homo_e_curve = np.zeros(n_points)

        for i in range(n_points):
            pc = solubilities[i]
            
            # 1. Predicted Best Fit (Existing Logic)
            sum_r = 0.0
            sum_e_kernel = 0.0
            
            for j in range(self.n):
                vaq = self.vq_ratios[j]
                if pc + vaq == 0:
                    r_val = 0
                else:
                    r_val = pc / (pc + vaq)
                    
                # Retention: Sum(Q * R)
                # Shunt (j=0, vaq=0) -> R=1.
                if j == 0: r_val = 1.0
                
                sum_r += q_dist_norm[j] * r_val
                
                if vaq > 0: 
                    sum_e_kernel += q_dist_norm[j] * r_val * vaq
                    
            pred_r_curve[i] = sum_r
            if ve > 0:
                pred_e_curve[i] = sum_e_kernel * (qt / ve)
            
            # 2. Homogeneous Curve
            # R_homo = pc / (pc + homo_ratio)
            # This is "Ideal Compartment R".
            # BUT R is defined as Pa_curr / Pv_curr?
            # Overall R for homogeneous lung (with Shunt=0) is just the compartment R.
            # If we keep Deadspace:
            # E_homo = R_homo * (1 - deadspace)
            
            if pc + homo_ratio == 0:
                r_homo = 0
            else:
                r_homo = pc / (pc + homo_ratio)
            
            homo_r_curve[i] = r_homo
            homo_e_curve[i] = r_homo * (1 - deadspace) # Simple relationship if single compartment
                
        return solubilities, pred_r_curve, pred_e_curve, homo_r_curve, homo_e_curve

class VQForwardModel:
    """
    Implements the Forward (Direct) Problem:
    Given a defined V/Q distribution, calculate the resulting R and E values.
    """
    def __init__(self, n_compartments=50):
        self.n = n_compartments
        
        # Standard Axis Setup (Same as VQModel)
        self.vq_ratios = np.zeros(n_compartments)
        vq_lo = 0.005
        vq_hi = 100.0
        rnv = float(n_compartments - 1)
        dvq = np.log(vq_hi / vq_lo) / rnv
        
        for j in range(n_compartments):
            tj = float(j + 1)
            self.vq_ratios[j] = vq_lo * np.exp(dvq * (tj - 1.0))
            
        self.vq_ratios[0] = 0.0 # Shunt
        # self.vq_ratios[-1] = 10000.0 # Standard logic uses the log curve up to 100
        
    def generate_distribution(self, shunt_pct, deadspace_pct, 
                            q1_mean, q1_sd,
                            q2_mean=None, q2_sd=None, q2_flow_pct=0.0):
        """
        Generates a V/Q distribution based on log-normal parameters.
        Supports Bimodal Q distribution.
        Ventilation is DERIVED from Q * Ratio.
        """
        # 1. Normalize percentages
        shunt_f = shunt_pct / 100.0
        deadspace_f = deadspace_pct / 100.0
        
        remaining_flow = 1.0 - shunt_f
        # Note: We don't normalize V to remaining_vent here yet, 
        # we calculate absolute V then normalize later.
        
        # 2. Generate Log-Normal shapes for the main body (bins 1..49)
        # We work in Natural Log domain
        
        # Mode 1
        q1_mu = np.log(q1_mean)
        q1_sigma = q1_sd
        
        # Mode 2
        if q2_mean is not None and q2_flow_pct > 0:
            q2_mu = np.log(q2_mean)
            q2_sigma = q2_sd
            q2_fraction_of_body = q2_flow_pct / 100.0
            # Wait, interpretation of "Ratio of TOTAL blood flow in secondary to main mode"?
            # Or "Ratio of flow"?
            # Book says "QRATIO=0.05 (ratio of TOTAL blood flow in secondary to main mode)"
            # That implies Flow2 / Flow1 = 0.05 ? Or Flow2 / TotalFlow ?
            # "ratio of secondary to main" usually means F2/F1.
            # So F2 = R * F1.
            # Total F = F1 + F2 = F1 (1 + R).
            # So F1_share = 1 / (1+R). F2_share = R / (1+R).
            
            # Let's assume input is just "Percentage of Flow in Mode 2" for simplicity in UI,
            # or strictly follow "QRATIO" if user wants EXACT match.
            # User inputs "Flow Split (%)" in plan. Let's interpret as % of Q_body.
            f2 = q2_flow_pct / 100.0
            f1 = 1.0 - f2
        else:
            f1 = 1.0
            f2 = 0.0
        
        q_dist = np.zeros(self.n)
        
        # Iterate bins 1 to N-1 (Exclude Shunt)
        for j in range(1, self.n):
            x = self.vq_ratios[j]
            if x > 0:
                log_val = np.log(x)
                
                # Q Mode 1 PDF
                val1 = 0.0
                if q1_sigma > 0:
                    denom1 = x * q1_sigma * np.sqrt(2 * np.pi)
                    num1 = np.exp( - (log_val - q1_mu)**2 / (2 * q1_sigma**2) )
                    val1 = num1 / denom1
                    
                # Q Mode 2 PDF
                val2 = 0.0
                if f2 > 0 and q2_mean is not None and q2_sd > 0:
                     denom2 = x * q2_sd * np.sqrt(2 * np.pi)
                     num2 = np.exp( - (log_val - q2_mu)**2 / (2 * q2_sd**2) )
                     val2 = num2 / denom2
                
                # Combine
                q_dist[j] = f1 * val1 + f2 * val2

        # Normalize QMain to remaining_flow (Total Body Flow)
        sum_q_main = np.sum(q_dist)
        if sum_q_main > 0:
            q_dist = q_dist * (remaining_flow / sum_q_main)
            
        # Add Shunt
        q_dist[0] = shunt_f
        
        # 3. Derive Ventilation (V = Q * Ratio)
        # Consistent V/Q theory for Direct Problem
        v_dist = q_dist * self.vq_ratios
        
        # Normalize V Distribution
        # The sum of V_dist represents the alveolar ventilation associated with perfusion.
        # But we also have Deadspace (Pure V, Q=0).
        # We define Deadspace as a PERCENTAGE of Total Ventilation.
        # V_total = V_alveolar + V_deadspace.
        # V_deadspace = V_total * deadspace_f.
        # V_alveolar = V_total * (1 - deadspace_f).
        
        # So sum(v_dist) should correspond to (1 - deadspace_f) fraction.
        
        sum_v_alv = np.sum(v_dist)
        
        if sum_v_alv > 0:
            # Scale v_dist so it sums to exactly (1 - deadspace_f)
            scale_v = (1.0 - deadspace_f) / sum_v_alv
            v_dist = v_dist * scale_v
            
        # Now v_dist sums to (1-ds).
        # Deadspace is separate (handled in R/E calc implicitly or needs explicit bin?).
        # For plotting, maybe we want to see it? 
        # But deadspace has Ratio=Infinity. Can't put on plot easily.
        
        return q_dist, v_dist, deadspace_f
        
        return q_dist, v_dist_target, deadspace_f
        
    def calculate_re(self, q_dist, v_dist, deadspace_f, solubilities):
        """
        Calculate R and E vectors for given solubilities.
        For Direct Problem.
        """
        n_gases = len(solubilities)
        retentions = np.zeros(n_gases)
        excretions = np.zeros(n_gases)
        
        # Total V for normalization
        # v_dist sums to (1 - deadspace_f). 
        # Total ventilation = 1.0 (Unit normalized).
        
        for i in range(n_gases):
            sol = solubilities[i]
            
            r_sum = 0.0
            e_sum = 0.0
            
            for j in range(self.n):
                ratio = self.vq_ratios[j]
                
                if sol + ratio == 0:
                    term = 0 # R_j
                else:
                    term = sol / (sol + ratio)
                    
                # Retention (Weighted by Q)
                # Shunt (Ratio 0) -> R=1.
                if j == 0: term = 1.0
                
                r_sum += q_dist[j] * term
                
                # Excretion (Weighted by V)
                # Deadspace shouldn't be in v_dist loop?
                e_sum += v_dist[j] * term
                
            retentions[i] = r_sum
            excretions[i] = e_sum
            
        return retentions, excretions

# Add this method to VQModel also or share it? 
# VQModel is for Inverse. VQForwardModel is for Direct.
# The user wants smooth curves for Inverse Problem.
# So we need a method in VQModel to calculate R/E for arbitrary solubilities given a distribution.

def calculate_continuous_re(dist, vq_ratios, qt, ve, n_points=100):
   """
   Calculates smooth R and E curves for plotting for the Inverse Problem results.
   """
   # Sweep solubilities from 0.001 to 1000
   solubilities = np.logspace(-3, 3, n_points)
   pred_r_curve = np.zeros(n_points)
   pred_e_curve = np.zeros(n_points)
   
   n_compartments = len(dist)
   
   for i in range(n_points):
       pc = solubilities[i]
       
       sum_r = 0.0
       sum_e_kernel = 0.0
       
       for j in range(n_compartments):
           vaq = vq_ratios[j]
           if pc + vaq == 0:
               r_val = 0
           else:
               r_val = pc / (pc + vaq)
               
           # Retention: Sum(Q * R)
           # Shunt (j=0, vaq=0) -> R=1
           if j == 0: r_val = 1.0
           
           sum_r += dist[j] * r_val
           
           # Partial Excretion Kernel: R * VAQ
           # For Deadspace? Deadspace is separate in Inverse Model usually: 
           # E = (PE/PV) = ...
           # Here we are predicting E from Recovered Q.
           # E_pred = Sum(Q * VAQ * R) * (QT/VE)
           if vaq > 0: # Shunt contributes 0 to excretion
               sum_e_kernel += dist[j] * r_val * vaq
               
       pred_r_curve[i] = sum_r
       if ve > 0:
           pred_e_curve[i] = sum_e_kernel * (qt / ve)
           
   return solubilities, pred_r_curve, pred_e_curve
