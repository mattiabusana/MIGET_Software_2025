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
    gas_names: List[str] = field(default_factory=lambda: ["SF6", "Ethane", "Cyclopropane", "Enflurane", "Ether", "Acetone"])
    partition_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(6)) # PC at bath temp
    solubilities: np.ndarray = field(default_factory=lambda: np.zeros(6))     # Blood-Gas partition coeff at body temp? Or S?
    
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
        # Requires specific slopes for Human, Dog, Horse (SHORT.FOR data statements).
        # For now, simplistic implementation assuming Human if not specified.
        hum_slo = np.array([2950., 1374., 2025., 3016., 4066., 805.])
        # Using simplified Van't Hoff eq for now:
        # PCFACT(I) = EXP(SLP * (1/(TEMPB+273) - 1/(TBATH+273)))
        tt1 = 1.0 / (self.temp_body + 273.0)
        tt2 = 1.0 / (self.temp_bath + 273.0)
        
        # Assuming gases are sorted as in legacy: SF6, Ethane, Cyclo, Enflurane, Ether, Acetone
        # We should probably make this configurable.
        pcfact = np.exp(hum_slo[:n_gases] * (tt1 - tt2))
        pc_body = self.partition_coeffs * pcfact
        
        # 2. Solubility (Line 364)
        # FACT = (PBSEA - SVPB)/100 ? No wait, SHORT loops over runs.
        # Line 313: FACT=(PBSEA-SVPB)/100.0  where SVPB is SVP at Body Temp.
        svpb = 47.0 # Approximation or calc? SHORT calculates it:
        # BEXP=9.0578-2290.5/(273.15+TEMPB); SVPB=10.0**BEXP
        bexp = 9.0578 - 2290.5 / (273.15 + self.temp_body)
        svpb = 10.0 ** bexp
        fact = (self.pb_sea - svpb) / 100.0
        
        self.solubilities = pc_body / fact # S(I)
        
        # 3. Corrected Pressures
        # EXPFAC=(PB-SVPB)/(PBSEA-SVPB)
        exp_fac = (self.pb_sea - svpb) / (self.pb_sea - svpb) # Wait, uses PB vs PBSEA.
        # Usually PB is station pressure.
        # Assuming PB=PBSEA for now if not separate.
        
        # PAC, PEC, PVC calculation (Lines 365-372)
        # PEC(I)=GE(I)*PE(I)*EXPFAC
        # PAC(I) = GA(I)*PA(I)*(...corrections for machine volumes...)
        # Simplified:
        pac = self.ga * self.pa_raw
        pec = self.ge * self.pe_raw
        pvc = self.gv * self.pv_raw
        
        # Handle "Derived" values (Lines 940+)
        # If PA is missing but PV and PE exist...
        
        # 4. R and E (Lines 436-437)
        # R = Pa / Pv
        # E = Pe / Pv
        with np.errstate(divide='ignore', invalid='ignore'):
            self.retention = np.divide(pac, pvc)
            self.excretion = np.divide(pec, pvc)
            
        self.retention = np.nan_to_num(self.retention)
        self.excretion = np.nan_to_num(self.excretion)
        
        # 5. QT Calculation (Lines 438-447)
        # QTCALC(I) = VESTAR*E(I)/(PC(I)*(1.0-R(I))) 
        # VESTAR is VE corrected to STPD/BTPS?
        # VE = VEO * CORR
        
        # 6. Weights (Lines 451-472)
        # Complex error propagation.
        # For now, set uniform weights or based on crude error model.
        self.retention_weights = np.ones(n_gases) # Placeholder
        self.excretion_weights = np.ones(n_gases) # Placeholder

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
    
    mean_q: float = 0.0
    sd_q: float = 0.0
    mean_v: float = 0.0
    sd_v: float = 0.0

from scipy.optimize import nnls

class VQModel:
    """
    Implements the multiple inert gas elimination technique (MIGET)
    recovery of V/Q distributions.
    """
    def __init__(self, data: InertGasData, n_compartments=50, z_factor=40.0):
        self.data = data
        self.n = n_compartments
        self.z = z_factor
        
        # Log-spaced V/Q ratios from 0.005 to 100.0 (Standard)
        # Legacy uses VQLO=0.01, VQHI=100.0?
        # VQBOHR: VAQ(J) = VQLO*EXP(DVQ*(TJ-1.0))
        self.vq_ratios = np.logspace(np.log10(0.005), np.log10(100.0), n_compartments)
        self.vq_ratios[0] = 0.0 # Shunt
        # Note: Legacy usually has explicit Shunt (0) and Deadspace (inf) compartments
        # handled separately or as indices 1 and N.
        # Here we include Shunt as vq=0.
        # Deadspace is usually derived or handled as separate constraint?
        
    def solve(self, weight_by_retention=True) -> VQDistribution:
        """
        Recover the V/Q distribution using Ridge Regression with non-negativity constraint.
        Legacy: SMOOTH subroutine.
        """
        # 1. Setup Kernel Matrix A
        # A_ij = PC_i / (PC_i + VAQ_j) for Retention
        # A_ij = VAQ_j / (PC_i + VAQ_j) for Excretion ? No, E = R * VAQ/PC? 
        # Legacy SMOOTH uses: A(J,I) = PC(I) / (PC(I) + VAQ(J))
        
        n_gases = len(self.data.gas_names)
        A = np.zeros((n_gases, self.n))
        
        # We need to solve for Q (Blood Flow Distribution)
        # Measured R_i = Sum_j [ Q_j * PC_i / (PC_i + VAQ_j) ]
        
        pcs = self.data.partition_coeffs # Or Solubilities?
        # SHORT.FOR: S(I)=PCBODY(I)/FACT. 
        # But VQBOHR SMOOTH uses PC(I).
        
        for i in range(n_gases):
            pc = pcs[i]
            for j in range(self.n):
                vaq = self.vq_ratios[j]
                # Avoid div by zero
                if pc + vaq == 0:
                    val = 0
                else:
                    val = pc / (pc + vaq)
                A[i, j] = val
                
        # 2. Setup Measurement Vector b
        b = self.data.retention
        
        # 3. Weighting (Measurement Error)
        # Legacy applies weights to rows of A and b?
        # "DATA(I)=DATA(I)*WEIGHT(I)" (Line 639)
        # Yes, weighted least squares.
        weights = self.data.retention_weights
        # Apply weights
        A_w = A * weights[:, np.newaxis]
        b_w = b * weights
        
        # 4. Regularization (Smoothing Z)
        # Legacy: WT(J) = SQRT(Z * (1 + (QT/VE * VAQ)^2))
        # This penalizes high V/Q components? or smooths them?
        # It's added to the diagonal of A'A.
        # Equivalent to augmenting A with Diagonal(Wt).
        
        qt_ve_ratio = 1.0 # Default if not known
        if self.data.ve_measured > 0 and self.data.qt_measured > 0:
            qt_ve_ratio = self.data.qt_measured / self.data.ve_measured
            
        reg_weights = np.sqrt(self.z * (1.0 + (qt_ve_ratio * self.vq_ratios)**2))
        reg_matrix = np.diag(reg_weights)
        
        # Augment A and b
        A_aug = np.vstack([A_w, reg_matrix])
        b_aug = np.concatenate([b_w, np.zeros(self.n)])
        
        # 5. Solve NNLS
        q_dist, residual = nnls(A_aug, b_aug)
        
        # 6. Normalize
        # q_dist should sum to 1 (fractional) or QT?
        # Legacy seems to treat it as fractional then scales?
        total_q = np.sum(q_dist)
        if total_q > 0:
            q_dist_norm = q_dist / total_q
        else:
            q_dist_norm = q_dist
            
        # 7. Calculate Ventilation Distribution
        # V_j = Q_j * VAQ_j * (VE/QT)?
        # V = Q * VAQ * (QT/VE ratio for global scaling?)
        # Simply V_j = Q_j * VAQ_j (up to a scaling factor)
        v_dist = q_dist_norm * self.vq_ratios
        total_v = np.sum(v_dist)
        if total_v > 0:
            v_dist_norm = v_dist / total_v
        else:
            v_dist_norm = v_dist
            
        # Create Result Object
        dist = VQDistribution(
            compartments=self.n,
            blood_flow=q_dist_norm,
            ventilation=v_dist_norm,
            vaq_ratios=self.vq_ratios,
            shunt=q_dist_norm[0]
        )
        
        return dist
