
import numpy as np

# Constants from SHORT.FOR
HUMSLO = np.array([2950., 1374., 2025., 3016., 4066., 805.])
SOL_SF6_TEMP_SLOPE = 2950.

# Data from MISSISSIPPI (Run 1)
# Line 2: 1 752.0 37.0 38.0 0.03 0.0030
pb_sea = 752.0
elect_temp = 37.0
bath_temp = 38.0

# Line 4: PC
pc_bath = np.array([0.658E-02, 0.132E+00, 0.749E+00, 0.293E+01, 0.123E+02, 0.317E+03])
# Note: Is order SF6, Ethane, Cyclo, Enflurane, Ether, Acetone?
# Line 419 SHORT.FOR: SF6, ETHANE, CYCLO, ENFLURANE, ETHER, ACETONE.
# Matches.

# Line 5: PA (Arterial Raw)
pa_raw = np.array([80.8, 322.5, 1108.0, 990.0, 873.5, 522.4])

# Line 7: PE (Expired Raw, Line 7 of Run loop is PE?) 
# Wait, SHORT.FOR Loop:
# Line 5: PA
# Line 6: GA
# Line 7: PE
# Line 8: GE
# Line 9: PV
pe_raw = np.array([3588.0, 1320.0, 1167.0, 528.5, 420.8, 218.1])

# Line 9: PV (Venous Raw)
pv_raw = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Line 11: VE, QT, PB, TEMPB, TEMPR, VGA, VBA, VHA, VGV, VBV, VHV
# 14.40  7.40  752.0  38.7  38.7  10.04   8.55   0.61   0.00   1.00   0.00
ve = 14.40
qt = 7.40
pb = 752.0
temp_body = 38.7
temp_rot = 38.7
vga = 10.04
vba = 8.55
vha = 0.61
vgv = 0.00
vbv = 1.00
vhv = 0.00

# Params
ga = 1.0
ge = 1.0
gv = 1.0

# --- calculation ---

# 1. Temperature Correction (PCFACT)
# IF(IBATH.EQ.0) PCFACT=1.0. Assuming we check both cases.
# IF(ISPEC.EQ.1) (Man)
# TT1 = 1/(TEMPB+273)
# TT2 = 1/(TBATH+273)
tt1 = 1.0 / (temp_body + 273.0)
tt2 = 1.0 / (bath_temp + 273.0)
slopes = HUMSLO
pcfact = np.exp(slopes * (tt1 - tt2))
print("PCFACT (Temp Corr):", pcfact)

# PC Body
pc_body = pc_bath * pcfact

# 2. PAPV Correction
# IF IPAPV=1 (Arterial), PAPV=1.0
# IF IPAPV=2 (Venous), PAPV=0.95
papv_art = 1.0
papv_ven = 0.95

# 3. Headspace Correction (Arterial)
# PAC = GA * PA * (1 + VHA/VBA + VGA/(VBA*PC)) / (PCFACT * PAPV)
# Note: Divides by PCFACT.
def calc_pac(papv_val, apply_pcfact=True):
    pac_f = np.ones(6)
    if vba > 0 and vga > 0:
        term1 = vha / vba
        term2 = vga / (vba * pc_bath) # Uses PC Bath?
        pac_f = 1.0 + term1 + term2
    
    denom = papv_val
    if apply_pcfact:
        denom = denom * pcfact
        
    return ga * pa_raw * pac_f / denom

# 4. PV Calculation
# PV Raw is 0. So derived.
# Pv = Pa + VE*Pec / (QT * S)
# S = PCBODY (usually PCBODY/Fact, but Fact=(PB-SVPB)/100? No, S=PCBODY in Mass Balance line 505)
# Wait, SHORT.FOR Line 505: PAC(I) = PVC(I) - ... -> PVC = PAC + VE*PEC /(QT*PCBODY)
# Yes.
# PEC = GE * PE * EXPFAC
expfac = (pb - 47.0)/(760.0 - 47.0) # Approx
expfac = 1.0 # Assuming simple
pec = ge * pe_raw * expfac

def calculate_scenario(name, apply_temp, papv_val):
    pac = calc_pac(papv_val, apply_temp)
    
    # Calculate PVC (Derived)
    pvc = pac + (ve * pec) / (qt * pc_body)
    
    r = pac / pvc
    e = pec / pvc
    
    print(f"\n--- Scenario: {name} ---")
    print(f"PAC (Gas 1): {pac[0]:.4f}")
    print(f"PEC (Gas 1): {pec[0]:.4f}")
    print(f"PVC (Gas 1 derived): {pvc[0]:.4f}")
    print(f"R   (Gas 1): {r[0]:.5f}")
    print(f"Legacy Target: 0.01349")

# Run Scenarios
calculate_scenario("No Temp Corr, Arterial", False, 1.0)
calculate_scenario("Temp Corr, Arterial", True, 1.0)
calculate_scenario("No Temp Corr, Venous", False, 0.95)
calculate_scenario("Temp Corr, Venous", True, 0.95)

