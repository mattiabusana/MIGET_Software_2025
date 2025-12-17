import numpy as np
from dataclasses import dataclass

@dataclass
class BloodGasParams:
    hb: float = 15.0      # Hemoglobin (g/dL)
    hcrit: float = 45.0   # Hematocrit (%)
    temp: float = 37.0    # Temperature (C)
    dp50: float = 0.0     # P50 shift (Torr)
    sol_o2: float = 0.003 # O2 Solubility
    
    # Buffering params (from VQBOHR common block OXY1/2)?)
    aph: float = 0.0 # ??? Need to reverse engineer usage
    bph: float = 0.0
    
def calculate_ph(pco2: float, base_excess_c: float = 0.0, params: BloodGasParams = None) -> float:
    """
    Port of FUNCTION PH(PCO2, Y) from VQBOHR.FOR.
    Legacy code uses complex logic with APH, BPH variables which seem to be 
    related to buffer lines.
    
    Y passed to legacy PH seems to be related to O2 saturation effect on pH (Haldane).
    
    Legacy:
    PH=7.59 + Y - 0.2741*ALOG(PCO2/20.0)
    
    Where Y = 0.003*HB*(1.0-SATURA/100.0) in some calls.
    Or Y is iterating variable.
    """
    # Simplying for standard Kelman/Severinghaus for now if unable to decipher APH/BPH dynamic logic instantly.
    # However, to maintain fidelity, let's try to match the legacy formula structure.
    # The legacy PH function seems to interpolate between two buffer lines if APH/BPH are set?
    
    # Placeholder standard Henderson-Hasselbalch or similar
    # pH = 6.1 + log([HCO3-]/ (0.03*PCO2))
    # But legacy uses a linear log-pco2 relation (ASTRUP).
    
    ph = 7.59 - 0.2741 * np.log(pco2 / 20.0) 
    # Add Base Excess / Haldane effect correction if Y provided?
    return ph

def calculate_saturation(po2: float, pco2: float, ph: float, params: BloodGasParams) -> float:
    """
    Port of FUNCTION SATURA(PO2,PCO2,PHE) from VQBOHR.FOR.
    Standard Kelman subroutine likely.
    
    Legacy coeffs:
    A1=-8532.229
    A2=2121.401
    A3=-67.07399
    A4=935960.9
    A5=-31346.26
    A6=2396.167
    A7=-67.10441
    
    Virtual PO2 calculation:
    X = PO2 * 10**(0.024*(37-T) + 0.4*(pH-7.4) + 0.06*log(40/PCO2))
    """
    # Kelman Coefficients
    a1 = -8532.229
    a2 = 2121.401
    a3 = -67.07399
    a4 = 935960.9
    a5 = -31346.26
    a6 = 2396.167
    a7 = -67.10441

    if po2 < 0: po2 = 0
    
    # Virtual PO2 (correction for temp, pH, PCO2)
    # Note: Legacy p50 correction: X = 26.8 * X / (26.8 + DP50)
    
    temp_diff = 37.0 - params.temp
    ph_diff = ph - 7.4
    pco2_term = 0.06 * np.log10(40.0 / pco2) if pco2 > 0 else 0
    
    x = po2 * (10.0 ** (0.024 * temp_diff + 0.4 * ph_diff + pco2_term))
    
    # P50 shift
    # Standard P50 is 26.8 in this model?
    x = 26.8 * x / (26.8 + params.dp50)
    
    if x < 10.0:
        sat = 0.003683 * x + 0.000584 * x * x
    else:
        # Kelman equation
        num = x * (x * (x * (x + a3) + a2) + a1)
        den = x * (x * (x * (x + a7) + a6) + a5) + a4
        sat = num / den
        
    return sat * 100.0 # Return percentage

def calculate_co2_content(pco2: float, ph: float, saturation: float, params: BloodGasParams) -> float:
    """
    Port of FUNCTION CO2CON from VQBOHR.FOR
    Calculates total CO2 content (ml/100ml) in whole blood.
    """
    # Legacy logic:
    # P = 7.4 - PH
    # PK = 6.086 + 0.042*P + (38-T)*(0.00472+0.00139*P)
    # SOL = ...
    
    p = 7.4 - ph
    temp_diff = 37.0 - params.temp # Legacy uses 38?? "38.0-TEMP" in line 1006. Let's stick to legacy.
    # Wait, code says (38.0-TEMP).
    
    pk = 6.086 + 0.042 * p + (38.0 - params.temp) * (0.00472 + 0.00139 * p)
    
    # Solubility of CO2
    sol = 0.0307 + 0.00057 * temp_diff + 0.00002 * temp_diff**2
    
    # Oxygenation effect (Haldane)
    # DOX (Deoxygenated?) DR (Oxygenated? R-state?)
    dox = 0.59 + 0.2913 * p - 0.0844 * p * p
    dr = 0.664 + 0.2275 * p - 0.0938 * p * p
    
    # Interpolate based on saturation
    # DDD = DOX + (DR-DOX)*(1.0 - SAT/100) -- Wait, standard is:
    # Legacy line 1010: DDD=DOX+(DR-DOX)*(1.-SATN/100.0)
    # If SAT=100, DDD=DOX. If SAT=0, DDD=DR. 
    # Usually Deoxy Hb binds more CO2 (Haldane). So DR should be Deoxy?
    # DR > DOX usually? 0.664 > 0.59.
    # So if Sat=0 (Deoxy), we add full difference. Yes.
    
    ddd = dox + (dr - dox) * (1.0 - saturation / 100.0)
    
    # Plasma CO2 Content
    # CP = SOL * PCO2 * (1 + 10**(PH-PK))
    # This includes Dissolved + Bicarbonate
    cp = sol * pco2 * (1.0 + 10.0 ** (ph - pk))
    
    # Cell/Plasma distribution?
    # CCC = DDD * CP
    # CO2CON = (Hct*CCC*0.01 + (1-Hct*0.01)*CP) * 2.22
    # 2.22 converts mEq/L or mmol/L to ml/100ml? 
    # 22.2 ml/mmol roughly.
    
    ccc = ddd * cp
    hct_frac = params.hcrit / 100.0
    
    content = (hct_frac * ccc + (1.0 - hct_frac) * cp) * 2.22
    return content

def blood_gas_calc(po2: float, pco2: float, params: BloodGasParams) -> Tuple[float, float]:
    """
    Port of SUBROUTINE BLOOD.
    Iteratively calculates pH, Saturation, O2 Content, CO2 Content.
    Legacy BLOOD routine calls PH twice to handle Haldane effect on pH.
    """
    # 1. Initial pH guess based on PCO2 (ignoring Saturation effect Y=0)
    ph1 = calculate_ph(pco2, 0.0, params)
    
    # 2. Calculate Saturation with this pH
    sat1 = calculate_saturation(po2, pco2, ph1, params)
    
    # 3. Calculate "Y" correction (Haldane on pH)
    # Y=0.003*HB*(1.0-SATURA/100.0) (Line 987)
    y = 0.003 * params.hb * (1.0 - sat1 / 100.0)
    
    # 4. Re-calculate pH with Y
    # PH2=PH(PCO2,Y)
    # Legacy PH function: PH = 7.59 + Y - 0.2741*log(PCO2/20)
    # So we just add Y to our simple PH calc.
    ph2 = calculate_ph(pco2, 0.0, params) + y
    
    # 5. Final Saturation
    sat_final = calculate_saturation(po2, pco2, ph2, params)
    
    # 6. O2 Content (ml/100ml)
    # O2C = 0.0139*HB*SATRN + SO2*PO2 (Line 990)
    # 1.39 is Hufner's constant.
    o2_content = 1.39 * params.hb * (sat_final / 100.0) + params.sol_o2 * po2
    
    # 7. CO2 Content
    co2_content = calculate_co2_content(pco2, ph2, sat_final, params)
    
    return o2_content, co2_content

from typing import Tuple
