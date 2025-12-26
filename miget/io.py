import numpy as np
from typing import List, Tuple
from miget.core import InertGasData

class MigetIO:
    @staticmethod
    def parse_legacy_file(content: str) -> List[InertGasData]:
        """
        Parses the legacy Fortran-formatted output file (e.g., FILE7 or FILE9)
        as seen in SHORT_INPUT.
        
        Format appears to be:
        Line 1: Run Label (A60)
        Line 2: NRUNS, PBSEA, ELECT, TBATH, RER, SO2 (I4, F7.1, ...)
        Line 3: NGASES, NVAQS, ZZ, VQLO, VQHI (2I4, F5.1, 2X, F5.3, 2X, F5.1 ? Check SHORT.FOR)
        Line 4: Partition Coefficients (6(1PE12.3))
        Loop Runs:
          Line 5: PA (Arterial)
          Line 6: GA (Gain)
          Line 7: PE (Expired)
          Line 8: GE (Gain)
          Line 9: PV (Venous)
          Line 10: GV (Gain)
          Line 11: VE, QT, PB, TEMPB, TEMPR, VGA, VBA, VHA, VGV, VBV, VHV
          Line 12: HB, HCRIT, VO2, VCO2, ... (Blood Gas)
          Line 13: More Blood Gas?
        """
        lines = [l for l in content.splitlines() if l.strip()]
        data_pointer = 0
        
        def get_line():
            nonlocal data_pointer
            if data_pointer >= len(lines):
                return None
            line = lines[data_pointer]
            data_pointer += 1
            return line

        results = []
        
        # 1. Label
        label = get_line() # "rawdata2"
        
        # 2. Run Params
        line = get_line()
        if not line: return []
        parts = line.split()
        n_runs = int(parts[0])
        pb_sea = float(parts[1])
        elect_temp = float(parts[2])
        bath_temp = float(parts[3])
        # rer = float(parts[4])
        # so2 = float(parts[5])
        
        # 3. Gas Params
        # Line 3: 6  50  40.0  0.005  100.0 (NGAS, NVAQS, Z, VQLO, VQHI)
        line = get_line()
        parts = line.split()
        n_gases = int(parts[0])
        # n_vaqs = int(parts[1])
        z_val = float(parts[2]) if len(parts) > 2 else 40.0
        
        # 4. Partition Coefficients
        # Line 4: 6.580E-03 ...
        line = get_line()
        # Handle scientific notation split
        pc_values = np.array([float(x) for x in line.split()][:n_gases])
        
        # Loop over runs
        for i in range(n_runs):
            run_data = InertGasData()
            run_data.name = f"{label.strip()} - Run {i+1}"
            run_data.pb_sea = pb_sea
            run_data.temp_blood = elect_temp
            run_data.temp_bath = bath_temp
            run_data.z = z_val
            run_data.partition_coeffs = pc_values
            
            # PA (Arterial Peaks)
            line = get_line()
            # Heuristic for Repeated PCs (Oklahoma Format)
            # If line looks like PCs (small values, matches previous PCs) and i > 0
            # consume it and read next.
            temp_vals = np.array([float(x) for x in line.split()][:n_gases])
            if i > 0:
                 # Check if similar to pc_values
                 # Or just check first value magnitude? SF6 PC ~ 0.007. PA SF6 ~ 100.
                 if np.allclose(temp_vals, pc_values, rtol=0.1, atol=0.1):
                     # Likely repeated PCs
                     # Update PCs just in case?
                     pc_values = temp_vals
                     # Read next line for PA
                     line = get_line()
                     temp_vals = np.array([float(x) for x in line.split()][:n_gases])
            
            run_data.pa_raw = temp_vals
            
            # GA (Gains)
            line = get_line()
            run_data.ga = np.array([float(x) for x in line.split()][:n_gases])
            
            # PE (Expired Peaks)
            line = get_line()
            run_data.pe_raw = np.array([float(x) for x in line.split()][:n_gases])
            
            # GE (Gains)
            line = get_line()
            run_data.ge = np.array([float(x) for x in line.split()][:n_gases])
            
            # PV (Venous Peaks)
            line = get_line()
            run_data.pv_raw = np.array([float(x) for x in line.split()][:n_gases])
            
            # GV (Gains)
            line = get_line()
            run_data.gv = np.array([float(x) for x in line.split()][:n_gases])
            
            # Ventilation/Perfusion
            # Line 11: 14.40 7.40 ... (QT, VE, PB, TBODY, TEMPR, VGA, VBA...)
            # SHORT.FOR: READ (1,160) QT, VEO, PBAR, TEMPB, TEMPR
            # BUT: Data implies QT=7.4 (2nd Position) to match Low Shunt/Retention.
            # If we use QT=14.4 (1st), R becomes 5% (Too high).
            # So File Format IS 'VE, QT'.
            line = get_line()
            parts = line.split()
            run_data.ve_measured = float(parts[0]) # VE First
            run_data.qt_measured = float(parts[1]) # QT Second
            run_data.temp_body = float(parts[3]) # TEMPB
            
            # Check if we have volume data (Sample correction)
            if len(parts) >= 11:
                run_data.vga = float(parts[5])
                run_data.vba = float(parts[6])
                run_data.vha = float(parts[7])
                run_data.vgv = float(parts[8])
                run_data.vbv = float(parts[9])
                run_data.vhv = float(parts[10])
            else:
                # Default to 0? Or 1? If 0, correction formula might fail if division.
                # If VBA=0, formula VHA/VBA fails.
                # Assuming if missing, no correction needed? 
                pass
            
            # Blood Gas & Physiology
            # Line 12: 8.9  27.0 274.9 241.0 ... (HB, HCRIT, VO2, VCO2)
            # Try to parse Line 12
            line = get_line()
            if line:
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        run_data.hb = float(parts[0])
                        run_data.hcrit = float(parts[1])
                    if len(parts) >= 4:
                        run_data.vo2_measured = float(parts[2])
                        run_data.vco2_measured = float(parts[3])
                except ValueError:
                    pass # Keep defaults if parse error
            
            # Line 13: 79.0 41.0 7.44 ... (Measured Pxa, Pxv, etc?)
            # Currently unused, just consume
            _ = get_line()
            
            results.append(run_data)
            
        return results

    @staticmethod
    def parse_csv(content: str) -> List[InertGasData]:
        """
        Parses a CSV file with modern headers.
        Expected columns: run_id, gas_name, solubility, retention, excretion, ...
        """
        import pandas as pd
        from io import StringIO
        
        df = pd.read_csv(StringIO(content))
        results = []
        
        run_ids = df['run_id'].unique() if 'run_id' in df.columns else [0]
        
        for rid in run_ids:
            if 'run_id' in df.columns:
                sub_df = df[df['run_id'] == rid]
            else:
                sub_df = df
            
            data = InertGasData()
            data.name = str(rid)
            
            # Fill arrays
            if 'retention' in sub_df.columns:
                data.retention = sub_df['retention'].values
            if 'excretion' in sub_df.columns:
                data.excretion = sub_df['excretion'].values
            if 'solubility' in sub_df.columns:
                data.solubilities = sub_df['solubility'].values
                
            results.append(data)
            
        return results
