import numpy as np
from typing import List, Tuple
from miget.core import InertGasData

class MigetIO:
    @staticmethod
    def parse_legacy_file(content: str) -> List[InertGasData]:
        """
        Parses the legacy Fortran-formatted output file (e.g., FILE7 or FILE9).
        Ref: SHORT.FOR lines 675+ and header writes.
        """
        lines = content.strip().splitlines()
        data_pointer = 0
        
        def get_line():
            nonlocal data_pointer
            if data_pointer >= len(lines):
                return None
            line = lines[data_pointer]
            data_pointer += 1
            return line

        results = []
        
        # 1. Header: Run Parameters
        # READ(7,190) NRUNS,PBSEA,ELECT,TBATH,RER,SO2
        # FORMAT(I4,F7.1,2F6.1,F8.2,F8.4)
        line = get_line()
        if not line: return []
        parts = line.split() # Free format read for simplicity, though Fortran implies fixed width
        n_runs = int(parts[0])
        pb_sea = float(parts[1])
        elect_temp = float(parts[2])
        bath_temp = float(parts[3])
        # rer = float(parts[4])
        # so2 = float(parts[5])
        
        # 2. Header: Gas Parameters
        # READ(7,240) NGASES,NVAQS,ZZ,VQLO,VQHI
        line = get_line()
        parts = line.split()
        n_gases = int(parts[0])
        # n_vaqs = int(parts[1])
        
        # 3. Partition Coefficients
        # READ(7,250) (PC(I),I=1,NGASES)
        line = get_line()
        pc_values = np.array([float(x) for x in line.split()][:n_gases])
        
        # Loop over runs
        for i in range(n_runs):
            run_data = InertGasData()
            run_data.name = f"Run {i+1}"
            run_data.pb_sea = pb_sea
            run_data.temp_blood = elect_temp
            run_data.temp_bath = bath_temp
            run_data.partition_coeffs = pc_values
            
            # PA (Arterial Peaks)
            line = get_line()
            run_data.pa_raw = np.array([float(x) for x in line.split()][:n_gases])
            
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
            # READ(7,1220) VEO,QT,PB,TEMPB,TEMPR, ...
            line = get_line()
            parts = line.split()
            run_data.ve_measured = float(parts[0])
            run_data.qt_measured = float(parts[1])
            run_data.temp_body = float(parts[3]) # TEMPB
            
            # Blood Gas / O2 data (Optional in loop?)
            # SHORT.FOR: IF(PV(1).GT.0.0) WRITE...
            # We assume it exists for now or check PV
            line = get_line()
            # Parse blood gas data if needed
            
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
