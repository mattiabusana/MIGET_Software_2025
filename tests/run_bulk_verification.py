
import os
import glob
import re
import pandas as pd
import numpy as np
import sys
sys.path.append(os.getcwd())
from miget.io import MigetIO
from miget.core import VQModel

INPUT_DIR = "Data Study VentPerf/inputs"
OUTPUT_DIR = "Data Study VentPerf/outputs"

def parse_legacy_output(filepath):
    """
    Parses specific metrics from Legacy Output file.
    Returns dict: {shunt, deadspace, mean_q, sd_q, mean_v, sd_v}
    """
    metrics = {}
    with open(filepath, 'r', encoding='latin-1') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if "VA/Q OF ZERO" in line:
            # VA/Q OF ZERO                  0.000                ZERO
            parts = line.split()
            # Parts: VA/Q, OF, ZERO, 0.000, ZERO
            try:
                metrics['shunt'] = float(parts[3])
            except:
                pass
                
        if "VA/Q OF INFINITY" in line:
            # VA/Q OF INFINITY               ZERO               0.700
            parts = line.split()
            # Parts: VA/Q, OF, INFINITY, ZERO, 0.700
            try:
                metrics['deadspace'] = float(parts[4])
            except:
                pass
        
        if "MEAN OF BLOOD FLOW" in line:
            # MEAN OF BLOOD FLOW  DISTRIBUTION =   0.26
            val = line.split('=')[-1].strip()
            metrics['mean_q'] = float(val)
            
        if "2nd MOMENT OF BLOOD FLOW" in line:
             val = line.split('=')[-1].strip()
             metrics['var_q'] = float(val)
             metrics['sd_q'] = np.sqrt(float(val))

        if "MEAN OF VENTILATION" in line:
            val = line.split('=')[-1].strip()
            metrics['mean_v'] = float(val)

        if "2nd MOMENT OF VENTILATION" in line:
             val = line.split('=')[-1].strip()
             metrics['var_v'] = float(val)
             metrics['sd_v'] = np.sqrt(float(val))

        if "REMAINING" in line and "SUM OF SQUARES" in line:
             # REMAINING SUM OF SQUARES =  1.30E+02
             val = line.split('=')[-1].strip()
             try:
                 metrics['rss'] = float(val)
             except:
                 pass
             
    return metrics

def run_comparison():
    input_files = glob.glob(os.path.join(INPUT_DIR, "*.RAW")) + glob.glob(os.path.join(INPUT_DIR, "*.raw"))
    results = []
    
    print(f"Found {len(input_files)} input files.")
    
    for in_file in input_files:
        basename = os.path.basename(in_file)
        name_core = basename.lower().replace(".raw", "").replace("_a", "")
        
        out_file = os.path.join(OUTPUT_DIR, name_core)
        if not os.path.exists(out_file):
            print(f"Warning: No output file found for {basename}")
            continue
            
        try:
            with open(in_file, 'r', encoding='latin-1') as f:
                content = f.read()
            data_list = MigetIO.parse_legacy_file(content)
            run = data_list[0]
            run.species = "human"
            run.calculate_metrics()
            
            model = VQModel(run, z_factor=40.0)
            dist = model.solve()
            
            # Calculate Modern RSS
            weights = run.retention_weights
            r_pred = getattr(dist, 'predicted_retention', np.zeros(6))
            
            # If predicted_retention is all zeros (shouldn't be), warn
            if np.sum(r_pred) == 0 and np.sum(run.retention) > 0:
                 print(f"Warning: R_pred is zero for {name_core}")

            # Safe subtraction with numpy handling
            rss_modern = np.sum( ((r_pred - run.retention) * weights)**2 )
            
            legacy = parse_legacy_output(out_file)
            
            res = {
                "File": name_core,
                "L_RSS": legacy.get('rss', np.nan),
                "M_RSS": rss_modern,
                "L_Shunt": legacy.get('shunt', np.nan),
                "M_Shunt": dist.shunt,
                "L_Deadspace": legacy.get('deadspace', np.nan),
                "M_Deadspace": dist.deadspace
            }
            results.append(res)
            print(f"Processed {name_core}: L_RSS={res['L_RSS']:.1f} / M_RSS={res['M_RSS']:.1f}")
            
        except Exception as e:
            print(f"Error processing {basename}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save Results
    df = pd.DataFrame(results)
    df.to_csv("tests/bulk_results_rss.csv", index=False)
    
    print("\n--- RSS Analysis ---")
    df['RSS_Improv'] = df['L_RSS'] - df['M_RSS']
    better_fit = df[df['RSS_Improv'] > 0]
    print(f"Modern fit better in {len(better_fit)}/{len(df)} files.")
    print(f"Mean RSS Improvement: {df['RSS_Improv'].mean():.2f}")
    
    print("\n--- Top 10 RSS Improvements ---")
    print(df[['File', 'L_RSS', 'M_RSS', 'RSS_Improv']].sort_values('RSS_Improv', ascending=False).head(10))

if __name__ == "__main__":
    run_comparison()
