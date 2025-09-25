#!/usr/bin/env python3
"""
Simplified C6 Parameter Study using working c6_energy_comparison.py approach
"""

import os
import sys
import time
import subprocess
import shutil

def run_parameter_combination(threshold, max_cycle, output_file):
    """Run C6 comparison with specific parameters by modifying the working script"""
    
    print(f"\n{'='*80}")
    print(f"Running C6 Parameter Study: threshold={threshold}, max_cycle={max_cycle}")
    print(f"{'='*80}")
    
    # Create a temporary copy of the working script
    temp_script = f"temp_c6_comparison_{threshold}_{max_cycle}.py"
    
    # Copy the working script
    shutil.copy("c6_energy_comparison.py", temp_script)
    
    # Read the script content
    with open(temp_script, 'r') as f:
        content = f.read()
    
    # Modify the threshold and max_cycle values
    content = content.replace(
        "excitation_threshold = 0.0001",
        f"excitation_threshold = {threshold}"
    )
    content = content.replace(
        "vqe_max_cycle = 10",
        f"vqe_max_cycle = {max_cycle}"
    )
    
    # Write the modified script
    with open(temp_script, 'w') as f:
        f.write(content)
    
    # Run the modified script
    start_time = time.time()
    
    try:
        result = subprocess.run([
            "python", temp_script
        ], capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        total_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"Calculation completed successfully in {total_time:.2f} seconds")
            
            # Parse the output to extract results
            parse_and_write_results(result.stdout, threshold, max_cycle, output_file, total_time)
            
        else:
            print(f"Calculation failed with error:")
            print(result.stderr)
            write_error_results(threshold, max_cycle, output_file, result.stderr, total_time)
            
    except subprocess.TimeoutExpired:
        print("Calculation timed out after 30 minutes")
        write_timeout_results(threshold, max_cycle, output_file)
        
    except Exception as e:
        print(f"Error running calculation: {e}")
        write_error_results(threshold, max_cycle, output_file, str(e), 0)
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_script):
            os.remove(temp_script)

def parse_and_write_results(output, threshold, max_cycle, output_file, total_time):
    """Parse calculation output and write markdown results"""
    
    # Extract energy values from output
    results = {}
    lines = output.split('\n')
    
    casci_energy = None
    for line in lines:
        if "CASCI Energy:" in line:
            casci_energy = float(line.split()[-2])
        elif "LASSCF Energy:" in line and "LASSCF" not in results:
            results['LASSCF'] = {
                'energy': float(line.split()[-2]),
                'method': 'LASSCF',
                'converged': "converged" in line.lower()
            }
        elif "LAS-VQE (All) Final Energy:" in line:
            results['LAS_VQE_All'] = {
                'energy': float(line.split()[-2]),
                'method': 'LAS-VQE (All Excitations)',
                'converged': "converged" in line.lower()
            }
        elif "Protocol 1 Final Energy:" in line:
            results['Protocol_1'] = {
                'energy': float(line.split()[-2]),
                'method': 'LAS-VQE-NOSI Protocol 1',
                'converged': "converged" in line.lower()
            }
        elif "Protocol 2 Final Energy:" in line:
            results['Protocol_2'] = {
                'energy': float(line.split()[-2]),
                'method': 'LAS-VQE-NOSI Protocol 2', 
                'converged': "converged" in line.lower()
            }
    
    write_markdown_results(results, threshold, max_cycle, output_file, casci_energy, total_time, output)

def write_markdown_results(results, threshold, max_cycle, output_file, casci_energy, total_time, full_output):
    """Write detailed markdown results"""
    
    content = f"""# C6 Energy Comparison - Threshold {threshold}, Max Cycle {max_cycle}

## Overview

**Molecule**: C6 polyene chain  
**Basis**: STO-3G  
**Excitation Threshold**: {threshold}  
**VQE Max Cycles**: {max_cycle}  
**Total Calculation Time**: {total_time:.2f} seconds  
**Fragment Definition**: 
- Fragment 0: atoms (0,2), spin orbs (0,1,6,7)
- Fragment 1: atoms (10,11), spin orbs (2,3,8,9)  
- Fragment 2: atoms (3,1), spin orbs (4,5,10,11)

## Energy Results Summary

"""
    
    if casci_energy:
        content += f"**CASCI Reference Energy**: {casci_energy:.10f} hartree\n\n"
    
    content += "| Method | Energy (hartree) | ΔE (mEh) | Converged |\n"
    content += "|--------|------------------|----------|-----------|\n"
    
    # Sort results by energy (lowest first)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['energy'] if x[1]['energy'] is not None else float('inf'))
    
    for method_name, data in sorted_results:
        energy = data['energy']
        if energy is None:
            continue
            
        # Calculate energy difference from CASCI
        if casci_energy:
            delta_e = (energy - casci_energy) * 1000  # Convert to mEh
        else:
            delta_e = 0.0
        
        converged_symbol = "✓" if data.get('converged', False) else "✗"
        
        content += f"| **{data['method']}** | {energy:.10f} | {delta_e:.3f} | {converged_symbol} |\n"
    
    # Parameter effects analysis
    content += f"\n## Parameter Effects Analysis\n\n"
    content += f"### Excitation Threshold: {threshold}\n"
    if threshold <= 0.0001:
        content += "- **Low threshold**: Captures many excitations, potentially better accuracy but higher cost\n"
    else:
        content += "- **Higher threshold**: Fewer excitations, faster computation but may miss important correlations\n"
    
    content += f"\n### VQE Max Cycles: {max_cycle}\n"
    if max_cycle >= 1000:
        content += "- **High cycle count**: Better convergence but expensive computation\n"
    elif max_cycle >= 100:
        content += "- **Medium cycle count**: Balance between convergence and computational cost\n"
    else:
        content += "- **Low cycle count**: Fast computation but may not fully converge\n"
    
    # Technical insights
    content += f"\n## Technical Insights\n\n"
    
    if len(results) > 0:
        best_method = min(results.items(), key=lambda x: x[1]['energy'])[1]['method']
        best_energy = min(results.items(), key=lambda x: x[1]['energy'])[1]['energy']
        content += f"**Best Method**: {best_method} with energy {best_energy:.10f} hartree\n\n"
    
    converged_methods = [data['method'] for data in results.values() if data.get('converged', False)]
    unconverged_methods = [data['method'] for data in results.values() if not data.get('converged', False)]
    
    content += f"### Convergence Analysis\n"
    if converged_methods:
        content += f"**Converged**: {', '.join(converged_methods)}\n"
    if unconverged_methods:
        content += f"**Unconverged**: {', '.join(unconverged_methods)}\n"
    
    content += f"\n## Full Calculation Output\n\n"
    content += "```\n" + full_output + "\n```"
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"Results written to {output_file}")

def write_error_results(threshold, max_cycle, output_file, error_msg, total_time):
    """Write error results to markdown"""
    content = f"""# C6 Energy Comparison - Threshold {threshold}, Max Cycle {max_cycle} - ERROR

## Calculation Failed

**Threshold**: {threshold}  
**Max Cycle**: {max_cycle}  
**Total Time**: {total_time:.2f} seconds  

## Error Message

```
{error_msg}
```

This parameter combination failed to complete successfully.
"""
    with open(output_file, 'w') as f:
        f.write(content)
    print(f"Error results written to {output_file}")

def write_timeout_results(threshold, max_cycle, output_file):
    """Write timeout results to markdown"""
    content = f"""# C6 Energy Comparison - Threshold {threshold}, Max Cycle {max_cycle} - TIMEOUT

## Calculation Timed Out

**Threshold**: {threshold}  
**Max Cycle**: {max_cycle}  
**Timeout**: 30 minutes  

This parameter combination exceeded the time limit and was terminated.
The calculation may be too expensive for these parameters.
"""
    with open(output_file, 'w') as f:
        f.write(content)
    print(f"Timeout results written to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python c6_parameter_study_simple.py <threshold> <max_cycle> <output_file>")
        sys.exit(1)
    
    threshold = float(sys.argv[1])
    max_cycle = int(sys.argv[2])
    output_file = sys.argv[3]
    
    run_parameter_combination(threshold, max_cycle, output_file)