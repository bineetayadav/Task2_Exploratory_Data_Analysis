"""
run_pipeline.py
Master execution pipeline for Task 2: Exploratory Data Analysis (EDA).
Executes in sequence:
1. run_eda_analysis.py (Statistical computation, charts & tables generation)
2. build_notebook.py (Jupyter Notebook assembly and execution)
3. generate_pdf_report.py (Executive PDF compilation)
"""

import os
import sys
import subprocess
import time

def run_step(step_name: str, script_name: str, base_dir: str):
    print("\n" + "=" * 80)
    print(f"RUNNING: {step_name} ({script_name})")
    print("=" * 80)
    start_time = time.time()
    script_path = os.path.join(base_dir, "src", script_name)
    result = subprocess.run([sys.executable, script_path], cwd=base_dir, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:\n", result.stderr)
        
    if result.returncode != 0:
        raise RuntimeError(f"Step '{step_name}' failed with return code {result.returncode}!")
        
    elapsed = time.time() - start_time
    print(f" COMPLETED: {step_name} in {elapsed:.2f} seconds.")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("=" * 80)
    print("STARTING END-TO-END EDA PIPELINE — TASK 2")
    print(f"Base Directory: {base_dir}")
    print("=" * 80)
    total_start = time.time()

    run_step("1. Statistical Analysis & Visualizations", "run_eda_analysis.py", base_dir)
    run_step("2. Jupyter Notebook Construction & Execution", "build_notebook.py", base_dir)
    run_step("3. Executive PDF Report Compilation", "generate_pdf_report.py", base_dir)

    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print(f" ALL EDA TASKS COMPLETED SUCCESSFULLY in {total_time:.2f}s!")
    print("=" * 80)
    print("Deliverables Generated:")
    print(f" - Jupyter Notebook: {os.path.join(base_dir, 'notebooks', 'exploratory_data_analysis.ipynb')}")
    print(f" - Analytical Report (MD): {os.path.join(base_dir, 'reports', 'eda_summary_report.md')}")
    print(f" - Executive PDF Report: {os.path.join(base_dir, 'reports', 'eda_report.pdf')}")
    print(f" - Visualizations: {os.path.join(base_dir, 'outputs', 'figures')}")
    print(f" - Data Tables: {os.path.join(base_dir, 'outputs', 'tables')}")

if __name__ == "__main__":
    main()
