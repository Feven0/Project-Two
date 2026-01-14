
import subprocess
import sys
import os

def run_script(script_name, is_streamlit=False):
    print(f"\n--- Running {script_name} ---")
    try:
        if is_streamlit:

            subprocess.run([sys.executable, "-m", "streamlit", "run", script_name], check=True)
        else:
            subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        sys.exit(1)

if __name__ == "__main__":

    run_script("prepare_data.py")


    run_script("expected_danger_model.py")


    print("\nStarting the dashboard...")
    run_script("dashboard.py", is_streamlit=True)
