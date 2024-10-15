import subprocess
import sys

def run_script(script_name):
    print(f"Running {script_name}...")
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stderr)
        sys.exit(1)
    else:
        print(f"{script_name} completed successfully.")
        print(result.stdout)

# List of scripts to run in order
scripts = [
    "baseline_stats.py",
    "preliminary_models.py",
    "new_models.py",
    "save_data.py"
]

# Run each script in sequence
for script in scripts:
    run_script(script)

print("All scripts have been executed.")