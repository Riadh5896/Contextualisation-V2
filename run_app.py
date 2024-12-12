import subprocess

try:
    print("Running environment setup...")
    subprocess.run(["python", "setup_environment.py"], check=True, capture_output=True)

    print("Starting the Streamlit app...")
    subprocess.run(["streamlit", "run", "app.py"], check=True, capture_output=True)

except subprocess.CalledProcessError as e:
    print(f"Error: {e}")
    print(f"Return code: {e.returncode}")
    print(f"Standard output: {e.stdout.decode()}")
    print(f"Standard error: {e.stderr.decode()}")