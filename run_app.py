import subprocess

# Step 1: Run setup_environment.py
print("Running environment setup...")
subprocess.run(["python", "setup_environment.py"], check=True)

# Step 2: Run the Streamlit app
print("Starting the Streamlit app...")
subprocess.run(["streamlit", "run", "app.py"], check=True)