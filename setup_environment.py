import os
import subprocess
import sys

# Function to install a package using pip
def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# Check and install Ollama if not already installed
def setup_ollama():
    try:
        # Check if Ollama is installed by running a version check
        subprocess.run(["ollama", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Ollama is already installed.")
    except FileNotFoundError:
        print("Installing Ollama...")
        # Replace with the actual command to install Ollama
        subprocess.run(["brew", "install", "ollama"], check=True)

# Download and install the llama3.2 model
def install_llama_model():
    try:
        # Run a command to verify if llama3.2 is already installed
        subprocess.run(["ollama", "pull", "llama3.2"], check=True)
        print("Llama3.2 model is already installed.")
    except subprocess.CalledProcessError:
        print("Downloading and installing llama3.2 model...")
        subprocess.run(["ollama", "pull", "llama3.2"], check=True)

# Run the llama3.2 model
def run_llama_model():
    try:
        # Check if Ollama can start the llama3.2 model
        print("Starting llama3.2 model...")
        process = subprocess.Popen(["ollama", "run", "llama3.2"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Llama3.2 model is now running.")
        print("To stop it, use `kill` or terminate the process manually.")
        return process
    except Exception as e:
        print(f"An error occurred while running llama3.2: {e}")

# Run the setup process
def setup_environment():
    try:
        print("Setting up the environment...")
        setup_ollama()
        install_llama_model()
        print("Environment setup complete.")
        run_llama_model()  # Start the llama3.2 model after setup
    except Exception as e:
        print(f"An error occurred during setup: {e}")
