import os
import subprocess
import sys

def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def setup_ollama():
    try:
        subprocess.run(["ollama", "--version"], check=True, capture_output=True)
        print("Ollama is already installed.")
    except FileNotFoundError:
        print("Ollama not found. Please install it manually.")
        # Provide instructions or options for different OS
        if sys.platform == "darwin":
            print("On macOS, you can use Homebrew: `brew install ollama`")
        # Add instructions for other platforms (Linux, Windows)

def install_llama_model(model_name="llama3.2"):
    try:
        subprocess.run(["ollama", "pull", model_name], check=True, capture_output=True)
        print(f"{model_name} model is already installed.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {model_name}: {e}")
        print(e.stderr.decode())  # Print the error output

def run_llama_model(model_name="llama3.2"):
    try:
        print(f"Starting {model_name} model...")
        process = subprocess.Popen(
            ["ollama", "run", model_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print(f"{model_name} model is now running.")
        print("To stop it, use `kill` or terminate the process manually.")
        return process
    except Exception as e:
        print(f"An error occurred while running {model_name}: {e}")

def setup_environment():
    try:
        print("Setting up the environment...")
        setup_ollama()
        install_llama_model()  # You can now pass a model name here
        print("Environment setup complete.")
        run_llama_model()
    except Exception as e:
        print(f"An error occurred during setup: {e}")

if __name__ == "__main__":
    setup_environment()