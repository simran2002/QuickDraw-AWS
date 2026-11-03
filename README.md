# Update the instance and Install Python, Pip, and Git
sudo apt update && sudo apt install -y python3-pip python3.12-venv

# Create the project directory
mkdir ~/visionsketch && cd ~/visionsketch

# Setup the Virtual Environment
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn transformers pillow python-multipart

# Install the Hugging Face Hub library
pip install -U "huggingface_hub[cli]"

# Download the BEiT model weights
hf download kmewhort/beit-sketch-classifier --local-dir ./ai_model

# Start the server
python3 main.py