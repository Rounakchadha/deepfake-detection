#!/bin/bash

# This script automates the setup process for the deepfake detection system.

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting the setup process for the Deepfake Detection System...${NC}"

# Change to the project root directory
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR" || exit

# --- 1. Create a Python Virtual Environment ---
echo -e "\n${YELLOW}[Step 1/4] Creating a Python virtual environment named 'venv'...${NC}"
if [ -d "venv" ]; then
    echo "Virtual environment 'venv' already exists. Skipping creation."
else
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create the virtual environment. Please ensure you have python3 and venv installed."
        exit 1
    fi
fi

# --- 2. Activate the Virtual Environment ---
echo -e "\n${YELLOW}[Step 2/4] Activating the virtual environment...${NC}"
source venv/bin/activate
echo "To deactivate the environment later, simply run: deactivate"

# --- 3. Install Dependencies ---
echo -e "\n${YELLOW}[Step 3/4] Installing required Python packages from requirements.txt...${NC}"
./venv/bin/pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install all dependencies. Please check the 'requirements.txt' file and your internet connection."
    exit 1
fi
echo "Dependencies installed successfully."

# --- 4. Dataset Download Instructions ---
echo -e "\n${YELLOW}[Step 4/4] Dataset Download Instructions...${NC}"
echo "The deep learning models require large datasets that cannot be included in this repository."
echo "You will need to download them manually."
python data/download_datasets.py

echo -e "\n${GREEN}--- Setup Complete! ---
${NC}"
echo -e "You are now ready to train the models and run the application.\n"

# --- Next Steps ---
echo -e "${YELLOW}Next Steps:${NC}"
echo "1.  ${GREEN}Train a model:${NC}"
echo "    For example, to train MesoNet on the Celeb-DF dataset, run:"
echo "    python training/train.py --model MesoNet --dataset Celeb-DF --dataset-path data/Celeb-DF"
echo ""
echo "2.  ${GREEN}Run the frontend application:${NC}"
echo "    Once you have a trained model in the 'weights/' directory, start the Streamlit app:"
echo "    streamlit run frontend/app.py"
echo ""
