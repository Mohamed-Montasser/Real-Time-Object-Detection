#!/bin/bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt
