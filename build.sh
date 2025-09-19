#!/bin/bash

# Build script for Render
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Create models directory if it doesn't exist
mkdir -p models