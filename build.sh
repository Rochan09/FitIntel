#!/bin/bash

# Build script for Render
pip install --upgrade pip
pip install -r requirements.txt

# Create models directory if it doesn't exist
mkdir -p models