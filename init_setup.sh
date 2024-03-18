#!/bin/bash
# Create a virtual environment and install the project in editable mode.
echo "Creating the environment"
python3 -m venv env
source env/bin/activate
pip install -r requirements_devs.txt
echo "Environment ready: source env/bin/activate"
