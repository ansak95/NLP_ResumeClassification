#!/bin/bash

# Install requirements
echo "Installing the packages..."
pip install -q -r requirements.txt
echo "Done."

# Run the Streamlit app in the background
echo "Running the Streamlit app..."

# The URL provided to the user. Specify the "NOTEBOOK_URL"
echo "Access the Streamlit app at: https://{NOTEBOOK_URL}/proxy/8501/"
streamlit run app.py 

# Wait for a few seconds to allow the server to start
# sleep 5


