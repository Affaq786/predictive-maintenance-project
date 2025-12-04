Predictive Maintenance Project (ANN & Streamlit)

This project implements a dynamic Predictive Maintenance system using an Artificial Neural Network (ANN) and is deployed via a Streamlit web interface. It fulfills the requirements for Machine Learning, ANN, AI (deployment), and Optimization Techniques (Adam optimizer).

Project Structure

predictive-maintenance-project/
├── app.py              # Main Streamlit application and ML/ANN logic.
├── requirements.txt    # Python dependencies.
└── README.md           # This file.


Setup and Execution (VS Code)

Step 1: Clone or Create the Project

Create a new folder named predictive-maintenance-project.

Place app.py and requirements.txt inside this folder.

Open the folder in VS Code.

Step 2: Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies:

# Create environment
# Note: Use 'python' for Windows, and 'python3' for Linux/macOS
python -m venv venv 


Step 3: Activate the Virtual Environment

Activate the environment based on your operating system.

OS

Activation Command

Windows (PowerShell)

.\venv\Scripts\activate

Linux / macOS

source venv/bin/activate

Step 4: Install Dependencies

With the environment activated, install all required libraries:

  pip install -r requirements.txt


Step 5: Run the Application

Execute the Streamlit application from your terminal:

treamlit run app.py


This command will launch the application in your default web browser (usually at http://localhost:8501).

How to Use the Web Interface

Data Upload:

Find a suitable dataset (e.g., the AI4I 2020 Predictive Maintenance CSV or NASA C-MAPSS TXT files).

Click "Upload Dataset (.txt or .csv)" to upload your file.

Model Training (Auto-Detection):

Click the "Train Predictive Model (Auto-Detect)" button.

The system will automatically detect if the data is suitable for ANN Binary Classification (AI4I 2020) or LSTM RUL Regression (NASA C-MAPSS).

The model will train and display the final performance metrics, including Accuracy/Recall (Classification) or $R^2$/RMSE (Regression).

Interactive Prediction:

Use the sliders and number inputs in Section 2 to simulate real-time sensor readings for a machine.

Click "Predict Failure Risk" (Classification) or "Predict RUL" (Regression) to get an immediate prediction and risk assessment.