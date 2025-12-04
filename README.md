# ⚙️ Predictive Maintenance System (ANN + Streamlit)

This project delivers an intelligent Predictive Maintenance system built with Machine Learning and Deep Learning techniques. It automatically identifies whether data is meant for failure classification or RUL prediction and trains the appropriate model—ANN for static sensor data and LSTM for time-series sequences. Using Adam Optimization and Early Stopping, the system achieves strong performance with high Recall, high R², and low RMSE. The integrated Streamlit interface allows users to upload datasets, train models, and generate real-time failure risk or RUL predictions, making it a complete AI solution for Industry 4.0 machinery maintenance.

---

## 🎯 Purpose
To build an **AI-driven maintenance system** capable of:

- Predicting machine failure (classification)
- Estimating Remaining Useful Life (RUL) (regression)
- Supporting both ANN and LSTM deep learning models
- Offering real-time sensor-based predictions through an intuitive web UI

---

## 🧠 Core Features

✅ **Dataset Auto-Detection**  
✔ Detects whether the uploaded dataset is AI4I (classification) or NASA C-MAPSS (RUL regression)

✅ **ANN Model for Failure Classification**  
✔ Trains using Adam optimizer, multiple hidden layers, and early stopping  

✅ **LSTM Model for RUL Regression**  
✔ Automatically processes sequence data for life prediction  

✅ **Interactive Web Interface (Streamlit)**  
✔ Dataset upload  
✔ Model training  
✔ Real-time prediction using sliders  

✅ **Performance Metrics**  
✔ Accuracy, Precision, Recall (Classification)  
✔ R², RMSE (Regression)  

---

## ⚙️ Technologies Used

### 🔹 Machine Learning & Deep Learning  
- **TensorFlow / Keras** — ANN & LSTM architecture  
- **scikit-learn** — preprocessing, metrics  

### 🔹 Data Processing  
- **pandas**, **numpy**

### 🔹 Web Deployment  
- **Streamlit**

### 🔹 Visualization  
- **matplotlib**, **seaborn**

### 🔹 Optimization  
- **Adam optimizer**

---
### Project Flowchart
<img width="1000" height="900" alt="flowchart" src="https://github.com/user-attachments/assets/e52089b4-b49d-4305-9122-de6b60345064" />

---

## 📂 Project Structure
predictive-maintenance-project/

├── app.py                     # Main Streamlit application with ANN/LSTM logic

├── requirements.txt           # All dependencies

└── README.md                  # Documentation


---

## 📦 Installation & Run Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/predictive-maintenance-project

```
---
### 2️⃣ Navigate into the directory
```bash
cd predictive-maintenance-project
```
---
### 3️⃣ Create a virtual environment
```bash
python -m venv venv
```
---
### 4️⃣ Activate the environment

| OS          | Command                    |
| ----------- | -------------------------- |
| Windows     | `.\venv\Scripts\activate`  |
| Linux/macOS | `source venv/bin/activate` |

---

### 5️⃣ Install dependencies
```
pip install -r requirements.txt
```
---
### 6️⃣ Run the Streamlit App
```
streamlit run app.py
```
---

### ⭐ Your browser will open automatically at:
```
http://localhost:8501
```
---
### ▶️ Usage
### 🔹 Step 1 — Upload Dataset

Click “Upload Dataset (.txt or .csv)” and upload either:

AI4I 2020 Predictive Maintenance dataset (CSV)

NASA C-MAPSS dataset (TXT)

### 🔹 Step 2 — Train the Model

Click “Train Predictive Model (Auto-Detect)”
The system will:

Identify dataset type

Preprocess data

Train ANN (classification) or LSTM (regression)

Display performance metrics

### 🔹 Step 3 — Real-Time Prediction

## Using the prediction panel:

Adjust sliders for temperature, vibration, speed, load

Click “Predict Failure Risk” or “Predict RUL”

📊 Output will appear instantly on the dashboard.

---
### 💬 Sample Inputs & Outputs

Upload: AI4I_2020.csv

Upload: C-MAPSS_TRAIN_FD001.txt

Adjust sliders for simulated machine readings

Check predicted RUL or Failure Probability

<img width="1919" height="908" alt="Screenshot 2025-12-04 144523" src="https://github.com/user-attachments/assets/194233ce-3c38-4546-9ad1-713a844c52da" />
<img width="1750" height="637" alt="Screenshot 2025-12-04 150858" src="https://github.com/user-attachments/assets/1e7d50ef-a8f4-42ed-93b3-730108418662" />
<img width="1751" height="526" alt="Screenshot 2025-12-04 150909" src="https://github.com/user-attachments/assets/5a873696-7a7d-4665-89eb-9b118fc717a4" />
<img width="1752" height="436" alt="Screenshot 2025-12-04 150920" src="https://github.com/user-attachments/assets/e3b33261-dd9b-44f1-8ff0-5b8c3450c7f4" />
<img width="1357" height="608" alt="Screenshot 2025-12-04 150934" src="https://github.com/user-attachments/assets/ed4ef95f-9bde-464c-b875-fd3815c2d99f" />

---
### 📜 License
This project is licensed under the MIT License — free to use, modify, improve, and distribute with proper credit.

---
### 👨‍💻 Developed By

Chaitanya Bhosale

🔗 GitHub: https://github.com/Chaitanya5068

🔗 LinkedIn: https://www.linkedin.com/in/chaitanya-bhosale




