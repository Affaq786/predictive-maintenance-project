# -*- coding: utf-8 -*-
"""
Dynamic Predictive Maintenance System (Automated Architecture)

This version automatically detects the required ML task (Regression or Classification) 
based on the columns present in the uploaded dataset, providing a seamless user interface.

Features:
- Single file uploader for .csv or .txt (auto-detects delimiter).
- Automatic selection between LSTM RUL Regression and ANN Binary Classification.
- Enhanced validation checks and simplified UI.
"""
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, classification_report
# FIX: Added the necessary import for splitting data sets
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import sys

# --- GLOBAL CONSTANTS (RUL Regression - NASA C-MAPSS) ---
RUL_SEQUENCE_LENGTH = 50 
RUL_MAX_RUL = 130 
RUL_SENSOR_COLS = [f'sensor_{i}' for i in range(1, 22)]
RUL_SETTING_COLS = [f'setting_{i}' for i in range(1, 4)]
RUL_COLUMNS = ['unit_number', 'time_in_cycles'] + RUL_SETTING_COLS + RUL_SENSOR_COLS

# --- GLOBAL CONSTANTS (Binary Classification - AI4I 2020 / Generic) ---
CLASSIFICATION_TARGET = 'Machine failure'
CLASSIFICATION_NUM_FEATURES = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
CLASSIFICATION_CAT_FEATURES = ['Type']
CLASSIFICATION_ALL_FEATURES = CLASSIFICATION_NUM_FEATURES + CLASSIFICATION_CAT_FEATURES

# Global state for caching/model storage
if 'is_trained' not in st.session_state:
    st.session_state['is_trained'] = False
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'scaler_or_preprocessor' not in st.session_state:
    st.session_state['scaler_or_preprocessor'] = None
if 'task_type' not in st.session_state:
    st.session_state['task_type'] = None
if 'features' not in st.session_state:
    st.session_state['features'] = None

# --- CORE DATA LOADING (Universal) ---

@st.cache_data
def load_data(uploaded_file):
    """Loads the data, determining delimiter based on file extension."""
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    data = uploaded_file.getvalue().decode("utf-8")
    
    if file_extension == '.csv':
        df = pd.read_csv(io.StringIO(data), sep=',')
        return df
    
    elif file_extension == '.txt':
        # Try space-separated load (typical for NASA C-MAPSS)
        try:
            df = pd.read_csv(io.StringIO(data), sep='\s+', header=None)
            df = df.dropna(axis=1, how='all')
            # If column count matches NASA, assign NASA column names
            if df.shape[1] == len(RUL_COLUMNS):
                df.columns = RUL_COLUMNS
            return df
        except Exception:
            # Fallback for generic text file load
            return pd.read_csv(io.StringIO(data), sep='\s+')

    return None

# --- RUL REGRESSION LOGIC (NASA C-MAPSS) ---

def calculate_rul(df):
    """Calculates the Remaining Useful Life (RUL) for each cycle."""
    max_cycle = df.groupby('unit_number')['time_in_cycles'].max()
    merged = df.merge(max_cycle.rename('max_cycle'), left_on='unit_number', right_index=True)
    df['RUL'] = merged['max_cycle'] - merged['time_in_cycles']
    df['RUL'] = df['RUL'].apply(lambda x: min(x, RUL_MAX_RUL))
    return df

@st.cache_data
def generate_sequences(feature_array, rul_array, unit_numbers, sequence_length):
    """Generates time-series sequences for LSTM input."""
    X_sequences, y_targets = [], []
    df_temp = pd.DataFrame({'features': list(feature_array), 'rul': rul_array, 'unit': unit_numbers})
    
    for unit_id in df_temp['unit'].unique():
        unit_data = df_temp[df_temp['unit'] == unit_id]
        features = unit_data['features'].tolist()
        ruls = unit_data['rul'].tolist()
        
        for i in range(len(features) - sequence_length + 1):
            X_sequences.append(features[i:i + sequence_length])
            y_targets.append(ruls[i + sequence_length - 1])
            
    return np.array(X_sequences), np.array(y_targets)

def train_lstm_rul_model(df_train_raw):
    """Time-Series Regression for RUL using LSTM."""
    try:
        # 1. RUL Calculation 
        df_train = calculate_rul(df_train_raw.copy())
        
        # 2. Feature Selection and Scaling
        features = RUL_SETTING_COLS + RUL_SENSOR_COLS
        scaler = MinMaxScaler()
        df_train[features] = scaler.fit_transform(df_train[features])
        
        # 3. Sequence Generation
        X_seq, y_target = generate_sequences(df_train[features].values, df_train['RUL'].values, df_train['unit_number'].values, RUL_SEQUENCE_LENGTH)
        
        # 4. LSTM Model Definition
        input_shape = (RUL_SEQUENCE_LENGTH, X_seq.shape[2])
        model_lstm = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(units=50, return_sequences=False),
            Dropout(0.3),
            Dense(1, activation='linear')
        ], name='LSTM_RUL_Model')
        
        # 5. Compilation and Optimization
        model_lstm.compile(
            loss='mse', 
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
            metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )
        
        # 6. Training
        st.info("Training LSTM Model (Optimization via Adam & Early Stopping)...")
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_lstm.fit(X_seq, y_target, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stopping], verbose=0)
        
        # 7. Evaluation
        y_pred_train = model_lstm.predict(X_seq, verbose=0).flatten()
        rmse_train = np.sqrt(mean_squared_error(y_target, y_pred_train))
        r2_train = r2_score(y_target, y_pred_train)
        
        st.subheader("Training Performance Metrics (RUL Regression):")
        st.markdown(f"**Root Mean Squared Error (RMSE):** `{rmse_train:.4f}`")
        st.markdown(f"**R-squared ($R^2$) Score:** `{r2_train:.4f}`")

        return model_lstm, scaler, features
    
    except Exception as e:
        st.error(f"Error during LSTM training or RUL calculation: {e}")
        return None, None, None

# --- ANN CLASSIFICATION LOGIC (AI4I / Generic) ---

def train_ann_classification_model(df_train_raw):
    """Binary Classification for immediate failure using ANN."""
    try:
        # 1. Data Preprocessing (Scaling and Encoding)
        X = df_train_raw.drop([CLASSIFICATION_TARGET], axis=1, errors='ignore')
        y = df_train_raw[CLASSIFICATION_TARGET]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), CLASSIFICATION_NUM_FEATURES),
                ('cat', OneHotEncoder(handle_unknown='ignore'), CLASSIFICATION_CAT_FEATURES)
            ],
            remainder='drop'
        )
        
        X_processed = preprocessor.fit_transform(X)
        # train_test_split is now properly imported globally
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
        
        # 2. ANN Model Definition
        input_shape = X_train.shape[1]
        model_ann = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid') # Binary Classification Output
        ], name='ANN_Classification_Model')
        
        # 3. Compilation and Optimization
        model_ann.compile(
            optimizer='adam', 
            loss='binary_crossentropy', 
            metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
        )
        
        # 4. Training
        st.info("Training ANN Model (Optimization via Adam & Early Stopping)...")
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_ann.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stopping], verbose=0)

        # 5. Evaluation
        y_pred_prob = model_ann.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype("int32")
        
        loss, accuracy, recall = model_ann.evaluate(X_test, y_test, verbose=0)[:3]
        
        st.subheader("Training Performance Metrics (Binary Classification):")
        st.markdown(f"**Test Accuracy:** `{accuracy:.4f}`")
        st.markdown(f"**Test Recall (Critical Metric):** `{recall:.4f}`")
        st.text("Classification Report:")
        st.code(classification_report(y_test, y_pred, zero_division=0))
        
        # Confusion Matrix Plot
        cm = tf.math.confusion_matrix(y_test, y_pred).numpy()
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['No Failure (0)', 'Failure (1)'], yticklabels=['No Failure (0)', 'Failure (1)'])
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('Actual Label')
        ax.set_xlabel('Predicted Label')
        st.pyplot(fig) # 
        
        return model_ann, preprocessor, CLASSIFICATION_ALL_FEATURES
    
    except Exception as e:
        st.error(f"Error during ANN training or data splitting: {e}")
        return None, None, None


# --- CORE LOGIC: DETECT TASK TYPE ---

def detect_task_and_train(df_uploaded):
    """Tries to determine if the data is RUL or Classification and trains accordingly."""
    
    # --- 1. Attempt RUL (NASA C-MAPSS) Detection ---
    # Requires unit_number, time_in_cycles, and 21 sensor columns (26 total columns)
    is_rul_data = all(col in df_uploaded.columns for col in ['unit_number', 'time_in_cycles'] + [RUL_SENSOR_COLS[0]]) and df_uploaded.shape[1] == len(RUL_COLUMNS)

    # --- 2. Attempt Classification (AI4I 2020 / Generic) Detection ---
    # Requires 'Machine failure' and key sensor columns
    is_classification_data = CLASSIFICATION_TARGET in df_uploaded.columns and all(col in df_uploaded.columns for col in CLASSIFICATION_NUM_FEATURES + CLASSIFICATION_CAT_FEATURES)

    if is_rul_data:
        st.info("Structure identified as: **Time-Series Regression (NASA C-MAPSS)**.")
        return "REGRESSION", train_lstm_rul_model(df_uploaded)
    
    elif is_classification_data:
        st.info("Structure identified as: **Binary Classification (AI4I 2020/Generic)**.")
        return "CLASSIFICATION", train_ann_classification_model(df_uploaded)
    
    else:
        # Generate informative error message
        rul_missing = [col for col in ['unit_number', 'time_in_cycles'] + RUL_SENSOR_COLS[:3] if col not in df_uploaded.columns]
        class_missing = [col for col in [CLASSIFICATION_TARGET] + CLASSIFICATION_NUM_FEATURES[:3] if col not in df_uploaded.columns]
        
        error_msg = ("**Data structure incompatible with defined models.**\n\n"
                     "To run **RUL Regression (LSTM)**, the file must contain NASA C-MAPSS columns. Missing examples: " + ", ".join(rul_missing) + "...\n\n"
                     "To run **Binary Classification (ANN)**, the file must contain AI4I 2020 columns. Missing examples: " + ", ".join(class_missing) + "...")
        
        st.error(error_msg)
        return None, (None, None, None)


# --- STREAMLIT UI LAYOUT ---

st.set_page_config(page_title="Dynamic Predictive Maintenance System", layout="wide")

st.title("⚙️ Dynamic Predictive Maintenance System") 
st.markdown("""
_Upload your dataset (.csv or .txt) to automatically initiate the appropriate **ANN/LSTM** model training._
""")

# --- 1. DATA UPLOAD SECTION ---
st.header("1. Data Upload and Model Training")

# SINGLE, FLEXIBLE FILE UPLOADER
uploaded_file = st.file_uploader(
    "Upload Dataset (.txt or .csv)", 
    type=["txt", "csv"], 
    help="Upload space-delimited NASA data or comma-separated AI4I 2020 data."
)

if uploaded_file is not None:
    try:
        df_uploaded = load_data(uploaded_file)
        
        if df_uploaded is not None:
            st.success(f"Data Loaded: {uploaded_file.name}")
            st.dataframe(df_uploaded.head(), use_container_width=True)
            
            if st.button("Train Predictive Model (Auto-Detect)", type="primary"):
                with st.spinner('Detecting data structure and training model... This may take a few minutes.'):
                    task_type, (model, scaler_or_preprocessor, features) = detect_task_and_train(df_uploaded)
                    
                    if model:
                        st.session_state['model'] = model
                        st.session_state['scaler_or_preprocessor'] = scaler_or_preprocessor
                        st.session_state['features'] = features
                        st.session_state['is_trained'] = True
                        st.session_state['task_type'] = task_type
                        st.balloons()
        else:
            st.session_state['is_trained'] = False

    except Exception as e:
        st.error(f"An unexpected fatal error occurred during processing: {e}")
        st.session_state['is_trained'] = False
else:
    st.session_state['is_trained'] = False
    st.info("Awaiting dataset upload to begin model training.")


# --- 2. INTERACTIVE PREDICTION SECTION ---
if st.session_state['is_trained']:
    
    model = st.session_state['model']
    scaler_or_preprocessor = st.session_state['scaler_or_preprocessor']
    features = st.session_state['features']
    task_type = st.session_state['task_type']
    
    st.header(f"2. Interactive Real-time Prediction")
    
    if task_type == "REGRESSION":
        st.success("Model Status: **LSTM RUL Regression** trained and ready!")
        st.caption("Enter sensor values to predict **Remaining Useful Life (RUL)** in cycles.")
        
        # --- Input Fields for RUL REGRESSION (NASA C-MAPSS) ---
        st.subheader("Simulate Critical Sensor Readings")
        col_input = st.columns(4)
        last_cycle_inputs = {}
        # Only using simplified inputs, need to create the full sequence later
        last_cycle_inputs['sensor_2'] = col_input[0].slider("Sensor 2 (T24)", min_value=640.0, max_value=645.0, value=642.0, step=0.1)
        last_cycle_inputs['sensor_3'] = col_input[1].slider("Sensor 3 (T30)", min_value=1580.0, max_value=1620.0, value=1590.0, step=1.0)
        last_cycle_inputs['sensor_4'] = col_input[2].slider("Sensor 4 (T50)", min_value=1390.0, max_value=1420.0, value=1400.0, step=1.0)
        last_cycle_inputs['sensor_7'] = col_input[3].slider("Sensor 7 (P30)", min_value=550.0, max_value=560.0, value=555.0, step=0.1)
        predict_label = "Predict RUL"

        
    elif task_type == "CLASSIFICATION":
        st.success("Model Status: **ANN Binary Classification** trained and ready!")
        st.caption("Enter sensor values to predict **Immediate Failure Risk** (0 or 1).")
        
        # --- Input Fields for BINARY CLASSIFICATION (AI4I 2020 / Generic) ---
        st.subheader("Enter Machine Sensor Readings (Required Features)")
        col1, col2, col3 = st.columns(3)
        
        # NOTE: Using CLASSIFICATION_NUM_FEATURES and CLASSIFICATION_CAT_FEATURES for input labels
        with col1:
            product_type = st.selectbox("Product Quality Type", options=['L', 'M', 'H'], index=1)
            air_temp = st.slider("Air Temperature [K]", min_value=290.0, max_value=310.0, value=300.0, step=0.1)
        
        with col2:
            process_temp = st.slider("Process Temperature [K]", min_value=300.0, max_value=320.0, value=310.0, step=0.1)
            rotational_speed = st.number_input("Rotational Speed [rpm]", min_value=1200, max_value=2000, value=1500, step=10)
        
        with col3:
            torque = st.number_input("Torque [Nm]", min_value=10.0, max_value=70.0, value=40.0, step=0.5)
            tool_wear = st.slider("Tool Wear [min]", min_value=0, max_value=250, value=100, step=1)

        # Map Classification inputs to a dictionary structure
        last_cycle_inputs = {
            'Air temperature [K]': air_temp,
            'Process temperature [K]': process_temp,
            'Rotational speed [rpm]': rotational_speed,
            'Torque [Nm]': torque,
            'Tool wear [min]': tool_wear,
            'Type': product_type
        }
        predict_label = "Predict Failure Risk"
        
    # --- Prediction Execution ---
    predict_button = st.button(predict_label, key="predict_btn", type="secondary")
    
    if predict_button:
        with st.spinner('Calculating Prediction...'):
            try:
                if task_type == "REGRESSION":
                    # REGRESSION (LSTM) Prediction Logic
                    df_last_cycle = pd.DataFrame(index=[0])
                    
                    # Add simple inputs to DataFrame
                    for col_name, value in last_cycle_inputs.items():
                        df_last_cycle[col_name] = value
                    
                    # Fill in placeholders for setting/other sensor columns expected by NASA model
                    for col in features:
                        if col not in df_last_cycle.columns:
                            # Setting columns are 3-5, others are sensor columns
                            if col in RUL_SETTING_COLS:
                                df_last_cycle[col] = 0.0 # Settings are usually constants/dummy for prediction
                            elif col in RUL_SENSOR_COLS:
                                # Use a safe default value for less important sensors
                                df_last_cycle[col] = 500.0 if float(col.split('_')[1]) > 10 else 20.0
                            
                    df_last_cycle = df_last_cycle[features]
                    
                    # Scale, create sequence, and predict
                    input_scaled = scaler_or_preprocessor.transform(df_last_cycle.values)
                    X_pred = np.tile(input_scaled, (RUL_SEQUENCE_LENGTH, 1)).reshape(1, RUL_SEQUENCE_LENGTH, len(features))
                    predicted_rul = model.predict(X_pred, verbose=0)[0][0]
                    
                    # Display RUL Result
                    st.subheader("Prediction Result:")
                    if predicted_rul < 30:
                        st.error(f"⚠️ CRITICAL FAILURE ZONE: RUL is predicted to be {predicted_rul:.1f} cycles.")
                    else:
                        st.success(f"✅ Healthy Operation.")
                        st.metric("Predicted Remaining Useful Life (RUL)", f"{predicted_rul:.1f} Cycles", "High")
                        
                elif task_type == "CLASSIFICATION":
                    # CLASSIFICATION (ANN) Prediction Logic
                    input_data = pd.DataFrame([last_cycle_inputs])
                    input_processed = scaler_or_preprocessor.transform(input_data)
                    prediction_prob = model.predict(input_processed, verbose=0)[0][0]
                    prediction_class = 1 if prediction_prob > 0.5 else 0

                    st.subheader("Prediction Result:")
                    if prediction_class == 1:
                        st.warning(f"🚨 FAILURE ALERT: Immediate Action Recommended!")
                        st.markdown(f"**Predicted Probability of Failure:** `{prediction_prob:.2%}`")
                    else:
                        st.success(f"✅ Machine Operating Normally (Low Risk)")
                        st.markdown(f"**Predicted Probability of Failure:** `{prediction_prob:.2%}`")

            except Exception as e:
                
                st.error(f"Prediction failed due to processing error: {e}")