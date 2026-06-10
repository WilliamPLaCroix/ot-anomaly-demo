# OT Network Anomaly Detection with Autoencoders

An end-to-end machine learning pipeline for detecting anomalous behavior in Industrial Control System (ICS) network traffic using an autoencoder-based anomaly detection model.

## Overview

Industrial control networks typically exhibit highly regular communication patterns. Devices such as PLCs, sensors, and HMIs often communicate with the same endpoints at predictable intervals using a limited set of protocols.

This project demonstrates how unsupervised learning can be used to establish a baseline of normal network behavior and identify deviations that may indicate misconfiguration, malfunction, or malicious activity.

The system trains an autoencoder on normal network traffic and uses reconstruction error as an anomaly score during inference.

## Objectives

* Preprocess industrial network traffic data
* Engineer flow-level behavioral features
* Train an autoencoder on normal traffic
* Detect anomalous communications using reconstruction error
* Visualize anomaly scores through an interactive dashboard

## Dataset
 
This project uses the [Secure Water Treatment (SWaT) dataset](https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system/), a widely used benchmark for ICS security research.

The dataset contains:

* Normal operational periods
* Simulated attack scenarios
* Sensor and process measurements
* Industrial network communications

## Pipeline

### 1. Data Ingestion

Raw network records are loaded from the source dataset and transformed into a structured tabular format.

### 2. Feature Engineering

Example features include:

* Timestamp — Date and time of the recorded data point
* FIT101 — Flow Indicator Transmitter at stage 1
* LIT101 — Level Indicator Transmitter at stage 1
* MV101 — Motorized Valve at stage 1
* P101, P102 — Pumps at stage 1

Features are normalized prior to model training. Timestamp is not used for training.

### 3. Baseline Learning

The autoencoder is trained exclusively on normal traffic.

The model learns a compressed representation of expected network behavior.

### 4. Anomaly Detection

During inference:

1. The model reconstructs incoming observations
2. Reconstruction error is calculated as MSE
3. Observations exceeding a threshold are flagged as anomalies

### 5. Visualization  #TODO

A Streamlit dashboard displays:

* Traffic statistics
* Reconstruction error distributions
* Detected anomalies
* Temporal anomaly trends

## Model Architecture

A simple fully connected autoencoder is used:

Input Layer
→ Dense(25)
→ Dense(12)
→ Latent Space(5)
→ Dense(12)
→ Dense(25)
→ Reconstruction Layer

The objective is to minimize mean squared reconstruction error: mean((x - x_hat)^2)

## Example Workflow

Normal Traffic
↓
Feature Engineering
↓
Autoencoder Training
↓
Learn Baseline Behavior

New Traffic
↓
Feature Engineering
↓
Autoencoder Inference
↓
Reconstruction Error
↓
Anomaly Score
↓
Alert

## Repository Structure

```text
ot-anomaly-demo/
├── data/ # temporary data cache, full CSV should not live here
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── infer.py
├── models/
│   ├── autoencoder.pth # saved model checkpoint
│   ├── training_errors.pt # saved training errors for validation thresholding
│   └── autoencoder.py
├── app/
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/train.py
```

Training and evaluation take ~3min on NVIDIA GTX 1650 Ti

Launch the dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Results

The model learns baseline traffic patterns and identifies observations that differ substantially from normal operational behavior.
Detected anomalies should appear as spikes in reconstruction error and can be investigated through the dashboard interface.

Initial F1 score: 0.2930

![MVP Confusion Matrix][../data/confusion_matrix.png]

## Limitations

* Data leakage: MVP trains on normal network traffic, and evaluates on merged normal+attack traffic. All normal traffic already seen in training data
* Network structure: Autoencoder layer sizes have not been tuned
* Threshold value: MVP Error threshold set to 95th percentile of training errors. We expect 5% false negatives, current model gives ~10% false negatives, and ~40% false positives.
* Meaned training errors: training errors are batch-meaned (n=256) during training, so thresholding calculates percentile against batch-meaned errors, rather than sample-wise training errors.
* Raw features: MVP features only include raw inputs without any engineered features based on timeseries data. Network anomalies will likely be best detected by *change in network traffic*, rather than simply raw data at any given point. Feature engineering for statistical differences as well as change over time should improve model performance.
* No dimensionality reduction: MVP includes all 51 input features from original dataset without any EDA for feature selection (PCA, correlation/dependency, ANOVA F-value).

## Technologies

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib

## Motivation

This project was developed to explore machine learning applications in operational technology (OT) and industrial cybersecurity. The focus is on practical anomaly detection workflows that bridge data engineering, machine learning, and security monitoring.
