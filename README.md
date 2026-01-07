# 🛡️ AI-Powered Network Intrusion Detection System (AI-NIDS)

An interactive dashboard built with **Streamlit** for real-time network traffic analysis and intrusion detection using the **Random Forest** machine learning algorithm.

## ✨ Project Overview

This project simulates a Network Intrusion Detection System (NIDS) that leverages machine learning to distinguish between safe (benign) network traffic and potentially malicious traffic (intrusions/attacks).

The system uses synthesized network flow features (like `Flow Duration`, `Total Fwd Packets`, `Packet Length Mean`, etc.) to train a classification model. The goal is to provide a clear, interactive interface for training the model, evaluating its performance, and testing its detection capabilities.

## Project Link

[https://ainids-pdas.streamlit.app/](https://ainids-pdas.streamlit.app/)

### 🎯 Key Features

* **Interactive Dashboard:** Built with Streamlit for an easy-to-use interface.
* **Machine Learning Model:** Utilizes the **Random Forest Classifier** for high-accuracy binary classification.
* **Customizable Parameters:** Users can adjust the training split size and the number of estimators (trees) in the Random Forest.
* **Comprehensive Evaluation:** Displays accuracy, mean errors, classification reports, and **Confusion Matrices** for both training and test data.
* **Live Traffic Simulation:** An interactive section to input new packet features and get a real-time intrusion prediction from the trained AI model.
* **Real World NID (DDoS) Dataset:** Option for user to select between **Simulation** and **Real-time** to train the model with the simulated data or the real-world data respectively.

## 🛠️ Technology Stack

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Dashboard** | Streamlit | Frontend interface and interactivity. |
| **Machine Learning** | Scikit-learn (sklearn) | Model training (Random Forest) and performance metrics. |
| **Data Handling** | Pandas, NumPy | Data generation, manipulation, and feature engineering. |
| **Visualization** | Matplotlib, Seaborn | Generating confusion matrices and data visualizations. |

## 🚀 Getting Started

Follow these steps to set up and run the AI-NIDS dashboard locally.

### Prerequisites

You need **Python 3.x** installed on your system.

### Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone https://github.com/PERSONAL-PROJECTS-2000/AINIDS.git
    cd AINIDS
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    The project uses the following dependencies:
    ```bash
    pip install streamlit pandas numpy scikit-learn seaborn matplotlib
    ```

### Running the Dashboard

1.  **Save the provided code:** Save the Python code (e.g., in a file named `app.py`).

2.  **Run the Streamlit application:**
    ```bash
    streamlit run nmain.py
    ```
    This command will open the dashboard in your default web browser (usually at `http://localhost:8501`).

## 👨‍💻 How to Use the Dashboard

The application is divided into three main sections:

### ⚙️ Control Panel (Sidebar)

On the left sidebar, you can set the key parameters for the model:

* **Training Data Size (%):** Adjusts the percentage of the simulated dataset used for model training (e.g., 80% for training, 20% for testing).
* **Number of Trees (Random Forest):** Sets the `n_estimators` hyperparameter for the Random Forest Classifier, controlling the complexity and performance of the model.

### 1. Model Training

* Select the **"Mode"**.
* Click the **"Train Model Now"** button to train the Random Forest model using the current parameters.
* *Note: Model training must be completed before proceeding to testing and prediction.*

### 2. Performance Metrics

Once the model is trained, this section displays a detailed analysis of its performance:

* **Metrics:** Training Accuracy, Test Accuracy, and the count of detected threats.
* **Mean Errors:** Tables showing Mean Absolute Error (MAE) and Mean Squared Error (MSE).
* **Classification Reports:** Detailed precision, recall, and F1-score for each class (Safe/Malicious).
* **Confusion Matrices:** Visual representation of true positives, true negatives, false positives (Type I error), and false negatives (Type II error) for both data sets. 

### 3. Live Traffic Simulator (Test the AI)

This section allows you to manually input network flow features to test the trained model's prediction:

* **Input Fields:** Enter simulated values for `Flow Duration`, `Total Packets`, `Packet Length Mean`, and `Active Mean Time`.
* **Analyze Packet:** Click the button to get the model's classification:
    * **Success (Green):** Classified as **BENIGN (Safe)**.
    * **Error (Red):** Classified as **MALICIOUS TRAFFIC DETECTED!**

## 📂 Code Structure Highlights

* **`load_data()`:** Generates a synthetic dataset for demonstration or loads the real-time dataset. It ensures malicious (Label=1) data points have characteristic features (e.g., higher `Total_Fwd_Packets`, lower `Flow_Duration`) to make the intrusion detection task feasible for the AI.
* **`@SL.cache_data`:** Used for the data loading function to prevent re-running the data generation every time the app interacts, ensuring performance efficiency.
* **Random Forest (`RFC`):** The core algorithm chosen for its balance of high accuracy and interpretability in classification tasks like NIDS.
