import streamlit as SL
import pandas as pa
import numpy as nu
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report as CR, confusion_matrix as CM, ConfusionMatrixDisplay as CMD, accuracy_score as acs, mean_absolute_error as MAE, mean_squared_error as MSE
import seaborn as sb
import matplotlib.pyplot as mp

SL.set_page_config(page_title="AI NIDS Dashboard", layout="wide")

SL.title("AI-Powered Network Intrusion Detection System")
SL.markdown("""
### Project Overview
This system uses Machine Learning (**Random Forest Algorithm**) to analyze network traffic in 
real-time.
It classifies traffic into two categories:
* **Benign (0):** Safe, normal traffic.
* **Malicious (1):** Potential cyberattacks (DDoS, Port Scan, etc.).
""")

@SL.cache_data
def load_data():
    nu.random.seed(42)
    n_samples = 5000
    data = {
    'Destination_Port': nu.random.randint(1, 65535, n_samples),
    'Flow_Duration': nu.random.randint(100, 100000, n_samples),
    'Total_Fwd_Packets': nu.random.randint(1, 100, n_samples),
    'Packet_Length_Mean': nu.random.uniform(10, 1500, n_samples),
    'Active_Mean': nu.random.uniform(0, 1000, n_samples),
    'Label': nu.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    }
    df = pa.DataFrame(data)
    df.loc[df['Label'] == 1, 'Total_Fwd_Packets'] += nu.random.randint(50, 200, size=df[df['Label']==1].shape[0])
    df.loc[df['Label'] == 1, 'Flow_Duration'] = nu.random.randint(1, 1000, size=df[df['Label']==1].shape[0])
    return df
        
SL.sidebar.header("Control Panel")
SL.sidebar.info("Adjust the model parameters here.")
split_size = SL.sidebar.slider("Training Data Size (%)", 50, 90, 80)
n_estimators = SL.sidebar.slider("Number of Trees (Random Forest)", 10, 200, 100)

df = load_data()

X = df.drop('Label', axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = tts(X, y, test_size=(100-split_size)/100, 
random_state=42)

SL.divider()
with SL.container(border=True):
    SL.subheader("1. Model Training")
    if SL.button("Train Model Now"):
        with SL.spinner("Training Random Forest Classifier..."):
            model = RFC(n_estimators=n_estimators)
            model.fit(X_train, y_train)
            SL.session_state['model'] = model 
            SL.success("Training Complete!")
    if 'model' in SL.session_state:
        SL.success("Model is Ready for Testing")

with SL.container(border=True):
    SL.subheader("2. Performance Metrics")
    if 'model' in SL.session_state:
        model = SL.session_state['model']
        y_pred = model.predict(X_train)
        y_tpred = model.predict(X_test)
        acc = acs(y_train, y_pred)
        mae = MAE(y_train, y_pred)
        mse = MSE(y_train, y_pred)
        acct = acs(y_test, y_tpred)
        maet = MAE(y_test, y_tpred)
        mset = MSE(y_test, y_tpred)
        m1, m2, m3, m4, m5  = SL.columns(5)
        m1.metric("Total Samples: ", len(df))
        m2.metric("Training Accuracy: ", f"{acc*100:.2f}%")
        m3.metric("Detected Threats (training): ", nu.sum(y_pred))
        m4.metric("Test Accuracy: ", f"{acct*100:.2f}%")
        m5.metric("Detected Threats (test): ", nu.sum(y_tpred))
        acm = {'Mean Accuracy Error': [mae, maet], 'Mean Squared Error': [mse, mset]}
        mns = pa.DataFrame(acm, index=['Training Data', 'Test Data'])
        SL.write("#### MEAN ERRORS")
        SL.dataframe(mns, use_container_width=True)
        SL.divider()
        SL.write("### CLASSIFICATION REPORTS")
        SL.write("#### TRAINING DATA:- ")
        SL.dataframe(CR(y_train, y_pred, output_dict=True))
        SL.write("#### TEST DATA:- ")
        SL.dataframe(CR(y_test, y_tpred, output_dict=True))
        SL.divider()
        SL.write("### CONFUSION MATRICES")
        SL.write("#### TRAINING DATA:- ")
        cm = CM(y_train, y_pred)
        dis=CMD(confusion_matrix=cm, display_labels=['Safe(0)', 'Malicious(1)'])
        fig, ax = mp.subplots(figsize=(4, 2))
        dis.plot(ax=ax, cmap='Blues')
        mp.title("Training Data")
        SL.pyplot(fig)
        SL.write("\n\n\n")
        SL.write("#### TEST DATA:- ")
        cmt = CM(y_test, y_tpred)
        dist=CMD(confusion_matrix=cmt, display_labels=['Safe(0)', 'Malicious(1)'])
        figt, ax = mp.subplots(figsize=(4, 2))
        dist.plot(ax=ax, cmap='Reds')
        mp.title("Test Data")
        SL.pyplot(figt)
    else:
        SL.warning("Please train the model first.")

SL.divider()
with SL.container(border=True):
    SL.subheader("3. Live Traffic Simulator (Test the AI)")
    SL.write("Enter network packet details below to see if the AI flags it as an attack.")
    c1, c2, c3, c4 = SL.columns(4)
    p_dur = c1.number_input("Flow Duration (ms)", 0, 100000, 500)
    p_pkts = c2.number_input("Total Packets", 0, 500, 100)
    p_len = c3.number_input("Packet Length Mean", 0, 1500, 500)
    p_active = c4.number_input("Active Mean Time", 0, 1000, 50)
    if SL.button("Analyze Packet"):
        if 'model' in SL.session_state:
            model = SL.session_state['model']
            input_data = nu.array([[80, p_dur, p_pkts, p_len, p_active]])
            pred = model.predict(input_data)
            if pred[0] == 1:
                SL.error("ALERT: MALICIOUS TRAFFIC DETECTED!")
                SL.write("**Reason:** High packet count with low duration is suspicious.")
            else:
                SL.success("Traffic Status: BENIGN (Safe)")
        else:
            SL.error("Please train the model first!")