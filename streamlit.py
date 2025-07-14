#pip install streamlit
import streamlit as st
import pandas as pd
import joblib
import os
st.set_page_config(page_title="TeleCos Churn", layout="wide")

pipeline_path = "artefactos/preprocessor/preprocessor.pkl"
model_path = "artefactos/model/XGBoost.pkl"
encoder_path = "artefactos/preprocessor/label_encoder.pkl"

st.write("Directorio actual:", os.getcwd())
st.write("Archivos en artefactos/model:", os.listdir("artefactos/model"))
st.write("Existe modelo:", os.path.exists(model_path))


with open(pipeline_path, "rb") as file1:
    print(file1.read(100))
    

try:
    pipeline= joblib.load(pipeline_path)
    print("pipeline cargada")
    st.write("pipeline cargada")
except Exception as e:
    print(f'Error al cargar el pipeline {e}')
    

with open(model_path, "rb") as file2:
    print(file2.read(100))
try:
    model = joblib.load(model_path)
    print("modelo Cargado")
    st.write("modelo cargado")
except Exception as e:
    print(f"error al cargar modelo: {e}")


with open(encoder_path, "rb") as file3:
    print(file3.read(100))
try:
    encoder = joblib.load(encoder_path)
    print("codificador cargado")
    st.write("encoder cargado")
except Exception as e:
    print(f"fallo al cargar el encoder {e}")


###################################################################
st.title("Prediccion de Churn")
st.header("ingrese los datos")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Género", options=["Male", "Female"])
    SeniorCitizen = st.selectbox("Es adulto mayor (1=Sí, 0=No)", options=[0,1])
    Partner = st.selectbox("Tiene pareja?", options=["Yes", "No"])
    Dependents = st.selectbox("Tiene dependientes?", options=["Yes", "No"])
    tenure = st.slider("Meses con la empresa (Tenure)", min_value=1, max_value=72)

with col2:
    PhoneService = st.selectbox("Tiene servicio telefónico?", options=["Yes", "No"])
    MultipleLines = st.selectbox("Tiene múltiples líneas?", options=["No phone service", "No", "Yes"])
    InternetService = st.selectbox("Tipo de servicio de internet", options=["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Seguridad en línea?", options=["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("Respaldo en línea?", options=["No", "Yes", "No internet service"])

with col3:
    DeviceProtection = st.selectbox("Protección del dispositivo?", options=["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("Soporte técnico?", options=["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("Streaming de TV?", options=["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming de películas?", options=["No", "Yes", "No internet service"])
    Contract = st.selectbox("Tipo de contrato", options=["Month-to-month", "One year", "Two year"])

# Y si tienes más campos, puedes hacer más columnas (col4, col5...) o ponerlos abajo
PaperlessBilling = st.selectbox("Facturación sin papel?", options=["Yes", "No"])
PaymentMethod = st.selectbox("Método de pago", options=[
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

MonthlyCharges = st.number_input("Cobro mensual", min_value=18.0, max_value=118.75, value=50.0, step=0.1)
TotalCharges = st.number_input("Cobro total", min_value=18.0, max_value=8684.80, value=500.0, step=0.1)



if st.button("Predecir Churn"):
    
    input_data = pd.DataFrame(
        {
            "gender": [gender],
            "SeniorCitizen": [SeniorCitizen],
            "Partner": [Partner],
            "Dependents": [Dependents],
            "tenure": [tenure],
            "PhoneService": [PhoneService],
            "MultipleLines": [MultipleLines],
            "InternetService": [InternetService],
            "OnlineSecurity": [OnlineSecurity],
            "OnlineBackup": [OnlineBackup],
            "DeviceProtection": [DeviceProtection],
            "TechSupport": [TechSupport],
            "StreamingTV": [StreamingTV],
            "StreamingMovies": [StreamingMovies],
            "Contract": [Contract],
            "PaperlessBilling": [PaperlessBilling],
            "PaymentMethod": [PaymentMethod],
            "MonthlyCharges": [MonthlyCharges],
            "TotalCharges": [TotalCharges]
        }
    )
        
    st.dataframe(input_data)
    pipeline_data = pipeline.transform(input_data)
    prediction = model.predict(pipeline_data)


    if prediction[0] == 1:
        st.error("El cliente tiene riesgo de churn (abandono).")
    else:
        st.success("El cliente probablemente se quedará.")
