# import library yang dibutuhkan
import streamlit as st
import pandas as pd
import numpy  as np
import joblib



st.header('Model Deployment')
st.write("""
Created by Fitri Octaviani

Hck 006.
""")


# load best model
with open('rf_tuning.pkl','rb') as file_1:
    rf_tuning = joblib.load(file_1)


# define feature
features = ['age','anaemia','creatinine_phosphokinase',
    'high_blood_pressure','serum_creatinine']

def infer(df_infer):
    # predict result random forest model
    y_rf = rf_tuning.predict(df_infer)
    return y_rf 

st.header("Pasien yang Meninggal Namun Terprediksi Tidak Meninggal")

# artificial data infer
age                         = st.slider("Umur:",0,100)                 
anaemia                     = st.radio("Apakah penderita anaemia?",("Ya","Tidak"))
anaemia_                    = lambda anaemia: 1 if 'Ya' in anaemia else 0
creatinine_phosphokinase    = st.slider("Level enzim CPK dalam darah (mcg/L):",0,1000)
high_blood_pressure         = st.radio("Apakah penderita hipertensi?",("Ya","Tidak"))
high_blood_pressure_        = lambda high_blood_pressure: 1 if 'Ya' in high_blood_pressure else 0
serum_creatinine            = st.slider("Level serum creatinine dalam darah (mg/mL):",0.0,10.0)
   



if st.button("Submit"):
    F = {
    'age':age,
    'anaemia':anaemia_(anaemia),                    
    'creatinine_phosphokinase':creatinine_phosphokinase,         
    'high_blood_pressure':high_blood_pressure_(high_blood_pressure),                
    'serum_creatinine':serum_creatinine,             
    }
    
    # construct data inference dalam dataframe
    df_infer = pd.DataFrame(data=F,columns=features,index=[0])

    #panggil fungsi inference
    rf_pred = infer(df_infer)
    rf = ''

    # interpretasi hasil prediksi
    if rf_pred[0] == 0:
        rf = "Kemungkinan tidak meninggal"
    else:
        rf = "Kemungkinan meninggal"


    st.header(f"Hasil Prediksi: ")
    st.write(rf)