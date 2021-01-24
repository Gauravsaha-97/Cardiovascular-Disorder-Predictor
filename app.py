#Importing necessary libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

#Setting the header components

header = """
<div style="background-color:#ff751a;padding:10px;border-radius:10%;box-shadow: 5px 10px #888888">
<h1 style="color:white;text-align:center">Cardiovascular Disorder Predictor</h1>
</div>
<br>
</body>
"""
st.markdown(header, unsafe_allow_html=True)

#Reading the dataset('heart.csv')

heart_df = pd.read_csv('heart.csv')

#Seperating the dependent(Y) and Independent(X) features

X = heart_df.iloc[:,:-1].values
Y = heart_df.iloc[:,-1].values

#Preprocessing the data

scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

#Splitting the data into training and testing sets (Train size: 70%, Test size: 30%)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled,Y,test_size=0.3, random_state=47)

#Function to take input from the user

def user_input():
    header_text = """
    <h2 style="color:#4d1f00;font-weight:600">Please Enter Your Information</h1>
    </div>
    <br>
    <br>
    """
    st.markdown(header_text, unsafe_allow_html=True)

    age = st.number_input('Enter your Age',0,100,0)
    sex = st.selectbox('Select your Gender',("Male", "Female"))
    cp = st.radio('Chest Pain Type',("Typical angina","Atypical angina","Non-anginal pain","Asymptomatic"))
    trestbps = st.slider('Resting Blood Pressure(mm/Hg)',94,200,100)
    chol = st.slider('Serum Cholestoral(mg/dl)',100,700,150)
    fbs = st.radio('Fasting Blood Sugar',("Lower than 120 mg/dl", "Greater than 120 mg/dl"))
    restecg = st.radio('Resting Electrocardiographic Results',("Normal", "ST-T wave abnormality",'Left ventricular hypertrophy'))
    thalach =  st.slider('Maximum heart rate achieved',60,250,100)
    exang = st.radio('Exercise Induced Angina',("No", "Yes"))
    oldpeak = st.slider('ST depression induced by exercise relative to rest',0.0,8.0,4.0)
    slope = st.radio('The slope of the peak exercise ST segment',("Upsloping", "Flat", "Downsloping"))
    ca = st.slider('Number of major vessels (0-3) colored by Flourosopy',0,3,1)
    thal = st.radio('Thalassemia',('Normal','Fixed defect','Reversable defect'))

    user_data = {}
    user_input_data = {
        'Age': age, 'Sex': sex, 'Chest Pain Type': cp, 'Resting Blood Pressure(mm/Hg)': trestbps, 'Serum Cholestoral(mg/dl)': chol,'Fasting Blood Sugar':fbs,
    'Resting Electrocardiographic Results': restecg,'Maximum heart rate achieved':thalach,'Exercise Induced Angina':exang,'ST depression':oldpeak,
    'Slope':slope,'Number of major vessels':ca, 'Thalassemia': thal
    }

    user_data['Age'] = age

    if sex=="Male":
        user_data['Sex'] = 1
    else:
        user_data['Sex']=0

    if cp=="Typical angina":
        user_data['Chest Pain Type'] = 0
    elif cp=="Atypical angina":
        user_data['Chest Pain Type']=1
    elif cp=="Non-anginal pain":
        user_data['Chest Pain Type']=2
    elif cp=="Asymptomatic":
        user_data['Chest Pain Type']=3

    user_data['Resting Blood Pressure(mm/Hg)'] = trestbps

    user_data['Serum Cholestoral(mg/dl)'] = chol

    if fbs=="Lower than 120 mg/dl":
        user_data['Fasting Blood Sugar'] = 0
    elif fbs=="Greater than 120 mg/dl":
        user_data['Fasting Blood Sugar'] = 1

    if restecg=='Normal':
        user_data['Resting Electrocardiographic Results']=0
    elif restecg=='ST-T wave abnormality':
        user_data['Resting Electrocardiographic Results'] = 1
    elif restecg == 'Left ventricular hypertrophy':
        user_data['Resting Electrocardiographic Results'] = 2

    user_data['Maximum heart rate achieved'] = thalach

    if exang=='No':
        user_data['Exercise Induced Angina'] = 0
    elif exang=='Yes':
        user_data['Exercise Induced Angina'] = 1

    user_data['ST depression'] = oldpeak

    if slope=='Upsloping':
        user_data['Slope'] = 0
    elif slope == 'Flat':
        user_data['Slope'] = 1
    elif slope == 'Downsloping':
        user_data['Slope'] = 2

    user_data['Number of major vessels'] = ca

    if thal=='Normal':
        user_data['Thalassemia'] = 1
    elif thal=='Fixed defect':
        user_data['Thalassemia'] = 2
    elif thal=='Reversable defect':
        user_data['Thalassemia']=3

    features = pd.DataFrame(user_data, index = [0])
    user_input_features = pd.DataFrame(user_input_data, index = [0])

    return features, user_input_features

user_data_input, user_input_features = user_input()

#Initializing the KNN Classifier and do the prediction

classifier = KNeighborsClassifier(n_neighbors = 6)
classifier.fit(X_train, Y_train)
prediction = classifier.predict(user_data_input)

#Display the entered data and predicted results to the user

if st.button("Show Input Data"):
    user_input_header = """
        <h2 style="color:#4d1f00;font-weight:600">Please Check Your Details</h1>
        </div>
        <br>
        <br>
        """
    st.markdown(user_input_header, unsafe_allow_html=True)
    st.write(user_input_features)

if(st.button("Show Test Results")):
    st.subheader('Test Results:')
    if prediction == 0:
        st.success("Risk of Heart Disease: NO. Your heart is healthy")
    elif prediction == 1:
        st.error("Risk of Heart Disease: YES. Please consult a doctor")
