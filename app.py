import numpy as np

import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import taipy as taip
import taipy.gui.builder as tgb 

geography = None
gender = None
age = None
balance = None
credit_score = None
estimated_salary = None
tenure = None
noofproducts = None
has_cr_card = None
is_active_member = None
result = ""

model = tf.keras.models.load_model('model.h5')
with open('label_encoder_gender.pkl', 'rb') as file :
    LabelEncoderGender = pickle.load(file) 
with open('onehotencoder_geo.pkl', 'rb') as file :
    OneHotEncoderGeo = pickle.load(file)
with open('scaler.pkl', 'rb') as file :
    Scaler = pickle.load(file)

page = tgb.Page()

def reset_method(state):
    state.geography = None
    state.gender = None
    state.age = 18
    state.balance = None
    state.credit_score = None
    state.estimated_salary = None
    state.tenure = 0
    state.noofproducts = 0
    state.has_cr_card = None
    state.is_active_member = None


def model_prediction(state):
    global result
    print(state.geography, 
    state.gender, 
    state.age, 
    state.balance, 
    state.credit_score, 
    state.estimated_salary, 
    state.tenure, 
    state.noofproducts, 
    state.has_cr_card,
    state.is_active_member)

    if (state.geography is None  or  
    state.gender is None or
    state.age is None or
    state.balance is None or
    state.credit_score is None or
    state.estimated_salary is None or
    state.tenure is None or
    state.noofproducts is None or
    state.has_cr_card is None or
    state.is_active_member is None ):
        print("Invalid Input")
        return -1
    if state.is_active_member == "Yes":
        state.is_active_member = 1
    else:
        state.is_active_member = 0
    if state.has_cr_card == "Yes":
        state.has_cr_card = 1
    else:
        state.has_cr_card = 0
    input_data = {
    'CreditScore': int(state.credit_score),
    'Geography': state.geography,
    'Gender': state.gender,
    'Age': int(state.age),
    'Tenure': int(state.tenure),
    'Balance': int(state.balance),
    'NumOfProducts': int(state.noofproducts),
    'HasCrCard': state.has_cr_card,
    'IsActiveMember': state.is_active_member,
    'EstimatedSalary': int(state.estimated_salary),
    }
    geo_encoded = OneHotEncoderGeo.transform([[input_data['Geography']]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded,columns= OneHotEncoderGeo.get_feature_names_out(['Geography']))
    input_df = pd.DataFrame([input_data])
    input_df['Gender'] = LabelEncoderGender.transform(input_df['Gender'])
    input_df = pd.concat([input_df.drop("Geography", axis = 1),geo_encoded_df],axis = 1)
    ip_scaled = Scaler.transform(input_df)
    prediction = model.predict(ip_scaled)
    state.result = "The customer is not likely to churn"
    if(prediction[0][0] > 0.5):
        state.result = "The customer is likely to churn"

    # Update the page content based on the prediction
    return page.add(tgb.text("{result}"))









page.add(tgb.text("Select the Geography:"))
page.add(tgb.selector(dropdown= True,label="Select",value="{geography}",lov=["France", "Germany", "Spain"]))
page.add(tgb.text("Select the gender:"))
page.add(tgb.selector(dropdown= True,label="Select",value="{gender}",lov=["Male", "Female"]))
page.add(tgb.text("Select Your Current Age"))
page.add(tgb.text("{age}"))
page.add(tgb.slider(min=18, max=100, value="{age}", label="Age"))
page.add(tgb.text("Enter your Balance:"))
page.add(tgb.number(value="{balance}", label="Balance"))
page.add(tgb.text("Enter your Credit Score:"))
page.add(tgb.number(value="{credit_score}", label="Credit Score"))
page.add(tgb.text("Enter your Estimeted Salary:"))
page.add(tgb.number(value="{estimated_salary}", label="Estimated Salary"))
page.add(tgb.text("Tell Us Your account tenure Left"))
page.add(tgb.slider(min=0, max=10, value="{tenure}", label="Tenure"))
page.add(tgb.text("Tell us your number of products"))
page.add(tgb.slider(min=1, max=10, value="{noofproducts}", label="Number of Products"))
page.add(tgb.text("Did you have a credit card"))
page.add(tgb.toggle(value="{has_cr_card}", lov=["Yes", "No"]))
page.add(tgb.text("Are you currently a active member"))
page.add(tgb.toggle(value="{is_active_member}", lov=["Yes", "No"]))
page.add(tgb.button(label="Check My Prediction", on_action=model_prediction))
page.add(tgb.button(label="Reset", on_action=reset_method))
page.add(tgb.text("{result}"))



    


if __name__ == '__main__':
    taip.Gui(page).run(debug=True, use_reloader= True)