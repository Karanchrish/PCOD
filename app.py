# app.py (no changes needed to the Flask backend)
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model_joblib.pkl')

features = [
    'Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'Cycle(R/I)', 'Cycle length(days)',
    'Marraige Status (Yrs)', 'Pregnant(Y/N)', 'No. of abortions',
    'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)',
    'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)'
]

def get_risk_category(probability):
    if probability <= 20:
        return ("No Significant Risk", "You show minimal signs of PCOS. Maintain a healthy lifestyle.", "#4CAF50")
    elif probability <= 40:
        return ("Low Risk", "Be mindful of symptoms. Prevention is better than cure.", "#8BC34A")
    elif probability <= 60:
        return ("Moderate Risk", "You may want to consult a gynecologist for evaluation.", "#FFC107")
    elif probability <= 80:
        return ("High Risk", "Please consult a doctor for proper diagnosis and treatment.", "#FF9800")
    else:
        return ("Very High Risk", "Immediate medical consultation is strongly recommended.", "#F44336")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    input_data = pd.DataFrame([data])
    
    # Data processing
    numeric_features = ['Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'Cycle length(days)', 
                       'Marraige Status (Yrs)', 'No. of abortions']
    for feature in numeric_features:
        input_data[feature] = pd.to_numeric(input_data[feature])
        
    binary_features = ['Pregnant(Y/N)', 'Weight gain(Y/N)', 'hair growth(Y/N)', 
                     'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)', 
                     'Fast food (Y/N)', 'Reg.Exercise(Y/N)']
    for feature in binary_features:
        input_data[feature] = input_data[feature].astype(int)
        
    input_data['Cycle(R/I)'] = 1 if input_data['Cycle(R/I)'].iloc[0].lower() == 'irregular' else 0
    input_data = input_data[features]
    
    # Prediction
    probability = model.predict_proba(input_data)[0][1] * 100
    category, advice, color = get_risk_category(probability)
    
    result = {
        'probability': round(probability, 2),
        'category': category,
        'advice': advice,
        'color': color
    }
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
