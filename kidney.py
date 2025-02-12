import pickle
import streamlit as st
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Chronic Kidney Disease Predictor",
    page_icon="🏥",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    try:
        return pickle.load(open('model/model_knn.sav', 'rb'))
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'model_knn.sav' exists in the model directory.")
        return None

# Input validation functions
def validate_numeric_input(value, min_val, max_val, field_name):
    try:
        val = float(value)
        if val < min_val or val > max_val:
            return None, f"{field_name} should be between {min_val} and {max_val}"
        return val, None
    except ValueError:
        return None, f"{field_name} should be a numeric value"

def validate_categorical_input(value, valid_options, field_name):
    if value.lower() not in valid_options:
        return None, f"{field_name} should be one of: {', '.join(valid_options)}"
    return value.lower(), None

# Encoding mappings for categorical variables
categorical_encodings = {
    'red_blood_cells': {'normal': 1, 'abnormal': 0},
    'pus_cell': {'normal': 1, 'abnormal': 0},
    'pus_cell_clumps': {'present': 1, 'notpresent': 0},
    'bacteria': {'present': 1, 'notpresent': 0},
    'hypertension': {'yes': 1, 'no': 0},
    'diabetes_mellitus': {'yes': 1, 'no': 0},
    'coronary_artery_disease': {'yes': 1, 'no': 0},
    'appetite': {'good': 1, 'poor': 0},
    'pedal_edema': {'yes': 1, 'no': 0},
    'anemia': {'yes': 1, 'no': 0}
}

# Define input fields with their validation rules
input_categories = {
    'Demographic Data': {
        'age': {'type': 'numeric', 'min': 0, 'max': 120, 'label': 'Age'},
        'blood_pressure': {'type': 'numeric', 'min': 50, 'max': 200, 'label': 'Blood Pressure (mm Hg)'},
    },
    'Laboratory Results': {
        'specific_gravity': {'type': 'numeric', 'min': 1.005, 'max': 1.025, 'label': 'Specific Gravity'},
        'albumin': {'type': 'numeric', 'min': 0, 'max': 5, 'label': 'Albumin (g/dL)'},
        'sugar': {'type': 'numeric', 'min': 0, 'max': 5, 'label': 'Sugar Level'},
        'red_blood_cells': {'type': 'categorical', 'options': ['normal', 'abnormal'], 'label': 'Red Blood Cells'},
        'pus_cell': {'type': 'categorical', 'options': ['normal', 'abnormal'], 'label': 'Pus Cell'},
        'pus_cell_clumps': {'type': 'categorical', 'options': ['present', 'notpresent'], 'label': 'Pus Cell Clumps'},
        'bacteria': {'type': 'categorical', 'options': ['present', 'notpresent'], 'label': 'Bacteria'},
        'blood_glucose_random': {'type': 'numeric', 'min': 70, 'max': 400, 'label': 'Blood Glucose Random (mg/dL)'},
        'blood_urea': {'type': 'numeric', 'min': 10, 'max': 200, 'label': 'Blood Urea (mg/dL)'},
        'serum_creatinine': {'type': 'numeric', 'min': 0.4, 'max': 15, 'label': 'Serum Creatinine (mg/dL)'},
        'sodium': {'type': 'numeric', 'min': 100, 'max': 150, 'label': 'Sodium (mEq/L)'},
        'potassium': {'type': 'numeric', 'min': 2.5, 'max': 7, 'label': 'Potassium (mEq/L)'},
        'hemoglobin': {'type': 'numeric', 'min': 3.5, 'max': 17.5, 'label': 'Hemoglobin (g/dL)'},
        'packed_cell_volume': {'type': 'numeric', 'min': 15, 'max': 55, 'label': 'Packed Cell Volume (%)'},
        'white_blood_cell_count': {'type': 'numeric', 'min': 2000, 'max': 30000, 'label': 'White Blood Cell Count (/mm³)'},
        'red_blood_cell_count': {'type': 'numeric', 'min': 2, 'max': 8, 'label': 'Red Blood Cell Count (millions/mm³)'},
    },
    'Medical History': {
        'hypertension': {'type': 'categorical', 'options': ['yes', 'no'], 'label': 'Hypertension'},
        'diabetes_mellitus': {'type': 'categorical', 'options': ['yes', 'no'], 'label': 'Diabetes Mellitus'},
        'coronary_artery_disease': {'type': 'categorical', 'options': ['yes', 'no'], 'label': 'Coronary Artery Disease'},
        'appetite': {'type': 'categorical', 'options': ['good', 'poor'], 'label': 'Appetite'},
        'pedal_edema': {'type': 'categorical', 'options': ['yes', 'no'], 'label': 'Pedal Edema'},
        'anemia': {'type': 'categorical', 'options': ['yes', 'no'], 'label': 'Anemia'}
    }
}

# Create a flat list of field names in the correct order
input_fields = []
for category in input_categories.values():
    input_fields.extend(category.keys())

def encode_input(field_name, value):
    """Convert input values to numeric format"""
    if field_name in categorical_encodings:
        return categorical_encodings[field_name][value]
    return float(value)

def main():
    # Load model
    model = load_model()
    if not model:
        return

    # Title and description
    st.title('🏥 Chronic Kidney Disease Predictor')
    st.markdown("""
    This application helps predict the likelihood of chronic kidney disease based on various medical parameters.
    Please fill in all the fields below with the patient's information.
    """)

    # Create form
    with st.form("prediction_form"):
        input_values = {}
        
        # Create columns for each category
        for category_name, fields in input_categories.items():
            st.subheader(category_name)
            cols = st.columns(4)
            col_idx = 0
            
            for field_name, field_info in fields.items():
                with cols[col_idx]:
                    if field_info['type'] == 'categorical':
                        value = st.selectbox(
                            field_info['label'],
                            options=[''] + list(field_info['options']),
                            key=field_name
                        )
                    else:
                        value = st.text_input(field_info['label'], key=field_name)
                    input_values[field_name] = value
                
                col_idx = (col_idx + 1) % 4

        # Submit button
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Validate all inputs
        valid_inputs = {}
        has_errors = False

        # Flatten the categories to get all fields
        all_fields = {}
        for category in input_categories.values():
            all_fields.update(category)

        for field_name, value in input_values.items():
            field_info = all_fields[field_name]
            
            if not value:
                st.error(f"Please fill in {field_info['label']}")
                has_errors = True
                continue

            if field_info['type'] == 'numeric':
                val, error = validate_numeric_input(
                    value,
                    field_info['min'],
                    field_info['max'],
                    field_info['label']
                )
            else:
                val, error = validate_categorical_input(
                    value,
                    field_info['options'],
                    field_info['label']
                )

            if error:
                st.error(error)
                has_errors = True
            else:
                valid_inputs[field_name] = val

        if not has_errors:
            try:
                # Prepare input for model using the ordered list of fields
                input_array = [encode_input(field, valid_inputs[field]) for field in input_fields]
                
                # Make prediction
                prediction = model.predict([input_array])
                
                # Display result
                if prediction[0] == 1:
                    st.success("🟢 Patient is NOT likely to have Chronic Kidney Disease")
                else:
                    st.error("🔴 Patient is likely to have Chronic Kidney Disease")
                
                # Display warning
                st.warning("""
                    ⚠️ This prediction is for informational purposes only and should not be used as a substitute 
                    for professional medical advice. Please consult with a healthcare provider for proper diagnosis 
                    and treatment.
                    """)
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    main()