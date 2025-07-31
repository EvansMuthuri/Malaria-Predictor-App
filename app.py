import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
import base64
import os

# --- Helper Function for Text Symptom Parsing ---
def parse_text_symptoms(text_input, all_symptom_features):
    """
    Parses free-form text input to identify presence of symptoms.
    Returns a dictionary of symptoms (0 or 1) matching the model's features.
    """
    detected_symptoms = {s: 0 for s in all_symptom_features}
    text_input_lower = text_input.lower()

    # Define keywords for each symptom.
    # These keywords should map to your CSV column names (feature_names).
    symptom_keywords = {
        'fever': ['fever', 'hot', 'temperature', 'warm'],
        'headache': ['headache', 'head ache', 'head pain'],
        'chills': ['chills', 'shivering', 'cold sweats'],
        'fatigue': ['fatigue', 'tired', 'weary', 'exhausted', 'lack of energy'],
        'nausea_vomiting': ['nausea', 'vomiting', 'sick to stomach', 'throwing up', 'puking'], # Combined for your CSV
        'muscle_pain': ['muscle pain', 'body aches', 'muscle aches'],
        'diarrhea': ['diarrhea', 'loose stools'],
        'abdominal_pain': ['abdominal pain', 'stomach pain', 'belly ache'],
        'convulsions': ['convulsions', 'seizures', 'fits'],
        'coma': ['coma', 'unconscious', 'unresponsive'],
        'impaired_consciousness': ['impaired consciousness', 'confused', 'dizzy', 'drowsy', 'disoriented'],
        'anemia': ['anemia', 'pale', 'weak blood', 'low blood']
        # Add more if your CSV has more columns and you want to detect them
    }

    # Ensure symptom_keywords only includes features actually in the model's feature_names
    for feature in all_symptom_features:
        if feature in symptom_keywords: # Only process if we have keywords for this feature
            for keyword in symptom_keywords[feature]:
                if keyword in text_input_lower:
                    detected_symptoms[feature] = 1
                    break # Move to next symptom once a keyword is found
        # If a feature from the model is not in symptom_keywords, it defaults to 0 (not detected)

    return detected_symptoms

# --- Function to load and train model (from CSV) ---
@st.cache_resource # Cache the model loading/training for efficiency
def load_and_train_model():
    # Load the actual provided CSV data
    try:
        df_app = pd.read_csv('malaria_symptom_dataset.csv')
    except FileNotFoundError:
        st.error("Error: 'malaria_symptom_dataset.csv' not found. Please ensure the file is uploaded to your Colab environment.")
        st.stop() # Stop the app if data isn't available

    # Assuming 'Malaria' is the target column
    X_app = df_app.drop('Malaria', axis=1)
    y_app = df_app['Malaria']

    # Retrain the model with the same parameters as in the notebook
    model_app = DecisionTreeClassifier(max_depth=4, random_state=42, criterion='entropy', min_samples_leaf=5)
    model_app.fit(X_app, y_app)

    return model_app, X_app.columns.tolist() # Return model and feature names

# Load the model and feature names once
model, feature_names = load_and_train_model()

# --- Streamlit App Layout ---
st.set_page_config(
    page_title="ML for Kids: Predicting Malaria in Kenya", # Updated Title
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a better look (Orange theme)
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        # font-size: 3.2em;
        font-size: 20px;
        color: #FF6F00; /* Vibrant Orange */
        text-align: center;
        margin-bottom: 0.2em;
        font-weight: bold;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    /* Subheader styling */
    .subheader {
        font-size: 1.6em;
        color: #666666;
        text-align: center;
        margin-bottom: 1.8em;
        font-weight: normal;
    }
    /* Banner/Hero section styling */
    .hero-banner {
        background: linear-gradient(135deg, #FFA000 0%, #FF6F00 100%); /* Orange gradient */
        padding: 40px 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2em;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .hero-banner h1 {
        color: white;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .hero-banner p {
        font-size: 1.1em;
        opacity: 0.9;
    }

    /* Button styling */
    .stButton>button {
        background-color: #4CAF50; /* Green for action */
        color: white;
        padding: 12px 25px;
        border-radius: 8px;
        border: none;
        font-size: 1.2em;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease-in-out;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 3px 3px 8px rgba(0,0,0,0.3);
        transform: translateY(-2px);
    }
    /* Prediction box styling */
    .prediction-box {
        border-radius: 12px;
        padding: 25px;
        margin-top: 25px;
        font-size: 1.4em;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .prediction-positive {
        background-color: #ffebeb; /* Lighter red tint */
        color: #d90000; /* Stronger red */
        border: 2px solid #ff3333;
    }
    .prediction-negative {
        background-color: #ebffeb; /* Lighter green tint */
        color: #009900; /* Stronger green */
        border: 2px solid #33cc33;
    }
    .stInfo {
        background-color: #e0f2f7;
        border-left: 5px solid #2196F3;
        padding: 10px;
        border-radius: 5px;
        margin-top: 15px;
        font-size: 0.95em;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<h3 style='color: #FF6F00; text-align: center; margin-bottom: 0.2em; font-weight: bold; letter-spacing: -0.03em;'>Malaria Diagnosis Application</h3>", unsafe_allow_html=True)
# st.markdown("<p class='main-header'>Malaria Prediction Application</p>", unsafe_allow_html=True)
# st.markdown("<p class='subheader'>Predicting & Reducing Malaria Mortality in Kenya</p>", unsafe_allow_html=True)

# Hero Banner
st.markdown("""
<div class="hero-banner">
    <h1>Kids using Machine Learning to Predict Malaria in Kenya</h1>
    <p>Join us in using technology to make a difference in our community's health!</p>
</div>
""", unsafe_allow_html=True)



st.markdown("---")

# --- Input Method Selection ---
st.header("1. Choose How to Input Symptoms")
input_method = st.radio(
    "Select your preferred method:",
    ("Select from List (Checkboxes)", "Describe Symptoms (Text Input)"),
    key="input_method_radio"
)

# Initialize patient_symptoms dictionary with all features from the model
patient_symptoms_combined = {s: 0 for s in feature_names}

# --- Functionality 1: Checkbox Selection ---
if input_method == "Select from List (Checkboxes)":
    st.markdown("---")
    st.header("2. Tick the Symptoms You See")
    st.markdown("Please check all symptoms that apply to the patient:")

    num_cols = len(feature_names)
    cols = st.columns(3) # Use 3 columns for better layout

    for i, symptom in enumerate(feature_names):
        with cols[i % 3]: # Distribute checkboxes across 3 columns
            patient_symptoms_combined[symptom] = int(st.checkbox(symptom.replace('_', ' ').title(), value=False, key=f"cb_{symptom}"))

# --- Functionality 2: Text Input ---
elif input_method == "Describe Symptoms (Text Input)":
    st.markdown("---")
    st.header("2. Describe the Patient's Symptoms")
    st.markdown("Please list the symptoms the patient is experiencing in your own words (e.g., 'They have a high fever, a bad headache, and feel very tired.').")
    text_symptom_input = st.text_area("Type symptoms here:", height=100, key="text_symptom_area")

    if text_symptom_input:
        # Parse text input to update patient_symptoms_combined
        parsed_symptoms = parse_text_symptoms(text_symptom_input, feature_names)
        # Update patient_symptoms_combined with detected ones
        patient_symptoms_combined.update(parsed_symptoms)

        st.markdown("**Symptoms detected from your description:**")
        detected_count = 0
        for s, present in patient_symptoms_combined.items():
            if present == 1:
                st.write(f"- {s.replace('_', ' ').title()}")
                detected_count += 1
        if detected_count == 0:
            st.write("No specific symptoms detected from text. Please try different wording or use checkboxes.")

# --- Prediction Button ---
st.markdown("---")
st.header("3. Get Prediction")
predict_button = st.button("Predict Malaria Status")

# --- Prediction and Result Display ---
if predict_button:
    # Create DataFrame for prediction from the combined symptoms
    input_features_df = pd.DataFrame([patient_symptoms_combined])

    # Ensure columns are in the correct order for the model
    input_features_df = input_features_df[feature_names]

    prediction = model.predict(input_features_df)[0]
    prediction_proba = model.predict_proba(input_features_df)[0] # Get probabilities

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.markdown(f"<div class='prediction-box prediction-positive'>Malaria Predicted: <strong>YES</strong></div>", unsafe_allow_html=True)
        st.write(f"Based on the symptoms, there is a **{prediction_proba[1]*100:.2f}% likelihood** of malaria.")
        st.info("Please note: This is a machine learning prediction and not a medical diagnosis. Consult a healthcare professional for accurate diagnosis and treatment.")
    else:
        st.markdown(f"<div class='prediction-box prediction-negative'>Malaria Predicted: <strong>NO</strong></div>", unsafe_allow_html=True)
        st.write(f"Based on the symptoms, malaria is **{prediction_proba[0]*100:.2f}% unlikely**.")
        st.info("Please note: This is a machine learning prediction and not a medical diagnosis. If symptoms persist, consult a healthcare professional.")

    st.markdown("---")
    st.subheader("4. How the Model Made This Decision:")

    # --- Visualize Decision Path (Textual Description) ---
    # This function traces the path for a single prediction
    def get_decision_path(model, input_features, feature_names, class_names):
        # Apply the model to get the leaf node ID for the input
        leaf_id = model.apply(input_features)[0]
        # Get the path from the root to the leaf node
        node_indicator = model.decision_path(input_features)
        path = []

        # Iterate through nodes in the path
        for node_id in node_indicator.indices:
            # If we've reached the leaf node, stop
            if node_id == leaf_id:
                break

            # Get feature and threshold for decision nodes
            if model.tree_.feature[node_id] != tree._tree.TREE_LEAF:
                feature_idx = model.tree_.feature[node_id]
                threshold = model.tree_.threshold[node_id]
                feature_name = feature_names[feature_idx]
                value = input_features[feature_name].iloc[0]

                # Determine which branch was taken
                if value <= threshold:
                    decision = f"Is '{feature_name.replace('_', ' ').title()}' present (value: {value})? Yes (<= {threshold:.1f})"
                else:
                    decision = f"Is '{feature_name.replace('_', ' ').title()}' present (value: {value})? No (> {threshold:.1f})"
                path.append(decision)
            # No else needed, as we break once leaf_id is reached

        # Add the final prediction based on the leaf node's class distribution
        # Get the class distribution at the leaf node
        leaf_value = model.tree_.value[leaf_id][0]
        total_samples_at_leaf = leaf_value.sum()
        if total_samples_at_leaf > 0:
            probabilities = leaf_value / total_samples_at_leaf
            predicted_class_idx = probabilities.argmax()
            confidence = probabilities[predicted_class_idx] * 100
            final_prediction_text = class_names[predicted_class_idx]
            path.append(f"Reached a decision node with samples: {int(leaf_value[0])} 'No Malaria', {int(leaf_value[1])} 'Malaria'.")
            path.append(f"Final Prediction: {final_prediction_text} (Confidence: {confidence:.2f}%)")
        else:
            path.append("Could not determine final prediction path (leaf node had no samples).")

        return path

    class_names_for_path = ['No Malaria', 'Malaria']
    decision_path = get_decision_path(model, input_features_df, feature_names, class_names_for_path)

    for i, step in enumerate(decision_path):
        st.write(f"**{i+1}.** {step}")

    st.markdown("---")
    st.subheader("5. Full Decision Tree Visualization:")
    st.markdown("This diagram shows the entire decision-making process of the model. Each box represents a decision based on a symptom, leading to a prediction.")

    # Re-generate and display the full tree for context
    dot_data = tree.export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=['No Malaria', 'Malaria'],
        filled=True,
        rounded=True,
        special_characters=True,
        impurity=False, # Set to True if you want to see Gini/Entropy values
        proportion=True # Show proportions of samples in nodes
    )
    graph = graphviz.Source(dot_data)
    st.graphviz_chart(graph) # Streamlit's way to display graphviz

st.markdown("---")
st.markdown("Disclaimer: This application is for educational and demonstrative purposes only and should not be used for actual medical diagnosis. Always consult a qualified healthcare professional for any health concerns.")
