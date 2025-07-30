import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
import base64
import os # For potential future use with st.secrets

# --- Function to load and train model (or load pre-trained) ---
@st.cache_resource # Cache the model loading/training for efficiency
def load_and_train_model():
    # Load the actual provided CSV data
    try:
        df_app = pd.read_csv('malaria_symptom_dataset.csv')
    except FileNotFoundError:
        st.error("Error: 'malaria_symptom_dataset.csv' not found. Please ensure the file is uploaded to your Colab environment.")
        st.stop() # Stop the app if data isn't available

    X_app = df_app.drop('Malaria', axis=1) # Corrected column name
    y_app = df_app['Malaria']

    # Retrain the model with the same parameters as in the notebook
    # max_depth=4 and min_samples_leaf=5 to encourage more nuanced probabilities
    model_app = DecisionTreeClassifier(max_depth=4, random_state=42, criterion='entropy', min_samples_leaf=5)
    model_app.fit(X_app, y_app)

    return model_app, X_app.columns.tolist() # Return model and feature names

# Load the model and feature names once
model, feature_names = load_and_train_model()

# --- Streamlit App Layout ---
st.set_page_config(
    page_title="Malaria Symptom Predictor",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a better look
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3.2em; /* Slightly larger */
        color: #FF4B4B; /* A vibrant red */
        text-align: center;
        margin-bottom: 0.2em; /* Reduced margin for smoother flow */
        font-weight: bold;
        letter-spacing: -0.03em; /* Tighter letter spacing */
        line-height: 1.1;
    }
    /* Subheader styling */
    .subheader {
        font-size: 1.6em; /* Slightly larger */
        color: #666666; /* Softer grey */
        text-align: center;
        margin-bottom: 1.8em; /* Increased margin for separation */
        font-weight: normal;
    }
    /* Banner/Hero section styling */
    .hero-banner {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8C00 100%); /* Orange to Red gradient */
        padding: 40px 20px;
        border-radius: 15px; /* More rounded corners */
        text-align: center;
        color: white;
        margin-bottom: 2em;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2); /* More prominent shadow */
    }
    .hero-banner h1 {
        color: white;
        font-size: 2.5em; /* Adjusted size for banner */
        margin-bottom: 0.5em;
    }
    .hero-banner p {
        font-size: 1.1em;
        opacity: 0.9;
    }

    /* Button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 12px 25px; /* Slightly larger padding */
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
        transform: translateY(-2px); /* Slight lift effect */
    }
    /* Prediction box styling */
    .prediction-box {
        border-radius: 12px; /* More rounded */
        padding: 25px; /* More padding */
        margin-top: 25px;
        font-size: 1.4em; /* Slightly larger font */
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
    .stCheckbox {
        font-size: 1.1em;
        margin-bottom: 0.5em;
    }
    .stInfo {
        background-color: #e0f2f7; /* Light blue for info boxes */
        border-left: 5px solid #2196F3;
        padding: 10px;
        border-radius: 5px;
        margin-top: 15px;
        font-size: 0.95em;
    }
</style>
""", unsafe_allow_html=True)

# Header Section (Improved)
st.markdown("<p class='main-header'>Malaria Symptom Predictor</p>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Leveraging Machine Learning for Health Insights</p>", unsafe_allow_html=True)

# Hero Banner (Improved)
st.markdown("""
<div class="hero-banner">
    <h1>Empowering Health Decisions with Data</h1>
    <p>Simply select the patient's symptoms below and let our model provide an instant, data-driven analysis.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- Input Form for Symptoms ---
st.header("1. Enter Patient Symptoms")
st.markdown("Please check all symptoms that apply:")

# Use checkboxes for binary symptoms (0 or 1)
symptoms_input = {}
num_cols = len(feature_names) # Get number of symptom columns dynamically
cols = st.columns(min(3, num_cols)) # Create up to 3 columns for layout

for i, symptom in enumerate(feature_names):
    with cols[i % len(cols)]: # Distribute checkboxes across columns
        symptoms_input[symptom] = st.checkbox(symptom.replace('_', ' ').title(), value=False)

# Convert boolean checkboxes to 0/1 for prediction
input_features = pd.DataFrame([[int(symptoms_input[s]) for s in feature_names]], columns=feature_names)

# --- Prediction Button ---
st.markdown("---")
st.header("2. Get Prediction")
predict_button = st.button("Predict Malaria Status")

# --- Prediction and Result Display ---
if predict_button:
    # Ensure all features are present in the input_features DataFrame
    # This is important if the order of columns changes or some are missing
    for col in feature_names:
        if col not in input_features.columns:
            input_features[col] = 0 # Default to 0 if symptom not in input

    # Reorder columns to match training data
    input_features = input_features[feature_names]

    prediction = model.predict(input_features)[0]
    prediction_proba = model.predict_proba(input_features)[0] # Get probabilities

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
    st.subheader("3. How the Model Made This Decision:")

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
    decision_path = get_decision_path(model, input_features, feature_names, class_names_for_path)

    for i, step in enumerate(decision_path):
        st.write(f"**{i+1}.** {step}")

    st.markdown("---")
    st.subheader("4. Full Decision Tree Visualization:")
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

