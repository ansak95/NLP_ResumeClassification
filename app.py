import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
import joblib
import shap
import tempfile
import os
from utils import prepare_data
import json

# paths to the necessary utils
LABEL_ENCODER_PATH = 'models/label_encoder.pkl'
TFIDF_VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'
MODEL_PATH_TEMPLATE = 'models/{}_model.pkl'
EXPLAINER_PATH_TEMPLATE = 'models/{}_explainer.pkl'

def load_model_and_utilities(model_id):
    """
    Load the model and related utilities based on the provided model ID.
    """
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH_TEMPLATE.format(model_id))
    explainer = joblib.load(EXPLAINER_PATH_TEMPLATE.format(model_id))

    return label_encoder, tfidf_vectorizer, model, explainer


def visualize_shap_values(model, X, explainer, label_encoder, tfidf_vectorizer):
    """
    Generate and display SHAP values for the tree model in a Streamlit app.

    Args:
    model: Trained tree model.
    X: Features data used for the model.
    explainer: SHAP explainer object.
    label_encoder: LabelEncoder used to encode the labels.
    tfidf_vectorizer: Vectorizer used for feature transformation.
    """

    # Calculate SHAP values
    shap_values = explainer.shap_values(X.toarray())
    
    # Predict the class
    y_test_pred = model.predict(X)
    predicted_class = label_encoder.inverse_transform(y_test_pred)[0]

    # SHAP Summary Plot
    shap.summary_plot(shap_values[y_test_pred[0]], X.toarray(),
                      feature_names=tfidf_vectorizer.get_feature_names_out(),
                      class_names=label_encoder.classes_,
                      show=False, plot_type='dot')

    # Adjust the size of the graph
    plt.gcf().set_size_inches(14, 12)

    # Set the graph title
    plt.title(f'SHAP Values - Predicted class: {predicted_class}')

    # Display the plot in Streamlit
    st.pyplot(plt)


# Function to generate and return word cloud image
def get_wordcloud_image(text):
    """
    Generate and return a word cloud image from the given text.

    Args:
    text: Text to generate the word cloud from.

    Returns:
    Image object of the generated word cloud.
    """
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    img = BytesIO() # Create a BytesIO object to hold the image data
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(img, format='png')
    img.seek(0)
    return img

    
def main():
    st.sidebar.title('App Settings')
    model_id = st.sidebar.selectbox('Select Model', ['Random Forest'], key='model_select')
    st.sidebar.markdown("""
    Configure the settings for resume classification. Select the model and upload a resume in PDF format.
    """)
    
    st.title('Resume Classification')
    st.markdown("""
    This machine learning application classifies resumes into various job categories.
    Upload a resume in PDF format, and the app will process it and predict the most likely job category.
    """)
    
    
    if model_id == "Random Forest":
        model_id_ref = "rf"
    label_encoder, tfidf_vectorizer, model, explainer = load_model_and_utilities(model_id_ref)
    
    uploaded_file = st.file_uploader("Upload a Resume", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner('Processing the uploaded resume...'):
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                st.markdown("""### Understanding SHAP Values""")
                tmp_file.write(uploaded_file.read())
                file_path = tmp_file.name
                st.text_area("Text path", file_path, height=100)
                
            text = prepare_data.extract_text_from_pdf(file_path)
            cleaned_text = prepare_data.preprocess_text(text)
            st.text_area("Text path", file_path, height=100)
                        

            # Chargement des stopwords personnalisés à partir du fichier JSON
            with open('custom_stopwords.json', 'r') as file:
                custom_stopwords = json.load(file)

            cleaned_text = prepare_data.remove_custom_stopwords(cleaned_text, custom_stopwords)
            
            st.text_area("Text path", file_path, height=100)
            transformed_text = tfidf_vectorizer.transform([cleaned_text])
            # st.text_area("Processed Text", cleaned_text, height=300)
            st.success('Resume processed successfully.')
       
            

            # Assuming you have a 'Predict' button
            if st.button('Predict'):
                # Call the prediction function and get the transformed text
                prediction = model.predict(transformed_text)
                predicted_category = label_encoder.inverse_transform(prediction)[0]
                st.write(f"This resume seems to belong to the **{predicted_category}** category.")
            
            # Place the WordCloud and SHAP values explanations in separate columns
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Word Cloud")
                if st.button("Generate WordCloud", key='wordcloud'):
                    wordcloud_img = get_wordcloud_image(cleaned_text) # Generate word cloud
                    st.image(wordcloud_img, use_column_width=True) # Display the word cloud

            
            with col2:
                st.subheader("Model Insights")
                if st.button("Explain model with SHAP values?", key='shap'):
                    st.markdown("## Explanation of Results (SHAP Diagram)")

                    st.markdown("""
                    ### Understanding SHAP Values

                    SHAP (SHapley Additive exPlanations) values help in understanding the impact of each feature on the model's prediction. Each feature value is assigned a SHAP value which indicates how much that feature contributes, positively or negatively, to the final prediction. 
                    """)
                    # Show SHAP diagram for SHAP values        
                    visualize_shap_values(model, transformed_text, explainer, label_encoder, tfidf_vectorizer)
        
    else:
        st.text("Please upload a resume to begin.")
if __name__ == "__main__":
    main()
