import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import shap


# Function to train and save the Random Forest model
def train_rf_classifier(X_train, y_train, params, model_path):
    """
    Train and save a Random Forest classifier.

    Args:
    X_train, y_train: Training data and labels.
    params: Hyperparameters for the Random Forest model.
    model_path: Path to save the trained model.
    """
    rf_classifier = RandomForestClassifier(**params) #add seed later
    print("Training...")
    rf_classifier.fit(X_train, y_train)
    print("Done.")
    joblib.dump(rf_classifier, model_path)
    return rf_classifier



# Function to evaluate the model and print classification report
def evaluate_model(model, X, y, dataset_name, label_encoder):
    """
    Evaluate the model and display the classification report and confusion matrix.

    Args:
    model: Trained Random Forest model.
    X, y: Data and labels for evaluation.
    dataset_name: Name of the dataset for display.
    label_encoder: LabelEncoder to decode label classes.
    """
    y_pred = model.predict(X)

    # Confusion Matrix Visualization
    confusion_mat = confusion_matrix(y, y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()



def print_classification_report(model, X, y_true, label_encoder):
    """
    Print a formatted classification report.

    Args:
    model: The trained classifier model.
    X, y: Data and labels to evaluate.
    y_true: True labels.
    y_pred: Predicted labels.
    label_encoder: LabelEncoder used to decode the labels.
    """
    y_pred = model.predict(X)
    report = classification_report(y_true, y_pred, output_dict=True, target_names=label_encoder.classes_)
    report_df = pd.DataFrame(report).transpose()
    
    # Remove 'accuracy' - non necessary here
    report_df = report_df.drop(['accuracy'], errors='ignore')

    # display with background gradient
    display(report_df.style.background_gradient(cmap='viridis', axis=0))
    

def predict_on_test_set(model, X_test, y_test, label_encoder):
    """
    Predict on the test set and display true and predicted class names.

    Args:
    model: Trained classifier model.
    X_test, y_test: Test data and labels.
    label_encoder: LabelEncoder used to encode the labels.
    """
    y_test_pred = model.predict(X_test)
    y_test_pred_classes = label_encoder.inverse_transform(y_test_pred)[0]
    y_test_true_classes = label_encoder.inverse_transform([y_test])[0]
    # Predict probabilities
    y_test_pred_probs = model.predict_proba(X_test)[0]
    
    print(f"True Class: {y_test_true_classes}, Predicted Class: {y_test_pred_classes} with prediction probability: {max(y_test_pred_probs)*100:.2f}%")
    return y_test_pred_classes


# Function to predict the category 
def predict_and_display(text, model, tfidf_vectorizer, label_encoder):
    transformed_text = tfidf_vectorizer.transform([text])
    prediction = model.predict(transformed_text)
    prediction_proba = model.predict_proba(transformed_text)
    
    # Get the predicted category and its probability
    predicted_category = label_encoder.inverse_transform(prediction)[0]
    # predicted_category = prediction[0]
    # probability = max(prediction_proba[0])

    print(f"This resume seems to belong to the **{predicted_category}** category.")

    return transformed_text



def generate_shap_explained(model, X_train, explainer_path, model_path = ""):
    """
    Generate and save SHAP values for the Random Forest model.

    Args:
    model: Trained Random Forest model.
    X_train: Training data used for the model.
    model_path: Path to the trained model file.
    explainer_path: Path to save the SHAP explainer.
    """
    # Load the model if it's not passed directly
    if model is None:
        model = joblib.load(model_path)

    # Create the SHAP explainer
    explainer = shap.TreeExplainer(model, X_train.toarray())

    # Save the explainer
    joblib.dump(explainer, explainer_path)

    return explainer

def vizualize_shap_values(model, X, y, explainer, label_encoder, tfidf_vectorizer):
    """
    Generate and save SHAP values for the tree model.

    Args:
    model: Trained tree model.
    X: features data used for the model.
    explainer: SHAP explainer.
    label_encoder: LabelEncoder used to encode the labels.
    tfidf_vectorizer: vectorizer
    """
    
    shap_values = explainer.shap_values(X.toarray())
    y_test_pred = model.predict(X)
    y_test_pred_classes = label_encoder.inverse_transform(y_test_pred)[0]
    # y_test_true_classes = label_encoder.inverse_transform([y])[0]
    
    # Create a SHAP plot
    # For classification problems, there is a separate array of SHAP values for each possible outcome. 
    # In this case, we index in to get the SHAP values for the prediction of class "y_test_pred".
    
    shap.summary_plot(shap_values[y_test_pred[0]], X.toarray(), 
                      feature_names=tfidf_vectorizer.get_feature_names_out(), class_names=label_encoder.classes_,
                      show=False, plot_type='dot')

    # Adjust the size of the graph
    plt.gcf().set_size_inches(14, 12) # Adjust the dimensions to your preference

    # Graph title
    if y == None: 
        plt.title(f'Prevalence of SHAP characteristics with Predicted class: {y_test_pred_classes}')
    else : 
        y_test_true_classes = label_encoder.inverse_transform([y])[0]
        plt.title(f'Prevalence of SHAP characteristics - True class: {y_test_true_classes}, and Predicted class: {y_test_pred_classes}')

    # Display the graph
    plt.show()
    
    return shap_values
