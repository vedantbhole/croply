import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import pickle

# Load the data
crop = pd.read_csv("Crop_recommendation.csv")

# Create a dictionary for encoding crop labels
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6, 'orange': 7,
    'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13,
    'pomegranate': 14, 'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18,
    'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}

# Encode the labels
crop['label'] = crop['label'].map(crop_dict)

# Split features and target
X = crop.drop('label', axis=1)
y = crop['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models to compare
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'SGD': SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

# Dictionary to store results
results = {}

# # Print header for results
# print("\n" + "="*80)
# print("{:<20} {:<15} {:<25} {:<20}".format(
#     "Model", "Test Accuracy", "CV Mean Accuracy", "CV Std Dev"))
# print("="*80)

# Train and evaluate each model
for name, model in models.items():
    # print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'cv_scores': cv_scores,
        'model': model
    }
    
    # # Print formatted results
    # print("{:<20} {:<15.2f}% {:<25.2f}% ±{:<20.2f}%".format(
    #     name,
    #     accuracy * 100,
    #     cv_scores.mean() * 100,
    #     cv_scores.std() * 200  # 2 standard deviations
    # ))
    #
    # print("\nDetailed Classification Report:")
    # print(classification_report(y_test, y_pred, digits=4))
    # print("-"*80)

# Create a DataFrame for easy comparison
comparison_df = pd.DataFrame({
    'Test Accuracy (%)': [results[name]['accuracy'] * 100 for name in models.keys()],
    'CV Mean Accuracy (%)': [results[name]['cv_scores'].mean() * 100 for name in models.keys()],
    'CV Std Dev (%)': [results[name]['cv_scores'].std() * 200 for name in models.keys()]
}).round(2)
comparison_df.index = models.keys()

print("\nModel Comparison Summary:")
print(comparison_df)
print("\n")

# Visualize results with percentage labels
plt.figure(figsize=(12, 6))
accuracies = [results[name]['accuracy'] * 100 for name in models.keys()]
cv_means = [results[name]['cv_scores'].mean() * 100 for name in models.keys()]

x = np.arange(len(models))
width = 0.35

bars1 = plt.bar(x - width/2, accuracies, width, label='Test Accuracy')
bars2 = plt.bar(x + width/2, cv_means, width, label='CV Mean Accuracy')

# Add percentage labels on top of bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)

plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Model Comparison: Test Accuracy vs CV Mean Accuracy')
plt.xticks(x, models.keys(), rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Function for making predictions with all models
def recommend_crop_all_models(N, P, K, temperature, humidity, ph, rainfall):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)
    
    predictions = {}
    for name, model_info in results.items():
        prediction = model_info['model'].predict(input_scaled)[0]
        inverse_crop_dict = {v: k for k, v in crop_dict.items()}
        predictions[name] = inverse_crop_dict[prediction]
    
    return predictions

# Save all models and scaler
with open('crop_models_comparison.pkl', 'wb') as f:
    pickle.dump((results, scaler, crop_dict), f)

print("All models, scaler, and crop dictionary saved to 'crop_models_comparison.pkl'\n")

# Test cases
print("Model Predictions for Test Cases:")
print("="*80)

test_cases = [
    [90, 42, 43, 20.87, 82, 6.5, 202.93],
    [60, 55, 44, 23.00, 82, 6.5, 150.9],
    [85, 58, 41, 21.77, 80, 7.0, 226.65],
    [45, 80, 43, 38.87, 90.0, 6, 240.93]
]

for i, case in enumerate(test_cases, 1):
    print(f"\nTest Case {i}:")
    print(f"Features: N={case[0]}, P={case[1]}, K={case[2]}, Temperature={case[3]}°C, " 
          f"Humidity={case[4]}%, pH={case[5]}, Rainfall={case[6]}mm")
    print("-"*40)
    predictions = recommend_crop_all_models(*case)
    
    # Calculate consensus
    from collections import Counter
    consensus = Counter(predictions.values()).most_common(1)[0]
    
    print("{:<20} {:<15}".format("Model", "Predicted Crop"))
    print("-"*40)
    for model_name, prediction in predictions.items():
        print("{:<20} {:<15}".format(model_name, prediction))
    print("-"*40)
    print(f"Consensus: {consensus[0]} ({consensus[1]}/{len(predictions)} models agree)")
    print("="*80)

# Example of how to load and use the saved models
print("\nVerifying saved models:")
with open('crop_models_comparison.pkl', 'rb') as f:
    loaded_results, loaded_scaler, loaded_crop_dict = pickle.load(f)

# Test a single case with loaded models
test_case = test_cases[0]
predictions_loaded = recommend_crop_all_models(*test_case)
print("\nPredictions using loaded models (first test case):")
for model_name, prediction in predictions_loaded.items():
    print(f"{model_name}: {prediction}")

