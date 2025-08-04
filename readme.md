# 🌾 Intelligent Crop Recommendation System

Streamlit web app that suggests optimal crops based on soil nutrients (N, P, K, pH) and climate data (temperature, humidity, rainfall). Combines five ML models (Random Forest, SVM, SGD, Decision Tree, Logistic Regression) and integrates real-time weather via the Weatherbit API.

## 🚀 Quick Start

### 1.Clone & enter repo 

```bash
cd crop-recommendation-system
```
### 2. Create & activate virtual env
#### macOS/Linux:
```bash
python -m venv .venv 
source .venv/bin/activate   
```    
#### Windows: 
```bash
python -m venv .venv 
.venv\Scripts\activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Add environment variables

### 5. Train models (optional – pickle file already included)
```bash
python crop-prediction-comparison-new.py
```
### 6. Run the app
```bash
streamlit run app.py
```