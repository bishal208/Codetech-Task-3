# Codetech-Task-3

Name:- BISHAL KUMAR SHARMA 
Company:- CODETECH IT SOLUTIONS 
Domain:- DATA SCIENCE 
ID:-CT04DG1321
Duration:- 14TH JUNE TO 14TH JULY 2025
Mentor:- NEELA SANTOSH KUMAR 


# Iris Classification API

This project demonstrates a **complete Data Science workflow**, from **data collection and preprocessing**, to **model training and evaluation**, and finally to **deploying the trained model as an API using Flask**.  

It uses the classic Iris dataset to build a Logistic Regression classifier that predicts the species of an Iris flower given its sepal and petal measurements.

---

## Project Overview

- Data collection and preprocessing (with scaling)  
- Model training and evaluation (with accuracy report)  
- Model serialization using `joblib`  
- REST API implementation using Flask  
- Prediction endpoint that accepts JSON input and returns predictions

---

## Features

- Loads the Iris dataset from seabornâ€™s open dataset
- Preprocesses features and encodes target labels
- Splits the dataset into training and test sets
- Scales the features for better model performance
- Trains a Logistic Regression model
- Evaluates and prints test accuracy
- Saves the trained model and scaler to disk
- Flask API with endpoints:
  - `/` : Health check
  - `/predict` : Make predictions

---

## Requirements

- Python 3.7+
- Install dependencies using pip:
```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
flask
scikit-learn
pandas
joblib
```

---

## Running the Project

### 1. Train the Model
Run the training script:
```bash
python train_model.py
```
This will:
- Train the model
- Print the test accuracy (e.g., Model trained. Test Accuracy: 1.0000)
- Save the trained model as `iris_model.joblib`
- Save the scaler as `scaler.joblib`

### 2. Start the Flask API
Run the Flask app:
```bash
python app.py
```

The API runs at:
```
http://127.0.0.1:5000/
```

---

# API Endpoints

### Health Check
**GET /**  
Response:
```
   ¸ Iris Model API is running!
```

---

### Predict
**POST /predict**  
Send a JSON body with the 4 numeric features of the Iris flower:
```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Response:
```json
{
  "prediction": 0
}
```

Here `prediction` corresponds to:
- `0` â†’ setosa
- `1` â†’ versicolor
- `2` â†’ virginica

---

##‚ Directory Structure

```
- your-repo/
: app.py                # Flask API
: train_model.py        # Model training & evaluation
: iris_model.joblib     # Trained ML model (generated after training)
: scaler.joblib         # Scaler used in preprocessing (generated after training)
: requirements.txt      # Python dependencies
: README.md             # Project documentation
```

---
## Technologies Used

- [Python](https://www.python.org/)
- [Flask](https://flask.palletsprojects.com/) : for serving the API
- [scikit-learn](https://scikit-learn.org/) : ML model and preprocessing
- [pandas](https://pandas.pydata.org/) : data manipulation
- [joblib](https://joblib.readthedocs.io/) : model serialization

---
## Resources 
- youtube
- chatgpt
- google

