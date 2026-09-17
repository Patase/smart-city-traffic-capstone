# Smart City Traffic Intelligence – Part 3

## Overview

This project applies machine learning and AI techniques to traffic-volume data for traffic prediction, clustering, recommendation, deployment, and monitoring.

The main tasks include:

- Supervised machine learning
- Unsupervised machine learning
- Neural network and SHAP explainability
- MLflow experiment tracking
- Traffic recommendation system
- MLOps and FastAPI deployment
- Model monitoring
- Responsible and Sustainable AI

## Important Note: Accident Proxy Label

No real accident dataset was provided for this capstone.

For the classification task, a **proxy `high_risk` label** was created using:

- High or Severe congestion
- Together with severe weather or low-visibility weather

Therefore, the classification model predicts the **defined proxy high-risk condition**, not actual accident occurrence.

## Project Structure

```text
part3_machine_learning/
│
├── data/
│   └── part3_ml_data.csv
│
├── models/
│   ├── traffic_random_forest_v2.joblib
│   └── traffic_random_forest_v2_features.txt
│
├── results/
│   ├── classification_results.csv
│   ├── regression_results.csv
│   ├── kmeans_cluster_summary.csv
│   ├── association_rules.csv
│   ├── neural_network_results.csv
│   ├── shap_feature_importance.csv
│   ├── mlflow_model_comparison.csv
│   ├── model_versions.csv
│   ├── travel_recommendations.csv
│   ├── monitoring_results.csv
│   └── monitoring_status.txt
│
├── classification.py
├── regression.py
├── clustering.py
├── association_rules.py
├── deep_learning.py
├── mlflow_tracking.py
├── recommendation_system.py
├── model_versioning.py
├── deployment_model.py
├── api_app.py
├── monitoring.py
├── prepare_part3_data.py
├── mlflow.db
├── mlartifacts/
└── part3_ml.log
```

*File names can be adjusted if your local versions use slightly different names.*

## Main Results

### Classification

Two models were tested using the proxy `high_risk` label.

- Logistic Regression: Accuracy 97.82%, F1-score 0.9025, ROC AUC 0.9939
- Random Forest: Accuracy 98.38%, F1-score 0.9285, ROC AUC 0.9969

### Regression

Traffic volume was predicted using:

- Linear Regression: MAE 813.47, R² 0.7260
- Random Forest Regressor: MAE 269.96, R² 0.9448

### Deep Learning

A feed-forward neural network with hidden layers of 64, 32, and 16 neurons achieved:

- MAE 291.35
- R² 0.9409

SHAP showed that time-related variables, especially hour of day, were among the most important predictors.

### Recommendation System

The system recommends lower-traffic travel times between **06:00 and 22:00** to make the suggestions more practical.

It considers:

- Weekday or weekend
- Optional weather condition
- Historical traffic volume

The system also informs users that traffic between **22:00 and 06:00** is generally lower when supported by the historical data.

### MLOps

MLflow was used to track model experiments and results.

Model versions:

- v1.0 – Linear Regression
- v2.0 – Random Forest Regressor

The v2.0 model was deployed using FastAPI.

Model monitoring produced:

- Reference MAE: 274.56
- Current MAE: 265.36
- MAE drift: -3.35%
- Status: **PASS / Normal**

## How to Run

Run the scripts from the `part3_machine_learning` folder.

### Prepare Part 3 Data

```bash
python prepare_part3_data.py
```

### Supervised Learning

```bash
python classification.py
python regression.py
```

### Unsupervised Learning

```bash
python clustering.py
python association_rules.py
```

### Deep Learning and SHAP

```bash
python deep_learning.py
```

### MLflow

```bash
python mlflow_tracking.py
```

To open the MLflow interface:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

### Traffic Recommendation

Example:

```bash
python recommendation_system.py --day weekday
```

With weather:

```bash
python recommendation_system.py --day weekday --weather Rain
```

### Model Versioning

```bash
python model_versioning.py
```

### Prepare Deployment Model

```bash
python deployment_model.py
```

### FastAPI Deployment

Start the API:

```bash
python -m uvicorn api_app:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

### Model Monitoring

```bash
python monitoring.py
```

## Main Python Packages

The project uses:

- pandas
- numpy
- scikit-learn
- matplotlib
- mlxtend
- shap
- mlflow
- fastapi
- uvicorn
- joblib

Install missing packages using:

```bash
python -m pip install pandas numpy scikit-learn matplotlib mlxtend shap mlflow fastapi uvicorn joblib
```

## Limitations

- The dataset represents one traffic corridor and may not generalize to other locations.
- Some periods and weather conditions are not equally represented.
- `snow_1h` contains mostly zero values, so snowy conditions are poorly represented.
- The dataset is historical and traffic patterns may change over time.
- The `high_risk` classification target is only a proxy and must not be interpreted as actual accident prediction.
- The FastAPI and MLflow components are local simulations rather than full production deployments.

## Responsible AI

The model should be treated as a decision-support tool rather than an autonomous traffic-management system.

Human oversight, further validation, performance monitoring, and updated data would be required before real-world use.

A separate Bias and Fairness / Responsible AI report is included with the submission.
