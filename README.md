# Chronic Kidney Disease (CKD) Detection using Machine Learning

This project is a web-based application that predicts the likelihood and stage of Chronic Kidney Disease (CKD) based on a user's medical attributes using a trained Random Forest classifier. The app is built with Flask in Python and provides a simple user interface for input and result visualization.

## Project Objective

Early detection of Chronic Kidney Disease can significantly improve treatment outcomes. This project leverages machine learning to:
- Predict whether a patient is likely to have CKD.
- Classify the disease level as Low-level or High-level based on prediction probabilities.

## How It Works

1. User enters values for selected features through the web form.
2. Flask backend collects and preprocesses inputs.
3. The trained Random Forest model makes a probability prediction.
4. Based on probability thresholds:
   - `<= 0.5`: No Disease
   - `0.5 < p < 0.7`: Low-level Disease
   - `>= 0.7`: High-level Disease
5. Result is displayed back to the user.

