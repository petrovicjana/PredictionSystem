import numpy as np
import pandas as pd
import joblib
from django.shortcuts import render
import os


def home(request):
    return render(request, "home.html")


def predict(request):
    return render(request, "predict.html")


def result(request):
    model_path = os.path.join('model', 'linear_regression_model.pkl')

    # Load the pre-trained model
    model = joblib.load(model_path)

    # Get user inputs
    try:
        var1 = float(request.GET.get('n1'))
        var2 = float(request.GET.get('n2'))
        var3 = float(request.GET.get('n3'))
        var4 = float(request.GET.get('n4'))
        var5 = float(request.GET.get('n5'))
    except (ValueError, TypeError):
        return render(request, "predict.html", {"result2": "Invalid input values."})

    # Create DataFrame for prediction
    input_data = np.array([var1, var2, var3, var4, var5]).reshape(1, -1)

    # Predict
    pred = model.predict(input_data)
    pred = round(pred[0], 2)

    # Format the result
    price = f"The predicted price is ${pred:,.2f}"

    return render(request, "predict.html", {"result2": price})