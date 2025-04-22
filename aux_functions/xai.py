"""
xai.py

Utility functions for inspecting model predictions and visualizing local explanations.

Includes:
- single_prediction: prediction overview with threshold
- show_random_false_prediction: pick and display a false negative or false positive
- show_lime: export LIME explanation to HTML and open in browser

Author: Pablo Pimàs Verge
Created: 2025-04
License: CC 3.0
"""
import webbrowser
from pathlib import Path

def single_prediction(model, model_thresholded, threshold, X, y, observation_idx):
    """
    Displays the prediction probability and class for a specific observation, both with and without threshold adjustment.

    :param model: Trained model supporting predict_proba.
    :param model_thresholded: Wrapper model that applies a custom decision threshold.
    :param threshold: Threshold value used for classification.
    :param X: Feature set (DataFrame).
    :param y: Target variable (Series).
    :param observation_idx: Index of the observation to inspect.
    :return: None. Displays prediction info and the data row.
    """
    obs = X.iloc[[observation_idx]]
    true_label = y.iloc[observation_idx]
    display(obs)
    probs = model.predict_proba(obs)[0]
    prediction = model_thresholded.predict(obs)[0]
    print(f"Probabilidad de las clases de QoL: [Aceptable: {probs[0]:.2f}, Mejorable: {probs[1]:.2f}]")
    print(f"QoL predicha ajustada con umbral {threshold:.2f}: {prediction}")
    print(f"QoL real: {true_label}")


def show_random_false_prediction(model, predictions, threshold, X_test, y_test, negative=True):
    """
    Displays a randomly selected false negative or false positive, showing its features, true label,
    predicted probability, and adjusted prediction with threshold.

    :param model: Trained model supporting predict_proba.
    :param predictions: Array of predicted labels with threshold applied.
    :param threshold: Decision threshold used in classification.
    :param X_test: Test set features (DataFrame).
    :param y_test: True target labels (Series).
    :param negative: If True, search for false negatives. If False, search for false positives.
    :return: Index of the selected sample, or None if no misclassifications found.
    """
    error = "No false negatives found."

    if negative:
        falses = y_test[(predictions == 0) & (y_test == 1)]
    else:
        falses = y_test[(predictions == 1) & (y_test == 0)]
        error = "No false positives found."

    if falses.empty:
        print(error)
        return

    index = falses.sample(1).index[0]
    sample = X_test.loc[[index]]
    probas = model.predict_proba(sample)[0]

    display(sample)
    print(f"Predicción sin ajustar: {probas.round(2)}")
    print(f"Predicción ajustada con umbral ({threshold:.3f}): {int(probas[1] >= threshold)} ({probas[1]:.2f})")
    print(f"QoL real: {y_test.loc[index]}")
    print(f"Índice: {index}")

    return index


def show_lime(explanation, filename='lime_explanation.html'):
    """
    Saves a LIME explanation to an HTML file and opens it in the default browser.

    :param explanation: LIME explanation object with as_html() method.
    :param filename: Output filename (default: 'lime_explanation.html').
    :return: None. Saves the file and attempts to open it.
    """
    output_dir = Path('./lime_explanations')
    output_dir.mkdir(exist_ok=True)

    filepath = output_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(explanation.as_html())

    print(f"LIME explanation saved at: {filepath.resolve()}")

    opened = webbrowser.open(f"file://{filepath.resolve()}")
    if not opened:
        print("Could not open browser. Open it manually from the path above.")