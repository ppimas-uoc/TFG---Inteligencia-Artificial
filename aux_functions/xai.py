import tempfile
import webbrowser
import os
from pathlib import Path

def single_prediction(model, model_thresholded, threshold, X, y, observation_idx):
    obs = X.iloc[[observation_idx]]
    true_label = y.iloc[observation_idx]
    display(obs)
    probs = model.predict_proba(obs)[0]
    prediction = model_thresholded.predict(obs)[0]
    print(f"Probabilidad de las clases de QoL: [Aceptable: {probs[0]:.2f}, Mejorable: {probs[1]:.2f}]")
    print(f"QoL predicha ajustada con umbral {threshold:.2f}: {prediction}")
    print(f"QoL real: {true_label}")


def show_random_false_prediction(model, predictions, threshold, X_test, y_test, negative=True):
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
    output_dir = Path('./lime_explanations')
    output_dir.mkdir(exist_ok=True)

    filepath = output_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(explanation.as_html())

    print(f"LIME explanation saved at: {filepath.resolve()}")

    opened = webbrowser.open(f"file://{filepath.resolve()}")
    if not opened:
        print("Could not open browser. Open it manually from the path above.")