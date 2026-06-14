import pytest
from neuroscan.visualizer import plot_confidence_bar, plot_training_curves

def test_plot_confidence_bar():
    probs = {"glioma": 0.1, "meningioma": 0.7, "pituitary": 0.1, "notumor": 0.1}
    fig = plot_confidence_bar(probs)
    
    assert fig is not None
    # Check if the figure contains a bar trace
    assert any(trace.type == 'bar' for trace in fig.data)
    assert fig.layout.title.text == "Prediction Confidence (%)"

def test_plot_training_curves():
    history = {
        "accuracy": [0.5, 0.6, 0.7],
        "val_accuracy": [0.4, 0.5, 0.6],
        "loss": [1.0, 0.8, 0.5]
    }
    
    fig = plot_training_curves(history, metric="accuracy")
    assert fig is not None
    # Check if the figure contains scatter (line) traces
    assert any(trace.type == 'scatter' for trace in fig.data)
    assert len(fig.data) == 2 # train and val
    assert fig.layout.title.text == "Training Accuracy over Epochs"
