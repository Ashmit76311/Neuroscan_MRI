import plotly.graph_objects as go
import plotly.express as px

def plot_confidence_bar(probabilities_dict):
    """
    Plots a horizontal bar chart of classification confidence using Plotly.
    """
    classes = list(probabilities_dict.keys())
    probs = list(probabilities_dict.values())
    
    # Sort for better visualization
    sorted_idx = sorted(range(len(probs)), key=lambda k: probs[k])
    classes = [classes[i].upper() for i in sorted_idx]
    probs = [probs[i] * 100 for i in sorted_idx] # To percentage
    
    fig = go.Figure(go.Bar(
        x=probs,
        y=classes,
        orientation='h',
        marker=dict(
            color=probs,
            colorscale='Viridis'
        )
    ))
    
    fig.update_layout(
        title="Prediction Confidence (%)",
        xaxis_title="Confidence (%)",
        yaxis_title="Class",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=300
    )
    
    return fig

def plot_training_curves(history_dict, metric="accuracy"):
    """
    Plots training and validation curves.
    """
    epochs = list(range(1, len(history_dict[metric]) + 1))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history_dict[metric], mode='lines', name=f'Train {metric.capitalize()}'))
    
    if f"val_{metric}" in history_dict:
        fig.add_trace(go.Scatter(x=epochs, y=history_dict[f'val_{metric}'], mode='lines', name=f'Val {metric.capitalize()}'))
        
    fig.update_layout(
        title=f"Training {metric.capitalize()} over Epochs",
        xaxis_title="Epochs",
        yaxis_title=metric.capitalize(),
        template="plotly_dark"
    )
    
    return fig
