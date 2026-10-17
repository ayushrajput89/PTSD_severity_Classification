import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None
import pandas as pd
import numpy as np

def plot_class_distribution(df, class_col='severity_class'):
    if class_col not in df.columns:
        return None
    counts = df[class_col].value_counts().reset_index()
    counts.columns = [class_col, 'count']
    fig = px.bar(counts, x=class_col, y='count', 
                 color=class_col,
                 title='Distribution of Severity Classes',
                 template='plotly_dark')
    return fig

def plot_feature_distribution(df, feature_col, class_col='severity_class'):
    if feature_col not in df.columns or class_col not in df.columns:
        return None
    fig = px.box(df, x=class_col, y=feature_col, 
                 color=class_col,
                 title=f'Distribution of {feature_col} by Severity',
                 template='plotly_dark')
    return fig

def plot_correlation_heatmap(df, features_list):
    valid_features = [f for f in features_list if f in df.columns]
    if not valid_features:
        return None
    corr = df[valid_features].corr()
    fig = px.imshow(
        corr, 
        text_auto='.2f',
        title='Feature Correlation Heatmap',
        color_continuous_scale='YlOrRd',
        template='plotly_white',
        zmin=-1, zmax=1,
        aspect='auto'
    )
    fig.update_layout(
        height=650,
        margin=dict(l=60, r=60, t=50, b=60),
        xaxis=dict(tickangle=45, tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12))
    )
    return fig

def plot_wordcloud(texts, title=''):
    text_data = " ".join([str(t) for t in texts if pd.notna(t)])
    if not text_data.strip():
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='#0E1117', 
                          colormap='Blues').generate(text_data)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(title, color='white')
    fig.patch.set_facecolor('#0E1117')
    return fig

def plot_confidence_bars(probabilities, class_names):
    df = pd.DataFrame({'Class': class_names, 'Probability': probabilities})
    df = df.sort_values('Probability')
    fig = px.bar(df, x='Probability', y='Class', orientation='h',
                 title='Prediction Confidence',
                 template='plotly_white',
                 color='Probability', color_continuous_scale='Blues')
    fig.update_layout(xaxis_range=[0, 1])
    return fig

def plot_model_comparison_chart(results_dict):
    records = []
    for model, metrics in results_dict.items():
        if not isinstance(metrics, dict):
            continue
        for metric, val in metrics.items():
            if metric in ['accuracy', 'macro_f1', 'f1', 'precision', 'recall']:
                metric_label = 'Accuracy' if metric == 'accuracy' else ('Macro F1' if metric == 'macro_f1' else metric.capitalize())
                records.append({'Model': model, 'Metric': metric_label, 'Score': round(val, 4)})
    
    df = pd.DataFrame(records)
    if df.empty:
        return None
        
    fig = px.bar(df, x='Model', y='Score', color='Metric', barmode='group',
                 text_auto='.3f',
                 title='Model Performance Comparison',
                 template='plotly_white',
                 color_discrete_sequence=['#2B6CB0', '#ED8936'])
    fig.update_layout(yaxis_range=[0, 1], height=450)
    return fig

def plot_roc_curves_plotly(y_true, y_prob, class_names):
    from sklearn.metrics import roc_curve, auc
    fig = go.Figure()
    
    if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
        for i, class_name in enumerate(class_names):
            y_true_binary = (np.array(y_true) == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{class_name} (AUC = {roc_auc:.2f})', mode='lines'))
    else:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC = {roc_auc:.2f})', mode='lines'))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', mode='lines', line=dict(dash='dash', color='gray')))
    fig.update_layout(title='ROC Curves by Class', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                      template='plotly_white', height=450)
    return fig

def plot_confusion_matrix_plotly(cm, class_names):
    fig = px.imshow(cm, x=class_names, y=class_names, text_auto=True,
                    color_continuous_scale='Blues',
                    title='Confusion Matrix',
                    labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                    template='plotly_white')
    fig.update_layout(height=450)
    return fig
