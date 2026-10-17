import streamlit as st

def inject_custom_css():
    st.markdown("""
        <style>
        /* Professional card-like containers */
        .card {
            background-color: #1E1E1E;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            border-left: 4px solid #4EA8DE;
        }
        
        /* Severity badges */
        .badge-low {
            background-color: #2e7d32;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        .badge-moderate {
            background-color: #f57f17;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        .badge-high {
            background-color: #c62828;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        /* Custom header styling */
        h1, h2, h3 {
            color: #4EA8DE !important;
            font-family: "sans serif";
        }
        
        /* Smooth animations */
        .stButton>button {
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(78, 168, 222, 0.4);
        }
        
        /* Metric cards */
        .metric-card {
            background-color: #F8FAFC;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .metric-title {
            font-size: 12px;
            color: #64748B;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #0F172A;
            margin-top: 4px;
        }
        .metric-delta {
            font-size: 12px;
            color: #16A34A;
            font-weight: 500;
            margin-top: 2px;
        }
        </style>
    """, unsafe_allow_html=True)

def severity_badge(severity, confidence):
    severity_lower = str(severity).lower()
    if 'low' in severity_lower or severity_lower == '0':
        badge_class = 'badge-low'
        label = 'Low Severity'
    elif 'moderate' in severity_lower or severity_lower == '1':
        badge_class = 'badge-moderate'
        label = 'Moderate Severity'
    else:
        badge_class = 'badge-high'
        label = 'High Severity'
        
    return f"""
    <div style="display: flex; align-items: center; gap: 10px; margin: 10px 0;">
        <span class="{badge_class}">{label}</span>
        <span style="color: #AAA; font-size: 0.9em;">Confidence: {confidence:.1%}</span>
    </div>
    """

def metric_card(title, value, delta=None):
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ''
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """
