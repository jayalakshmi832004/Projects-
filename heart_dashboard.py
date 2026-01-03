import dash
import dash_bootstrap_components as dbc # Import Dash Bootstrap Components
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go # For more custom plots if needed
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import plotly.figure_factory as ff
import numpy as np

# --- Initialize Dash App with a Bootstrap Theme ---
app_heart = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX]) # Apply LUX theme
app_heart.title = "Heart Disease Dashboard"

# Plotly template for charts
PLOTLY_TEMPLATE = "plotly_white"


# --- 1. Load your Heart Data (same as before) ---
try:
    heart_df = pd.read_csv('cleaned_heart_data.csv')
except FileNotFoundError:
    print("Error: 'cleaned_heart_data (1).csv' not found. Using dummy data.")
    heart_df = pd.DataFrame({
        'age': np.random.randint(10000, 25000, 200),
        'gender': np.random.choice([1, 2], 200),
        'height': np.random.randint(150, 190, 200),
        'weight': np.random.uniform(50, 100, 200),
        'ap_hi': np.random.randint(100, 180, 200),
        'ap_lo': np.random.randint(60, 100, 200),
        'cholesterol': np.random.choice([1, 2, 3], 200),
        'gluc': np.random.choice([1, 2, 3], 200),
        'smoke': np.random.choice([0, 1], 200),
        'alco': np.random.choice([0, 1], 200),
        'active': np.random.choice([0, 1], 200),
        'cardio': np.random.choice([0, 1], 200)
    })

# --- 2. Data Preprocessing (Enhanced for clarity and robustness) ---
if 'age' in heart_df.columns:
    heart_df['age_years'] = (heart_df['age'] / 365.25).round().astype(int)

categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
numerical_features = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo']
TARGET = 'cardio'

if TARGET not in heart_df.columns and not heart_df.empty:
    print(f"Target column '{TARGET}' not found. Adding dummy target.")
    heart_df[TARGET] = np.random.choice([0, 1], heart_df.shape[0])
elif heart_df.empty:
    # Create a minimal df for app layout if initial load failed completely
     heart_df = pd.DataFrame(columns=numerical_features + categorical_features + [TARGET, 'age_years', 'gender_label', 'cholesterol_label', 'gluc_label'])


# Label mapping for visualization
map_gender = {1: 'Female', 2: 'Male'}
map_cholesterol = {1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'}
map_gluc = {1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'}

heart_df['gender_label'] = heart_df['gender'].map(map_gender).fillna('Other') if 'gender' in heart_df else 'N/A'
heart_df['cholesterol_label'] = heart_df['cholesterol'].map(map_cholesterol).fillna('Unknown') if 'cholesterol' in heart_df else 'N/A'
heart_df['gluc_label'] = heart_df['gluc'].map(map_gluc).fillna('Unknown') if 'gluc' in heart_df else 'N/A'


# --- 3. Model Training (same as before, ensure X_model uses existing cols) ---
X = heart_df.drop(TARGET, axis=1, errors='ignore')
y = heart_df[TARGET] if TARGET in heart_df else pd.Series(dtype='int')

existing_model_features = [f for f in numerical_features + categorical_features if f in X.columns]
X_model = X[existing_model_features].copy()

for col in numerical_features:
    if col in X_model.columns:
        X_model[col] = X_model[col].fillna(X_model[col].median())
for col in categorical_features:
    if col in X_model.columns:
        X_model[col] = X_model[col].fillna(X_model[col].mode()[0])
        X_model[col] = X_model[col].astype('category')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [f for f in numerical_features if f in X_model.columns]),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [f for f in categorical_features if f in X_model.columns])
    ],
    remainder='drop'
)

model_trained = False
accuracy, precision, recall, f1, roc_auc = 0, 0, 0, 0, 0
conf_matrix = np.zeros((2, 2))
pipeline = None # Initialize pipeline

if not X_model.empty and not y.empty and len(y.unique()) > 1 and existing_model_features:
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=0.2, random_state=42, stratify=y)
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', LogisticRegression(solver='liblinear', random_state=42))])
        pipeline.fit(X_train, y_train)
        y_pred_test = pipeline.predict(X_test)
        y_pred_proba_test = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, zero_division=0)
        recall = recall_score(y_test, y_pred_test, zero_division=0)
        f1 = f1_score(y_test, y_pred_test, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba_test)
        conf_matrix = confusion_matrix(y_test, y_pred_test)
        model_trained = True
    except Exception as e:
        print(f"Error during model training: {e}")
else:
    print("Skipping model training due to insufficient data or features.")


# --- 4. Define App Layout with Dash Bootstrap Components ---

# Helper function for creating styled cards
def create_card(title, content, color="light", className=""):
    return dbc.Card(
        [
            dbc.CardHeader(html.H5(title, className="mb-0"), className=f"bg-{color} text-white" if color not in ["light", "white"] else ""),
            dbc.CardBody(content)
        ],
        className=f"shadow-sm {className}", # Add a subtle shadow
        style={"marginBottom": "20px"}
    )

# Dropdown options
hist_dropdown_options = [{'label': col, 'value': col} for col in ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo'] if col in heart_df.columns]
bar_dropdown_options = [{'label': col, 'value': col} for col in ['gender_label', 'cholesterol_label', 'gluc_label', 'smoke', 'alco', 'active'] if col in heart_df.columns]


# Navigation Bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("GitHub", href="#", disabled=True)), # Placeholder
    ],
    brand="Cardiovascular Disease Dashboard",
    brand_href="#",
    color="primary",
    dark=True,
    className="mb-4"
)

# Data Exploration Section
exploration_content = dbc.Row([
    dbc.Col(md=6, children=[
        html.Label("Select X-axis for Distribution:", className="form-label"),
        dbc.Select(
            id='hist-x-axis-heart',
            options=hist_dropdown_options,
            value=hist_dropdown_options[0]['value'] if hist_dropdown_options else None,
            className="mb-2"
        ),
        dcc.Graph(id='distribution-histogram-heart', config={'displayModeBar': False}) # Hide mode bar for cleaner look
    ]),
    dbc.Col(md=6, children=[
        html.Label("Feature for Grouped Counts:", className="form-label"),
        dbc.Select(
            id='bar-feature-heart',
            options=bar_dropdown_options,
            value=bar_dropdown_options[0]['value'] if bar_dropdown_options else None,
            className="mb-2"
        ),
        dcc.Graph(id='categorical-bar-chart-heart', config={'displayModeBar': False})
    ]),
    dbc.Col(md=12, className="mt-3", children=[
        dcc.Graph(id='age-bp-scatter-heart', config={'displayModeBar': False})
    ])
])

# Prediction Section
model_metrics = [
    dbc.Row([dbc.Col(html.Strong("Accuracy:")), dbc.Col(f"{accuracy:.3f}")]),
    dbc.Row([dbc.Col(html.Strong("Precision:")), dbc.Col(f"{precision:.3f}")]),
    dbc.Row([dbc.Col(html.Strong("Recall:")), dbc.Col(f"{recall:.3f}")]),
    dbc.Row([dbc.Col(html.Strong("F1-Score:")), dbc.Col(f"{f1:.3f}")]),
    dbc.Row([dbc.Col(html.Strong("ROC AUC:")), dbc.Col(f"{roc_auc:.3f}")]),
]

prediction_inputs = dbc.Row([
    dbc.Col(dbc.Input(id='pred-age', type='number', placeholder='Age (Years)'), md=3),
    dbc.Col(dbc.Input(id='pred-ap_hi', type='number', placeholder='Systolic BP'), md=3),
    dbc.Col(dbc.Input(id='pred-ap_lo', type='number', placeholder='Diastolic BP'), md=3),
    dbc.Col(dbc.Select(id='pred-cholesterol', options=[{'label': l, 'value': v} for v,l in map_cholesterol.items()], placeholder='Cholesterol'), md=3),
    # Add more inputs for gender, gluc, smoke, alco, active, height, weight for a complete prediction
], className="g-2 mb-3") # g-2 for gutter spacing

prediction_content = html.Div([
    dbc.Row([
        dbc.Col(md=6, children=[
            html.H6("Model Performance (Test Set):", className="mb-3"),
            html.P(f"Model: Logistic Regression (Trained: {model_trained})", className="text-muted small"),
            *model_metrics, # Unpack the list of rows
            html.H6("Confusion Matrix:", className="mt-4"),
            dcc.Graph(id='confusion-matrix-heart', config={'displayModeBar': False}, style={"height": "300px"})
        ]),
        dbc.Col(md=6, children=[
            html.H6("Make a New Prediction:"),
            prediction_inputs,
            dbc.Button('Predict Disease Risk', id='predict-button-heart', color="success", className="w-100 mb-3"),
            dbc.Alert(id='prediction-output-heart', color="info", dismissable=True, is_open=False, duration=8000) # Auto-dismiss alert
        ])
    ])
])


app_heart.layout = dbc.Container(fluid=True, children=[
    navbar,
    create_card("Patient Data Exploration", exploration_content),
    create_card("Disease Prediction & Model Insights", prediction_content)
])


# --- 5. Define Callbacks (Logic largely same, update plot styling) ---
@app_heart.callback(
    Output('distribution-histogram-heart', 'figure'),
    [Input('hist-x-axis-heart', 'value')]
)
def update_histogram(selected_col):
    fig = go.Figure() # Use graph_objects for more control
    if not selected_col or selected_col not in heart_df.columns:
        fig.update_layout(title_text="Please select a feature", template=PLOTLY_TEMPLATE, xaxis_showgrid=False, yaxis_showgrid=False, xaxis_zeroline=False, yaxis_zeroline=False)
        return fig
    
    if TARGET not in heart_df.columns:
        fig = px.histogram(heart_df, x=selected_col, marginal="box", template=PLOTLY_TEMPLATE,
                           title=f"Distribution of {selected_col}")
        fig.update_layout(bargap=0.1)
        return fig

    # Create two traces for the histogram, one for each cardio status
    df_no_disease = heart_df[heart_df[TARGET] == 0]
    df_disease = heart_df[heart_df[TARGET] == 1]

    fig.add_trace(go.Histogram(x=df_no_disease[selected_col], name='No Disease', marker_color='#1f77b4', opacity=0.75))
    fig.add_trace(go.Histogram(x=df_disease[selected_col], name='Disease', marker_color='#ff7f0e', opacity=0.75))

    fig.update_layout(
        title_text=f"Distribution of {selected_col} by Cardio Status",
        xaxis_title_text=selected_col,
        yaxis_title_text="Count",
        barmode='overlay', # Overlay histograms
        bargap=0.1,
        template=PLOTLY_TEMPLATE,
        legend_title_text='Status',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

@app_heart.callback(
    Output('categorical-bar-chart-heart', 'figure'),
    [Input('bar-feature-heart', 'value')]
)
def update_barchart(selected_feature):
    if not selected_feature or selected_feature not in heart_df.columns:
        return px.bar(title=f"Select a feature.", template=PLOTLY_TEMPLATE).update_layout(xaxis_showgrid=False, yaxis_showgrid=False)

    if TARGET not in heart_df.columns:
        counts_df = heart_df[selected_feature].value_counts().reset_index()
        counts_df.columns = [selected_feature, 'count']
        if counts_df.empty: return px.bar(title=f"No data for {selected_feature}", template=PLOTLY_TEMPLATE)
        fig = px.bar(counts_df, x=selected_feature, y='count', color=selected_feature,
                     title=f"Counts of {selected_feature}", template=PLOTLY_TEMPLATE)
        fig.update_layout(showlegend=False, bargap=0.2)
        return fig

    try:
        grouped_df = heart_df.groupby([selected_feature, TARGET]).size().reset_index(name='count')
        if grouped_df.empty: return px.bar(title=f"No data for {selected_feature} by {TARGET}", template=PLOTLY_TEMPLATE)
        
        # Map TARGET to string for legend
        grouped_df[TARGET] = grouped_df[TARGET].map({0: 'No Disease', 1: 'Disease'})

        fig = px.bar(grouped_df, x=selected_feature, y='count', color=TARGET,
                     barmode='group', title=f"Counts of {selected_feature} by Cardio Status",
                     template=PLOTLY_TEMPLATE, color_discrete_map={'No Disease': '#1f77b4', 'Disease': '#ff7f0e'})
        fig.update_layout(bargap=0.2, legend_title_text='Status', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        return fig
    except Exception as e:
        return px.bar(title=f"Error: {e}", template=PLOTLY_TEMPLATE)


@app_heart.callback(
    Output('age-bp-scatter-heart', 'figure'),
    Input('hist-x-axis-heart', 'value') # Dummy trigger
)
def update_scatter_plot(_):
    if 'age_years' not in heart_df.columns or 'ap_hi' not in heart_df.columns:
        return px.scatter(title="Required columns missing for scatter plot.", template=PLOTLY_TEMPLATE)
    
    if TARGET not in heart_df.columns:
        fig = px.scatter(heart_df, x='age_years', y='ap_hi',
                         title="Age vs. Systolic Blood Pressure",
                         labels={'age_years': 'Age (Years)', 'ap_hi': 'Systolic BP (ap_hi)'},
                         template=PLOTLY_TEMPLATE, marginal_y="violin", marginal_x="box")
        return fig
    
    # Map TARGET to string for legend
    temp_df = heart_df.copy()
    temp_df[TARGET] = temp_df[TARGET].map({0: 'No Disease', 1: 'Disease'})

    fig = px.scatter(temp_df, x='age_years', y='ap_hi', color=TARGET,
                     title="Age vs. Systolic Blood Pressure by Cardio Status",
                     labels={'age_years': 'Age (Years)', 'ap_hi': 'Systolic BP (ap_hi)'},
                     template=PLOTLY_TEMPLATE, marginal_y="violin", marginal_x="box",
                     color_discrete_map={'No Disease': '#1f77b4', 'Disease': '#ff7f0e'},
                     hover_data=['weight', 'cholesterol_label']) # Add more info to hover
    fig.update_layout(legend_title_text='Status', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_traces(marker=dict(size=7, opacity=0.7))
    return fig

@app_heart.callback(
    Output('confusion-matrix-heart', 'figure'),
    Input('predict-button-heart', 'n_clicks') # Also trigger on initial load
)
def display_confusion_matrix(_):
    z = conf_matrix
    x_labels = ['Predicted No Disease', 'Predicted Disease']
    y_labels = ['Actual No Disease', 'Actual Disease']

    if not model_trained or z.shape != (2,2):
        z = np.array([[0,0],[0,0]]) # Ensure it's a numpy array for ff
        title = 'Confusion Matrix (Model Not Trained)'
    else:
        title = 'Confusion Matrix (Test Set)'

    z_text = [[str(int(y_val)) for y_val in x_val] for x_val in z]
    fig = ff.create_annotated_heatmap(z, x=x_labels, y=y_labels, annotation_text=z_text,
                                      colorscale='Blues', showscale=False) # No need for colorscale bar
    fig.update_layout(title_text=title, template=PLOTLY_TEMPLATE, title_x=0.5)
    return fig

@app_heart.callback(
    [Output('prediction-output-heart', 'children'),
     Output('prediction-output-heart', 'is_open'),
     Output('prediction-output-heart', 'color')],
    [Input('predict-button-heart', 'n_clicks')],
    [State('pred-age', 'value'),
     State('pred-ap_hi', 'value'),
     State('pred-ap_lo', 'value'),
     State('pred-cholesterol', 'value')]
    # Add State for other inputs like gender, gluc, smoke, alco, active, height, weight if you add them to layout
)
def make_prediction(n_clicks, age, ap_hi, ap_lo, cholesterol_val):
    if n_clicks == 0:
        return "", False, "info"
    if not model_trained or pipeline is None:
        return "Model not trained. Cannot predict.", True, "danger"

    # Basic validation
    if any(v is None for v in [age, ap_hi, ap_lo, cholesterol_val]):
        return "Please fill all prediction input fields.", True, "warning"

    sample_data = {}
    # Populate with provided inputs
    sample_data['age_years'] = age
    sample_data['ap_hi'] = ap_hi
    sample_data['ap_lo'] = ap_lo
    sample_data['cholesterol'] = cholesterol_val # This should be numeric 1,2,3

    # Fill missing model features with medians/modes from the original dataset used for training
    for feat in existing_model_features:
        if feat not in sample_data:
            if feat in numerical_features and feat in heart_df.columns: # check original df for median/mode
                sample_data[feat] = heart_df[feat].median()
            elif feat in categorical_features and feat in heart_df.columns:
                sample_data[feat] = heart_df[feat].mode()[0]
            else: # Fallback if feature is critical and not handled (e.g. 'gender' which is categorical)
                  # For a real app, ensure all inputs are taken or handled systematically
                if feat == 'gender': sample_data[feat] = heart_df['gender'].mode()[0] if 'gender' in heart_df else 1
                elif feat == 'gluc': sample_data[feat] = heart_df['gluc'].mode()[0] if 'gluc' in heart_df else 1
                elif feat == 'smoke': sample_data[feat] = 0
                elif feat == 'alco': sample_data[feat] = 0
                elif feat == 'active': sample_data[feat] = 1
                elif feat == 'height': sample_data[feat] = heart_df['height'].median() if 'height' in heart_df else 170
                elif feat == 'weight': sample_data[feat] = heart_df['weight'].median() if 'weight' in heart_df else 70


    try:
        input_df = pd.DataFrame([sample_data])
        input_df = input_df[existing_model_features] # Ensure correct feature order

        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1] # Prob of class 1 (disease)

        pred_label = "Heart Disease Likely" if prediction == 1 else "Heart Disease Unlikely"
        alert_color = "danger" if prediction == 1 else "success"
        result_text = f"{pred_label} (Risk Score: {probability:.2f})"
        return result_text, True, alert_color
    except Exception as e:
        return f"Prediction error: {e}", True, "danger"


# --- Run the App ---
if __name__ == '__main__':
    app_heart.run(debug=True, port=8051)
