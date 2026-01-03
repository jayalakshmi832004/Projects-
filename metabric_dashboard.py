import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import plotly.figure_factory as ff
import numpy as np

# --- Initialize Dash App with a Bootstrap Theme ---
app_metabric = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app_metabric.title = "METABRIC Breast Cancer Dashboard"

PLOTLY_TEMPLATE = "plotly_white"
METABRIC_DATA_FILE = 'cleaned_metabric.csv' # Ensure this file is in the same directory

# --- 1. Load METABRIC Data ---
try:
    metabric_df_original = pd.read_csv(METABRIC_DATA_FILE, low_memory=False)
    print(f"Successfully loaded '{METABRIC_DATA_FILE}'")
    metabric_df = metabric_df_original.copy()

    TARGET_METABRIC = 'death_from_cancer_binary'
    if 'death_from_cancer' in metabric_df.columns:
        metabric_df[TARGET_METABRIC] = metabric_df['death_from_cancer'].apply(lambda x: 1 if x == 'Died of Disease' else 0)
        print(f"Target '{TARGET_METABRIC}' created. Distribution:\n{metabric_df[TARGET_METABRIC].value_counts(normalize=True)}")
    else:
        # If 'death_from_cancer' is missing, try to use 'overall_survival' if it's binary
        if 'overall_survival' in metabric_df.columns and metabric_df['overall_survival'].isin([0,1]).all():
            print("Warning: 'death_from_cancer' column not found. Using 'overall_survival' as binary target.")
            metabric_df[TARGET_METABRIC] = metabric_df['overall_survival']
        else:
            raise FileNotFoundError("Suitable target column ('death_from_cancer' or binary 'overall_survival') not found for METABRIC.")


except FileNotFoundError as e:
    print(f"Error: {e}. Using dummy data for METABRIC app.")
    metabric_df = pd.DataFrame({
        'age_at_diagnosis': np.random.uniform(30, 90, 200),
        'lymph_nodes_examined_positive': np.random.randint(0, 15, 200),
        'mutation_count': np.random.randint(0, 50, 200),
        'tumor_size': np.random.uniform(5, 100, 200),
        'nottingham_prognostic_index': np.random.uniform(1, 6, 200),
        'type_of_breast_surgery': np.random.choice(['MASTECTOMY', 'BREAST CONSERVING', 'N/A'], 200),
        'cellularity': np.random.choice(['Low', 'Moderate', 'High', 'N/A'], 200),
        'chemotherapy': np.random.choice([0, 1], 200),
        'pam50_+_claudin-low_subtype': np.random.choice(['LumA', 'LumB', 'Her2', 'Basal', 'Normal', 'NC'], 200),
        'er_status': np.random.choice(['Positive', 'Negative'], 200),
        'her2_status': np.random.choice(['Positive', 'Negative'], 200),
        'hormone_therapy': np.random.choice([0, 1], 200),
        'neoplasm_histologic_grade': np.random.choice([1, 2, 3, np.nan], 200), # Allow NaN for grade
        'tumor_stage': np.random.choice([0, 1, 2, 3, 4, np.nan], 200), # Allow NaN for stage
        'death_from_cancer_binary': np.random.choice([0, 1], 200)
    })
    TARGET_METABRIC = 'death_from_cancer_binary'

# --- 2. Data Preprocessing for METABRIC ---
numerical_features_metabric = ['age_at_diagnosis', 'lymph_nodes_examined_positive', 'mutation_count', 'tumor_size', 'nottingham_prognostic_index']
categorical_features_metabric = [
    'type_of_breast_surgery', 'cellularity', 'chemotherapy',
    'pam50_+_claudin-low_subtype', 'er_status', 'her2_status',
    'hormone_therapy', 'neoplasm_histologic_grade', 'tumor_stage'
]

# Filter features to those actually present in the loaded DataFrame
numerical_features_metabric = [f for f in numerical_features_metabric if f in metabric_df.columns]
categorical_features_metabric = [f for f in categorical_features_metabric if f in metabric_df.columns]

# Create labels for visualization
metabric_df['chemotherapy_label'] = metabric_df['chemotherapy'].map({0: 'No', 1: 'Yes'}).fillna('Unknown') if 'chemotherapy' in metabric_df else 'N/A'
metabric_df['hormone_therapy_label'] = metabric_df['hormone_therapy'].map({0: 'No', 1: 'Yes'}).fillna('Unknown') if 'hormone_therapy' in metabric_df else 'N/A'

# --- 3. Model Training for METABRIC (Simplified) ---
X_metabric_original = metabric_df.drop([TARGET_METABRIC, 'death_from_cancer'] if 'death_from_cancer' in metabric_df else [TARGET_METABRIC], axis=1, errors='ignore')
y_metabric = metabric_df[TARGET_METABRIC] if TARGET_METABRIC in metabric_df else pd.Series(dtype='int')

existing_model_features_metabric = [f for f in numerical_features_metabric + categorical_features_metabric if f in X_metabric_original.columns]

if not existing_model_features_metabric:
    print("Error: No features available for METABRIC model training. Check CSV columns and feature definitions.")
    X_model_metabric = pd.DataFrame()
else:
    X_model_metabric = X_metabric_original[existing_model_features_metabric].copy()

numerical_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

for col in [f for f in numerical_features_metabric if f in X_model_metabric.columns]:
    X_model_metabric[col] = numerical_imputer.fit_transform(X_model_metabric[[col]]).ravel()
for col in [f for f in categorical_features_metabric if f in X_model_metabric.columns]:
    X_model_metabric[col] = categorical_imputer.fit_transform(X_model_metabric[[col]]).ravel()
    X_model_metabric[col] = X_model_metabric[col].astype('category') # Convert after imputation

preprocessor_metabric = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [f for f in numerical_features_metabric if f in X_model_metabric.columns]),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [f for f in categorical_features_metabric if f in X_model_metabric.columns])
    ],
    remainder='drop' # Drop any columns not specified (e.g. original label columns)
)

model_trained_metabric = False
accuracy_m, precision_m, recall_m, f1_m, roc_auc_m = 0, 0, 0, 0, 0
conf_matrix_m = np.zeros((2, 2))
pipeline_metabric = None

if not X_model_metabric.empty and not y_metabric.empty and len(y_metabric.unique()) > 1 and existing_model_features_metabric:
    try:
        # Stratify only if there are enough samples in each class for the smallest split
        stratify_option = y_metabric if y_metabric.value_counts().min() >= 2 else None
        X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_model_metabric, y_metabric, test_size=0.2, random_state=42, stratify=stratify_option)

        pipeline_metabric = Pipeline(steps=[('preprocessor', preprocessor_metabric),
                                            ('classifier', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'))])
        pipeline_metabric.fit(X_train_m, y_train_m)
        y_pred_test_m = pipeline_metabric.predict(X_test_m)
        
        # Check if predict_proba is available and target is binary
        if hasattr(pipeline_metabric, "predict_proba") and len(y_test_m.unique()) <=2 :
             y_pred_proba_test_m = pipeline_metabric.predict_proba(X_test_m)[:, 1]
             roc_auc_m = roc_auc_score(y_test_m, y_pred_proba_test_m)
        else:
            y_pred_proba_test_m = None # For multi-class or if proba not available
            roc_auc_m = 0.0 # Cannot calculate ROC AUC without probabilities for binary case

        accuracy_m = accuracy_score(y_test_m, y_pred_test_m)
        precision_m = precision_score(y_test_m, y_pred_test_m, zero_division=0)
        recall_m = recall_score(y_test_m, y_pred_test_m, zero_division=0)
        f1_m = f1_score(y_test_m, y_pred_test_m, zero_division=0)
        conf_matrix_m = confusion_matrix(y_test_m, y_pred_test_m)
        model_trained_metabric = True
        print("METABRIC model trained successfully.")
    except Exception as e:
        print(f"Error during METABRIC model training: {e}")
else:
    print("Skipping METABRIC model training due to insufficient data, features, or target classes.")


# --- 4. Define METABRIC App Layout ---
navbar_metabric = dbc.NavbarSimple(
    brand="METABRIC Breast Cancer Dashboard", brand_href="#", color="info", dark=True, className="mb-4"
)

def create_metabric_card(title, content, color="light", className=""):
    return dbc.Card([
        dbc.CardHeader(html.H5(title, className="mb-0"), className=f"bg-{color} text-white" if color not in ["light", "white"] else ""),
        dbc.CardBody(content)
    ], className=f"shadow-sm {className}", style={"marginBottom": "20px"})

hist_dropdown_options_m = [{'label': col.replace('_', ' ').title(), 'value': col} for col in numerical_features_metabric if col in metabric_df.columns]
bar_dropdown_options_m = [{'label': col.replace('_', ' ').title(), 'value': col} for col in ['pam50_+_claudin-low_subtype', 'er_status', 'her2_status', 'tumor_stage', 'cellularity', 'neoplasm_histologic_grade', 'type_of_breast_surgery'] if col in metabric_df.columns]
scatter_x_options_m = hist_dropdown_options_m
scatter_y_options_m = hist_dropdown_options_m

exploration_content_m = dbc.Row([
    dbc.Col(md=4, children=[
        dbc.Label("Numerical Feature for Distribution:"),
        dbc.Select(id='hist-x-axis-metabric', options=hist_dropdown_options_m, value=hist_dropdown_options_m[0]['value'] if hist_dropdown_options_m else None, className="mb-2"),
        dcc.Graph(id='distribution-histogram-metabric', config={'displayModeBar': False})
    ]),
    dbc.Col(md=4, children=[
        dbc.Label("Categorical Feature for Counts:"),
        dbc.Select(id='bar-feature-metabric', options=bar_dropdown_options_m, value=bar_dropdown_options_m[0]['value'] if bar_dropdown_options_m else None, className="mb-2"),
        dcc.Graph(id='categorical-bar-chart-metabric', config={'displayModeBar': False})
    ]),
    dbc.Col(md=4, children=[
        dbc.Label("Scatter Plot X-axis:"),
        dbc.Select(id='scatter-x-metabric', options=scatter_x_options_m, value=scatter_x_options_m[0]['value'] if scatter_x_options_m else None, className="mb-1"),
        dbc.Label("Scatter Plot Y-axis:"),
        dbc.Select(id='scatter-y-metabric', options=scatter_y_options_m, value=scatter_y_options_m[1]['value'] if len(scatter_y_options_m) > 1 else (scatter_y_options_m[0]['value'] if scatter_y_options_m else None) , className="mb-2"),
        dcc.Graph(id='feature-scatter-metabric', config={'displayModeBar': False})
    ]),
    dbc.Col(md=12, className="mt-3", children=[
        html.H5("Sample of Cleaned METABRIC Data"),
        dash_table.DataTable(
            id='metabric-data-table',
            columns=[{"name": i.replace('_',' ').title(), "id": i} for i in metabric_df[numerical_features_metabric + categorical_features_metabric + [TARGET_METABRIC]].head().columns if i in metabric_df.columns],
            data=metabric_df[[col for col in numerical_features_metabric + categorical_features_metabric + [TARGET_METABRIC] if col in metabric_df.columns]].head(10).to_dict('records'),
            page_size=5,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px', 'fontSize': '12px'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        )
    ])
])

model_metrics_m_display = [
    dbc.Row([dbc.Col(html.Strong("Accuracy:")), dbc.Col(f"{accuracy_m:.3f}")]),
    dbc.Row([dbc.Col(html.Strong("Precision (Died of Disease):")), dbc.Col(f"{precision_m:.3f}")]),
    dbc.Row([dbc.Col(html.Strong("Recall (Died of Disease):")), dbc.Col(f"{recall_m:.3f}")]),
    dbc.Row([dbc.Col(html.Strong("F1-Score (Died of Disease):")), dbc.Col(f"{f1_m:.3f}")]),
    dbc.Row([dbc.Col(html.Strong("ROC AUC:")), dbc.Col(f"{roc_auc_m:.3f}")]),
]

prediction_inputs_m_list = []
for feature in existing_model_features_metabric: # Use features that model was trained on
    feature_label = feature.replace("_", " ").title()
    input_id = f'pred-metabric-{feature}'
    
    if feature in numerical_features_metabric:
        default_val_num = X_model_metabric[feature].median() if feature in X_model_metabric and not X_model_metabric.empty else 0
        prediction_inputs_m_list.append(
            dbc.Col([
                dbc.Label(f"{feature_label}:", html_for=input_id, className="mb-0 small"),
                dbc.Input(id=input_id, type='number', value=round(default_val_num, 2) if pd.notna(default_val_num) else None, placeholder=f"e.g., {default_val_num:.1f}")
            ], md=4, className="mb-2")
        )
    elif feature in categorical_features_metabric:
        # Get unique values from the *original* df for dropdown options, after imputation for default
        unique_vals = metabric_df[feature].dropna().unique() if feature in metabric_df else ['N/A']
        options = [{'label': str(val), 'value': val} for val in unique_vals]
        default_val_cat = X_model_metabric[feature].mode()[0] if feature in X_model_metabric and not X_model_metabric.empty else (options[0]['value'] if options else None)
        prediction_inputs_m_list.append(
            dbc.Col([
                dbc.Label(f"{feature_label}:", html_for=input_id, className="mb-0 small"),
                dbc.Select(id=input_id, options=options, value=default_val_cat, placeholder=f"Select {feature_label}")
            ], md=4, className="mb-2")
        )

prediction_content_m = html.Div([
    dbc.Row([
        dbc.Col(md=6, children=[
            html.H6("Model Performance (Test Set - Predicting Death from Cancer):", className="mb-3"),
            html.P(f"Model: Logistic Regression (Trained: {model_trained_metabric})", className="text-muted small"),
            *model_metrics_m_display,
            html.H6("Confusion Matrix:", className="mt-4"),
            dcc.Graph(id='confusion-matrix-metabric', config={'displayModeBar': False}, style={"height": "300px"})
        ]),
        dbc.Col(md=6, children=[
            html.H6("Make a New Prediction:"),
            html.Small("Fields are pre-filled with median/mode values from the training data.", className="text-muted d-block mb-2"),
            dbc.Row(prediction_inputs_m_list),
            dbc.Button('Predict Cancer Mortality Risk', id='predict-button-metabric', color="success", className="w-100 mb-3 mt-2"),
            dbc.Alert(id='prediction-output-metabric', color="info", dismissable=True, is_open=False, duration=10000)
        ])
    ])
])

app_metabric.layout = dbc.Container(fluid=True, children=[
    navbar_metabric,
    create_metabric_card("METABRIC Data Exploration", exploration_content_m, color="info"),
    create_metabric_card("Cancer Outcome Prediction & Model Insights", prediction_content_m, color="info")
])

# --- 5. Define METABRIC Callbacks (Callbacks for EDA plots are similar to heart app, adapt as needed) ---
@app_metabric.callback(
    Output('distribution-histogram-metabric', 'figure'),
    [Input('hist-x-axis-metabric', 'value')]
)
def update_histogram_metabric(selected_col):
    fig = go.Figure()
    if not selected_col or selected_col not in metabric_df.columns or TARGET_METABRIC not in metabric_df.columns:
        fig.update_layout(title_text="Select feature or target missing", template=PLOTLY_TEMPLATE, xaxis_showgrid=False, yaxis_showgrid=False)
        return fig

    df_no_event = metabric_df[metabric_df[TARGET_METABRIC] == 0]
    df_event = metabric_df[metabric_df[TARGET_METABRIC] == 1]

    fig.add_trace(go.Histogram(x=df_no_event[selected_col], name='Not Died from Cancer', marker_color='#1f77b4', opacity=0.75))
    fig.add_trace(go.Histogram(x=df_event[selected_col], name='Died from Cancer', marker_color='#ff7f0e', opacity=0.75))

    fig.update_layout(
        title_text=f"Distribution of {selected_col.replace('_', ' ').title()} by Target",
        xaxis_title_text=selected_col.replace('_', ' ').title(),
        yaxis_title_text="Count",
        barmode='overlay', bargap=0.1, template=PLOTLY_TEMPLATE,
        legend_title_text='Status', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

@app_metabric.callback(
    Output('categorical-bar-chart-metabric', 'figure'),
    [Input('bar-feature-metabric', 'value')]
)
def update_barchart_metabric(selected_feature):
    if not selected_feature or selected_feature not in metabric_df.columns or TARGET_METABRIC not in metabric_df.columns:
        return px.bar(template=PLOTLY_TEMPLATE).update_layout(title_text="Select feature or target missing", xaxis_showgrid=False, yaxis_showgrid=False)

    try:
        grouped_df = metabric_df.groupby([selected_feature, TARGET_METABRIC]).size().reset_index(name='count')
        if grouped_df.empty:
            return px.bar(template=PLOTLY_TEMPLATE).update_layout(title_text=f"No data for {selected_feature} by Target")

        grouped_df[TARGET_METABRIC] = grouped_df[TARGET_METABRIC].map({0: 'Not Died from Cancer', 1: 'Died from Cancer'})
        fig = px.bar(grouped_df, x=selected_feature, y='count', color=TARGET_METABRIC,
                     barmode='group', title=f"Counts of {selected_feature.replace('_', ' ').title()} by Target",
                     template=PLOTLY_TEMPLATE, color_discrete_map={'Not Died from Cancer': '#1f77b4', 'Died from Cancer': '#ff7f0e'})
        fig.update_layout(bargap=0.2, legend_title_text='Status', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                          xaxis_title=selected_feature.replace('_', ' ').title())
        return fig
    except Exception as e:
        print(f"Error in categorical bar chart for METABRIC: {e}")
        return px.bar(template=PLOTLY_TEMPLATE).update_layout(title_text=f"Error generating chart: {e}")


@app_metabric.callback(
    Output('feature-scatter-metabric', 'figure'),
    [Input('scatter-x-metabric', 'value'),
     Input('scatter-y-metabric', 'value')]
)
def update_scatter_metabric(x_col, y_col):
    fig = go.Figure()
    if not x_col or not y_col or x_col not in metabric_df.columns or y_col not in metabric_df.columns or TARGET_METABRIC not in metabric_df.columns:
        fig.update_layout(title_text="Select X & Y features or target missing", template=PLOTLY_TEMPLATE, xaxis_showgrid=False, yaxis_showgrid=False)
        return fig
    
    temp_df = metabric_df.copy()
    temp_df['Target_Label'] = temp_df[TARGET_METABRIC].map({0: 'Not Died from Cancer', 1: 'Died from Cancer'})

    fig = px.scatter(temp_df, x=x_col, y=y_col, color='Target_Label',
                     title=f"{x_col.replace('_',' ').title()} vs. {y_col.replace('_',' ').title()} by Target",
                     labels={x_col: x_col.replace('_',' ').title(), y_col: y_col.replace('_',' ').title()},
                     template=PLOTLY_TEMPLATE, marginal_y="violin", marginal_x="box",
                     color_discrete_map={'Not Died from Cancer': '#1f77b4', 'Died from Cancer': '#ff7f0e'},
                     hover_data=[col for col in temp_df.columns if col not in [x_col, y_col, 'Target_Label', TARGET_METABRIC] and metabric_df[col].nunique()<20][:5] # Show some other relevant data on hover
                     )
    fig.update_layout(legend_title_text='Status', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_traces(marker=dict(size=7, opacity=0.7))
    return fig


@app_metabric.callback(
    Output('confusion-matrix-metabric', 'figure'),
    Input('predict-button-metabric', 'n_clicks')
)
def display_confusion_matrix_metabric(_):
    z = conf_matrix_m
    x_labels = ['Predicted Not Died of Cancer', 'Predicted Died of Cancer']
    y_labels = ['Actual Not Died of Cancer', 'Actual Died of Cancer']

    if not model_trained_metabric or not isinstance(z, np.ndarray) or z.shape != (2,2): # Added isinstance check
        z = np.array([[0,0],[0,0]])
        title = 'Confusion Matrix (Model Not Trained or Error)'
    else:
        title = 'Confusion Matrix (Test Set)'
    
    z_text = [[str(int(y_val)) for y_val in x_val] for x_val in z]
    fig_cm = ff.create_annotated_heatmap(z, x=x_labels, y=y_labels, annotation_text=z_text,
                                       colorscale='Blues', showscale=False)
    fig_cm.update_layout(title_text=title, template=PLOTLY_TEMPLATE, title_x=0.5)
    return fig_cm

metabric_pred_input_states = []
# Ensure this list matches the order and IDs of inputs created in the layout
for feature in existing_model_features_metabric:
    metabric_pred_input_states.append(State(f'pred-metabric-{feature}', 'value'))

@app_metabric.callback(
    [Output('prediction-output-metabric', 'children'),
     Output('prediction-output-metabric', 'is_open'),
     Output('prediction-output-metabric', 'color')],
    [Input('predict-button-metabric', 'n_clicks')],
    metabric_pred_input_states
)
def make_prediction_metabric(n_clicks, *input_values):
    if n_clicks == 0:
        return "", False, "info"
    if not model_trained_metabric or pipeline_metabric is None:
        return "Model not trained. Cannot predict.", True, "danger"

    sample_data = {}
    has_user_input = False
    for i, feature_name in enumerate(existing_model_features_metabric):
        user_val = input_values[i]
        if user_val is not None: # User provided input for this field
            has_user_input = True
            try:
                if feature_name in numerical_features_metabric:
                    sample_data[feature_name] = float(user_val)
                else: # Categorical
                    sample_data[feature_name] = user_val # dbc.Select returns the value directly
            except ValueError:
                return f"Invalid input for {feature_name.replace('_',' ').title()}. Please enter a valid value.", True, "warning"
        else: # User did not provide input, use default (median/mode from X_model_metabric)
            if feature_name in X_model_metabric.columns:
                if feature_name in numerical_features_metabric:
                    sample_data[feature_name] = X_model_metabric[feature_name].median()
                else: # Categorical
                    sample_data[feature_name] = X_model_metabric[feature_name].mode()[0]
            else: # Should not happen if existing_model_features_metabric is defined correctly
                 return f"Configuration error: Feature {feature_name} definition missing for default value.", True, "danger"


    if not has_user_input: # If all fields were left blank by user (and thus filled by defaults)
         return "Please provide at least one input value or confirm defaults for prediction.", True, "info"


    try:
        input_df = pd.DataFrame([sample_data])
        # Ensure feature order and presence matches training
        input_df_ordered = input_df[existing_model_features_metabric]

        # Ensure categorical types match if necessary (astype('category') was done on X_model_metabric)
        # The OneHotEncoder should handle string categories properly
        for col in [f for f in categorical_features_metabric if f in input_df_ordered.columns]:
            # If your pipeline expects category dtype, ensure it. Otherwise, OHE usually handles strings.
            # If values are from dropdowns, they should be of the correct type already.
            pass

        prediction = pipeline_metabric.predict(input_df_ordered)[0]
        
        pred_label = "High Risk of Death from Cancer" if prediction == 1 else "Lower Risk of Death from Cancer"
        alert_color = "danger" if prediction == 1 else "success"
        result_text = f"{pred_label}"

        if hasattr(pipeline_metabric, "predict_proba") and y_pred_proba_test_m is not None: # Check if proba was calculated during training
            probability = pipeline_metabric.predict_proba(input_df_ordered)[0][1] # Prob of class 1
            result_text += f" (Predicted Risk Score: {probability:.2f})"
        
        return result_text, True, alert_color
    except Exception as e:
        print(f"Error in METABRIC prediction callback: {e}")
        return f"Prediction error: Check input types or model compatibility. Details: {e}", True, "danger"

# --- Run the METABRIC App ---
if __name__ == '__main__':
    app_metabric.run(debug=True, port=8052)
