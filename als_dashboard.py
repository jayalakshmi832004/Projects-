import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# --- Initialize Dash App with a Bootstrap Theme ---
app_als = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])
app_als.title = "ALS GWAS Explorer"

# Plotly template for charts
PLOTLY_TEMPLATE = "plotly_white"

# --- 1. Load your ALS Data ---
file_path = 'cleaned_als_data.csv' # Or your full path
try:
    als_df = pd.read_csv(file_path)
    print(f"Successfully loaded '{file_path}'")
    # Basic data type sanity checks (adjust as per your actual data)
    if 'CHR' in als_df.columns:
        als_df['CHR'] = als_df['CHR'].astype(str)
    if 'BP' in als_df.columns:
        als_df['BP'] = pd.to_numeric(als_df['BP'], errors='coerce')
    if 'P' in als_df.columns:
        als_df['P'] = pd.to_numeric(als_df['P'], errors='coerce')
    if 'Effect' in als_df.columns:
        als_df['Effect'] = pd.to_numeric(als_df['Effect'], errors='coerce')

except FileNotFoundError:
    print(f"Error: '{file_path}' not found. Using dummy data.")
    als_df = pd.DataFrame({
        'CHR': [str(i) for i in np.random.randint(1, 23, 2000)] + ['X', 'Y'] * 50,
        'BP': np.random.randint(1, 2000000, 2100),
        'P': np.random.uniform(1e-10, 1, 2100),
        'Effect': np.random.normal(0, 0.1, 2100),
        'effectAlleleFreq': np.random.uniform(0.01, 0.99, 2100),
        'SNP_ID': [f'rs{i}' for i in range(2100)] # Adding SNP IDs for hover
    })
    # Add some more significant SNPs to dummy data for better visualization
    if not als_df.empty:
        significant_indices = np.random.choice(als_df.index, min(10, len(als_df)), replace=False)
        als_df.loc[significant_indices, 'P'] = np.random.uniform(1e-9, 1e-7, len(significant_indices))

# --- 2. Data Preprocessing for Visualization (FutureWarning Free) ---
# Initialize placeholder for data if loading fails completely
processed_als_df = pd.DataFrame()
tick_positions = []
tick_labels = []
SIGNIFICANCE_THRESHOLD_PVAL = 5e-8 # Default
SIGNIFICANCE_THRESHOLD_LOG10 = -np.log10(SIGNIFICANCE_THRESHOLD_PVAL) # Default

if not als_df.empty:
    # Make a copy to ensure all modifications are on a new DataFrame, avoiding SettingWithCopyWarning
    current_df = als_df.copy()

    # Calculate -log10(P)
    current_df = current_df.dropna(subset=['P'])
    if not current_df.empty:
        current_df['P_log10'] = -np.log10(current_df['P'])
        current_df['P_log10'] = current_df['P_log10'].replace([np.inf, -np.inf], np.nan)
        current_df = current_df.dropna(subset=['P_log10', 'BP', 'CHR'])

    # Genome-wide significance line
    SIGNIFICANCE_THRESHOLD_PVAL = 5e-8
    SIGNIFICANCE_THRESHOLD_LOG10 = -np.log10(SIGNIFICANCE_THRESHOLD_PVAL)

    # Prepare for Manhattan plot (cumulative position)
    if not current_df.empty and 'CHR' in current_df.columns:
        chr_map = {str(i): i for i in range(1, 23)}
        chr_map.update({'X': 23, 'Y': 24, 'MT': 25, 'x': 23, 'y': 24, 'mt': 25})
        current_df['chr_num_temp'] = current_df['CHR'].astype(str).str.upper().map(chr_map)
        current_df = current_df.dropna(subset=['chr_num_temp'])

        if not current_df.empty:
            current_df['chr_num_temp'] = current_df['chr_num_temp'].astype(int)
            current_df = current_df.sort_values(['chr_num_temp', 'BP'])

            # Calculate cumulative base pair position safely
            temp_bp_cumulative = pd.Series(index=current_df.index, dtype='float64')
            chromosome_centers = {}
            # tick_positions and tick_labels re-initialized inside if needed

            chr_lengths = current_df.groupby('chr_num_temp')['BP'].max()
            valid_chr_nums = sorted(current_df['chr_num_temp'].unique())
            
            cumulative_starts = chr_lengths.shift(1).cumsum()
            # Use .ffill() as per FutureWarning for method='ffill'
            cumulative_starts = cumulative_starts.reindex(valid_chr_nums).ffill().fillna(0)

            current_tick_positions = []
            current_tick_labels = []

            for chr_num in valid_chr_nums:
                mask = (current_df['chr_num_temp'] == chr_num)
                current_cumulative_start = cumulative_starts.get(chr_num, 0)
                # Assign to the temporary series
                temp_bp_cumulative.loc[mask] = current_df.loc[mask, 'BP'] + current_cumulative_start
                
                if mask.any(): # Ensure there are rows for this chromosome
                    # Use the values from temp_bp_cumulative for center calculation
                    chr_min_cumulative_val = temp_bp_cumulative.loc[mask].min()
                    chr_max_cumulative_val = temp_bp_cumulative.loc[mask].max()
                    
                    if pd.notna(chr_min_cumulative_val) and pd.notna(chr_max_cumulative_val):
                        chromosome_centers[chr_num] = (chr_min_cumulative_val + chr_max_cumulative_val) / 2
                        current_tick_positions.append(chromosome_centers[chr_num])
                        # Find original chromosome name robustly
                        original_chr_name = 'Unknown'
                        for k, v in chr_map.items():
                            if v == chr_num:
                                original_chr_name = k
                                break
                        current_tick_labels.append(original_chr_name)
            
            current_df['BP_cumulative'] = temp_bp_cumulative # Assign back the calculated cumulative positions
            tick_positions = current_tick_positions
            tick_labels = current_tick_labels

            unique_chroms_sorted = sorted(current_df['chr_num_temp'].unique())
            color_map = {chrom: ('#0d6efd' if i % 2 == 0 else '#6c757d') for i, chrom in enumerate(unique_chroms_sorted)}
            current_df['color_group'] = current_df['chr_num_temp'].map(color_map)
            processed_als_df = current_df # Final processed DataFrame
        else: # current_df became empty after chr_num_temp processing
            processed_als_df = current_df # which is empty
            processed_als_df['BP_cumulative'] = pd.Series(dtype='float64') # Ensure column exists
            processed_als_df['color_group'] = pd.Series(dtype='str')      # Ensure column exists
    else: # CHR column missing or df was empty initially
        if not current_df.empty:
            current_df['BP_cumulative'] = current_df['BP'] if 'BP' in current_df else pd.Series(dtype='float64')
            current_df['color_group'] = '#0d6efd'
        processed_als_df = current_df
else: # Original als_df was empty
    processed_als_df = pd.DataFrame(columns=['CHR', 'BP', 'P', 'P_log10', 'Effect', 'effectAlleleFreq', 'SNP_ID', 'BP_cumulative', 'color_group', 'chr_num_temp'])

# Columns for the table, ensure they exist in the processed DataFrame
cols_for_table = ['SNP_ID', 'CHR', 'BP', 'P', 'Effect', 'effectAlleleFreq']
cols_for_table = [c for c in cols_for_table if c in processed_als_df.columns]


# --- 3. Define App Layout with Dash Bootstrap Components ---
navbar = dbc.NavbarSimple(
    brand="ALS Genome-Wide Association Study Explorer",
    brand_href="#",
    color="primary",
    dark=True,
    className="mb-4"
)

# Manhattan Plot Controls
manhattan_controls = dbc.CardBody([
    dbc.Row([
        dbc.Col([
            html.Label("Max -log10(P) Y-axis:", className="form-label"),
            dcc.Slider(
                id='pvalue-zoom-slider-als',
                min=0,
                max=max(30, int(processed_als_df['P_log10'].max() + 5)) if not processed_als_df.empty and 'P_log10' in processed_als_df and not processed_als_df['P_log10'].empty else 30,
                step=1,
                value=max(15, int(processed_als_df['P_log10'].max() + 5)) if not processed_als_df.empty and 'P_log10' in processed_als_df and not processed_als_df['P_log10'].empty else 15,
                marks={i: str(i) for i in range(0, max(30, int(processed_als_df['P_log10'].max() + 6)) if not processed_als_df.empty and 'P_log10' in processed_als_df and not processed_als_df['P_log10'].empty else 31, 5)},
                tooltip={"placement": "bottom", "always_visible": False}
            )
        ], width=12)
    ])
])

# Other Plots Controls
other_plots_controls = dbc.CardBody([
    html.Label("Filter by Chromosome (for scatter plot):", className="form-label"),
    dbc.Select(
        id='chromosome-dropdown-als',
        options=[{'label': 'All Chromosomes', 'value': 'all'}] +
                [{'label': f'CHR {c}', 'value': c} for c in sorted(processed_als_df['CHR'].unique()) if 'CHR' in processed_als_df and pd.notna(c)],
        value='all',
    )
])

# Top SNPs Table Controls
table_controls = dbc.CardBody([
    html.Label("P-value threshold for Top SNPs table:", className="form-label"),
    dbc.Input(id='pvalue-threshold-table-als', type='number', value=SIGNIFICANCE_THRESHOLD_PVAL, step=1e-9, min=0, max=1, placeholder="e.g., 5e-8")
])

app_als.layout = dbc.Container(fluid=True, children=[
    navbar,
    dbc.Row([
        dbc.Col(md=12, children=[
            dbc.Card([
                dbc.CardHeader(html.H5("Manhattan Plot", className="mb-0")),
                manhattan_controls,
                dcc.Loading(dcc.Graph(id='manhattan-plot-als', style={"height": "60vh"}), type="circle")
            ], className="shadow-sm mb-4")
        ])
    ]),
    dbc.Row([
        dbc.Col(md=6, children=[
            dbc.Card([
                dbc.CardHeader(html.H5("P-value Distribution", className="mb-0")),
                dcc.Graph(id='pvalue-histogram-als')
            ], className="shadow-sm mb-4")
        ]),
        dbc.Col(md=6, children=[
            dbc.Card([
                dbc.CardHeader(html.H5("Effect Size vs. P-value", className="mb-0")),
                other_plots_controls,
                dcc.Graph(id='effect-vs-pvalue-scatter-als')
            ], className="shadow-sm mb-4")
        ])
    ]),
    dbc.Row([
        dbc.Col(md=12, children=[
            dbc.Card([
                dbc.CardHeader(html.H5("Top Associated SNPs", className="mb-0")),
                table_controls,
                html.Div(id='significant-snps-table-als-container',
                         children=dash_table.DataTable(
                             id='significant-snps-table-als',
                             columns=[{"name": i, "id": i} for i in cols_for_table], # Uses dynamically checked cols_for_table
                             page_size=10,
                             sort_action="native",
                             filter_action="native",
                             style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                             style_cell={'textAlign': 'left', 'padding': '5px', 'minWidth': '80px', 'width': '120px', 'maxWidth': '180px', 'overflow': 'hidden', 'textOverflow': 'ellipsis'},
                             style_table={'overflowX': 'auto'},
                         )
                )
            ], className="shadow-sm mb-4")
        ])
    ])
])

# --- 4. Define Callbacks ---
@app_als.callback(
    Output('manhattan-plot-als', 'figure'),
    [Input('pvalue-zoom-slider-als', 'value')]
)
def update_manhattan_plot(max_y_zoom):
    df_to_plot = processed_als_df # Use the fully processed DataFrame

    if df_to_plot.empty or 'BP_cumulative' not in df_to_plot.columns or df_to_plot['BP_cumulative'].isnull().all():
        fig = go.Figure()
        fig.update_layout(title_text="Manhattan Plot (No data available or cumulative BP missing)", template=PLOTLY_TEMPLATE, xaxis_showgrid=False, yaxis_showgrid=False)
        return fig

    fig = go.Figure()

    # Group by numerical chromosome for plotting
    if 'chr_num_temp' in df_to_plot.columns:
        for chr_num_val, group in df_to_plot.groupby('chr_num_temp'):
            color = group['color_group'].iloc[0] if 'color_group' in group and not group['color_group'].empty else '#0d6efd'
            
            customdata_cols = ['SNP_ID', 'P', 'Effect']
            present_custom_cols = [col for col in customdata_cols if col in group.columns]
            customdata_df = group[present_custom_cols]

            hovertemplate_parts = []
            if 'SNP_ID' in present_custom_cols: hovertemplate_parts.append("<b>SNP:</b> %{customdata[0]}")
            if 'P' in present_custom_cols: hovertemplate_parts.append("<b>P-value:</b> %{customdata[1]:.2e}") # Adjust index based on actual order
            if 'Effect' in present_custom_cols: hovertemplate_parts.append("<b>Effect:</b> %{customdata[2]:.3f}") # Adjust index

            # Rebuild customdata indices for template string
            template_indices = {}
            current_idx = 0
            for col_name in ['SNP_ID', 'P', 'Effect']: # Fixed order for template
                if col_name in present_custom_cols:
                    template_indices[col_name] = current_idx
                    current_idx +=1
            
            hovertemplate_str = []
            if 'SNP_ID' in template_indices: hovertemplate_str.append(f"<b>SNP:</b> %{{customdata[{template_indices['SNP_ID']}]}}")
            if 'P' in template_indices: hovertemplate_str.append(f"<b>P-value:</b> %{{customdata[{template_indices['P']}]}}:.2e")
            if 'Effect' in template_indices: hovertemplate_str.append(f"<b>Effect:</b> %{{customdata[{template_indices['Effect']}]}}:.3f")


            hovertemplate_str.extend([
                "<b>Position:</b> CHR %{text}:%{customdata[BP_idx]}", # We'd need to pass BP to customdata if not using x for it
                "-log10(P): %{y:.2f}<extra></extra>"
            ])
            
            # Simplified hover for now, more complex template needs careful customdata construction
            hover_text = (
                "<b>SNP:</b> %{customdata[0]}<br>"
                "<b>P-value:</b> %{customdata[1]:.2e}<br>"
                "<b>-log10(P):</b> %{y:.2f}<br>"
                "<b>Position:</b> CHR %{text}"
                "<extra></extra>"
            ) if 'SNP_ID' in present_custom_cols and 'P' in present_custom_cols else "<b>-log10(P):</b> %{y:.2f}<extra></extra>"


            fig.add_trace(go.Scattergl(
                x=group['BP_cumulative'],
                y=group['P_log10'],
                mode='markers',
                marker=dict(color=color, size=5, opacity=0.7),
                name=f'CHR {group["CHR"].iloc[0]}' if 'CHR' in group and not group['CHR'].empty else 'Unknown CHR',
                customdata=customdata_df.to_numpy() if not customdata_df.empty else None,
                text=group['CHR'] if 'CHR' in group else None, # Original CHR name for hover text
                hovertemplate=hover_text
            ))
    else: # Fallback if chr_num_temp is not available
        fig.add_trace(go.Scattergl(x=df_to_plot['BP_cumulative'], y=df_to_plot['P_log10'], mode='markers'))

    
    fig.add_hline(y=SIGNIFICANCE_THRESHOLD_LOG10, line_dash="dash", line_color="red",
                  annotation_text="GWAS Significance (5e-8)", annotation_position="bottom right")

    fig.update_layout(
        title="Manhattan Plot of ALS GWAS Results",
        xaxis_title="Genomic Position (Cumulative)",
        yaxis_title="-log10(P-value)",
        xaxis=dict(
            tickmode='array',
            tickvals=tick_positions, # Use global tick_positions
            ticktext=tick_labels,   # Use global tick_labels
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(range=[0, max_y_zoom], showgrid=True, gridwidth=0.5, gridcolor='rgba(200,200,200,0.5)'),
        showlegend=False,
        template=PLOTLY_TEMPLATE,
        dragmode='pan'
    )
    return fig

@app_als.callback(
    Output('pvalue-histogram-als', 'figure'),
    Input('chromosome-dropdown-als', 'value') # Dummy input for now
)
def update_pvalue_histogram(_):
    df_to_plot = processed_als_df
    if df_to_plot.empty or 'P' not in df_to_plot.columns:
        return px.histogram(title="P-value Distribution (No Data)", template=PLOTLY_TEMPLATE)
    
    fig = px.histogram(df_to_plot, x='P', nbins=100, title="P-value Distribution", opacity=0.8)
    fig.update_layout(xaxis_title="P-value", yaxis_title="Frequency", template=PLOTLY_TEMPLATE, bargap=0.1)
    return fig

@app_als.callback(
    Output('effect-vs-pvalue-scatter-als', 'figure'),
    [Input('chromosome-dropdown-als', 'value')]
)
def update_effect_scatter(selected_chr_val):
    df_to_plot = processed_als_df
    if df_to_plot.empty or 'Effect' not in df_to_plot.columns or 'P_log10' not in df_to_plot.columns:
        return px.scatter(title="Effect Size vs. P-value (No Data)", template=PLOTLY_TEMPLATE)

    filtered_df = df_to_plot
    title_suffix = "All Chromosomes"
    if selected_chr_val != 'all' and 'CHR' in df_to_plot.columns:
        # Ensure selected_chr_val is compared correctly if it's numeric or string
        filtered_df = df_to_plot[df_to_plot['CHR'].astype(str) == str(selected_chr_val)]
        title_suffix = f"Chromosome {selected_chr_val}"
    
    if filtered_df.empty:
         return px.scatter(title=f"No data for {title_suffix}", template=PLOTLY_TEMPLATE)

    hover_data_cols = ['SNP_ID', 'CHR', 'BP', 'P', 'effectAlleleFreq']
    hover_data_cols_present = [col for col in hover_data_cols if col in filtered_df.columns]

    fig = px.scatter(filtered_df, x='Effect', y='P_log10',
                     color='effectAlleleFreq' if 'effectAlleleFreq' in filtered_df.columns else None,
                     color_continuous_scale=px.colors.sequential.Viridis,
                     title=f"Effect Size vs. -log10(P-value) for {title_suffix}",
                     labels={'Effect': 'Effect Size (Beta)', 'P_log10': '-log10(P-value)', 'effectAlleleFreq': 'Effect Allele Freq.'},
                     hover_data=hover_data_cols_present,
                     template=PLOTLY_TEMPLATE,
                     opacity=0.7)
    fig.add_hline(y=SIGNIFICANCE_THRESHOLD_LOG10, line_dash="dash", line_color="red")
    fig.update_traces(marker=dict(size=6))
    return fig

@app_als.callback(
    Output('significant-snps-table-als', 'data'),
    [Input('pvalue-threshold-table-als', 'value')]
)
def update_significant_snps_table(p_threshold_val):
    df_to_use = processed_als_df
    if df_to_use.empty or 'P' not in df_to_use.columns:
        return []
        
    # Prepare a default empty DataFrame structure matching cols_for_table
    empty_df_structure = pd.DataFrame(columns=cols_for_table)

    if p_threshold_val is None or p_threshold_val == "":
        return empty_df_structure.to_dict('records')

    try:
        p_thresh = float(p_threshold_val)
        significant_df = df_to_use[df_to_use['P'] < p_thresh].sort_values('P')
        # Ensure only defined columns are selected for the table
        return significant_df[cols_for_table].head(100).to_dict('records')
    except ValueError: # Invalid threshold input
        return empty_df_structure.to_dict('records')
    except Exception as e:
        print(f"Error updating table: {e}")
        return []


# --- Run the App ---
if __name__ == '__main__':
    app_als.run(debug=True, port=8050)
