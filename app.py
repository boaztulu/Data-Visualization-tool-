# app.py
import dash
from dash import html, dcc, Output, Input, State, callback_context, dash_table
import dash_bootstrap_components as dbc

from layouts.base_layout import create_base_layout
from components.preprocessing import preprocessing_layout
from components.visualization import visualization_layout
from components.ml import ml_layout

import base64
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import urllib.parse

from dash.dependencies import Input, Output, State, ALL, MATCH

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)

app.title = "Boaz Tulu Data Visualization Project"

app.layout = create_base_layout()

# Callback to handle file upload and store data
@app.callback(
    [Output('uploaded-data', 'data'),
     Output('upload-status', 'children')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
)
def handle_upload(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename.lower():
                # Suppress DtypeWarning by setting low_memory=False
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), low_memory=False)
                # Store the data in JSON format
                data = df.to_json(orient='split')
                return data, html.Div(f"File '{filename}' uploaded successfully.", style={'color': 'green'})
            else:
                return None, html.Div('Please upload a CSV file.', style={'color': 'red'})
        except Exception as e:
            print(e)
            return None, html.Div('There was an error processing the file.', style={'color': 'red'})
    return None, ''

# Callback to update page content based on active tab and uploaded data
@app.callback(
    Output('page-content', 'children'),
    [Input('tabs', 'active_tab'),
     Input('uploaded-data', 'data')]
)
def update_content(active_tab, data):
    if data is None:
        return html.Div('Please upload a CSV file to get started.')
    else:
        df = pd.read_json(data, orient='split')
        # Prepare file info
        file_info_card = dbc.Card(
            dbc.CardBody([
                html.H5("File Information", className="card-title"),
                html.P(f'File Name: Uploaded CSV'),
                html.P(f'Total Rows: {df.shape[0]}'),
                html.P(f'Total Columns: {df.shape[1]}'),
            ]),
            style={"margin": "20px"}
        )
        if active_tab == 'tab-preprocess':
            return html.Div([
                file_info_card,
                preprocessing_layout(data)
            ])
        elif active_tab == 'tab-visualize':
            return html.Div([
                file_info_card,
                visualization_layout(data)
            ])
        elif active_tab == 'tab-ml':
            return html.Div([
                file_info_card,
                ml_layout(data)
            ])
        else:
            return html.Div('Please select a tab.')

# Callback for Visualization Analysis
@app.callback(
    Output('viz-graph-output', 'children'),
    Input('generate-viz-graph', 'n_clicks'),
    State('viz-x-axis', 'value'),
    State('viz-y-axis', 'value'),
    State('viz-graph-type', 'value'),
    State('uploaded-data', 'data'),
)
def generate_viz_graph(n_clicks, x_axis, y_axis, graph_type, data):
    if n_clicks and x_axis and y_axis and graph_type and data:
        df = pd.read_json(data, orient='split')
        if graph_type == 'scatter':
            fig = px.scatter(df, x=x_axis, y=y_axis, title='Scatter Plot')
        elif graph_type == 'line':
            fig = px.line(df, x=x_axis, y=y_axis, title='Line Plot')
        elif graph_type == 'bar':
            fig = px.bar(df, x=x_axis, y=y_axis, title='Bar Chart')
        elif graph_type == 'histogram':
            fig = px.histogram(df, x=x_axis, y=y_axis, title='Histogram')
        elif graph_type == 'box':
            fig = px.box(df, x=x_axis, y=y_axis, title='Box Plot')
        elif graph_type == 'violin':
            fig = px.violin(df, x=x_axis, y=y_axis, title='Violin Plot')
        elif graph_type == 'area':
            fig = px.area(df, x=x_axis, y=y_axis, title='Area Plot')
        else:
            return html.Div("Invalid graph type selected.")
        return dcc.Graph(figure=fig)
    else:
        return html.Div("Please select X-axis, Y-axis, and Graph Type.")

# Callback for Correlation Analysis
@app.callback(
    Output('correlation-graph-output', 'children'),
    Input('generate-correlation-graph', 'n_clicks'),
    State('correlation-features', 'value'),
    State('correlation-graph-type', 'value'),
    State('uploaded-data', 'data'),
)
def generate_correlation_graph(n_clicks, features, graph_type, data):
    if n_clicks and features and graph_type and data:
        df = pd.read_json(data, orient='split')
        df = df[features]
        if graph_type == 'heatmap':
            corr = df.corr()
            fig = px.imshow(corr, text_auto=True, title='Correlation Heatmap')
        elif graph_type == 'scatter_matrix':
            fig = px.scatter_matrix(df, dimensions=features, title='Scatter Matrix')
        elif graph_type == 'corr_matrix':
            corr = df.corr()
            fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index))
            fig.update_layout(title='Correlation Matrix')
        elif graph_type == 'pair_plot':
            fig = px.scatter_matrix(df, dimensions=features, title='Pair Plot')
        else:
            return html.Div("Invalid graph type selected.")
        return dcc.Graph(figure=fig)
    else:
        return html.Div("Please select features and Graph Type.")

# Callback for Multidimensional Graph Visualization
@app.callback(
    Output('multi-graph-output', 'children'),
    Input('generate-multi-graph', 'n_clicks'),
    State('multi-graph-type', 'value'),
    State('multi-x-axis', 'value'),
    State('multi-y-axis', 'value'),
    State('multi-z-axis', 'value'),
    State('multi-color-dim', 'value'),
    State('uploaded-data', 'data'),
)
def generate_multi_graph(n_clicks, graph_type, x_axis, y_axis, z_axis, color_dim, data):
    if n_clicks and graph_type and x_axis and y_axis and z_axis and data:
        df = pd.read_json(data, orient='split')
        if graph_type == '3d_scatter':
            fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color=color_dim, title='3D Scatter Plot')
        elif graph_type == '3d_surface':
            fig = go.Figure(data=[go.Surface(z=df[z_axis], x=df[x_axis], y=df[y_axis])])
            fig.update_layout(title='3D Surface Plot', autosize=True)
        else:
            return html.Div("Invalid graph type selected.")
        return dcc.Graph(figure=fig)
    else:
        return html.Div("Please select all required axes and Graph Type.")

# Callback to update training features options based on target feature selection
@app.callback(
    Output('ml-training-features', 'options'),
    Input('ml-target-feature', 'value'),
    State('uploaded-data', 'data'),
)
def update_training_features_options(target_feature, data):
    if data:
        df = pd.read_json(data, orient='split')
        all_features = df.columns.tolist()
        options = [{'label': f, 'value': f, 'disabled': f == target_feature} for f in all_features]
        return options
    else:
        return []

# Callback to update target feature options based on training features selection
@app.callback(
    Output('ml-target-feature', 'options'),
    Input('ml-training-features', 'value'),
    State('uploaded-data', 'data'),
)
def update_target_feature_options(training_features, data):
    if data:
        df = pd.read_json(data, orient='split')
        all_features = df.columns.tolist()
        training_features = training_features or []
        options = [{'label': f, 'value': f, 'disabled': f in training_features} for f in all_features]
        return options
    else:
        return []

# Callback to update train-test split display
@app.callback(
    Output('train-size-output', 'children'),
    Output('test-size-output', 'children'),
    Input('train-test-slider', 'value'),
    State('uploaded-data', 'data'),
)
def update_train_test_split(value, data):
    if data:
        df = pd.read_json(data, orient='split')
        total_rows = df.shape[0]
        train_rows = int(total_rows * value / 100)
        test_rows = total_rows - train_rows
        return f"Training Size: {train_rows} rows ({value}%)", f"Testing Size: {test_rows} rows ({100 - value}%)"
    else:
        return "", ""

# Callback to show model selection after clicking 'Next'
@app.callback(
    Output('model-selection', 'style'),
    Input('ml-next-button', 'n_clicks'),
    State('ml-training-features', 'value'),
    State('ml-target-feature', 'value'),
)
def show_model_selection(n_clicks, training_features, target_feature):
    if n_clicks and training_features and target_feature:
        return {'display': 'block'}
    else:
        return {'display': 'none'}

# Callback for opening and closing hyperparameter modal
@app.callback(
    [Output('hyperparameter-modal', 'is_open'),
     Output('hyperparameter-modal-body', 'children'),
     Output('selected-model-name', 'data')],
    [Input({'type': 'model-button', 'index': ALL}, 'n_clicks'),
     Input('close-hyperparameter-modal', 'n_clicks')],
    [State({'type': 'model-button', 'index': ALL}, 'id'),
     State('hyperparameter-modal', 'is_open')]
)
def toggle_hyperparameter_modal(n_clicks_list, close_clicks, ids, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return is_open, dash.no_update, dash.no_update
    else:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if 'close-hyperparameter-modal' in triggered_id:
            return False, dash.no_update, dash.no_update
        else:
            for i, n_clicks in enumerate(n_clicks_list):
                if n_clicks:
                    model_name = ids[i]['index']
                    # Generate hyperparameter inputs based on model_name
                    if model_name == 'Linear Regression':
                        content = html.Div([
                            dbc.FormGroup([
                                dbc.Label("Fit Intercept:"),
                                dbc.RadioItems(
                                    options=[
                                        {"label": "True", "value": True},
                                        {"label": "False", "value": False},
                                    ],
                                    value=True,
                                    id='fit-intercept',
                                ),
                            ]),
                        ])
                    elif model_name == 'Decision Tree Regressor':
                        content = html.Div([
                            dbc.FormGroup([
                                dbc.Label("Max Depth:"),
                                dbc.Input(id='max-depth', type='number', placeholder='Enter max depth'),
                            ]),
                            dbc.FormGroup([
                                dbc.Label("Min Samples Split:"),
                                dbc.Input(id='min-samples-split', type='number', placeholder='Enter min samples split'),
                            ]),
                        ])
                    # Add hyperparameters for other models...
                    else:
                        content = html.Div(f"No hyperparameters available for {model_name}")
                    return True, content, model_name
            return is_open, dash.no_update, dash.no_update

# Callback for opening and closing training summary modal
@app.callback(
    [Output('training-summary-modal', 'is_open'),
     Output('training-summary-body', 'children')],
    [Input('train-button', 'n_clicks'),
     Input('close-training-summary', 'n_clicks')],
    [State('training-summary-modal', 'is_open'),
     State('selected-model-name', 'data'),
     State('ml-training-features', 'value'),
     State('ml-target-feature', 'value'),
     State('train-test-slider', 'value'),
     State('uploaded-data', 'data'),
     State('fit-intercept', 'value'),
     State('max-depth', 'value'),
     State('min-samples-split', 'value')]
)
def toggle_training_summary_modal(train_clicks, close_clicks, is_open, model_name, training_features, target_feature, train_size, data, fit_intercept, max_depth, min_samples_split):
    ctx = callback_context
    if not ctx.triggered:
        return is_open, dash.no_update
    else:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if 'close-training-summary' in triggered_id:
            return False, dash.no_update
        elif 'train-button' in triggered_id and train_clicks:
            if data and training_features and target_feature:
                total_rows = pd.read_json(data, orient='split').shape[0]
                train_rows = int(total_rows * train_size / 100)
                test_rows = total_rows - train_rows
                hyperparameters = {}
                if model_name == 'Linear Regression':
                    hyperparameters['fit_intercept'] = fit_intercept
                elif model_name == 'Decision Tree Regressor':
                    hyperparameters['max_depth'] = max_depth
                    hyperparameters['min_samples_split'] = min_samples_split
                # Collect other hyperparameters as needed
                summary = html.Div([
                    html.P(f"Model: {model_name}"),
                    html.P(f"Training Size: {train_rows} rows ({train_size}%)"),
                    html.P(f"Testing Size: {test_rows} rows ({100 - train_size}%)"),
                    html.P(f"Training Features: {', '.join(training_features)}"),
                    html.P(f"Target Feature: {target_feature}"),
                    html.P(f"Hyperparameters: {hyperparameters}"),
                ])
                return True, summary
            else:
                return is_open, html.Div("Please complete all steps before training.")
        else:
            return is_open, dash.no_update

# Callback to update missing values modal
@app.callback(
    Output('missing-values-modal', 'is_open'),
    Output('missing-values-content', 'children'),
    [Input('show-missing-values', 'n_clicks'),
     Input('close-missing-values', 'n_clicks')],
    [State('missing-values-modal', 'is_open'),
     State('uploaded-data', 'data')]
)
def toggle_missing_values_modal(n1, n2, is_open, data):
    ctx = callback_context
    if not ctx.triggered:
        return is_open, dash.no_update
    else:
        if data:
            df = pd.read_json(data, orient='split')
            missing_df = df.isnull().sum().reset_index()
            missing_df.columns = ['Feature', 'Missing Values']
            fig = px.bar(missing_df, x='Feature', y='Missing Values', title='Missing Values per Feature')
            content = dcc.Graph(figure=fig)
        else:
            content = html.Div("No data available.")
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if 'show-missing-values' in triggered_id:
            return True, content
        elif 'close-missing-values' in triggered_id:
            return False, dash.no_update
        else:
            return is_open, dash.no_update

# Callback to update outliers modal
@app.callback(
    Output('outliers-modal', 'is_open'),
    Output('outlier-feature-list', 'children'),
    [Input('show-outliers', 'n_clicks'),
     Input('close-outliers', 'n_clicks')],
    [State('outliers-modal', 'is_open'),
     State('uploaded-data', 'data')]
)
def toggle_outliers_modal(n1, n2, is_open, data):
    ctx = callback_context
    if not ctx.triggered:
        return is_open, dash.no_update
    else:
        if data and 'show-outliers' in ctx.triggered[0]['prop_id']:
            df = pd.read_json(data, orient='split')
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            feature_buttons = []
            for feature in numeric_cols:
                button = dbc.Button(
                    feature,
                    id={'type': 'outlier-feature-button', 'index': feature},
                    n_clicks=0,
                    color='secondary',
                    style={'width': '100%', 'marginBottom': '5px'}
                )
                feature_buttons.append(button)
            return True, feature_buttons
        elif 'close-outliers' in ctx.triggered[0]['prop_id']:
            return False, dash.no_update
        else:
            return is_open, dash.no_update

# Callback to update outlier boxplot
@app.callback(
    Output('outlier-boxplot', 'figure'),
    Input({'type': 'outlier-feature-button', 'index': ALL}, 'n_clicks'),
    State({'type': 'outlier-feature-button', 'index': ALL}, 'id'),
    State('uploaded-data', 'data'),
)
def update_outlier_boxplot(n_clicks_list, ids, data):
    if not data:
        return {}
    ctx = callback_context
    if not ctx.triggered:
        return {}
    else:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if 'outlier-feature-button' in triggered_id:
            triggered_id = eval(triggered_id)
            feature = triggered_id['index']
            df = pd.read_json(data, orient='split')
            fig = px.box(df, y=feature, title=f'Box Plot of {feature}')
            return fig
        else:
            return {}

# Callback to toggle scaling modal
@app.callback(
    Output('scaling-modal', 'is_open'),
    [Input('feature-scaling', 'n_clicks'),
     Input('close-scaling', 'n_clicks')],
    State('scaling-modal', 'is_open'),
)
def toggle_scaling_modal(n1, n2, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return is_open
    else:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if 'feature-scaling' in triggered_id:
            return True
        elif 'close-scaling' in triggered_id:
            return False
        else:
            return is_open

# Populate features in the scaling modal
@app.callback(
    Output('scaling-features', 'options'),
    Input('scaling-modal', 'is_open'),
    State('uploaded-data', 'data'),
)
def populate_scaling_features(is_open, data):
    if is_open and data:
        df = pd.read_json(data, orient='split')
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        return [{'label': col, 'value': col} for col in numeric_cols]
    return []

# Callback to apply scaling
@app.callback(
    Output('preprocessed-data', 'data'),
    Input('apply-scaling', 'n_clicks'),
    State('scaling-method', 'value'),
    State('scaling-features', 'value'),
    State('preprocessed-data', 'data'),
    State('uploaded-data', 'data'),
)
def apply_scaling(n_clicks, method, features, preprocessed_data, uploaded_data):
    if n_clicks and method and features:
        if preprocessed_data:
            df = pd.read_json(preprocessed_data, orient='split')
        else:
            df = pd.read_json(uploaded_data, orient='split')
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        scaler = None
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        if scaler:
            df_scaled = df.copy()
            df_scaled[features] = scaler.fit_transform(df[features])
            return df_scaled.to_json(orient='split')
        else:
            return preprocessed_data or uploaded_data
    else:
        return preprocessed_data or uploaded_data

if __name__ == "__main__":
    app.run_server(debug=True)
