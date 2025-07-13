# components/visualization.py
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd

def visualization_layout(data):
    if data is None:
        return html.Div("Please upload data to use this page.")
    else:
        df = pd.read_json(data, orient='split')
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        all_cols = df.columns

        layout = dbc.Container([
            html.H3("Data Visualization", className='text-center', style={'marginTop': '20px'}),

            # First Part: Visualization Analysis
            html.H4("Visualization Analysis"),
            dbc.Row([
                dbc.Col([
                    html.Label("Select X-axis:"),
                    dcc.Dropdown(
                        id='viz-x-axis',
                        options=[{'label': col, 'value': col} for col in all_cols],
                        placeholder='Select X-axis',
                        style={'width': '100%'}
                    ),
                ], width=3),
                dbc.Col([
                    html.Label("Select Y-axis:"),
                    dcc.Dropdown(
                        id='viz-y-axis',
                        options=[{'label': col, 'value': col} for col in all_cols],
                        placeholder='Select Y-axis',
                        style={'width': '100%'}
                    ),
                ], width=3),
                dbc.Col([
                    html.Label("Select Graph Type:"),
                    dcc.Dropdown(
                        id='viz-graph-type',
                        options=[
                            {'label': 'Scatter Plot', 'value': 'scatter'},
                            {'label': 'Line Plot', 'value': 'line'},
                            {'label': 'Bar Chart', 'value': 'bar'},
                            {'label': 'Histogram', 'value': 'histogram'},
                            {'label': 'Box Plot', 'value': 'box'},
                            {'label': 'Violin Plot', 'value': 'violin'},
                            {'label': 'Area Plot', 'value': 'area'},
                        ],
                        placeholder='Select Graph Type',
                        style={'width': '100%'}
                    ),
                ], width=3),
                dbc.Col([
                    html.Br(),
                    dbc.Button('Generate Graph', id='generate-viz-graph', n_clicks=0, color='primary'),
                ], width=3, style={'paddingTop': '5px'}),
            ], className='my-4'),
            dbc.Row([
                dbc.Col([
                    html.Div(id='viz-graph-output')
                ], width=12),
            ]),

            html.Hr(),

            # Second Part: Correlation Analysis
            html.H4("Correlation Analysis"),
            dbc.Row([
                dbc.Col([
                    html.Label("Select Features for Correlation:"),
                    dcc.Dropdown(
                        id='correlation-features',
                        options=[{'label': col, 'value': col} for col in numeric_cols],
                        multi=True,
                        placeholder='Select Features',
                        style={'width': '100%'}
                    ),
                ], width=6),
                dbc.Col([
                    html.Label("Select Correlation Graph Type:"),
                    dcc.Dropdown(
                        id='correlation-graph-type',
                        options=[
                            {'label': 'Correlation Heatmap', 'value': 'heatmap'},
                            {'label': 'Scatter Matrix', 'value': 'scatter_matrix'},
                            {'label': 'Correlation Matrix', 'value': 'corr_matrix'},
                            {'label': 'Pair Plot', 'value': 'pair_plot'},
                        ],
                        placeholder='Select Graph Type',
                        style={'width': '100%'}
                    ),
                ], width=3),
                dbc.Col([
                    html.Br(),
                    dbc.Button('Generate Correlation Graph', id='generate-correlation-graph', n_clicks=0, color='primary'),
                ], width=3, style={'paddingTop': '5px'}),
            ], className='my-4'),
            dbc.Row([
                dbc.Col([
                    html.Div(id='correlation-graph-output')
                ], width=12),
            ]),

            html.Hr(),

            # Third Part: Multidimensional Graph Visualization
            html.H4("Multidimensional Graph Visualization"),
            dbc.Row([
                dbc.Col([
                    html.Label("Select Graph Type:"),
                    dcc.Dropdown(
                        id='multi-graph-type',
                        options=[
                            {'label': '3D Scatter Plot', 'value': '3d_scatter'},
                            {'label': '3D Surface Plot', 'value': '3d_surface'},
                        ],
                        placeholder='Select Graph Type',
                        style={'width': '100%'}
                    ),
                ], width=3),
                dbc.Col([
                    html.Label("Select X-axis:"),
                    dcc.Dropdown(
                        id='multi-x-axis',
                        options=[{'label': col, 'value': col} for col in numeric_cols],
                        placeholder='Select X-axis',
                        style={'width': '100%'}
                    ),
                ], width=2),
                dbc.Col([
                    html.Label("Select Y-axis:"),
                    dcc.Dropdown(
                        id='multi-y-axis',
                        options=[{'label': col, 'value': col} for col in numeric_cols],
                        placeholder='Select Y-axis',
                        style={'width': '100%'}
                    ),
                ], width=2),
                dbc.Col([
                    html.Label("Select Z-axis:"),
                    dcc.Dropdown(
                        id='multi-z-axis',
                        options=[{'label': col, 'value': col} for col in numeric_cols],
                        placeholder='Select Z-axis',
                        style={'width': '100%'}
                    ),
                ], width=2),
                dbc.Col([
                    html.Label("Select Color Dimension (Optional):"),
                    dcc.Dropdown(
                        id='multi-color-dim',
                        options=[{'label': col, 'value': col} for col in all_cols],
                        placeholder='Select Color Dimension',
                        style={'width': '100%'}
                    ),
                ], width=3),
            ], className='my-4'),
            dbc.Row([
                dbc.Col([
                    dbc.Button('Generate Multidimensional Graph', id='generate-multi-graph', n_clicks=0, color='primary'),
                ], width=12),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id='multi-graph-output')
                ], width=12),
            ]),
        ])
        return layout
