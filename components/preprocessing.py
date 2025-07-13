# components/preprocessing.py
from dash import html, dcc
import dash_bootstrap_components as dbc

def preprocessing_layout(data):
    layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("Data Preprocessing", className='text-center', style={'marginTop': '20px'}),
            ], width=12),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Button('Show Missing Values', id='show-missing-values', n_clicks=0, color='info', className='mr-2'),
            ], width=4),
            dbc.Col([
                dbc.Button('Show Outliers', id='show-outliers', n_clicks=0, color='info', className='mr-2'),
            ], width=4),
            dbc.Col([
                dbc.Button('Feature Scaling', id='feature-scaling', n_clicks=0, color='info'),
            ], width=4),
        ], className='my-4'),
        dbc.Row([
            dbc.Col([
                html.H5("Handle Missing Values:"),
                dcc.Dropdown(
                    id='missing-method',
                    options=[
                        {'label': 'Mean Imputation', 'value': 'mean'},
                        {'label': 'Median Imputation', 'value': 'median'},
                        {'label': 'Mode Imputation', 'value': 'mode'},
                        {'label': 'Drop Missing Values', 'value': 'drop'}
                    ],
                    placeholder='Select a method',
                    style={'width': '100%'}
                ),
            ], width=6),
            dbc.Col([
                html.H5("Handle Outliers:"),
                dcc.Dropdown(
                    id='outlier-method',
                    options=[
                        {'label': 'Remove Outliers', 'value': 'remove'},
                        {'label': 'Cap Outliers', 'value': 'cap'},
                        {'label': 'Do Nothing', 'value': 'none'}
                    ],
                    placeholder='Select a method',
                    style={'width': '100%'}
                ),
            ], width=6),
        ], className='my-4'),
        dbc.Row([
            dbc.Col([
                dbc.Button('Apply Preprocessing and Download Preprocessed Data', id='apply-preprocessing', n_clicks=0, color='success'),
                html.A(
                    ' Download Preprocessed Data',
                    id='download-link',
                    download="preprocessed_data.csv",
                    href="#",
                    target="_blank",
                    className='btn btn-link ml-2',
                    style={'marginTop': '20px'}
                ),
            ], width=12),
        ], className='my-4'),
        # Modals
        dbc.Modal(
            [
                dbc.ModalHeader("Missing Values"),
                dbc.ModalBody(id='missing-values-content'),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-missing-values", className="ml-auto")
                ),
            ],
            id="missing-values-modal",
            size="lg",
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Outliers"),
                dbc.ModalBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Click on a Feature to View Outliers:"),
                            html.Div(id='outlier-feature-list', style={'overflowY': 'scroll', 'height': '400px'})
                        ], width=4),
                        dbc.Col([
                            dcc.Graph(id='outlier-boxplot')
                        ], width=8),
                    ])
                ]),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-outliers", className="ml-auto")
                ),
            ],
            id="outliers-modal",
            size="lg",
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Feature Scaling"),
                dbc.ModalBody([
                    html.H5("Select Features to Scale:"),
                    dcc.Checklist(
                        id='scaling-features',
                        options=[],
                        value=[],
                        labelStyle={'display': 'block'}
                    ),
                    html.H5("Select Scaling Method:", style={'marginTop': '20px'}),
                    dcc.RadioItems(
                        id='scaling-method',
                        options=[
                            {'label': 'Standardization (Z-score)', 'value': 'standard'},
                            {'label': 'Normalization (Min-Max)', 'value': 'minmax'},
                            {'label': 'Robust Scaling', 'value': 'robust'},
                        ],
                        value='standard',
                        labelStyle={'display': 'block'}
                    ),
                ]),
                dbc.ModalFooter(
                    [
                        dbc.Button("Apply Scaling", id="apply-scaling", className="mr-2"),
                        dbc.Button("Close", id="close-scaling", className="ml-auto"),
                    ]
                ),
            ],
            id="scaling-modal",
            size="lg",
        ),
    ])
    return layout
