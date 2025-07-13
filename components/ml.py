# components/ml.py
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd

def ml_layout(data):
    if data is None:
        return html.Div("Please upload data to use this page.")
    else:
        df = pd.read_json(data, orient='split')
        all_features = df.columns.tolist()

        layout = dbc.Container([
            html.H3("Machine Learning", className='text-center', style={'marginTop': '20px'}),
            dbc.Row([
                dbc.Col([
                    html.H5("Select Training Features:"),
                    dcc.Dropdown(
                        id='ml-training-features',
                        options=[{'label': f, 'value': f} for f in all_features],
                        multi=True,
                        placeholder='Select training features',
                        style={'width': '100%'}
                    ),
                ], width=6),
                dbc.Col([
                    html.H5("Select Target Feature:"),
                    dcc.Dropdown(
                        id='ml-target-feature',
                        options=[{'label': f, 'value': f} for f in all_features],
                        multi=False,
                        placeholder='Select target feature',
                        style={'width': '100%'}
                    ),
                ], width=6),
            ], className='my-4'),
            dbc.Row([
                dbc.Col([
                    html.H5("Train/Test Split:"),
                    dcc.Slider(
                        id='train-test-slider',
                        min=10,
                        max=90,
                        step=5,
                        value=70,
                        marks={i: f'{i}%' for i in range(10, 91, 10)},
                    ),
                    html.Div(id='train-size-output'),
                    html.Div(id='test-size-output'),
                ], width=12),
            ], className='my-4'),
            dbc.Row([
                dbc.Col([
                    dbc.Button('Next', id='ml-next-button', n_clicks=0, color='primary'),
                ], width=12),
            ], className='my-2'),
            html.Div(id='model-selection', style={'display': 'none'}, children=[
                html.H4("Select Model", className='text-center', style={'marginTop': '20px'}),
                dbc.Row([
                    dbc.Col([
                        html.H5("Regression Models:"),
                        dbc.ButtonGroup(
                            [
                                dbc.Button("Linear Regression", id={'type': 'model-button', 'index': 'Linear Regression'}),
                                dbc.Button("Decision Tree Regressor", id={'type': 'model-button', 'index': 'Decision Tree Regressor'}),
                                dbc.Button("Random Forest Regressor", id={'type': 'model-button', 'index': 'Random Forest Regressor'}),
                                dbc.Button("Support Vector Regressor", id={'type': 'model-button', 'index': 'Support Vector Regressor'}),
                                dbc.Button("KNN Regressor", id={'type': 'model-button', 'index': 'KNN Regressor'}),
                                dbc.Button("AdaBoost Regressor", id={'type': 'model-button', 'index': 'AdaBoost Regressor'}),
                                dbc.Button("Gradient Boosting Regressor", id={'type': 'model-button', 'index': 'Gradient Boosting Regressor'}),
                            ],
                            vertical=True,
                        ),
                    ], width=6),
                    dbc.Col([
                        html.H5("Classification Models:"),
                        dbc.ButtonGroup(
                            [
                                dbc.Button("Logistic Regression", id={'type': 'model-button', 'index': 'Logistic Regression'}),
                                dbc.Button("Decision Tree Classifier", id={'type': 'model-button', 'index': 'Decision Tree Classifier'}),
                                dbc.Button("Random Forest Classifier", id={'type': 'model-button', 'index': 'Random Forest Classifier'}),
                                dbc.Button("Support Vector Classifier", id={'type': 'model-button', 'index': 'Support Vector Classifier'}),
                                dbc.Button("KNN Classifier", id={'type': 'model-button', 'index': 'KNN Classifier'}),
                                dbc.Button("AdaBoost Classifier", id={'type': 'model-button', 'index': 'AdaBoost Classifier'}),
                                dbc.Button("Gradient Boosting Classifier", id={'type': 'model-button', 'index': 'Gradient Boosting Classifier'}),
                            ],
                            vertical=True,
                        ),
                    ], width=6),
                ]),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Set Hyperparameters"),
                        dbc.ModalBody(id='hyperparameter-modal-body'),
                        dbc.ModalFooter([
                            dbc.Button("Train", id="train-button", className="mr-auto", color='success', style={'fontSize': '20px', 'width': '150px'}),
                            dbc.Button("Close", id="close-hyperparameter-modal", className="ml-auto"),
                        ]),
                    ],
                    id='hyperparameter-modal',
                    size='lg',
                ),
            ]),
            # Training Summary Modal
            dbc.Modal(
                [
                    dbc.ModalHeader("Training Summary"),
                    dbc.ModalBody(id='training-summary-body'),
                    dbc.ModalFooter([
                        dbc.Button("Close", id="close-training-summary", className="ml-auto"),
                    ]),
                ],
                id='training-summary-modal',
                size='lg',
            ),
        ])
        return layout
