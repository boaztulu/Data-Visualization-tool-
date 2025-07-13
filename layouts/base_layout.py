# layouts/base_layout.py
from dash import html, dcc
import dash_bootstrap_components as dbc

def create_base_layout():
    navbar = dbc.NavbarSimple(
        brand="Boaz Tulu Data Visualization Project",
        brand_href="#",
        color="primary",
        dark=True,
        fixed="top",
    )

    upload_component = dbc.Container([
        dbc.Row([
            dbc.Col([
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select CSV File', style={'color': 'blue', 'text-decoration': 'underline'})
                    ]),
                    style={
                        'width': '100%',
                        'height': '100px',
                        'lineHeight': '100px',
                        'borderWidth': '2px',
                        'borderStyle': 'dashed',
                        'borderRadius': '10px',
                        'textAlign': 'center',
                        'backgroundColor': 'rgba(255,255,255,0.9)'
                    },
                    multiple=False
                ),
                html.Div(id='upload-status', style={'marginTop': '10px'})
            ], width=12)
        ], justify='center', style={'marginTop': '80px'})
    ])

    tabs = dbc.Tabs(
        [
            dbc.Tab(label="Preprocessing", tab_id="tab-preprocess"),
            dbc.Tab(label="Visualization", tab_id="tab-visualize"),
            dbc.Tab(label="ML", tab_id="tab-ml"),
        ],
        id="tabs",
        active_tab="tab-preprocess",
        className="mt-3"
    )

    content = html.Div(id="page-content", children=[])

    layout = html.Div([
        navbar,
        upload_component,
        tabs,
        content,
        dcc.Store(id='uploaded-data'),
        dcc.Store(id='preprocessed-data'),
        dcc.Store(id='selected-model-name'),
    ])
    return layout
