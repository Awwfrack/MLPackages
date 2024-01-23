import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import os

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
import sys

sys.path.insert(0, project_dir)

dash.register_page(__name__, path="/q4_home")

q4_tables_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Tables", className="card-title"),
            html.P("Record table samples" ""),
            dbc.Button("Start", color="primary", href="/q4_table_form", class_name="mt-auto",),
        ],
        class_name="d-flex flex-column",
    ),
)

q4_texts_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Texts", className="card-title"),
            html.P("Record text samples"),
            dbc.Button("Start", color="primary", href="/q4_text_form", class_name="mt-auto",),
        ],
        class_name="d-flex flex-column",
    ),
)

q4_manual_texts_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Manual Texts", className="card-title"),
            dcc.Markdown(
                """
                Record text samples manually. 
                
                **ONLY** do this if text extraction does not work properly.
                Please visit ***Texts*** page first.
                """
            ),
            dbc.Button("Start", color="primary", href="/q4_manual_text_form", class_name="mt-auto",),
        ],
        class_name="d-flex flex-column",
    ),
)

q4_home_cards = dbc.CardGroup([q4_tables_card, q4_texts_card, q4_manual_texts_card])

layout = html.Div(
    [
        html.Br(),
        html.H1("Q4 Home", id="q4-home-title"),
        dbc.Tooltip(
            "Record net zero targets",
            target="q4-home-title",
            placement="bottom-start",
        ),
        html.Br(),
        q4_home_cards,
    ]
)
