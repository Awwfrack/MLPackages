import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import os

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
import sys

sys.path.insert(0, project_dir)


dash.register_page(__name__, path="/q31_home")

q31_tables_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Tables", className="card-title"),
            html.P("Record table samples" ""),
            dbc.Button("Start", color="primary", href="/q31_table_form", class_name="mt-auto",),
        ],
        class_name="d-flex flex-column",
    )
)

q31_texts_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Texts", className="card-title"),
            html.P("Record text samples"),
            dbc.Button("Start", color="primary", href="/q31_text_form", class_name="mt-auto",),
        ],
        class_name="d-flex flex-column",
    )
)

q31_manual_texts_card = dbc.Card(
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
            dbc.Button("Start", color="primary", href="/q31_manual_text_form", class_name="mt-auto",),
        ],
        class_name="d-flex flex-column",
    )
)

q31_home_cards = dbc.CardGroup([q31_tables_card, q31_texts_card, q31_manual_texts_card])

layout = dbc.Container(
    [
        html.Br(),
        html.H1("Q31 Home", id="q31-home-title"),
        dbc.Tooltip(
            "Record emissions samples",
            target="q31-home-title",
            placement="bottom-start",
        ),
        html.Br(),
        q31_home_cards,
    ]
)
