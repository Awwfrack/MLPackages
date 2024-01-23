from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import html, register_page, callback, Input, Output

register_page(__name__, path="/questions_home")

# ----------------------------- Create Cards -----------------------------
q31_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Q3-1", className="card-title"),
            html.P("Record emissions samples"),
            dbc.Button(
                "Start",
                color="primary",
                href="/q31_home",
                id="home-q31-start-button",
                # disabled=True,
                class_name="mt-auto",
            ),
        ],
        class_name="d-flex flex-column",
    ),
    id="home-q31-card",
)

q32_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Q3-2", className="card-title"),
            html.P("Record what has been excluded from the emission calculations"),
            dbc.Button(
                "Start",
                color="primary",
                href="/q32_home",
                id="home-q32-start-button",
                # disabled=True,
                class_name="mt-auto",
            ),
        ],
        class_name="d-flex flex-column",
    ),
    id="home-q32-card",
)

q4_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Q4", className="card-title"),
            html.P("Record net zero targets"),
            dbc.Button(
                "Start",
                color="primary",
                href="/q4_home",
                id="home-q4-start-button",
                # disabled=True,
                class_name="mt-auto",
            ),
        ],
        class_name="d-flex flex-column",
    ),
    id="home-q4-card",
)

cards = dbc.CardGroup([q31_card, q32_card, q4_card])

# ------------------------------ Layout --------------------------------
layout = html.Div(
    [
        html.Br(),
        html.H2(id="questions-home-company"),
        html.Br(),
        html.H4(id="questions-home-report"),
        html.Br(),
        cards,
    ]
)


# ------------------------------ Callbacks --------------------------------
@callback(
    Output("questions-home-company", "children"),
    Output("questions-home-report", "children"),
    Input("data-store", "data"),
)
def display_company_report(data_store):
    if (data_store is None) | (data_store == []):
        raise PreventUpdate

    return data_store["company"], data_store["report"]
