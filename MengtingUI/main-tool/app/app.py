import dash_bootstrap_components as dbc
from dash import Dash, html, page_container, dcc, Input, Output, callback, ctx, State
from dash.exceptions import PreventUpdate
import os
from dash_iconify import DashIconify
import pandas as pd
from pathlib import Path
import dash_auth
import yaml
project_dir = os.path.dirname(__file__)
import sys
sys.path.insert(0, project_dir)
from src import cache
import flask

conf_path = os.path.join(os.path.dirname(__file__), "conf.yaml")
with open(conf_path, "r") as file:
    config = yaml.safe_load(file)

# CMD Run
# gunicorn app:server -b 10.154.0.93:8090 --timeout 600 --workers 8 --preload

GCS_BUCKET = config["gcs_bucket"]
VALID_USERNAME_PASSWORD_PAIRS = config["valid_user_name_password_pairs"]

# * Initiate the app
server = flask.Flask(__name__) # define flask app.server

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SIMPLEX],
    use_pages=True,
    suppress_callback_exceptions=True,
    server=server
)

server = app.server

cache.init_app(app.server)

# * Authenticate
auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)

# * ----------------------------- Navigation Bar ------------------------
# * Units
units_button_section = html.Div(
    [
        dbc.Button(
            DashIconify(icon="arcticons:units", color="white", width=30),
            id=f"app-units-button",
            color="secondary",
            outline=True,
            n_clicks=0,
            size="sm",
        ),
        dbc.Tooltip(
            "Click to view the units explainations",
            target=f"app-units-button",
            placement="right",
        ),
    ],
    className="g-0 ms-2 me-2 flex-nowrap align-self-center",
)


navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(
            dbc.NavLink(
                "Home", href="/questions_home", disabled=False, id="app-home-nav"
            )
        ),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Q3-1", href="/q31_home"),
                dbc.DropdownMenuItem("Tables", href="/q31_table_form"),
                dbc.DropdownMenuItem("Texts", href="/q31_text_form"),
                dbc.DropdownMenuItem("Manual Texts", href="/q31_manual_text_form"),
            ],
            nav=True,
            in_navbar=True,
            label="Q3-1",
            id="app-q31-nav",
            disabled=False,
        ),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Q3-2", href="/q32_home"),
                dbc.DropdownMenuItem("Tables", href="/q32_table_form"),
                dbc.DropdownMenuItem("Texts", href="/q32_text_form"),
                dbc.DropdownMenuItem("Manual Texts", href="/q32_manual_text_form"),
            ],
            nav=True,
            in_navbar=True,
            label="Q3-2",
            id="app-q32-nav",
            disabled=False,
        ),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Q4", href="/q4_home"),
                dbc.DropdownMenuItem("Tables", href="/q4_table_form"),
                dbc.DropdownMenuItem("Texts", href="/q4_text_form"),
                dbc.DropdownMenuItem("Manual Texts", href="/q4_manual_text_form"),
            ],
            nav=True,
            in_navbar=False,
            label="Q4",
            id="app-q4-nav",
        ),
        units_button_section,
        dbc.Button(
            "Restart", color="light", outline=True, id="app-restart", class_name="ms-2"
        ),
        dbc.Button("Submit", color="light", id="app-submit", class_name="ms-2"),
    ],
    brand=dbc.Row(
        [
            dbc.Col(
                html.Img(
                    src=r"/assets/HSBC_MASTERBRAND_LOGO_RGB_NEG.png",
                    height="35px",
                    alt="image",
                )
            ),
            dbc.Col(dbc.NavbarBrand("Q&A Labelling", className="ms-2")),
        ],
        align="center",
        className="g-0",
    ),
    # brand_href="/",
    color="dark",
    dark=True,
    id="app-navbar",
)

# * --------------------------- Units Modal --------------------------------------
parent_dir = Path(__file__).parent
units_df = pd.read_csv(os.path.join(parent_dir, "src", "units.csv"))
units_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Standardized Units Overview")),
        dbc.ModalBody(
            [
                dbc.Table.from_dataframe(
                    units_df, striped=True, bordered=True, hover=True
                )
            ]
        ),
    ],
    id="app-units-modal",
    centered=True,
    is_open=False,
    size="lg",
)

# * ----------------------------- Submission Confirmation ------------------------
submit_confirmation_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Declaration")),
        dbc.ModalBody(
            [
                dbc.Alert(
                    "No labels are tagged for this report. Please click 'Cancel' and start tagging.",
                    color="info",
                    id="app-submit-no-lables-alert",
                    dismissable=True,
                    is_open=False,
                ),
                """By submitting the samples, I confirm that I have recorded all the samples in the tables and text snippets for Q3-1, Q3-2, Q4.""",
            ]
        ),
        dbc.ModalFooter(
            dbc.Button(
                "Confirm", color="primary", id="app-final-submit", className="ms-auto"
            )
        ),
    ],
    id="app-submit-confirmation",
    centered=True,
    is_open=False,
)

# * ----------------------------- Restart Confirmation ------------------------
restart_confirmation_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Restart Warning")),
        dbc.ModalBody(
            [
                """By clicking "Confirm," you will be directed to the main page where you can select another company and report for tagging. Your current recorded samples have not been submitted yet, but you will have the option to retrieve your progress at a later stage. Are you sure you want to proceed?"""
            ]
        ),
        dbc.ModalFooter(
            dbc.Button(
                "Confirm", color="primary", id="app-final-restart", className="ms-auto"
            )
        ),
    ],
    id="app-restart-confirmation",
    centered=True,
    is_open=False,
)

# * ------------------------------- Loader ---------------------------------
loader = dcc.Loading(
    id="app-submit-loader",
    children=[html.Div([html.Div(id="app-submit-loader-text")])],
    type="circle",
)

# * ------------------------------- Layout -----------------------------------
app.layout = html.Div(
    [
        navbar,
        dbc.Container(page_container),
        submit_confirmation_modal,
        restart_confirmation_modal,
        units_modal,
        loader,
        dcc.Location(id="app-after-submit-url"),
        dcc.Location(id="app-after-restart-url"),
        dcc.Store(
            id="data-store", data=[], storage_type="session"
        ),  # 'local' or 'session'
    ]
)


# * ---------------------------------------------------------------------------------------
# *                                     Callbacks
# * ---------------------------------------------------------------------------------------
# ! Open units modal
@callback(
    Output("app-units-modal", "is_open", allow_duplicate=True),
    Input("app-units-button", "n_clicks"),
    prevent_initial_call=True,
)
def open_submit_confirmation_modal(units):
    if ctx.triggered_id == "app-units-button":
        return True
    return False


# ! Open submission confirmation modal
@callback(
    Output("app-submit-confirmation", "is_open", allow_duplicate=True),
    Input("app-submit", "n_clicks"),
    prevent_initial_call=True,
)
def open_submit_confirmation_modal(submit):
    if ctx.triggered_id == "app-submit":
        return True
    return False


# ! Inside the modal - Submit vs Cancel
@callback(
    Output("app-submit-confirmation", "is_open", allow_duplicate=True),
    Output("app-submit-loader-text", "children"),
    Output("app-submit-no-lables-alert", "is_open"),
    Output("app-after-submit-url", "href"),
    Output("app-after-submit-url", "refresh"),
    Output("data-store", "data"),
    Input("app-final-submit", "n_clicks"),
    State("data-store", "data"),
    prevent_initial_call=True,
)
def confirm_submission(submit, data_store):
    if submit == 0:
        raise PreventUpdate

    if ctx.triggered_id == "app-final-submit":
        if (data_store is None) | (data_store == []):
            return True, " ", True, None, False, data_store
        labels_path = data_store["labels_path"]
        company_path = os.path.join(data_store["local_store"], data_store["company"])
        gcs_company_folder = os.path.join(GCS_BUCKET, data_store["company"], "")

        if os.path.exists(labels_path):
            # copy new table jsons to the labels folder
            table_jsons_path = os.path.join(company_path, f"{data_store['report']}_json")
            dest_table_jsons_path = os.path.join(labels_path, f"{data_store['report']}_json")
            if os.path.exists(dest_table_jsons_path):
                os.system(f"rm -rf '{dest_table_jsons_path}'")
            if os.path.exists(table_jsons_path):
                os.system(f"cp -r '{table_jsons_path}' '{dest_table_jsons_path}'")
            
            # copy the labels folder to gcs
            os.system(f"gsutil -m cp -r '{labels_path}' '{gcs_company_folder}'")
            
            os.system(f"rm -rf '{company_path}'")
            
            return False, " ", False, "/", True, []
        else:
            return True, " ", True, None, False, data_store

    return True, " ", False, None, False, data_store


# ! Open restart confirmation modal
@callback(
    Output("app-restart-confirmation", "is_open", allow_duplicate=True),
    Input("app-restart", "n_clicks"),
    prevent_initial_call=True,
)
def open_restart_confirmation_modal(restart):
    if restart == 0:
        raise PreventUpdate

    if ctx.triggered_id == "app-restart":
        return True
    return False


# ! Confirms restarting
@callback(
    Output("app-restart-confirmation", "is_open", allow_duplicate=True),
    Output("app-after-restart-url", "href"),
    Output("app-after-restart-url", "refresh"),
    Input("app-final-restart", "n_clicks"),
    State("data-store", "data"),
    prevent_initial_call=True,
)
def confirm_restart(restart, data_store):
    if restart == 0:
        raise PreventUpdate

    if ctx.triggered_id == "app-final-restart":
        return False, "/", True
    return True, None, False


# Run app
if __name__ == "__main__":
    app.run_server(
        debug=False,
        host="10.154.0.95",
        port=8050,
        # use_reloader=False,
        # dev_tools_hot_reload=False
    )
