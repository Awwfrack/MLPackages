from pathlib import Path
import dash_bootstrap_components as dbc
from dash import (
    html,
    dcc,
    Input,
    Output,
    callback,
    ctx,
    register_page,
    no_update,
)
from dash.exceptions import PreventUpdate
import os
from dash_iconify import DashIconify

app_path = os.path.dirname(__file__)
import sys

sys.path.insert(0, app_path)
from src.load_table import get_table_paths, postprocess_tables, concatenate_tables
from src.pdf_data import DocumentLoader, postprocess_func
from src.commons import glob_re
import json
import pandas as pd
from src.commons import extract_image, Box
import jsonlines
from glob import glob
import sys
import subprocess

import yaml

conf_path = os.path.join(os.path.dirname((os.path.dirname(__file__))), "conf.yaml")
with open(conf_path, "r") as file:
    config = yaml.safe_load(file)


def extract_text_with_annotations(document_path, table_paths):
    doc = DocumentLoader(document=document_path)

    dpi = 200

    # Get table bboxes
    page_configs = []
    if len(table_paths) > 0:
        for table_path in table_paths:
            with open(table_path) as f:
                table_json = json.load(f)

            if isinstance(table_json["table_id"], list):
                table_path = [
                    os.path.join(Path(table_path).parent, f"{tid}.json")
                    for tid in table_json["table_id"]
                ]
            else:
                table_path = [table_path]

            for tp in table_path:
                with open(tp) as f:
                    t = json.load(f)
                    print(f"opened {tp}")

                bbox = t["bounding_box"]
                page_num = t["page_no"]

                image, _ = extract_image(document_path, page_num, dpi=dpi)
                bbox = Box(*bbox).apply_scale(image.size).to_array()

                page_configs.append({"page_num": page_num, "bboxes": bbox})

        page_df = pd.DataFrame(page_configs)
        # groupby
        page_df = page_df.groupby("page_num")["bboxes"].apply(list).reset_index()
        # turn into a json
        bbox_configs = page_df.to_dict("records")
        # turn into the expected format by the text extraction function
        for i in bbox_configs:
            i["bboxes"] = [{"text": "", "bbox": j} for j in i["bboxes"]]

        print("applying annotation")
        # apply annotations
        doc.apply_annotation(
            bboxes_configs=bbox_configs, verbose=False, is_image_bbox=True, dpi=dpi
        )
        print("annotation applied")

    # Get extracted text at block level
    doc.get_text(
        page_nums=None,
        level="blocks",
        # token_length=30,
        # token_overlap_size=0,
        metadata=True,
        verbose=False,
    )

    # Apply postprocessing function
    doc_data = doc.save(postprocess_func)

    return doc_data


def get_company_options(gcp_path):
    proc = subprocess.Popen(
        ["gcloud", "storage", "ls", f"{gcp_path}"],
        stdout=subprocess.PIPE,
        shell=False,
    )
    (out, error) = proc.communicate()
    companies_paths = out.decode().split("/\n")
    companies_paths = [i for i in companies_paths if i != ""]
    companies = [Path(i).stem for i in companies_paths]
    return companies


register_page(__name__, path="/")

# -------------------------- Search Companies ----------------------------
GCS_BUCKET = config["gcs_bucket"]
LOCAL_STORE = config["local_store"]
COMPANIES = get_company_options(GCS_BUCKET)

# ------------------------------ Loaders --------------------------------
loader = dcc.Loading(
    id="home-loader",
    children=[html.Div([html.Div(id="home-loader-text")])],
    type="circle",
    color="#d9230f",
)

report_loader = dcc.Loading(
    id="home-report-loader",
    children=[html.Div([html.Div(id="home-report-loader-text")])],
    type="circle",
    color="#d9230f",
)

company_loader = dcc.Loading(
    id="home-company-loader",
    children=[html.Div([html.Div(id="home-company-loader-text")])],
    type="circle",
    color="#d9230f",
)

loaders = [loader, report_loader, company_loader]


# ------------------------------ Layout --------------------------------
def create_layout():
    COMPANIES = get_company_options(GCS_BUCKET)
    refresh_button_section = html.Div(
        [
            dbc.Button(
                DashIconify(icon="pepicons-pop:refresh", color="gray", width=23),
                id=f"home-company-input-refresh-button",
                color="light",
                n_clicks=0,
                size="sm",
            ),
            dbc.Tooltip(
                "Click to retrieve the last companies",
                target=f"home-company-input-refresh-button",
                placement="right",
            ),
        ],
        className="g-0 ms-2 me-2 flex-nowrap align-self-center",
    )

    company_input = html.Div(
        [
            dbc.Label("Company"),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            options=COMPANIES,
                            id="home-company-input",
                            placeholder="Select a company for tagging",
                            persistence=False,
                            # persistence_type="session",
                            # className="mb-2",
                        ),
                        width="11",
                    ),
                    dbc.Col(refresh_button_section, width="1"),
                ],
                className="g-0",
            ),
        ]
    )

    report_input = html.Div(
        [
            dbc.Label("Report"),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id="home-report-input",
                            disabled=True,
                            multi=False,
                            persistence=False,
                            # persistence_type="session",
                            placeholder="Select a report for tagging",
                        ),
                        width="11",
                    )
                ],
                className="g-0",
            ),
        ]
    )

    return html.Div(
        [
            dcc.Location(id="home-after-begin"),
            dbc.Alert(
                "This report does not have a table_overview.json in the table jsons directory. Please check and fix it before continuing tagging.",
                id="home-table-overview-alert",
                is_open=False,
                fade=False,
                color="danger",
                dismissable=True,
            ),
            dbc.Alert(
                "Please note this report does not have a table jsons folder.",
                id="home-table-jsons-alert",
                is_open=False,
                fade=False,
                color="warning",
                dismissable=True,
            ),
            dbc.Container(
                [
                    dbc.Row([html.Img(src=r"assets/banner.jpeg", alt="image", width="auto")]),
                    html.Br(),
                    company_input,
                    html.Br(),
                    report_input,
                    html.Br(),
                    html.Br(),
                    html.Div(
                        dbc.Button(
                            "Begin",
                            color="primary",
                            outline=False,
                            disabled=True,
                            id="home-label-start",
                        ),
                        className="d-grid gap-2 col-2 mx-auto",
                    ),
                ],
                fluid=True,
                className="py-3",
            ),
        ]
        + loaders,
        className="h-100 p-5 bg-light border rounded-3 mt-4",
    )


layout = create_layout


# * ---------------------------------------------------------------------------------------
# *                                     Callbacks
# * ---------------------------------------------------------------------------------------
# ! Getting the companies dropdown
@callback(
    Output("home-company-input", "options"),
    Output("home-company-loader-text", "children"),
    Input("home-company-input-refresh-button", "n_clicks"),
)
def refresh_company_options(n_clicks):
    if n_clicks == 0:
        raise PreventUpdate

    if ctx.triggered_id == "home-company-input-refresh-button":
        companies = get_company_options(GCS_BUCKET)
        return companies, " "
    else:
        raise PreventUpdate


# ! Clear report value
@callback(
    Output("home-report-input", "value"),
    Input("home-company-input", "value"),
    prevent_initial_call=True,
)
def refresh_report_value(company):
    if ctx.triggered_id == "home-company-input":
        return ""
    else:
        raise PreventUpdate


@callback(
    Output("home-report-input", "options"),
    Output("home-report-input", "disabled"),
    Output("home-report-loader", "children"),
    Input("home-company-input", "value"),
    prevent_initial_call=True,
)
def get_report_options(company):
    if (company is None) | (company == ""):
        return no_update, True, " "

    proc = subprocess.Popen(
        ["gcloud", "storage", "ls", f"{GCS_BUCKET}/{company}/*.pdf"],
        stdout=subprocess.PIPE,
        shell=False,
    )
    (out, error) = proc.communicate()
    report_paths = out.decode().split("\n")
    report_paths = [i for i in report_paths if i != ""]
    reports = [Path(i).stem for i in report_paths]
    return reports, False, " "


@callback(
    Output("app-q4-nav", "disabled"),
    Output("app-q31-nav", "disabled"),
    Output("app-q32-nav", "disabled"),
    Output("app-home-nav", "disabled"),
    Output("home-label-start", "disabled"),
    Output("home-table-overview-alert", "is_open"),
    Output("home-table-jsons-alert", "is_open"),
    Output("data-store", "data", allow_duplicate=True),
    Output("home-loader-text", "children"),
    Input("home-report-input", "value"),
    Input("home-company-input", "value"),
    prevent_initial_call=True,
)
def postprocess(report, company):
    table_overview_alert = False
    table_jsons_alert = False
    disabled = True

    if (company is None) | (company == "") | (report is None) | (report == ""):
        return [True] * 5 + [False, False, [], " "]

    # Create temp local folder
    if not os.path.exists(LOCAL_STORE):
        os.makedirs(LOCAL_STORE)

    # Copy from GCS bucket
    company_folder = os.path.join(LOCAL_STORE, company)
    if not os.path.exists(company_folder):
        os.system(f"gsutil -m cp -r '{GCS_BUCKET}/{company}' '{LOCAL_STORE}'")

    run = (
        True if os.path.exists(os.path.join(company_folder, f"{report}.pdf")) else False
    )
    if not run:
        return [True] * 5 + [False, False, [], " "]

    # Create folder per question
    labels_path = os.path.join(LOCAL_STORE, company, f"{report}_labels")
    for q_id in ["q31", "q32", "q4"]:
        question_folder = os.path.join(labels_path, q_id, "")
        if not os.path.exists(question_folder):
            os.makedirs(question_folder)

    # Table processing
    table_folder = os.path.join(LOCAL_STORE, company, f"{report}_json")
    table_overview_path = os.path.join(table_folder, "table_overview.json")

    if os.path.exists(table_folder):
        raw_table_paths = glob_re(table_folder, r".+_table\d+.*.json")
        num_tables = len(raw_table_paths)
        if (not os.path.exists(table_overview_path)) & (num_tables > 0):
            table_overview_alert = True
            disabled = [True] * 5
            data_store = []
            return disabled + [table_overview_alert, table_jsons_alert, data_store, " "]
        else:
            postprocess_tables(table_folder)
            print("concatenating")
            concatenate_tables(table_folder)
            table_paths = get_table_paths(table_folder, table_category=["Emission", "Emissions", "emission", "emissions"])
    else:
        table_jsons_alert = True
        table_paths = []

    # Text processing
    report_path = os.path.join(LOCAL_STORE, company, f"{report}.pdf")
    report_extracted_text_path = os.path.join(
        LOCAL_STORE, company, f"{report}_labels", "extracted_text"
    )
    if not os.path.exists(report_extracted_text_path):
        os.makedirs(report_extracted_text_path)

    report_data_path = os.path.join(report_extracted_text_path, "report_data.jsonl")

    # only extract when it has not been extracted before
    if not os.path.exists(report_data_path):
        report_data = extract_text_with_annotations(report_path, table_paths)
        with jsonlines.open(report_data_path, mode="w") as writer:
            writer.write_all(report_data)

    data_store = {
        "company": company,
        "report": report,
        "local_store": LOCAL_STORE,
        "table_paths": table_paths,
        "report_data_path": report_data_path,
        "labels_path": labels_path,
    }

    disabled = [False] * 5
    return disabled + [table_overview_alert, table_jsons_alert, data_store, " "]


@callback(
    Output("home-after-begin", "href"),
    Output("home-after-begin", "refresh"),
    Input("home-label-start", "n_clicks"),
)
def begin_tagging(n_clicks):
    if n_clicks == 0:
        raise PreventUpdate

    if ctx.triggered_id == "home-label-start":
        return "/questions_home", True
    else:
        raise PreventUpdate
