import dash_bootstrap_components as dbc
from .commons import create_form
import dash
from dash_iconify import DashIconify
import dash_ag_grid as dag
import dash_mantine_components as dmc
import pandas as pd
from typing import List
from dash import html, Input, Output, callback, ctx, State, dash_table, ctx, no_update, dcc
from dash.exceptions import PreventUpdate
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import json
import jsonlines
import time
from dash_canvas import DashCanvas, utils
import math
import os

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
from src.load_table import load_table_image
from src.commons import create_form
from src import cache
import json
import pandas as pd
from typing import List
from functools import partial
from pathlib import Path
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import cv2
from PIL import Image
import re
import time

def create_input(id_prefix, **kwargs) -> dbc.InputGroup:
    """
    This function creates the input. It creates
        - User input field
        - Evidence field
            - Evidence input field (read only)
            - Add evidence button
            - Clear evidence button
    """
    name = kwargs.pop("name")
    id = kwargs.pop("id")
    comp_type = kwargs.pop("comp_type")
    comp_kwargs = kwargs.pop("comp_kwargs", {})
    try:
        disabled = comp_kwargs["disabled"]
    except:
        disabled = False
    tooltip = kwargs.pop("tooltip", "")
    required = kwargs.pop("required", True)

    if comp_type == "dropdown":
        user_input = dbc.Select(
            **comp_kwargs,
            required=required,
            placeholder="",
            persistence=True,
            persistence_type="session",
            id=f"{id_prefix}-{id}-input",
            style={"width": "110px"},
        )
    elif comp_type == "input":
        user_input = dbc.Input(
            **comp_kwargs,
            required=required,
            autoFocus=False,
            persistence=True,
            persistence_type="session",
            id=f"{id_prefix}-{id}-input",
            style={"width": "110px"},
        )
    elif comp_type == "text-area":
        user_input = dbc.Textarea(
            required=required,
            autoFocus=False,
            persistence=True,
            persistence_type="session",
            id=f"{id_prefix}-{id}-input",
            style={"width": "110px"},
        )

    input_group = dbc.InputGroup(
        [
            dbc.InputGroupText(
                name, id=f"{id_prefix}-{id}-name", style={"width": "210px"}
            ),
            user_input,
            dbc.Tooltip(tooltip, target=f"{id_prefix}-{id}-name", placement="top"),
            dbc.Input(
                type="text",
                # placeholder = "",
                required=False if disabled else True,
                disabled = disabled,
                autoFocus=False,
                name=f"{id_prefix}-{id}-evidence",
                readonly=True,
                persistence=True,
                persistence_type="session",
                id=f"{id_prefix}-{id}-evidence",
                style={"width": "110px"},
            ),
            dbc.Button(
                "+",
                color="dark",
                disabled=disabled,
                id=f"{id_prefix}-{id}-evidence-add",
            ),
            dbc.Button(
                "-",
                color="dark",
                outline=True,
                disabled=disabled,
                id=f"{id_prefix}-{id}-evidence-clear",
            ),
        ],
        className="mb-3",
    )

    return input_group


def create_layout(
    id_prefix: str,
    # ---------- Form -------------
    inputs: List[dict],
    n_col: int = 1,
    main_evidence_id: str = "value",
):
    # * ---------------------------------------- Form ----------------------------------------------
    user_inputs = []
    for i in inputs:
        user_input = create_input(
            id_prefix=id_prefix, **i
        )  # value, evidence, add and clear set
        user_inputs.append(user_input)

    form = create_form(user_inputs, n_col=n_col, id_prefix=id_prefix)

    # * ---------------------------------------- Table --------------------------------------------
    tables_pagination = html.Div(
        [
            dbc.Progress(id=f"{id_prefix}-table-form-progress", striped=True),
            html.Br(),
            dbc.Pagination(
                id=f"{id_prefix}-table-input",
                previous_next=True,
                # active_page = 1,
                first_last=True,
                fully_expanded=False,
                max_value=1,
                className="justify-content-center",
            ),
        ],
        className="row",
        style={"width": "100%"},
    )

    # * table image
    table_collapse = html.Div(
        [
            dbc.Button(
                DashIconify(icon="feather:info", color="#373a3c", width=30),
                id=f"{id_prefix}-table-form-table-collapse-button",
                # className="d-flex align-items-center",
                color="light",
                n_clicks=0,
                outline=True,
                size="sm",
            ),
            dbc.Tooltip(
                "Click to view the table details",
                target=f"{id_prefix}-table-form-table-collapse-button",
                placement="right",
            ),
            dbc.Collapse(id=f"{id_prefix}-table-form-table-collapse", is_open=True),
        ]
    )

    table = html.Div(
        [
            html.Div(children="", id=f"{id_prefix}-table-header"),
            html.Div(
                [
                    dbc.Button(
                        "Evidence in Header",
                        color="primary",
                        outline=True,
                        id=f"{id_prefix}-header-evidence",
                    )
                ],
                className="d-grid gap-2",
            ),
            dag.AgGrid(
                id=f"{id_prefix}-table",
                defaultColDef={"resizable": True, "sortable": False, "filter": False},
                dashGridOptions={
                    "columnHoverHighlight": True,
                    "enableCellTextSelection": True,
                    # 'pinnedTopRowData': data[:nhead]
                },
                columnSize="responsiveSizeToFit",
                style={"height": 610, "width": "100%"},
            ),
            html.Div(
                [
                    dbc.Button(
                        "Evidence in Footer",
                        color="primary",
                        outline=True,
                        id=f"{id_prefix}-footer-evidence",
                    )
                ],
                className="d-grid gap-2",
            ),
            html.Div(children="", id=f"{id_prefix}-table-footer"),
        ]
    )

    # * ----------------------------------------- Recorded Samples ----------------------------------------
    samples_df = pd.DataFrame(
        {
            **{i["name"]: [] for i in inputs},
            **{f"{i['name']} Evidence": [] for i in inputs},
            **{"Page Number Evidence": [], "Sample ID": []},
        }
    )

    samples_table = dbc.Row(
        [
            dbc.Col(
                [
                    html.H5("Recorded Samples", className="text-center"),
                    dash_table.DataTable(
                        id=f"{id_prefix}-samples-table-container",
                        data=[],
                        columns=[
                            {"name": i_3, "id": i_3} for i_3 in samples_df.columns
                        ],
                        style_table={"overflow": "scroll", "height": 400},
                        style_cell={"textAlign": "left"},
                        row_deletable=True,
                        row_selectable="multi",
                        editable=False,
                        persistence=False,
                        sort_action="native",
                        filter_action="native",
                        hidden_columns=[f"{i['name']} Evidence" for i in inputs]
                        + ["Page Number Evidence", "Sample ID"]
                        # persistence_type="session"
                    ),
                    html.Div(
                        className="text-center", id=f"{id_prefix}-table-form-last-save"
                    ),
                ],
                width={"size": 12, "offset": 0, "order": 1},
            )
        ],
        className="mt-3",
    )

    # * ----------------------------------------- Modal ----------------------------------------
    modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Sample Details")),
            dbc.ModalBody(id=f"{id_prefix}-table-form-modal-body"),
            dbc.ModalFooter(
                dbc.Button(
                    "Edit",
                    id=f"{id_prefix}-table-form-modal-edit",
                    className="ms-auto",
                    n_clicks=0,
                )
            ),
        ],
        size="xl",
        is_open=False,
        fullscreen=True,
        scrollable=True,
        id=f"{id_prefix}-table-form-modal",
    )

    edit_alert_modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Editting Disabled")),
            dbc.ModalBody(id=f"{id_prefix}-table-form-alert-modal-body"),
        ],
        is_open=False,
        id=f"{id_prefix}-table-form-alert-modal",
    )

    deleted_sample_alert_modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Program Exists for Sample")),
            dbc.ModalBody(id=f"{id_prefix}-table-form-deleted-alert-modal-body"),
        ],
        is_open=False,
        id=f"{id_prefix}-table-form-deleted-alert-modal",
    )

    # * ----------------------------------------- Layout ----------------------------------------
    evidence_alerts = [
        dbc.Alert(
            f"A value is provided for {i['id']} but no evidence is provided. Please pick an evidence.",
            color="danger",
            id=f"{id_prefix}-{i['id']}-evidence-alert",
            dismissable=True,
            is_open=False,
        )
        for i in inputs
    ]

    layout = evidence_alerts + [
        dbc.Alert(
            "There is no table to label. Please move on to the next section.",
            color="info",
            id=f"{id_prefix}-table-form-notable-alert",
            dismissable=True,
            is_open=False,
        ),
        dbc.Alert(
            "Value should only have 1 evidence. Please generate 1 sample per Value evidence.",
            color="danger",
            id=f"{id_prefix}-{main_evidence_id}-alert",
            dismissable=True,
            is_open=False,
        ),
        dbc.Alert(
            "This cell has been recorded. Please record another cell if it is not intentional.",
            color="danger",
            id=f"{id_prefix}-{main_evidence_id}-duplicate-alert",
            dismissable=True,
            is_open=False,
        ),
        html.Br(),
        html.H1(f"{id_prefix.upper()} Table Evidence"),
        html.Br(),
        tables_pagination,
        dbc.Spinner(
            color="primary",
            fullscreen=True,
            id=f"{id_prefix}-table-form-spinner",
            size="lg",
        ),
        dbc.Row([dbc.Col(table_collapse, align="end")], align="end"),
        dbc.Row(
            [
                dbc.Col([table], width=7),
                dbc.Col(
                    [html.H5("Add a New Sample", className="text-center"), form],
                    width=5,
                ),
            ]
        ),
        # table_image_modal,
        html.Br(),
        html.Br(),
        dmc.Divider(variant="dotted"),
        html.Br(),
        edit_alert_modal,
        deleted_sample_alert_modal
    ]

    return layout, samples_table, modal


# * ---------------------------------------------------------------------------------------
# *                                     Callbacks
# * ---------------------------------------------------------------------------------------
# @cache.memoize(3600)
def rebuild_table(table_path):
    time.sleep(0.1)
    # load labels to get header and footer
    with open(table_path) as f:
        labels = json.load(f)

    header = f"Table Header: {labels['table_header']['text']}"
    footer = f"Table Footer: {labels['table_footer']['text']}"
    df = pd.DataFrame(labels["df"])
    header_inds = labels["header_inds"]

    data = df.to_dict("records")
    rowData = data
    columnDefs = [{"field": str(i)} for i in df.columns]

    # highlight table headers
    if len(header_inds)>0:
        getRowStyle = {
            "styleConditions": [
                {
                    "condition": f"params.rowIndex === {i}",
                    "style": {"backgroundColor": "#373a3c", "color": "white"},
                }
                for i in header_inds
            ],
            "defaultStyle": {"backgroundColor": "white", "color": "black"},
        }
    else:
        getRowStyle = {}

    return header, footer, rowData, columnDefs, getRowStyle


def create_callbacks(
    id_prefix: str,
    inputs: List[dict],
    main_evidence_cell: str = "Value",
    main_evidence_id: str = "value",
):
    # * ------------------------------------- Tables ---------------------------------
    # ! Retrieve the table options
    @callback(
        Output(f"{id_prefix}-table-input", "max_value"),
        Output(f"{id_prefix}-table-input", "previous_next"),
        Output(f"{id_prefix}-table-input", "first_last"),
        Output(f"{id_prefix}-table-form-notable-alert", "is_open"),
        Output(f"{id_prefix}-table-form-spinner", "children"),
        Input("data-store", "data"),
    )
    def get_tables_options(data_store):
        if (data_store is not None) & (data_store is not []) & (len(data_store) > 0):
            table_paths = data_store["table_paths"]
            if len(table_paths) == 0:
                return 1, False, False, True, " "
            else:
                return len(table_paths), True, True, False, " "
        else:
            raise PreventUpdate

    # ! Display the progress
    @callback(
        Output(f"{id_prefix}-table-form-progress", "value"),
        Input(f"{id_prefix}-table-input", "active_page"),
        State("data-store", "data"),
    )
    def display_progress(table_id, data_store):
        if (table_id is not None) & (table_id != ""):
            if len(data_store["table_paths"])>0:
                table_paths = data_store["table_paths"]
                total = len(table_paths)
                pct = (table_id / total) * 100
                return pct
        else:
            return 0

    # ! Display the selected table
    @callback(
        Output(f"{id_prefix}-table-header", "children"),
        Output(f"{id_prefix}-table-footer", "children"),
        Output(f"{id_prefix}-table", "rowData"),
        Output(f"{id_prefix}-table", "columnDefs"),
        Output(f"{id_prefix}-table", "getRowStyle"),
        State("data-store", "data"),
        Input(f"{id_prefix}-table-input", "active_page"),
        Input(f"{id_prefix}-form-submit", "n_clicks"),
        Input(f"{id_prefix}-samples-table-container", "data"),
        prevent_initial_call=True,
    )
    def get_table(data_store, table_id, submit, sample_data):
        print("GET TABLE CALLED")
        if (
            (table_id == "")
            | (table_id is None)
            | (data_store is None)
            | (len(data_store["table_paths"])==0)
            | (len(data_store) == 0)
        ):
            df = pd.DataFrame()
            header = ""
            footer = ""
            columnDefs = []
            df = pd.DataFrame()
            rowData = df.to_dict("records")
            getRowStyle = {}
        else:
            table_paths = data_store["table_paths"]
            table_path = table_paths[table_id - 1]
            header, footer, rowData, columnDefs, getRowStyle = rebuild_table(table_path)

            # read already recorded samples
            labels_path = data_store["labels_path"]
            question_folder = os.path.join(labels_path, id_prefix, "")
            table_name = Path(data_store["table_paths"][table_id - 1]).stem
            sample_df_path = os.path.join(question_folder, f"{table_name}_labels.jsonl")

            if os.path.exists(sample_df_path):
                with jsonlines.open(sample_df_path, "r") as jsonl_f:
                    value_evidence = []
                    for sample in jsonl_f:
                        if isinstance(sample[f"{main_evidence_cell} Evidence"], str):
                            sample[f"{main_evidence_cell} Evidence"] = eval(
                                sample[f"{main_evidence_cell} Evidence"]
                            )
                        value_evidence.append(sample[f"{main_evidence_cell} Evidence"])
                ve = []
                for j in value_evidence:
                    if j is not None:
                        for i in j:
                            ve.append(i)
                # grey out those cells
                for coldef in columnDefs:
                    coldef["cellStyle"] = {}
                    cellStyle = []
                    for elist in ve:
                        e = elist[1]
                        if (e[0] == "header") | (e[0] == "footer"):
                            pass
                        else:
                            row_ind, col_ind = e[0], e[1]
                            if str(col_ind) == coldef["field"]:
                                cell_style = {
                                    "condition": f"params.rowIndex==={row_ind}",
                                    "style": {
                                        "backgroundColor": "#daeace",
                                        "color": "black",
                                    },
                                }
                                cellStyle.append(cell_style)
                    coldef["cellStyle"]["styleConditions"] = cellStyle

        return header, footer, rowData, columnDefs, getRowStyle

    # * -------------------------------------- Form ------------------------------------------
    # ! Select evidence
    def get_evidence(
        cell,
        evidence,
        evidence_add,
        evidence_add_timestamp,
        evidence_clear,
        evidence_clear_timestamp,
        header_evidence,
        header_evidence_timestamp,
        footer_evidence,
        footer_evidence_timestamp,
        id,
    ):
        if (
            (cell is None)
            & (header_evidence_timestamp is None)
            & (footer_evidence_timestamp is None)
        ):
            raise PreventUpdate

        button_clicked = ctx.triggered_id

        if button_clicked == f"{id_prefix}-{id}-evidence-clear":
            evidence = ""

        if button_clicked == f"{id_prefix}-{id}-evidence-add":
            if (cell is None) & (header_evidence is None) & (footer_evidence is None):
                raise PreventUpdate

            if cell is None:
                cell_timestamp = 0
            else:
                cell_timestamp = cell["timestamp"]

            if header_evidence_timestamp is None:
                header_evidence_timestamp = 0

            if footer_evidence_timestamp is None:
                footer_evidence_timestamp = 0

            ts = {
                "cell": cell_timestamp,
                "header": header_evidence_timestamp,
                "footer": footer_evidence_timestamp,
            }
            most_recent = max(ts, key=ts.get)

            if most_recent == "cell":
                new_evidence = f"{cell['rowIndex']}-{cell['colId']}"
            elif most_recent == "header":
                new_evidence = f"header"
            elif most_recent == "footer":
                new_evidence = f"footer"

            if evidence is not None:
                if evidence != "":
                    all_evidence = evidence.split(", ")
                else:
                    all_evidence = []

                all_evidence.append(new_evidence)
                all_evidence = list(set(all_evidence))

                return ", ".join(all_evidence)
            else:
                return new_evidence
        else:
            return evidence

    for input_dict in inputs:
        input_id = input_dict["id"]

        callback(
            Output(f"{id_prefix}-{input_id}-evidence", "value", allow_duplicate=True),
            Input(f"{id_prefix}-table", "cellClicked"),
            Input(f"{id_prefix}-{input_id}-evidence", "value"),
            Input(f"{id_prefix}-{input_id}-evidence-add", "n_clicks"),
            Input(f"{id_prefix}-{input_id}-evidence-add", "n_clicks_timestamp"),
            Input(f"{id_prefix}-{input_id}-evidence-clear", "n_clicks"),
            Input(f"{id_prefix}-{input_id}-evidence-clear", "n_clicks_timestamp"),
            Input(f"{id_prefix}-header-evidence", "n_clicks"),
            Input(f"{id_prefix}-header-evidence", "n_clicks_timestamp"),
            Input(f"{id_prefix}-footer-evidence", "n_clicks"),
            Input(f"{id_prefix}-footer-evidence", "n_clicks_timestamp"),
            prevent_initial_call=True,
        )(partial(get_evidence, id=input_id))

    # ! Clear form
    @callback(
        [
            Output(f"{id_prefix}-{i['id']}-input", "value", allow_duplicate=True)
            for i in inputs
        ]
        + [
            Output(f"{id_prefix}-{i['id']}-evidence", "value", allow_duplicate=True)
            for i in inputs
        ],
        Input(f"{id_prefix}-form-clear", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_form(n_clicks):
        if ctx.triggered_id == f"{id_prefix}-form-clear":
            return [None] * len(inputs) * 2
        else:
            raise PreventUpdate

    # * ------------------------------------ Table Image --------------------------------------
    @cache.memoize(3600)
    def get_dashcanvas(table_path):
        table_image, page_num = load_table_image(table_path, dpi=200)
        table_name = Path(table_path).stem
        table_name = re.findall(r"table\d+.*", table_name)[0]

        table_array = cv2.cvtColor(table_image, cv2.COLOR_BGR2RGB)
        table_array = utils.array_to_data_url(table_array)
        table_image = DashCanvas(
            image_content=table_array,
            tool = "select",
            hide_buttons=["pan", "line", "pencil", "rectangle", "undo", "select"],
            goButtonTitle=" ",
            width=900,
        )
        
        return table_name, table_image, page_num
    
    # ! Display image
    @callback(
        Output(f"{id_prefix}-table-form-table-collapse", "children"),
        Output(f"{id_prefix}-table-form-table-collapse", "title"),
        Output(f"{id_prefix}-table-form-table-collapse", "is_open"),
        Input(f"{id_prefix}-table-form-table-collapse-button", "n_clicks"),
        State("data-store", "data"),
        Input(f"{id_prefix}-table-input", "active_page"),
        State(f"{id_prefix}-table-form-table-collapse", "is_open")
    )
    def display_collapse(n_clicks, data_store, table_id, is_open):
        if (n_clicks>0)&(ctx.triggered_id==f"{id_prefix}-table-form-table-collapse-button"):
            is_open = not is_open
        
        if (ctx.triggered_id==f"{id_prefix}-table-form-table-collapse-button") | (ctx.triggered_id==f"{id_prefix}-table-input"):
            if isinstance(data_store, dict):
                if (table_id is not None) & (table_id != ""):
                    if len(data_store["table_paths"])>0:
                        # * Load the sample data
                        table_paths = data_store["table_paths"]
                        table_path = table_paths[table_id - 1]

                        with open(table_path, "r") as f:
                            table_json = json.load(f)

                        table_id = table_json["table_id"]
                        if isinstance(table_id, list):
                            table_paths = [
                                os.path.join(Path(table_path).parent, f"{tid}.json")
                                for tid in table_id
                            ]
                        else:
                            table_paths = [table_path]

                        rows = []
                        num_tables = len(table_paths)
                        for table_path in table_paths:
                            table_name, table_image, page_num = get_dashcanvas(table_path)

                            rows.append(
                                dbc.Row(
                                    [
                                        dbc.Row(html.H5(table_name)),
                                        dbc.Row(html.Div(f"Page Number: {page_num}")),
                                        dbc.Row(
                                            table_image,
                                            className=f"h-{math.floor(100/num_tables)}",
                                        ),
                                    ]
                                )
                            )

                        body = html.Div(
                            dbc.Container(rows, fluid=True)
                            )
                        
                        return body, table_name, is_open
        
        return no_update, no_update, is_open

    # * ---------------------------------  Recorded Samples -----------------------------------
    # ! Display upon table selection
    @callback(
        Output(f"{id_prefix}-samples-table-container", "data", allow_duplicate=True),
        Output(f"{id_prefix}-table-form-last-save", "children", allow_duplicate=True),
        Input(f"{id_prefix}-table-input", "active_page"),
        State("data-store", "data"),
        State(f"{id_prefix}-samples-table-container", "columns"),
        prevent_initial_call=True,
    )
    def display_samples_df(table_id, data_store, columns):
        if ctx.triggered_id == f"{id_prefix}-table-input":
            if (table_id is not None) & (table_id!=""):
                if len(data_store["table_paths"])>0:
                    question_folder = os.path.join(data_store["labels_path"], id_prefix, "")
                    table_name = Path(data_store["table_paths"][table_id - 1]).stem
                    sample_df_path = os.path.join(question_folder, f"{table_name}_labels.jsonl")

                    if os.path.exists(sample_df_path):
                        with jsonlines.open(sample_df_path, "r") as jsonl_f:
                            saved_samples = []
                            for obj in jsonl_f:
                                values = {
                                    k: v
                                    for k, v in obj.items()
                                    if k
                                    in [
                                        c["id"]
                                        for c in columns
                                        if (" Evidence" not in c["id"])
                                        & ("Sample ID" not in c["id"])
                                        & ("Page Number ID" not in c["id"])
                                    ]
                                }
                                evidence = {
                                    k: str(v)
                                    for k, v in obj.items()
                                    if k in [c["id"] for c in columns if " Evidence" in c["id"]]
                                }
                                sample_id = {k: v for k, v in obj.items() if k == "Sample ID"}
                                sample = {**values, **evidence, **sample_id}
                                saved_samples.append(sample)
                        last_save = time.ctime(os.path.getmtime(sample_df_path))
                    else:
                        saved_samples = []
                        last_save = ""
                    return saved_samples, last_save
        raise PreventUpdate

    # ! Display upon sample deletion
    @callback(
        Output(f"{id_prefix}-samples-table-container", "data", allow_duplicate=True),
        Output(f"{id_prefix}-table-form-last-save", "children", allow_duplicate=True),
        Output(f"{id_prefix}-table-form-deleted-alert-modal", "is_open"),
        Output(f"{id_prefix}-table-form-deleted-alert-modal-body", "children"),
        State(f"{id_prefix}-table-input", "active_page"),
        State(f"{id_prefix}-samples-table-container", "data_previous"),
        Input(f"{id_prefix}-samples-table-container", "data_timestamp"),
        State(f"{id_prefix}-samples-table-container", "data"),
        State("data-store", "data"),
        prevent_initial_call=True,
    )
    def delete_sample(table_id, data_previous, edit_time, data_current, data_store):
        open_alert_modal = False
        modal_body = []

        if data_previous is not None:
            question_folder = os.path.join(data_store["labels_path"], id_prefix, "")
            table_name = Path(data_store["table_paths"][table_id - 1]).stem
            sample_df_path = os.path.join(question_folder, f"{table_name}_labels.jsonl")
            disk_last_save = os.path.getmtime(sample_df_path)

            if edit_time > disk_last_save:

                if id_prefix == "q31":
                    current_sample_ids = [i["Sample ID"] for i in data_current]
                    previous_sample_ids = [i["Sample ID"] for i in data_previous]

                    missing_id = [i for i in previous_sample_ids if i not in current_sample_ids][0]
                    # CHECK IF PROGRAM WITH SAMPLE EXISTS:
                    programs_df = os.path.join(
                        question_folder, f"{table_name}_programs_labels.jsonl"
                    )
                    programs_to_show = []
                    if os.path.exists(programs_df):
                        with jsonlines.open(programs_df, "r") as jsonl_f:
                            for sample in [i for i in jsonl_f]:
                                print(sample)
                                if missing_id in sample["Sample IDs"]:
                                    sample_data = {"Program": sample["Program"], "Answer": sample["Answer"]}
                                    programs_to_show.append(sample_data)

                    if len(programs_to_show)>0:
                        open_alert_modal = True
                        modal_body = html.Div(
                            [
                                dbc.Alert("CAUTION: A program has been found using this sample. Please delete the program.", color="danger",),
                                dbc.Row(
                                    [
                                        html.H5("Programs Found", className="text-center"),
                                        dash_table.DataTable(
                                            data=programs_to_show,
                                            columns=[{"name": i_3, "id": i_3} for i_3 in ["Program", "Answer"]],
                                            style_table={"overflow": "scroll", "height": 400},
                                            style_cell={"textAlign": "left"},
                                            row_deletable=False,
                                            editable=False,
                                            persistence=False,
                                        ),
                                    ]
                                )

                            ]
                        )

                
                # Now save new samples
                samples = []
                for sample in data_current:
                    values = {
                        k: v
                        for k, v in sample.items()
                        if (" Evidence" not in k)
                        & ("Sample ID" not in k)
                        & ("Page Number Evidence" not in k)
                    }
                    evidence = {
                        k: eval(v) for k, v in sample.items() if " Evidence" in k
                    }
                    sample_id = {k: v for k, v in sample.items() if k == "Sample ID"}
                    sample = {**values, **evidence, **sample_id}
                    samples.append(sample)
                with jsonlines.open(sample_df_path, mode="w") as writer:
                    writer.write_all(samples)
                print("deleted:", samples)
            last_save = time.ctime(os.path.getmtime(sample_df_path))
            return data_current, last_save, open_alert_modal, modal_body
        else:
            raise PreventUpdate

    # * ---------------------------------- Sample Modal ---------------------------------------
    # ! Display recorded sample details
    @callback(
        Output(f"{id_prefix}-table-form-modal-body", "children", allow_duplicate=True),
        Output(f"{id_prefix}-table-form-modal", "is_open", allow_duplicate=True),
        Input(f"{id_prefix}-samples-table-container", "active_cell"),
        State(f"{id_prefix}-samples-table-container", "derived_viewport_data"),
        State("data-store", "data"),
        State(f"{id_prefix}-table-input", "active_page"),
        prevent_initial_call=True
    )
    def display_modal(active_cell, table_data, data_store, table_id):
        if active_cell:
            row_id = active_cell["row"]
            row = table_data[row_id]

            # * Load the sample data
            question_folder = os.path.join(data_store["labels_path"], id_prefix, "")
            table_paths = data_store["table_paths"]

            if table_paths is []:
                raise PreventUpdate

            table_path = table_paths[table_id - 1]
            table_name = Path(data_store["table_paths"][table_id - 1]).stem
            sample_df_path = os.path.join(question_folder, f"{table_name}_labels.jsonl")
            # # if file exists, then we retrieve the sample
            # with jsonlines.open(sample_df_path, "r") as jsonl_f:
            #     rows = [obj for obj in jsonl_f]
            # row = rows[row_id]

            header, footer, rowData, columnDefs, getRowStyle = rebuild_table(table_path)
            for coldef in columnDefs:
                coldef["cellStyle"] = {}
                cellStyle = []
                for inp in inputs:
                    column_name = inp["name"]
                    color = inp["color"]
                    evidence = row[f"{column_name} Evidence"]
                    if isinstance(evidence, str):
                        evidence = eval(evidence)
                    if evidence is None:
                        pass
                    else:
                        for elist in evidence:
                            e = elist[1]
                            if not e == [""]:
                                if e[0] == "header":
                                    header_style = {"background-color": color}
                                elif e[0] == "footer":
                                    footer_style = {"background-color": color}
                                else:
                                    row_ind, col_ind = e[0], e[1]
                                    if str(col_ind) == coldef["field"]:
                                        cell_style = {
                                            "condition": f"params.rowIndex==={row_ind}",
                                            "style": {
                                                "backgroundColor": color,
                                                "color": "black",
                                            },
                                        }
                                        cellStyle.append(cell_style)

                coldef["cellStyle"]["styleConditions"] = cellStyle
            try:
                header_style
            except:
                header_style = {}
            try:
                footer_style
            except:
                footer_style = {}

            q31_table_form_modal_input_groups = []
            for i in inputs:
                name = i["name"]
                display_evi = row[f"{name} Evidence"]
                if isinstance(display_evi, str):
                    display_evi = eval(display_evi)
                if display_evi is not None:
                    display_evi = ", ".join(
                        ["-".join([str(j) for j in ev[1]]) for ev in display_evi]
                    )

                q31_table_form_modal_input_group = dbc.InputGroup(
                    [
                        dbc.InputGroupText(
                            i["name"],
                            style={"background-color": i["color"], "width": "40%"},
                        ),
                        dbc.Input(
                            value=row[name],
                            readonly=True,
                            persistence=True,
                            persistence_type="memory",
                            style={"width": "35%"},
                        ),
                        dbc.Input(
                            value=display_evi,
                            readonly=True,
                            persistence=True,
                            persistence_type="memory",
                            style={"width": "25%"},
                        ),
                    ],
                    className="mb-3",
                )
                q31_table_form_modal_input_groups.append(
                    dbc.Row([q31_table_form_modal_input_group], className="g-3")
                )
            q31_table_from_model_form = dbc.Form(q31_table_form_modal_input_groups)

            modal_body = html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                children=header, style=header_style
                                            ),
                                            dag.AgGrid(
                                                rowData=rowData,
                                                columnDefs=columnDefs,
                                                getRowStyle=getRowStyle,
                                                defaultColDef={
                                                    "resizable": True,
                                                    "sortable": False,
                                                    "filter": False,
                                                },
                                                dashGridOptions={
                                                    "columnHoverHighlight": True,
                                                    "enableCellTextSelection": True,
                                                },
                                                columnSize="responsiveSizeToFit",
                                                style={"height": 610, "width": "100%"},
                                            ),
                                            html.Div(
                                                children=footer, style=footer_style
                                            ),
                                        ]
                                    )
                                ]
                            ),
                            dbc.Col([q31_table_from_model_form], width=6),
                        ]
                    )
                ]
            )

            return modal_body, True
        else:
            return [], False


    # * ---------------------------------- Edit sample ---------------------------------------
    if id_prefix == "q31":
        @callback(
            [
                Output(f"{id_prefix}-{i['id']}-input", "value", allow_duplicate=True)
                for i in inputs
            ] +[
                Output(f"{id_prefix}-{i['id']}-evidence", "value", allow_duplicate=True)
                for i in inputs
            ] +
            [
                Output(f"{id_prefix}-table-form-modal", "is_open"),
                Output(f"{id_prefix}-table-form-alert-modal", "is_open")
            ] + 
            [
                Output(f"{id_prefix}-samples-table-container", "data", allow_duplicate=True),
                Output(f"{id_prefix}-table-form-last-save", "children", allow_duplicate=True)
            ] + [
                Output(f"{id_prefix}-table-form-alert-modal-body", "children", allow_duplicate=True)
            ],
            Input(f"{id_prefix}-table-form-modal-edit", "n_clicks"),
            State(f"{id_prefix}-samples-table-container", "active_cell"),
            State("data-store", "data"),
            State(f"{id_prefix}-table-input", "active_page"),
            State(f"{id_prefix}-samples-table-container", "derived_viewport_data"),
            prevent_initial_call=True
        )
        def edit_sample(n_clicks, active_cell, data_store, table_id, viewport_data):
            
            if ctx.triggered_id == f"{id_prefix}-table-form-modal-edit":
                print("EDITTING")
                question_folder = os.path.join(data_store["labels_path"], id_prefix, "")
                table_name = Path(data_store["table_paths"][table_id - 1]).stem
                sample_df_path = os.path.join(question_folder, f"{table_name}_labels.jsonl")

                row_id = active_cell["row"]

                # * Load the sample data
                row = viewport_data[row_id]
                
                # CHECK IF PROGRAM WITH SAMPLE EXISTS:
                programs_df = os.path.join(
                    question_folder, f"{table_name}_programs_labels.jsonl"
                )
                programs_to_show = []
                if os.path.exists(programs_df):
                    with jsonlines.open(programs_df, "r") as jsonl_f:
                        for sample in [i for i in jsonl_f]:
                            print(sample)
                            if row["Sample ID"] in sample["Sample IDs"]:
                                sample_data = {"Program": sample["Program"], "Answer": sample["Answer"]}
                                programs_to_show.append(sample_data)

                if len(programs_to_show)>0:
                    open_alert_modal = True
                    modal_body = html.Div(
                        [
                            dbc.Alert("CAUTION: A program has been found using this sample. Please delete the program before editing this sample.", color="danger",),
                            dbc.Row(
                                [
                                    html.H5("Programs Found", className="text-center"),
                                    dash_table.DataTable(
                                        data=programs_to_show,
                                        columns=[{"name": i_3, "id": i_3} for i_3 in ["Program", "Answer"]],
                                        style_table={"overflow": "scroll", "height": 400},
                                        style_cell={"textAlign": "left"},
                                        row_deletable=False,
                                        editable=False,
                                        persistence=False,
                                    ),
                                ]
                            )

                        ]
                    )
                    form_values = [None]*len(inputs)
                    form_evidences = [None]*len(inputs)

                    last_save = os.path.getmtime(sample_df_path)

                
                else:
                    row = viewport_data.pop(row_id)
                    open_alert_modal = False
                    modal_body = []

                    form_values = []
                    form_evidences = []
                    for i in inputs:
                        name = i["name"]
                        row_value = row[name]
                        row_evi = row[f"{name} Evidence"]
                        if isinstance(row_evi, str):
                            row_evi = eval(row_evi)
                        if row_evi is not None:
                            row_evi = ", ".join(
                                ["-".join([str(j) for j in ev[1]]) for ev in row_evi]
                            )

                        form_values.append(row_value)
                        form_evidences.append(row_evi)

                    # Delete sample
                    samples = []
                    for sample in viewport_data:
                        values = {
                            k: v
                            for k, v in sample.items()
                            if (" Evidence" not in k)
                            & ("Sample ID" not in k)
                            & ("Page Number Evidence" not in k)
                        }
                        evidence = {
                            k: eval(v) for k, v in sample.items() if " Evidence" in k
                        }
                        sample_id = {k: v for k, v in sample.items() if k == "Sample ID"}
                        sample = {**values, **evidence, **sample_id}
                        samples.append(sample)

                    with jsonlines.open(sample_df_path, mode="w") as writer:
                        writer.write_all(samples)
                    last_save = time.ctime(os.path.getmtime(sample_df_path))
                    

                return form_values + form_evidences + [False, open_alert_modal] + [viewport_data, last_save] + [modal_body]
            
            else:
                PreventUpdate

    else:
        @callback(
            [
                Output(f"{id_prefix}-{i['id']}-input", "value", allow_duplicate=True)
                for i in inputs
            ] +[
                Output(f"{id_prefix}-{i['id']}-evidence", "value", allow_duplicate=True)
                for i in inputs
            ] +
            [Output(f"{id_prefix}-table-form-modal", "is_open")] + 
            [
                Output(f"{id_prefix}-samples-table-container", "data", allow_duplicate=True),
                Output(f"{id_prefix}-table-form-last-save", "children", allow_duplicate=True)
            ],
            Input(f"{id_prefix}-table-form-modal-edit", "n_clicks"),
            State(f"{id_prefix}-samples-table-container", "active_cell"),
            State("data-store", "data"),
            State(f"{id_prefix}-table-input", "active_page"),
            State(f"{id_prefix}-samples-table-container", "derived_viewport_data"),
            prevent_initial_call=True
        )
        def edit_sample(n_clicks, active_cell, data_store, table_id, viewport_data):
            if ctx.triggered_id == f"{id_prefix}-table-form-modal-edit":
                row_id = active_cell["row"]

                # * Load the sample data
                row = viewport_data.pop(row_id)

                form_values = []
                form_evidences = []
                for i in inputs:
                    name = i["name"]
                    row_value = row[name]
                    row_evi = row[f"{name} Evidence"]
                    if isinstance(row_evi, str):
                        row_evi = eval(row_evi)
                    if row_evi is not None:
                        row_evi = ", ".join(
                            ["-".join([str(j) for j in ev[1]]) for ev in row_evi]
                        )

                    form_values.append(row_value)
                    form_evidences.append(row_evi)

                # Delete sample
                question_folder = os.path.join(data_store["labels_path"], id_prefix, "")
                table_name = Path(data_store["table_paths"][table_id - 1]).stem
                sample_df_path = os.path.join(question_folder, f"{table_name}_labels.jsonl")
                samples = []
                for sample in viewport_data:
                    values = {
                        k: v
                        for k, v in sample.items()
                        if (" Evidence" not in k)
                        & ("Sample ID" not in k)
                        & ("Page Number Evidence" not in k)
                    }
                    evidence = {
                        k: eval(v) for k, v in sample.items() if " Evidence" in k
                    }
                    sample_id = {k: v for k, v in sample.items() if k == "Sample ID"}
                    sample = {**values, **evidence, **sample_id}
                    samples.append(sample)

                with jsonlines.open(sample_df_path, mode="w") as writer:
                    writer.write_all(samples)
                last_save = time.ctime(os.path.getmtime(sample_df_path))

                return form_values + form_evidences + [False] + [viewport_data, last_save]
            
            else:
                PreventUpdate




# ! Check if evidence is provided
def check_evidence(value, evidence):
    if (value != "not specified") & (value != "") & (value is not None):
        if (evidence is None) | (evidence == ""):
            return False
        else:
            return True
    else:
        if (evidence is not None) & (evidence != ""):
            return False
        else:
            return True
