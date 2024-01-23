import os
from uuid import uuid4
from src.commons import get_page_num
from src.text_form_utils import create_layout, create_callbacks
import dash
from dash import (
    html,
    Input,
    Output,
    callback,
    ctx,
    State,
    dash_table,
    ctx,
    ALL,
)
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash.exceptions import PreventUpdate
import pandas as pd
import json
import jsonlines
import time
from dash_iconify import DashIconify
from src.inputs import q31_inputs as inputs


def create_program_layout(id_prefix, form_type):
    # * -------------------- Program ----------------
    program_button = html.Div(
        [
            dbc.Button(
                "Add a Program",
                color="primary",
                size="lg",
                id=f"{id_prefix}-{form_type}-form-record-program-button",
            )
        ],
        className="d-grid gap-2 mb-2",
    )

    program_modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Add a New Program")),
            dbc.ModalBody(id=f"{id_prefix}-{form_type}-form-program-modal-body"),
        ],
        size="xl",
        is_open=False,
        id=f"{id_prefix}-{form_type}-form-program-modal",
    )

    programs_df = pd.DataFrame({"Program": [], "Answer": [], "Sample IDs": []})

    programs_table = dbc.Row(
        [
            dbc.Col(
                [
                    html.H5("Recorded Programs", className="text-center"),
                    dash_table.DataTable(
                        id=f"{id_prefix}-programs-{form_type}-container",
                        data=[],
                        columns=[{"name": i_3, "id": i_3} for i_3 in programs_df.columns],
                        style_table={"overflow": "scroll", "height": 400},
                        style_cell={"textAlign": "left"},
                        row_deletable=True,
                        editable=False,
                        persistence=False,
                        sort_action="native",
                        filter_action="native",
                        hidden_columns=["Sample IDs"]
                    ),
                    html.Div(
                        className="text-center",
                        id=f"{id_prefix}-{form_type}-form-program-last-save",
                    ),
                ],
                width={"size": 12, "offset": 0, "order": 1},
            )
        ],
        className="mt-3",
    )
    
    # * -------------------- Program Details ----------------
    program_details_modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Program Details")),
            dbc.ModalBody(id=f"{id_prefix}-{form_type}-form-program-details-modal-body"),
        ],
        size="xl",
        scrollable=True,
        is_open=False,
        id=f"{id_prefix}-{form_type}-form-program-details-modal",
    )

    return program_button, program_modal, programs_table, program_details_modal


def create_program_callbacks(id_prefix, form_type, inputs):
    # -------------------- Program Modal --------------------
    # ! Alert when program add is clicked but no rows are selected
    @callback(
        Output(f"{id_prefix}-{form_type}-form-record-program-button", "disabled"),
        Input(f"{id_prefix}-samples-{form_type}-container", "selected_rows"),
        State(f"{id_prefix}-samples-{form_type}-container", "data"),
    )
    def disable_program_start_button(selected_rows, data):
        # do not start if less than 2 rows are selected
        if selected_rows is None:
            return True
        if len(selected_rows) <= 1:
            return True
        else:
            rows = [data[i] for i in selected_rows]
            
            # do not start when less than 2 absolute rows are selected
            absolute_rows = [i for i in rows if i["Absolute or Intensity"] == "absolute"]
            if len(absolute_rows) <= 1:
                return True
            
            # do not start when greater than 1 intensity rows are selected
            if len(absolute_rows) < len(rows):
                return True

            rows_std_units = [i["Standardized Unit"] for i in rows]
            if not "%" in rows_std_units:
                # do not start when less than 2 category rows are selected
                category_rows = [
                    i
                    for i in rows
                    if (i["Category Name"] != "") & (i["Category Name"] is not None)
                ]
                if len(category_rows) <= 1:
                    return True
                if len(category_rows) < len(rows):
                    return True

                # do not start when the other values are different
                cols = [
                    i["name"]
                    for i in inputs
                    if i["name"] not in [
                        "Value", "Category Name", "Raw Unit", "Additional Information",
                        "Percentage Type", "Base Year", "Base Scope"
                        ]
                ]
                for col in cols:
                    vals = [i[col] for i in category_rows]

                    if len(list(set(vals))) > 1:
                        return True
            else:
                # do not start when the other values are different
                cols = [
                    i["name"]
                    for i in inputs
                    if i["name"] not in [
                        "Value", "Category Name", "Raw Unit", "Standardized Unit", "Additional Information", 
                        "Scope", "Year", "Percentage Type", "Base Year", "Base Scope"
                        ]
                ]
                for col in cols:
                    vals = [i[col] for i in rows]

                    if len(list(set(vals))) > 1:
                        return True
                    
                return False

        return False
    
    arithmetic_ops = ["+", "-", "*", "/", "**", "(", ")"]
    arithmeti_buttons = dmc.Group(
        [html.Div(["Arithemtic Operations: "])]
        + [
            dbc.Button(
                op.title(),
                outline=True,
                id={
                    "type": f"{id_prefix}-{form_type}-form-arithmetic-operations-button",
                    "index": i,
                },
                color="primary",
            )
            for i, op in enumerate(arithmetic_ops)
        ]
    )


    @callback(
        Output(f"{id_prefix}-{form_type}-form-program-input", "value", allow_duplicate=True),
        Output("data-store", "data", allow_duplicate=True),
        Input(
            {"type": f"{id_prefix}-{form_type}-form-arithmetic-values-button", "index": ALL},
            "n_clicks",
        ),
        Input(
            {"type": f"{id_prefix}-{form_type}-form-arithmetic-operations-button", "index": ALL},
            "n_clicks",
        ),
        Input(
            {"type": f"{id_prefix}-{form_type}-form-arithmetic-ints-button", "index": ALL},
            "n_clicks",
        ),
        State(
            {"type": f"{id_prefix}-{form_type}-form-arithmetic-values-button", "index": ALL},
            "children",
        ),
        State(
            {"type": f"{id_prefix}-{form_type}-form-arithmetic-operations-button", "index": ALL},
            "children",
        ),
        State(
            {"type": f"{id_prefix}-{form_type}-form-arithmetic-ints-button", "index": ALL},
            "children",
        ),
        State(f"{id_prefix}-{form_type}-form-program-input", "value"),
        State("data-store", "data"),
        prevent_initial_call=True,
    )
    def display_program(
        value_clicked, op_clicked, int_clicked, value, op, ints, program, data_store
    ):
        if ctx.triggered[0]["prop_id"] == ".":
            raise PreventUpdate
        try:
            button_index = eval(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
            button_id = eval(ctx.triggered[0]["prop_id"].split(".")[0])["type"]
            if button_id == f"{id_prefix}-{form_type}-form-arithmetic-operations-button":
                p = op[button_index]
            elif button_id == f"{id_prefix}-{form_type}-form-arithmetic-values-button":
                p = value[button_index]
            elif button_id == f"{id_prefix}-{form_type}-form-arithmetic-ints-button":
                p = ints[button_index]
            else:
                raise PreventUpdate
        except:
            raise PreventUpdate
        program = "" if program is None else program

        data_store["last_program"] = program
        return program + " " + p, data_store


    @callback(
        Output(f"{id_prefix}-{form_type}-form-program-input", "value", allow_duplicate=True),
        Output("data-store", "data", allow_duplicate=True),
        Input(f"{id_prefix}-{form_type}-form-program-summation", "n_clicks"),
        State(
            {"type": f"{id_prefix}-{form_type}-form-arithmetic-values-button", "index": ALL},
            "children",
        ),
        State("data-store", "data"),
        prevent_initial_call=True,
    )
    def program_auto_summation(summation, values, data_store):
        if ctx.triggered_id == f"{id_prefix}-{form_type}-form-program-summation":
            program = " + ".join(values)
            data_store["last_program"] = program
            return program, data_store
        else:
            raise PreventUpdate


    @callback(
        Output(f"{id_prefix}-{form_type}-form-program-input", "value", allow_duplicate=True),
        Output("data-store", "data", allow_duplicate=True),
        Input(f"{id_prefix}-{form_type}-form-program-average", "n_clicks"),
        State(
            {"type": f"{id_prefix}-{form_type}-form-arithmetic-values-button", "index": ALL},
            "children",
        ),
        State("data-store", "data"),
        prevent_initial_call=True,
    )
    def program_auto_average(average, values, data_store):
        if ctx.triggered_id == f"{id_prefix}-{form_type}-form-program-average":
            num = len(values)
            program = f"({' + '.join(values)}) / {num}"
            data_store["last_program"] = program
            return program, data_store
        else:
            raise PreventUpdate


    @callback(
        Output(f"{id_prefix}-{form_type}-form-program-input", "value", allow_duplicate=True),
        Output("data-store", "data", allow_duplicate=True),
        Input(f"{id_prefix}-{form_type}-form-program-undo", "n_clicks"),
        State(f"{id_prefix}-{form_type}-form-program-input", "value"),
        State("data-store", "data"),
        prevent_initial_call=True,
    )
    def program_undo(undo, program, data_store):
        if ctx.triggered_id == f"{id_prefix}-{form_type}-form-program-undo":
            program = "" if program is None else program
            program = " ".join(program.split(" ")[:-1])
            return program, data_store
        else:
            raise PreventUpdate


    @callback(
        Output(f"{id_prefix}-{form_type}-form-program-input", "value", allow_duplicate=True),
        Input(f"{id_prefix}-{form_type}-form-program-clear", "n_clicks"),
        State(f"{id_prefix}-{form_type}-form-program-input", "value"),
        State("data-store", "data"),
        prevent_initial_call=True,
    )
    def program_clear(clear, program, data_store):
        if ctx.triggered_id == f"{id_prefix}-{form_type}-form-program-clear":
            program = ""
            return program
        else:
            raise PreventUpdate


    if form_type == "text":
        # ! Display add program page
        @callback(
            Output(f"{id_prefix}-{form_type}-form-program-modal-body", "children"),
            Output(f"{id_prefix}-{form_type}-form-program-modal", "is_open", allow_duplicate=True),
            Input(f"{id_prefix}-{form_type}-form-record-program-button", "n_clicks"),
            State(f"{id_prefix}-samples-{form_type}-container", "selected_rows"),
            State(f"{id_prefix}-samples-{form_type}-container", "data"),
            State("data-store", "data"),
            prevent_initial_call=True,
        )
        def display_program_modal(n_clicks, selected_rows, data, data_store):
            if ctx.triggered_id == f"{id_prefix}-{form_type}-form-record-program-button":
                rows = [data[i] for i in selected_rows]

                # * ---------------------------- Retrieve Values as Buttons -----------------------
                values = []
                for i in rows:
                    if i["Standardized Unit"] is not None:
                        if i["Standardized Unit"]=="%":
                            value = i["Value"]/100
                        else:
                            value = i["Value"]
                        values.append(str(value))
                    else:
                        values.append("null")
                
                values_buttons = dmc.Group(
                    [html.Div(["Emission Values: "])]
                    + [
                        dbc.Button(
                            v.title(),
                            outline=True,
                            id={
                                "type": f"{id_prefix}-{form_type}-form-arithmetic-values-button",
                                "index": i,
                            },
                            color="primary",
                        )
                        for i, v in enumerate(values)
                    ]
                )

                ints = [str(i + 1) for i in range(len(values))]
                ints_buttons = dmc.Group(
                    [html.Div(["Integer Values: "])]
                    + [
                        dbc.Button(
                            v.title(),
                            outline=True,
                            id={
                                "type": f"{id_prefix}-{form_type}-form-arithmetic-ints-button",
                                "index": i,
                            },
                            color="primary",
                        )
                        for i, v in enumerate(ints)
                    ]
                )

                # Get evidence info
                evidence_data = []
                evidence_cols = ["Text Evidence", "Page Number Evidence"]
                for row in rows:
                    evidence_data.append({i: row[i] for i in evidence_cols})
                evidence_table = dash_table.DataTable(
                    id=f"{id_prefix}-{form_type}-form-program-selected-evidence-table",
                    data=evidence_data,
                    columns=[{"name": i_3, "id": i_3} for i_3 in evidence_cols],
                    style_cell={"textAlign": "left"},
                    persistence=False,
                    editable=False,
                    style_data={
                        "whiteSpace": "normal",
                        "height": "auto",
                    },
                )

                # * ------------------------------------ Modal body --------------------------------
                modal_body = html.Div(
                    [
                        html.H5("Selected Evidences", className="text-center"),
                        dbc.Row([evidence_table]),
                        dmc.Divider(variant="solid"),
                        dbc.Alert(
                            id=f"{id_prefix}-{form_type}-form-program-alert",
                            color="danger",
                            dismissable=True,
                            is_open=False,
                        ),
                        dbc.Alert(
                            "These rows have got a program recorded. Please record other rows instead.",
                            color="danger",
                            id=f"{id_prefix}-{form_type}-program-duplicate-alert",
                            dismissable=True,
                            is_open=False,
                        ),
                        html.Br(),
                        arithmeti_buttons,
                        html.Br(),
                        values_buttons,
                        html.Br(),
                        ints_buttons,
                        html.Br(),
                        dmc.Group(
                            [
                                dmc.Button(
                                    "",
                                    leftIcon=DashIconify(
                                        icon="ic:baseline-undo", width=20, color="#d9230f"
                                    ),
                                    variant="subtle",
                                    id=f"{id_prefix}-{form_type}-form-program-undo",
                                ),
                                dmc.Button(
                                    "",
                                    leftIcon=DashIconify(
                                        icon="mdi:clear", width=20, color="#d9230f"
                                    ),
                                    variant="subtle",
                                    id=f"{id_prefix}-{form_type}-form-program-clear",
                                ),
                                dmc.Button(
                                    "",
                                    leftIcon=DashIconify(
                                        icon="carbon:chart-average", width=20, color="#d9230f"
                                    ),
                                    variant="subtle",
                                    id=f"{id_prefix}-{form_type}-form-program-average",
                                ),
                                dmc.Button(
                                    "",
                                    leftIcon=DashIconify(
                                        icon="tabler:sum", width=20, color="#d9230f"
                                    ),
                                    variant="subtle",
                                    id=f"{id_prefix}-{form_type}-form-program-summation",
                                ),
                            ]
                        ),
                        dbc.Input(
                            className="mb-3",
                            readonly=True,
                            placeholder="Please write a program to calculate the total emission",
                            id=f"{id_prefix}-{form_type}-form-program-input",
                            size="lg",
                            persistence=False,
                        ),
                        html.Div(id=f"{id_prefix}-{form_type}-form-program-answer"),
                        html.Br(),
                        html.Div(
                            [
                                dbc.Col(
                                    dbc.Button(
                                        "Submit", id=f"{id_prefix}-{form_type}-form-program-submit"
                                    )
                                )
                            ],
                            className="d-grid gap-2 d-md-flex justify-content-md-end",
                        ),
                    ]
                )

                return modal_body, True
            else:
                return [], False


        # ! Submit program
        @callback(
            Output(f"{id_prefix}-{form_type}-form-program-answer", "children"),
            Output(f"{id_prefix}-{form_type}-form-program-alert", "children"),
            Output(f"{id_prefix}-{form_type}-form-program-alert", "is_open"),
            Output(f"{id_prefix}-programs-{form_type}-container", "data", allow_duplicate=True),
            Output(
                f"{id_prefix}-{form_type}-form-program-last-save", "children", allow_duplicate=True
            ),
            Output(f"{id_prefix}-{form_type}-form-program-modal", "is_open", allow_duplicate=True),
            Output(f"{id_prefix}-{form_type}-program-duplicate-alert", "is_open"),
            Input(f"{id_prefix}-{form_type}-form-program-submit", "n_clicks"),
            State(f"{id_prefix}-{form_type}-form-program-input", "value"),
            State(f"{id_prefix}-samples-{form_type}-container", "selected_rows"),
            State("data-store", "data"),
            State(f"{id_prefix}-samples-{form_type}-container", "data"),
            State(f"{id_prefix}-programs-{form_type}-container", "data"),
            State(f"{id_prefix}-programs-{form_type}-container", "columns"),
            prevent_initial_call=True,
        )
        def submit_program(
            submit, program, selected_rows, data_store, samples_data, data, columns
        ):
            modal_open = True
            duplicate_alert_open = False

            # Get program labels
            if "manual" in id_prefix:
                just_id = id_prefix.split("-")[0]
                question_folder = os.path.join(data_store["labels_path"], just_id, "")
                sample_df_path = os.path.join(question_folder, "manual_text_programs_labels.jsonl")
            else:
                question_folder = os.path.join(data_store["labels_path"], id_prefix, "")
                sample_df_path = os.path.join(question_folder, "text_programs_labels.jsonl")

            if os.path.exists(sample_df_path):
                last_save = time.ctime(os.path.getmtime(sample_df_path))
            else:
                last_save = " "

            if (program is not None) & (program != ""):
                if ctx.triggered_id == f"{id_prefix}-{form_type}-form-program-submit":
                    try:
                        answer = eval(program)
                        sample_ids = [samples_data[i]["Sample ID"] for i in selected_rows]

                        # * check if these sample ids have been recorded before
                        recorded_sample_ids = [eval(i["Sample IDs"]) for i in data]
                        for recorded_sample_id in recorded_sample_ids:
                            if sorted(recorded_sample_id) == sorted(sample_ids):
                                duplicate_alert_open = True
                                return (
                                    f"Answer: {answer}",
                                    "Duplicated Programs Detected",
                                    False,
                                    data,
                                    last_save,
                                    modal_open,
                                    duplicate_alert_open,
                                )

                        str_row = {
                            c["id"]: r
                            for c, r in zip(columns, [program, answer, str(sample_ids)])
                        }
                        data.append(str_row)

                        row = {
                            c["id"]: r for c, r in zip(columns, [program, answer, sample_ids])
                        }
                        with jsonlines.open(sample_df_path, mode="a") as writer:
                            writer.write(row)

                        last_save = time.ctime(os.path.getmtime(sample_df_path))
                        modal_open = False
                        return f"Answer: {answer}", " ", False, data, last_save, modal_open, duplicate_alert_open
                    except Exception as e:
                        return (
                            f"Answer: Error!",
                            f"Error: {e}",
                            True,
                            data,
                            last_save,
                            modal_open,
                            duplicate_alert_open
                        )

            return " ", " ", False, data, last_save, modal_open, duplicate_alert_open


        @callback(
            Output(f"{id_prefix}-programs-{form_type}-container", "data", allow_duplicate=True),
            Output(
                f"{id_prefix}-{form_type}-form-program-last-save", "children", allow_duplicate=True
            ),
            Input("data-store", "data"),
            Input(f"{id_prefix}-programs-{form_type}-container", "data_previous"),
            State(f"{id_prefix}-programs-{form_type}-container", "data"),
            prevent_initial_call="initial_duplicate",
        )
        def display_recorded_programs(data_store, previous_data, data):
            if data_store != []:
                
                if "manual" in id_prefix:
                    just_id = id_prefix.split("-")[0]
                    question_folder = os.path.join(data_store["labels_path"], just_id, "")
                    sample_df_path = os.path.join(question_folder, "manual_text_programs_labels.jsonl")
                else:
                    question_folder = os.path.join(data_store["labels_path"], id_prefix, "")
                    sample_df_path = os.path.join(question_folder, "text_programs_labels.jsonl")

                if os.path.exists(sample_df_path):
                    with jsonlines.open(sample_df_path, "r") as jsonl_f:
                        saved_data = []
                        for sample in [i for i in jsonl_f]:
                            sample["Sample IDs"] = str(sample["Sample IDs"])
                            saved_data.append(sample)
                    last_save = time.ctime(os.path.getmtime(sample_df_path))
                else:
                    saved_data = []
                    last_save = " "

                if (previous_data is not None) and (len(previous_data) > len(data)):
                    data_cleaned = []
                    for sample in data:
                        values = {k: v for k, v in sample.items()}
                        values["Sample IDs"] = eval(values["Sample IDs"])
                        data_cleaned.append(values)

                    with jsonlines.open(sample_df_path, mode="w") as writer:
                        writer.write_all(data_cleaned)
                        last_save = time.ctime(os.path.getmtime(sample_df_path))

                    saved_data = data

                return saved_data, last_save
            raise PreventUpdate

    return arithmeti_buttons


