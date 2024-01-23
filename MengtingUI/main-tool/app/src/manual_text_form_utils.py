import random
from .commons import (
    split_sentence,
    filter_text_for_keywords,
)
from .pdf_data import DocumentLoader, display_pdf_page
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash import html, dash_table, dcc, Input, Output, callback, ctx, State, ALL, MATCH
from dash.exceptions import PreventUpdate
from dash_canvas import DashCanvas
from typing import List
import numpy as np
import pandas as pd
import os
import jsonlines
import json
from src import cache
import time



def create_input(id_prefix: str, **kwargs) -> dbc.InputGroup:
    """
    This function creates the input. It creates
        - User input field
        - Evidence field
            - Evidence input field (read only)
            - Clear evidence button
    """
    name = kwargs.pop("name")
    id = kwargs.pop("id")
    comp_type = kwargs.pop("comp_type")
    comp_kwargs = kwargs.pop("comp_kwargs", {})
    tooltip = kwargs.pop("tooltip", "")
    required = kwargs.pop("required", True)

    if comp_type == "dropdown":
        user_input = dbc.Select(
            **comp_kwargs,
            required=required,
            placeholder="",
            persistence=True,
            persistence_type="session",
            id=f"{id_prefix}-manual-{id}-text-input",
        )
    elif comp_type == "input":
        user_input = dbc.Input(
            **comp_kwargs,
            required=required,
            autoFocus=False,
            persistence=True,
            persistence_type="session",
            id=f"{id_prefix}-manual-{id}-text-input",
        )
    elif comp_type == "text-area":
        user_input = dbc.Textarea(
            required=required,
            autoFocus=False,
            persistence=True,
            persistence_type="session",
            id=f"{id_prefix}-manual-{id}-text-input",
        )


    input_group = dbc.InputGroup(
        [
            dbc.InputGroupText(
                name, id=f"{id_prefix}-manual-text-{id}-name", style={"width": "35%"}
            ),
            user_input,
            dbc.Tooltip(tooltip, target=f"{id_prefix}-manual-text-{id}-name", placement="top"),
        ],
        className="mb-2",
    )

    return input_group



def create_form(id_prefix: str, inputs: List[dict], n_col: int = 1):
    user_inputs = []
    for i in inputs:
        user_input = create_input(id_prefix=id_prefix, **i)
        user_inputs.append(user_input)

    if n_col > len(user_inputs):
        raise ValueError(
            f"n_col must be smaller than the number of user_inputs which is {len(user_inputs)}"
        )
    m = np.arange(len(user_inputs)).reshape(-1, n_col)

    rows = []
    for irow in m:
        row = dbc.Row([dbc.Col(user_inputs[icol]) for icol in irow], className="g-3")
        rows.append(row)

    evidence_inputs = [
        dbc.InputGroup(
            [
                dbc.InputGroupText(
                    "Text Evidence",
                    id=f"{id_prefix}-manual-text-evidence-name",
                    style={"width": "35%", "background-color": "secondary"},
                ),
                dbc.Tooltip(
                    "Select all text evidences needed to support the above facts.",
                    target=f"{id_prefix}-manual-text-evidence-name",
                    placement="top",
                ),
                dbc.Textarea(
                    id=f"{id_prefix}-manual-text-evidence-input",
                    required=True,
                    autoFocus=False,
                    persistence=True,
                    persistence_type="session",
                ),
            ],
            className="mb-2",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText(
                    "Page Number",
                    style={"width": "35%", "background-color": "secondary"},
                ),
                dbc.Input(
                    id=f"{id_prefix}-manual-text-evidence-page-number",
                    type="number",
                    inputmode="numeric",
                    min=0.0,
                    required=True,
                    autoFocus=False,
                    persistence=True,
                    persistence_type="session",
                ),
            ],
            className="mb-2",
        ),
    ]

    form_rows = [
        dbc.Row(
            [
                dbc.Col(rows, style={"width": "49%"}),
                dbc.Col(evidence_inputs, style={"width": "49%"}),
            ]
        )
    ]

    form = dbc.Form(
        form_rows
        + [
            dbc.Row(
                html.Div(
                    [
                        dbc.Button(
                            "Clear",
                            color="primary",
                            outline=True,
                            id=f"{id_prefix}-manual-text-form-clear",
                            n_clicks=0,
                            className="w-50",
                        ),
                        dbc.Button(
                            "Submit",
                            color="primary",
                            id=f"{id_prefix}-manual-text-form-submit",
                            n_clicks=0,
                            className="w-50",
                        ),
                    ],
                    className="gap-2 d-md-flex",
                )
            )
        ],
        id=f"{id_prefix}-manual-text-form",
        className="g-3",
    )

    return form


def create_layout(id_prefix: str, inputs: List[dict], n_col: int = 1):

    # * -------------------------------------- Text Carousel --------------------------------------
    text_carousel = dbc.Card(
            dbc.CardBody(
                dcc.Markdown(
                    """
                    Please open the selected report in your preferred PDF viewer and proceed to fill out the form by 
                    inputting the text evidence manually.
                    """
                )
            )
        )
    
    # * ---------------------------------------- Form ----------------------------------------------
    form = create_form(id_prefix, inputs, n_col)

    form_card = dbc.Card(
        dbc.CardBody([html.H5("Add a New Sample", className="text-center"), form])
    )

    # * ------------------------------------- Recorded Samples -------------------------------------
    hidden_columns = [
        "Text Evidence",
        "Page Number Evidence",
        "Sample ID",
    ]
    samples_df = pd.DataFrame(
        {**{i["name"]: [] for i in inputs}, **{i: [] for i in hidden_columns}}
    )

    samples_table = dbc.Row(
        [
            dbc.Col(
                [
                    html.H5("Recorded Samples", className="text-center"),
                    dash_table.DataTable(
                        id=f"{id_prefix}-manual-samples-text-container",
                        data=[],
                        columns=[
                            {"name": i_3, "id": i_3} for i_3 in samples_df.columns
                        ],
                        style_table={"overflow": "scroll", "height": 600},
                        style_cell={"textAlign": "left"},
                        row_deletable=True,
                        row_selectable="multi",
                        editable=False,
                        persistence=False,
                        sort_action="native",
                        filter_action="native",
                        hidden_columns=hidden_columns,
                        # persistence_type="session"
                    ),
                    html.Div(
                        className="text-center", id=f"{id_prefix}-manual-text-form-last-save"
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
            dbc.ModalBody(id=f"{id_prefix}-manual-text-form-modal-body"),
            dbc.ModalFooter(
                dbc.Button(
                    "Edit",
                    id=f"{id_prefix}-manual-text-form-modal-edit",
                    className="ms-auto",
                    n_clicks=0,
                )
            ),
        ],
        size="xl",
        scrollable=True,
        is_open=False,
        fullscreen=False,
        id=f"{id_prefix}-manual-text-form-modal",
    )

    edit_alert_modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Editting Disabled")),
            dbc.ModalBody(id=f"{id_prefix}-manual-text-form-alert-modal-body"),
        ],
        is_open=False,
        id=f"{id_prefix}-manual-text-form-alert-modal",
    )

    # * ----------------------------------------- Layout ----------------------------------------
    layout_header = dbc.Row(
        [
            html.H1(f"{id_prefix.upper()} Manual Text Evidence")
        ]
    )

    layout = [
        html.Br(),
        layout_header,
        html.Br(),
        text_carousel,
        html.Br(),
        form_card,
        html.Br(),
        dmc.Divider(variant="dotted"),
        html.Br(),
        edit_alert_modal
    ]

    return layout, samples_table, modal



# * ---------------------------------------------------------------------------------------
# *                                     Callbacks
# * ---------------------------------------------------------------------------------------


def create_callbacks(id_prefix: str, inputs: List[dict]):
    # * -------------------------------------- Form ------------------------------------------
    # ! Clear form
    @callback(
        [
            Output(f"{id_prefix}-manual-{i['id']}-text-input", "value", allow_duplicate=True)
            for i in inputs
        ]
        + [
            Output(f"{id_prefix}-manual-text-evidence-input", "value", allow_duplicate=True),
            Output(f"{id_prefix}-manual-text-evidence-page-number", "value", allow_duplicate=True),
        ],
        Input(f"{id_prefix}-manual-text-form-clear", "n_clicks"),
        Input(f"{id_prefix}-manual-text-form-submit", "n_clicks"),
        prevent_initial_call=True
    )
    def clear_form(n_clicks, submit):
        if ctx.triggered_id in [f"{id_prefix}-manual-text-form-clear", f"{id_prefix}-manual-text-form-submit"]:
            return (
                [None] * (len(inputs) + 2)
            )
        else:
            raise PreventUpdate
        

    # * ---------------------------------- Sample Modal ---------------------------------------
    # ! Display recorded sample details
    @callback(
        Output(f"{id_prefix}-manual-text-form-modal-body", "children"),
        Output(f"{id_prefix}-manual-text-form-modal", "is_open"),
        Input(f"{id_prefix}-manual-samples-text-container", "active_cell"),
        State("data-store", "data"),
    )
    def display_modal(active_cell, data_store):
        if active_cell:
            row_id = active_cell["row"]

            # Load sample data
            question_folder = os.path.join(data_store["labels_path"], id_prefix, "")
            sample_df_path = os.path.join(question_folder, "manual_text_labels.jsonl")


            # if file exists, then we retrieve the sample
            with jsonlines.open(sample_df_path, "r") as jsonl_f:
                rows = [obj for obj in jsonl_f]
            row = rows[row_id]

            page_num = row["Page Number Evidence"]

            # Get filled form
            form_with_answers = []
            for i in inputs:
                name = i["name"]
                row_input_group = dbc.InputGroup(
                    [
                        dbc.InputGroupText(name, style={"width": "35%"}),
                        dbc.Input(
                            value=row[name],
                            readonly=True,
                            persistence=True,
                            persistence_type="memory",
                        ),
                    ],
                    className="mb-2",
                )
                form_with_answers.append(dbc.Row([row_input_group], className="g-3"))

            text_evidence = row["Text Evidence"]
            page_number_evidence = row["Page Number Evidence"]

            evidence_inputs = [
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(
                            "Text Evidence",
                            style={"width": "35%", "background-color": "secondary"},
                        ),
                        dbc.Textarea(
                            value=text_evidence,
                            autoFocus=False,
                            readonly=True,
                        ),
                    ],
                    className="mb-2",
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(
                            "Page Number",
                            style={"width": "35%", "background-color": "secondary"},
                        ),
                        dbc.Input(
                            value=page_number_evidence,
                            type="number",
                            inputmode="numeric",
                            autoFocus=False,
                            readonly=True,
                        ),
                    ]
                ),
            ]

            form_rows = [
                dbc.Row(
                    [
                        dbc.Col(form_with_answers, style={"width": "49%"}),
                        dbc.Col(evidence_inputs, style={"width": "49%"}),
                    ]
                )
            ]

            modal_form = dbc.Form(
                form_rows,
                className="g-3",
            )

            

            modal_body = html.Div(
                [
                    html.H5("Recorded Sample", className="text-center"),
                    modal_form,
                ]
            )

            

            return modal_body, True
        else:
            return [], False

    # * ---------------------------------- Edit sample ---------------------------------------
    if id_prefix == "q31":
        @callback(
            [
                Output(f"{id_prefix}-manual-{i['id']}-text-input", "value", allow_duplicate=True)
                for i in inputs
            ] + 
            [
                Output(f"{id_prefix}-manual-text-evidence-input", "value", allow_duplicate=True),
                Output(f"{id_prefix}-manual-text-evidence-page-number", "value", allow_duplicate=True),
            ] +
            [
                Output(f"{id_prefix}-manual-text-form-modal", "is_open", allow_duplicate=True),
                Output(f"{id_prefix}-manual-text-form-alert-modal", "is_open", allow_duplicate=True)
            ] + 
            [
                Output(f"{id_prefix}-manual-samples-text-container", "data", allow_duplicate=True),
                Output(f"{id_prefix}-manual-text-form-last-save", "children", allow_duplicate=True)
            ] + [
                Output(f"{id_prefix}-manual-text-form-alert-modal-body", "children", allow_duplicate=True)
            ],
            Input(f"{id_prefix}-manual-text-form-modal-edit", "n_clicks"),
            State(f"{id_prefix}-manual-samples-text-container", "active_cell"),
            State("data-store", "data"),
            State(f"{id_prefix}-manual-samples-text-container", "data"),
            prevent_initial_call=True
        )
        def edit_sample(n_clicks, active_cell, data_store, current_samples):
            if ctx.triggered_id == f"{id_prefix}-manual-text-form-modal-edit":
                print("EDITTING")
                question_folder = os.path.join(data_store["labels_path"], id_prefix, "")
                sample_df_path = os.path.join(question_folder, "manual_text_labels.jsonl")

                row_id = active_cell["row"]

                # * Load the sample data
                row = current_samples[row_id]
                
                # CHECK IF PROGRAM WITH SAMPLE EXISTS:
                programs_df = os.path.join(
                    question_folder, "manual_text_programs_labels.jsonl"
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
                    form_evidences = [None]*2

                    last_save = os.path.getmtime(sample_df_path)

                else:
                    row = current_samples.pop(row_id)
                    open_alert_modal = False
                    modal_body = []

                    form_values = [row[i["name"]] for i in inputs]
                    form_evidences = [row["Text Evidence"], row["Page Number Evidence"]]

                    print(form_values)
                    print(form_evidences)

                    # Delete sample
                    samples = []
                    for sample in current_samples:
                        values = {k: v for k, v in sample.items()}
                        samples.append(values)

                    with jsonlines.open(sample_df_path, mode="w") as writer:
                        writer.write_all(samples)
                    last_save = time.ctime(os.path.getmtime(sample_df_path))

                return form_values + form_evidences + [False, open_alert_modal] + [current_samples, last_save] + [modal_body]
            
            else:
                PreventUpdate


    else:
        @callback(
            [
                Output(f"{id_prefix}-manual-{i['id']}-text-input", "value", allow_duplicate=True)
                for i in inputs
            ] +
            [
                Output(f"{id_prefix}-manual-text-evidence-input", "value", allow_duplicate=True),
                Output(f"{id_prefix}-manual-text-evidence-page-number", "value", allow_duplicate=True),
            ] +
            [
                Output(f"{id_prefix}-manual-text-form-modal", "is_open", allow_duplicate=True),
            ] + 
            [
                Output(f"{id_prefix}-manual-samples-text-container", "data", allow_duplicate=True),
                Output(f"{id_prefix}-manual-text-form-last-save", "children", allow_duplicate=True)
            ],
            Input(f"{id_prefix}-manual-text-form-modal-edit", "n_clicks"),
            State(f"{id_prefix}-manual-samples-text-container", "active_cell"),
            State("data-store", "data"),
            State(f"{id_prefix}-manual-samples-text-container", "data"),
            prevent_initial_call=True
        )
        def edit_sample(n_clicks, active_cell, data_store, current_samples):
            if ctx.triggered_id == f"{id_prefix}-manual-text-form-modal-edit":
                print("EDITTING")
                question_folder = os.path.join(data_store["labels_path"], id_prefix, "")
                sample_df_path = os.path.join(question_folder, "manual_text_labels.jsonl")

                row_id = active_cell["row"]

                # * Load the sample data
            
                row = current_samples.pop(row_id)

                form_values = [row[i["name"]] for i in inputs]
                form_evidences = [row["Text Evidence"], row["Page Number Evidence"]]

                print(form_values)
                print(form_evidences)

                # Delete sample
                with jsonlines.open(sample_df_path, mode="w") as writer:
                    writer.write_all(current_samples)
                last_save = time.ctime(os.path.getmtime(sample_df_path))

                return form_values + form_evidences + [False] + [current_samples, last_save]
            
            else:
                PreventUpdate





