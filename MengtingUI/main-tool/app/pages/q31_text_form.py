import os
from uuid import uuid4
from src.commons import get_page_num
from src.text_form_utils import create_layout, create_callbacks
from src.programs_utils import create_program_callbacks, create_program_layout
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

# Register the page
ID_PREFIX = "q31"
dash.register_page(__name__, path=f"/{ID_PREFIX}_text_form")


# * ------------------------------------ Parameters ------------------------------------
# Opening JSON file
keywords_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "keywords.json")
keywords_file = open(keywords_path)
keywords_dict = json.load(keywords_file)
keywords_file.close()


# * --------------------------------- Create Layout -----------------------------------
alerts = [
    dbc.Alert(
        "Market or Location-Based should not be applied to scopes other than 2.",
        color="danger",
        id=f"{ID_PREFIX}-text-form-market-alert",
        dismissable=True,
        is_open=False,
    ),
    dbc.Alert(
        "Either a Raw Unit is provided but no Standardized Unit is selected or a Standardized Unit is selected but no Raw Unit is provided. Please fix it before continuing.",
        color="danger",
        id=f"{ID_PREFIX}-text-form-std-unit-alert",
        dismissable=True,
        is_open=False,
    ),
    dbc.Alert(
        "Please select at least 2 rows that contain absolute emission with Category Name specified to start recording a program",
        color="danger",
        id=f"{ID_PREFIX}-text-program-start-alert",
        dismissable=True,
        is_open=False,
    ),
    dbc.Alert(
        "Percentage value exceeding 100. Please change it if it's not intentional.",
        id=f"{ID_PREFIX}-text-form-percentage-value-alert",
        dismissable=True,
        is_open=False,
        color="danger",
    )
]


# * -------------------- Layout ----------------
common_layout, samples_table, modal = create_layout(
    id_prefix=ID_PREFIX, inputs=inputs, n_col=1
)
program_button, program_modal, programs_table, program_details_modal = create_program_layout(ID_PREFIX, "text")

deleted_sample_alert_modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Program Exists for Sample")),
            dbc.ModalBody(id=f"{ID_PREFIX}-text-form-deleted-alert-modal-body"),
        ],
        is_open=False,
        id=f"{ID_PREFIX}-text-form-deleted-alert-modal",
    )

layout = dbc.Container(
    alerts
    + common_layout
    + [
        dbc.Tabs(
            [
                dbc.Tab(
                    [samples_table, program_button, modal, program_modal],
                    label="Recorded Samples",
                ),
                dbc.Tab(
                    [programs_table, program_details_modal], label="Recorded Programs"
                ),
            ]
        ),
        deleted_sample_alert_modal
    ],
    fluid=True,
)


# * ---------------------------------------------------------------------------------------
# *                                     Callbacks
# * ---------------------------------------------------------------------------------------
create_callbacks(id_prefix=ID_PREFIX, inputs=inputs)
create_program_callbacks(id_prefix=ID_PREFIX, form_type="text", inputs=inputs)

# * ------------------------------- Enable Percentage Form Fields --------------------------------
percentage_input_ids = ["percentage-type", "base-year", "base-scope"]
@callback(
    [Output(f"{ID_PREFIX}-{i}-text-input", "disabled") for i in percentage_input_ids],
    [Output(f"{ID_PREFIX}-{i}-text-input", "value") for i in percentage_input_ids],
    Input(f"{ID_PREFIX}-std-unit-text-input", "value"),
    [Input(f"{ID_PREFIX}-{i}-text-input", "value") for i in percentage_input_ids],
)
def disable_percentage_inputs(std_unit, pct_type, base_year, base_scope):
    if std_unit is None:
        raise PreventUpdate
    elif std_unit == "":
        raise PreventUpdate
    else:
        if std_unit=="%":
            disabled = False
            return [disabled]*3+[pct_type, base_year, base_scope]
        else:
            disabled = True
            return [disabled]*3+[None]*3

# check whether all the percentage fields are filled if std unit is %
def valid_percentage(std_unit, pct_type, base_year, base_scope):
    valid = True
    if std_unit is None:
        pass
    elif std_unit=="%":
        items = [pct_type, base_year, base_scope]
        for i in items:
            if i is None:
                valid = False
            elif i=="":
                valid = False
        
    return valid

def check_percentage_value(std_unit, value):
    valid = True
    if std_unit is None:
        pass
    elif std_unit=="%":
        if value is not None:
            if value!="":
                if value>100.:
                    valid = False
    return valid

#! Check if the value of % is exceeding 100
@callback(
    Output(f"{ID_PREFIX}-text-form-percentage-value-alert", "is_open"),
    Input(f"{ID_PREFIX}-value-text-input", "value"),
    Input(f"{ID_PREFIX}-std-unit-text-input", "value")
)
def valid_percentage_value(value, std_unit):
    valid = check_percentage_value(std_unit, value)
        
    if not valid:
        return True
    else:
        return False

# * -------------------------------------- Alert Checks ------------------------------------------
# ! Check if market is only applicable to scope 2
def check_market(scope, market):
    if "2" not in scope:
        if market != "not specified":
            return False
    return True


@callback(
    Output(f"{ID_PREFIX}-market-text-input", "valid"),
    Output(f"{ID_PREFIX}-text-form-market-alert", "is_open"),
    Input(f"{ID_PREFIX}-market-text-input", "value"),
    State(f"{ID_PREFIX}-scope-text-input", "value"),
)
def valid_market_input(market, scope):
    if (scope is not None) & (scope != ""):
        valid = check_market(scope, market)
        if not valid:
            return False, True
        else:
            return True, False
    else:
        raise PreventUpdate

#! Check if standardised unit and raw unit match
@callback(
    Output(f"{ID_PREFIX}-std-unit-text-input", "valid"),
    Output(f"{ID_PREFIX}-text-form-std-unit-alert", "is_open"),
    Input(f"{ID_PREFIX}-std-unit-text-input", "value"),
    Input(f"{ID_PREFIX}-raw-unit-text-input", "value"),
)
def valid_unit_input(std_unit, raw_unit):
    if (raw_unit is not None) & (raw_unit != ""):
        if (std_unit is None) | (std_unit == "") | (std_unit == "not specified"):
            return False, True
        else:
            return True, False
    else:
        if (std_unit is not None) & (std_unit != "") & (std_unit != "not specified"):
            return False, True
        else:
            return True, False

# Enable submit button
@callback(
    Output(f"{ID_PREFIX}-text-form-submit", "disabled"),
    [
        Input(f"{ID_PREFIX}-{i['id']}-text-input", "value")
        for i in inputs
        if i["required"] == True
    ],
    [Input(f"{ID_PREFIX}-{i}-text-input", "value") for i in percentage_input_ids],
    Input(f"{ID_PREFIX}-text-evidence-input", "children"),
    Input(f"{ID_PREFIX}-market-text-input", "valid"),
    Input(f"{ID_PREFIX}-std-unit-text-input", "valid"),
)
def disable_submit(
    year,
    scope,
    absolute,
    value,
    raw_unit,
    std_unit,
    market,
    percentage_type,
    base_year,
    base_scope,
    text_evidence,
    valid_market,
    valid_unit,
):
    disabled = False

    # * If any of the value is not filled, then cannot submit
    for i in [year, scope, absolute, market, std_unit]:
        if (i is None) | (i == ""):
            print(f"Check failed - value not filled {i}")
            disabled = True

    # * If evidence is not filled, then cannot submit
    if (text_evidence is None) | (text_evidence == "") | (text_evidence == []):
        print(f"Check failed - evidence not filled")
        disabled = True

    # * If any of the validation didn't pass, then cannot submit
    specific_checks = {
        "market": valid_market,
        "unit": valid_unit,
    }
    for name, vld in specific_checks.items():
        if not vld:
            print(f"Check failed - specific check for {name} failed ")
            disabled = True
            
    # * If all percentage-related fields are filled
    if not valid_percentage(std_unit, percentage_type, base_year, base_scope):
        print(f"Check failed - not all percentage-related fields are filled")
        disabled = True

    return disabled


# * -------------------------------------- Recorded Samples ------------------------------------------
# Display samples
@callback(
    Output(f"{ID_PREFIX}-samples-text-container", "data"),
    Output(f"{ID_PREFIX}-text-form-last-save", "children"),
    Output(f"{ID_PREFIX}-text-form-deleted-alert-modal", "is_open"),
    Output(f"{ID_PREFIX}-text-form-deleted-alert-modal-body", "children"),
    Input("data-store", "data"),
    Input(f"{ID_PREFIX}-text-form-submit", "n_clicks"),
    Input(f"{ID_PREFIX}-samples-text-container", "data_previous"),
    State(f"{ID_PREFIX}-samples-text-container", "data"),
    State(f"{ID_PREFIX}-samples-text-container", "columns"),
    State(f"{ID_PREFIX}-text-form-pagination", "active_page"),
    State(f"{ID_PREFIX}-text-evidence-input", "children"),
    [State(f"{ID_PREFIX}-{i['id']}-text-input", "value") for i in inputs],
    State(f"{ID_PREFIX}-text-form-report-extracted-data-store", "data"),
)
def display_samples_df(
    data_store,
    n_click,
    samples_previous,
    samples_current,
    columns,
    active_page,
    text_evidence,
    year,
    scope,
    absolute,
    value,
    raw_unit,
    std_unit,
    market,
    category,
    additional,
    percentage_type,
    base_year,
    base_scope,
    report_data,
):
    if data_store!=[]:
        # Get samples path
        question_folder = os.path.join(data_store["labels_path"], ID_PREFIX, "")
        sample_df_path = os.path.join(question_folder, "text_labels.jsonl")

        open_alert_modal = False
        modal_body = []

        # if file exists, then we read and display
        if os.path.exists(sample_df_path):
            with jsonlines.open(sample_df_path, "r") as jsonl_f:
                saved_samples = []
                for obj in jsonl_f:
                    sample = {
                        k: v for k, v in obj.items() if k in [c["id"] for c in columns]
                    }
                    print(sample["Text Evidence"])
                    sample["Text Evidence"] = str(sample["Text Evidence"])
                    sample["Bbox Evidence"] = str(sample["Bbox Evidence"])
                    saved_samples.append(sample)
            last_save = time.ctime(os.path.getmtime(sample_df_path))

        else:
            saved_samples = []
            last_save = ""

        # * Submit form and save
        if ctx.triggered_id == f"{ID_PREFIX}-text-form-submit":
            # Get clean text evidence
            print("We clean the text evidences:")
            clean_text_evidence = []
            for ev in text_evidence:
                if isinstance(ev, dict):
                    print(ev["props"]["children"])
                    clean_text_evidence.append(ev["props"]["children"])
            print("Cleaned text evidence:")
            print(clean_text_evidence)

            # Get bboxes
            page_sentences = report_data[active_page - 1][1]
            bbox_evidence = []
            for s in page_sentences:
                if s["text"] in clean_text_evidence:
                    bbox_evidence.append(s["bbox"])

            evidences = {
                "Page Number Evidence": get_page_num(report_data, active_page),
                "Text Evidence": clean_text_evidence,
                "Bbox Evidence": bbox_evidence,
            }
            str_evidences = {
                "Page Number Evidence": get_page_num(report_data, active_page),
                "Text Evidence": str(clean_text_evidence),
                "Bbox Evidence": str(bbox_evidence),
            }

            value_columns = [
                c["id"] for c in columns if " Evidence" or "Sample ID" not in c["id"]
            ]
            values = {
                c: r
                for c, r in zip(
                    value_columns,
                    [
                        year,
                        scope,
                        absolute,
                        value,
                        raw_unit,
                        std_unit,
                        market,
                        category,
                        additional,
                        percentage_type,
                        base_year,
                        base_scope
                    ],
                )
            }
            sample_id_row = {"Sample ID": str(uuid4())}
            sample = {**values, **evidences, **sample_id_row}
            str_sample = {**values, **str_evidences, **sample_id_row}
            saved_samples.append(str_sample)

            with jsonlines.open(sample_df_path, mode="a") as writer:
                writer.write(sample)

            last_save = time.ctime(os.path.getmtime(sample_df_path))

        # If row is deleted, overwrite samples path
        elif (samples_previous is not None) and (
            len(samples_previous) > len(samples_current)
        ):

            current_sample_ids = [i["Sample ID"] for i in samples_current]
            previous_sample_ids = [i["Sample ID"] for i in samples_previous]

            missing_id = [i for i in previous_sample_ids if i not in current_sample_ids][0]

            # CHECK IF PROGRAM WITH SAMPLE EXISTS:
            programs_df = os.path.join(
                question_folder, "text_programs_labels.jsonl"
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



            print("Row was deleted")
            print("These are the current rows:")
            samples_cleaned = []
            for sample in samples_current:
                values = {k: v for k, v in sample.items()}
                values["Text Evidence"] = eval(values["Text Evidence"])
                values["Bbox Evidence"] = eval(values["Bbox Evidence"])
                samples_cleaned.append(values)
            print()

            with jsonlines.open(sample_df_path, mode="w") as writer:
                writer.write_all(samples_cleaned)
                last_save = time.ctime(os.path.getmtime(sample_df_path))

            saved_samples = samples_current

        return saved_samples, f"Last saved: {last_save}", open_alert_modal, modal_body
    raise PreventUpdate



# -------------------- Program Details Modal --------------------
# ! Display recorded sample details
@callback(
    Output(f"{ID_PREFIX}-text-form-program-details-modal-body", "children"),
    Output(f"{ID_PREFIX}-text-form-program-details-modal", "is_open"),
    Input(f"{ID_PREFIX}-programs-text-container", "active_cell"),
    State(f"{ID_PREFIX}-samples-text-container", "data"),
    State(f"{ID_PREFIX}-programs-text-container", "data"),
    State("data-store", "data"),
)
def display_program_details_modal(active_cell, samples, programs, data_store):
    if active_cell:
        row_id = active_cell["row"]
        program_row = programs[row_id]

        # Retrieve program info
        sample_ids = program_row["Sample IDs"]

        # Get program samples evidence
        program_samples = [i for i in samples if i["Sample ID"] in sample_ids]

        samples_forms = []
        for sample in program_samples:
            form_with_answers = []
            for i in inputs:
                name = i["name"]
                row_input_group = dbc.InputGroup(
                    [
                        dbc.InputGroupText(name, style={"width": 180}),
                        dbc.Input(
                            value=sample[name],
                            readonly=True,
                            persistence=True,
                            persistence_type="memory",
                        ),
                    ],
                    className="mb-2",
                )
                form_with_answers.append(dbc.Row([row_input_group], className="g-3"))

            sample_form = dbc.Form(form_with_answers)
            text_evidence = eval(sample["Text Evidence"])
            sample_text_evidence = html.Ul([html.Li(i) for i in text_evidence])

            samples_forms.append(
                dbc.Col(
                    [
                        html.H5("Recorded Sample", className="text-center"),
                        sample_form,
                        html.Br(),
                        html.H5("Text Evidence", className="text-center"),
                        dbc.Card(sample_text_evidence, body=True),
                    ]
                )
            )

        modal_body = html.Div(
            [
                dbc.Row(samples_forms),
                html.Br(),
                dmc.Divider(variant="solid"),
                html.Br(),
                html.Div(f"Program: {program_row['Program']}"),
                html.Div(f"Answer: {program_row['Answer']}"),
            ]
        )

        return modal_body, True
    else:
        return [], False
