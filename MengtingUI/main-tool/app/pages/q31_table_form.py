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
    no_update,
)
from dash.exceptions import PreventUpdate
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import jsonlines
import time

import os

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
from src.table_form_utils import (
    create_layout,
    create_callbacks,
    rebuild_table,
    check_evidence,
)
from src.programs_utils import create_program_callbacks, create_program_layout
from src.inputs import q31_inputs as inputs
from src.commons import remove_punctuation
import pandas as pd
from pathlib import Path
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import re
from uuid import uuid4
import json

ID_PREFIX = "q31"
MAIN_EVIDENCE_ID = "value"

# Register the page
dash.register_page(__name__, path=f"/{ID_PREFIX}_table_form")

# * --------------------------------- Create Layout -----------------------------------

# * -------------------- Program ----------------
alerts = [
    dbc.Alert(
        "Market or Location-Based should not be applied to scopes other than 2.",
        color="danger",
        id=f"{ID_PREFIX}-market-alert",
        dismissable=True,
        is_open=False,
    ),
    dbc.Alert(
        "Either a Raw Unit is provided but no Standardized Unit is selected or a Standardized Unit is selected but no Raw Unit is provided. Please fix it before continue.",
        color="danger",
        id=f"{ID_PREFIX}-std-unit-alert",
        dismissable=True,
        is_open=False,
    ),
    dbc.Alert(
        "Please select at least 2 rows that contain absolute emission with Category Name specified to start recording a program",
        color="danger",
        id=f"{ID_PREFIX}-program-start-alert",
        dismissable=True,
        is_open=False,
    ),
    dbc.Alert(
        id=f"{ID_PREFIX}-value-evidence-value-alert",
        dismissable=True,
        is_open=False,
        color="warning",
    ),
    dbc.Alert(
        "Percentage value exceeding 100. Please change it if it is not intentional.",
        id=f"{ID_PREFIX}-percentage-value-alert",
        dismissable=True,
        is_open=False,
        color="danger",
    )
]



# * -------------------- Layout ----------------
common_layout, samples_table, modal = create_layout(
    id_prefix=ID_PREFIX, inputs=inputs, n_col=1
)
program_button, program_modal, programs_table, program_details_modal = create_program_layout(ID_PREFIX, "table")


layout = html.Div(
    alerts
    + common_layout
    + [
        dbc.Tabs(
            [
                dbc.Tab(
                    [samples_table, program_button, modal, program_modal],
                    label="Recorded Samples",
                ),
                dbc.Tab([programs_table], label="Recorded Programs"),
            ]
        ),
        program_details_modal,
    ]
)

# * ---------------------------------------------------------------------------------------
# *                                     Callbacks
# * ---------------------------------------------------------------------------------------
create_callbacks(
    id_prefix=ID_PREFIX,
    inputs=inputs,
    main_evidence_cell="Value",
    main_evidence_id=MAIN_EVIDENCE_ID,
)
arithmeti_buttons = create_program_callbacks(id_prefix=ID_PREFIX, form_type="table", inputs=inputs)

# * ------------------------------- Enable Percentage Form Fields --------------------------------
percentage_input_ids = ["percentage-type", "base-year", "base-scope"]
@callback(
    [Output(f"{ID_PREFIX}-{i}-input", "disabled") for i in percentage_input_ids],
    [Output(f"{ID_PREFIX}-{i}-evidence", "disabled") for i in percentage_input_ids],
    [Output(f"{ID_PREFIX}-{i}-evidence-add", "disabled") for i in percentage_input_ids],
    [Output(f"{ID_PREFIX}-{i}-evidence-clear", "disabled") for i in percentage_input_ids],
    [Output(f"{ID_PREFIX}-{i}-input", "value") for i in percentage_input_ids],
    [Output(f"{ID_PREFIX}-{i}-evidence", "value") for i in percentage_input_ids],
    [Output(f"{ID_PREFIX}-{i}-evidence-add", "value") for i in percentage_input_ids],
    [Output(f"{ID_PREFIX}-{i}-evidence-clear", "value") for i in percentage_input_ids],
    Input(f"{ID_PREFIX}-std-unit-input", "value"),
    [Input(f"{ID_PREFIX}-{i}-input", "value") for i in percentage_input_ids],
    [Input(f"{ID_PREFIX}-{i}-evidence", "value") for i in percentage_input_ids],
    [Input(f"{ID_PREFIX}-{i}-evidence-add", "value") for i in percentage_input_ids],
    [Input(f"{ID_PREFIX}-{i}-evidence-clear", "value") for i in percentage_input_ids],
)
def disable_percentage_inputs(*args):
    std_unit = args[0]
    if std_unit is None:
        raise PreventUpdate
    elif std_unit == "":
        raise PreventUpdate
    else:
        if std_unit=="%":
            disabled = False
            return [disabled]*3*4 + list(args[1:])
        else:
            disabled = True
            return [disabled]*3*4 + [None]*len(args[1:])

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
    Output(f"{ID_PREFIX}-percentage-value-alert", "is_open"),
    Input(f"{ID_PREFIX}-value-input", "value"),
    Input(f"{ID_PREFIX}-std-unit-input", "value")
)
def valid_percentage_value(value, std_unit):
    valid = check_percentage_value(std_unit, value)
        
    if not valid:
        return True
    else:
        return False

# ! Check if market is only applicable to scope 2
def check_market(scope, market):
    if "2" not in scope:
        if market != "not specified":
            return False
    return True


@callback(
    Output(f"{ID_PREFIX}-market-input", "valid"),
    Output(f"{ID_PREFIX}-market-alert", "is_open"),
    Input(f"{ID_PREFIX}-market-input", "value"),
    State(f"{ID_PREFIX}-scope-input", "value"),
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
    Output(f"{ID_PREFIX}-std-unit-input", "valid"),
    Output(f"{ID_PREFIX}-std-unit-alert", "is_open"),
    Input(f"{ID_PREFIX}-std-unit-input", "value"),
    Input(f"{ID_PREFIX}-raw-unit-input", "value"),
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


# ! Check if the evidence of Value contains the value
@callback(
    Output(f"{ID_PREFIX}-value-evidence-value-alert", "is_open"),
    Output(f"{ID_PREFIX}-value-evidence-value-alert", "children"),
    Input(f"{ID_PREFIX}-value-evidence", "value"),
    Input(f"{ID_PREFIX}-value-input", "value"),
    Input(f"{ID_PREFIX}-value-evidence-add", "n_clicks"),
    State(f"{ID_PREFIX}-table-input", "active_page"),
    State(f"data-store", "data"),
)
def check_value_evidence_value(raw_evidence, value, add, table_id, data_store):
    if (value is None) | (value == "") | (add is None):
        raise PreventUpdate
    value = str(value)

    if add == 0:
        raise PreventUpdate

    if (raw_evidence is not None) & (raw_evidence != ""):
        # Get the value evidence
        evidence = raw_evidence.split(", ")
        if len(evidence) > 1:
            raise PreventUpdate
        else:
            evidence = evidence[0]

        # Load the table
        table_paths = data_store["table_paths"]
        if table_paths is []:
            raise PreventUpdate
        table_path = table_paths[table_id - 1]
        header, footer, rowData, columnDefs, getRowStyle = rebuild_table(table_path)

        # Check if the value is inside the evidence
        valid = True
        if evidence == "header":
            evidence_value = header
            header = remove_punctuation(header)
            if value not in header:
                valid = False
        elif evidence == "footer":
            evidence_value = footer
            footer = remove_punctuation(footer)
            if value not in footer:
                valid = False
        else:
            row_ind, col_ind = int(evidence.split("-")[0]), int(evidence.split("-")[1])
            cell_text = str(list(rowData[row_ind].values())[col_ind])
            evidence_value = cell_text
            if value not in cell_text:
                valid = False

        if not valid:
            alert_message = f"The added evidence of 'Value' which is '{raw_evidence}' ({evidence_value}) does not contain the answer {value}. Please double check if the evidence is correct before submitting the form."
        else:
            alert_message = ""

        return not valid, alert_message
    else:
        raise PreventUpdate


# ! Enable the form submit button
@callback(
    Output(f"{ID_PREFIX}-form-submit", "disabled"),
    [Input(f"{ID_PREFIX}-{i['id']}-input", "value") for i in inputs]
    + [Input(f"{ID_PREFIX}-{i['id']}-evidence", "value") for i in inputs],
    State(f"{ID_PREFIX}-table", "rowData"),
    Input(f"{ID_PREFIX}-market-input", "valid"),
    Input(f"{ID_PREFIX}-std-unit-input", "valid"),
    Input(f"{ID_PREFIX}-value-input", "valid"),
    prevent_initial_call=True
)
def disable_submit(
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
    year_evidence,
    scope_evidence,
    absolute_evidence,
    value_evidence,
    raw_unit_evidence,
    std_unit_evidence,
    market_evidence,
    category_evidence,
    additional_evidence,
    percentage_type_evidence,
    base_year_evidence,
    base_scope_evidence,
    table,
    valid_market,
    valid_unit,
    valid_value_evidence
):
    # * If a table is not selected, then cannot submit
    if (table is None) | (table == "") | (table == []):
        return True
    else:
        disabled = False
        # * If any of the value is not filled, then cannot submit
        for i in [year, scope, absolute, market]:
            if (i is None) | (i == ""):
                print(f"Check failed - value not filled {i}")
                disabled = True
        # * If any evidence is not filled, then cannot submit
        values = [year, scope, absolute, raw_unit, market, category, additional,
                percentage_type, base_year, base_scope]
        evidence = [
            year_evidence,
            scope_evidence,
            absolute_evidence,
            raw_unit_evidence,
            market_evidence,
            category_evidence,
            additional_evidence,
            percentage_type_evidence,
            base_year_evidence,
            base_scope_evidence,
        ]
        for val, evi in zip(values, evidence):
            valid = check_evidence(val, evi)
            if not valid:
                # print(f"Check failed - evidence not filled {val}")
                disabled = True

        # * If any of the validation didn't pass, then cannot submit
        specific_checks = {
            "market": valid_market,
            "unit": valid_unit,
            "value": valid_value_evidence,
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


# ! Display upon form submission
@callback(
    Output(f"{ID_PREFIX}-samples-table-container", "data", allow_duplicate=True),
    Output(f"{ID_PREFIX}-table-form-last-save", "children", allow_duplicate=True),
    Output(f"{ID_PREFIX}-value-duplicate-alert", "is_open", allow_duplicate=True),
    Input(f"{ID_PREFIX}-form-submit", "n_clicks"),
    State(f"{ID_PREFIX}-table-input", "active_page"),
    State("data-store", "data"),
    State(f"{ID_PREFIX}-samples-table-container", "data"),
    State(f"{ID_PREFIX}-samples-table-container", "columns"),
    [State(f"{ID_PREFIX}-{i['id']}-input", "value") for i in inputs]
    + [State(f"{ID_PREFIX}-{i['id']}-evidence", "value") for i in inputs],
    prevent_initial_call=True,
)
def add_sample(
    submit,
    table_ind,
    data_store,
    samples_current,
    columns,
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
    year_evidence,
    scope_evidence,
    absolute_evidence,
    value_evidence,
    raw_unit_evidence,
    std_unit_evidence,
    market_evidence,
    category_evidence,
    additional_evidence,
    percentage_type_evidence,
    base_year_evidence,
    base_scope_evidence,
):
    # print("columns", columns)
    dup_alert = False
    if ctx.triggered_id == f"{ID_PREFIX}-form-submit":
        question_folder = os.path.join(data_store["labels_path"], ID_PREFIX, "")
        table_path = data_store["table_paths"][table_ind - 1]
        table_name = Path(table_path).stem
        table_id = int(re.search(r".+_table(\d+)", table_name).group(1))
        sample_df_path = os.path.join(question_folder, f"{table_name}_labels.jsonl")

        evidences = [
            year_evidence,
            scope_evidence,
            absolute_evidence,
            value_evidence,
            raw_unit_evidence,
            std_unit_evidence,
            market_evidence,
            category_evidence,
            additional_evidence,
            percentage_type_evidence,
            base_year_evidence,
            base_scope_evidence
        ]
        final_evidences = []
        for e in evidences:
            if e == "":
                es = None
            elif e is not None:
                e = e.split(", ")
                es = []
                for ei in e:
                    ee = ei.split("-")
                    try:
                        ee = [int(i) for i in ee]
                    except:
                        pass
                    es.append([table_id, ee])
            else:
                es = None
            final_evidences.append(es)

        # Avoid adding the same sample again
        if samples_current != []:
            str_value_evidence = str(final_evidences[3])
            if str_value_evidence in [i["Value Evidence"] for i in samples_current]:
                # return no_update, no_update, True
                dup_alert = True

        value_row = {
            c["id"]: r
            for c, r in zip(
                columns,
                [str(year), str(scope), absolute, value, raw_unit, std_unit, market, category, additional, percentage_type, base_year, base_scope],
            )
        }
        str_evidence_row = {
            c: str(r)
            for c, r in zip(
                [f"{col['id']} Evidence" for col in columns], final_evidences
            )
        }
        evidence_row = {
            c: r
            for c, r in zip(
                [f"{col['id']} Evidence" for col in columns], final_evidences
            )
        }

        with open(table_path, "r") as f:
            table_json = json.load(f)

        str_page_num_row = {"Page Number Evidence": str(table_json["page_no"])}
        page_num_row = {"Page Number Evidence": table_json["page_no"]}

        sample_id_row = {"Sample ID": str(uuid4())}

        row = {**value_row, **evidence_row, **page_num_row, **sample_id_row}
        str_row = {**value_row, **str_evidence_row, **str_page_num_row, **sample_id_row}

        samples_current.append(str_row)

        with jsonlines.open(sample_df_path, mode="a") as writer:
            writer.write(row)
        last_save = time.ctime(os.path.getmtime(sample_df_path))

        return samples_current, last_save, dup_alert
    else:
        raise PreventUpdate


# * ---------------------------------- Program Modal ---------------------------------------
# ! Display add program page
@callback(
    Output(f"{ID_PREFIX}-table-form-program-modal-body", "children"),
    Output(f"{ID_PREFIX}-table-form-program-modal", "is_open", allow_duplicate=True),
    Input(f"{ID_PREFIX}-table-form-record-program-button", "n_clicks"),
    State(f"{ID_PREFIX}-samples-table-container", "selected_rows"),
    State(f"{ID_PREFIX}-samples-table-container", "data"),
    State("data-store", "data"),
    State(f"{ID_PREFIX}-table-input", "active_page"),
    prevent_initial_call=True,
)
def display_program_modal(n_clicks, selected_rows, data, data_store, table_id):
    if ctx.triggered_id == f"{ID_PREFIX}-table-form-record-program-button":
        rows = [data[i] for i in selected_rows]

        # * ----------------------------- Highlight Cells --------------------------------
        # Get the value evidence
        value_evidence = [eval(i["Value Evidence"]) for i in rows]
        value_evidence = [i for j in value_evidence for i in j]
        value_evidence = [i[1] for i in value_evidence]

        # Load the table
        table_paths = data_store["table_paths"]

        if table_paths is []:
            raise PreventUpdate

        table_path = table_paths[table_id - 1]

        header, footer, rowData, columnDefs, getRowStyle = rebuild_table(table_path)

        # Highlight value evidence cells
        color = "SkyBlue"
        for coldef in columnDefs:
            coldef["cellStyle"] = {}
            cellStyle = []
            for e in value_evidence:
                if e[0] == "header":
                    header_style = {"background-color": color}
                elif e[0] == "footer":
                    footer_style = {"background-color": color}
                else:
                    row_ind, col_ind = e[0], e[1]
                    if str(col_ind) == coldef["field"]:
                        cell_style = {
                            "condition": f"params.rowIndex==={row_ind}",
                            "style": {"backgroundColor": color, "color": "black"},
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
                        "type": f"{ID_PREFIX}-table-form-arithmetic-values-button",
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
                        "type": f"{ID_PREFIX}-table-form-arithmetic-ints-button",
                        "index": i,
                    },
                    color="primary",
                )
                for i, v in enumerate(ints)
            ]
        )

        # * ------------------------------------ Modal body --------------------------------
        modal_body = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Div(children=header, style=header_style),
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
                                            style={"height": 400, "width": "100%"},
                                        ),
                                        html.Div(children=footer, style=footer_style),
                                    ]
                                )
                            ]
                        )
                    ]
                ),
                html.Br(),
                dmc.Divider(variant="solid"),
                dbc.Alert(
                    id=f"{ID_PREFIX}-table-form-program-alert",
                    color="danger",
                    dismissable=True,
                    is_open=False,
                ),
                dbc.Alert(
                    "These rows have got a program recorded. Please record other rows instead.",
                    color="danger",
                    id=f"{ID_PREFIX}-program-duplicate-alert",
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
                            id=f"{ID_PREFIX}-table-form-program-undo",
                        ),
                        # dmc.Button("", leftIcon=DashIconify(icon="ic:baseline-redo", width=30), variant="subtle", id=f"{ID_PREFIX}-table-form-program-redo"),
                        dmc.Button(
                            "",
                            leftIcon=DashIconify(
                                icon="mdi:clear", width=20, color="#d9230f"
                            ),
                            variant="subtle",
                            id=f"{ID_PREFIX}-table-form-program-clear",
                        ),
                        dmc.Button(
                            "",
                            leftIcon=DashIconify(
                                icon="carbon:chart-average", width=20, color="#d9230f"
                            ),
                            variant="subtle",
                            id=f"{ID_PREFIX}-table-form-program-average",
                        ),
                        dmc.Button(
                            "",
                            leftIcon=DashIconify(
                                icon="tabler:sum", width=20, color="#d9230f"
                            ),
                            variant="subtle",
                            id=f"{ID_PREFIX}-table-form-program-summation",
                        ),
                    ]
                ),
                dbc.Input(
                    className="mb-3",
                    readonly=True,
                    placeholder="Please write a program to calculate the total emission",
                    id=f"{ID_PREFIX}-table-form-program-input",
                    size="lg",
                    persistence=False,
                ),
                html.Div(id=f"{ID_PREFIX}-table-form-program-answer"),
                html.Br(),
                html.Div(
                    [
                        dbc.Col(
                            dbc.Button(
                                "Submit", id=f"{ID_PREFIX}-table-form-program-submit"
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
    Output(f"{ID_PREFIX}-table-form-program-answer", "children"),
    Output(f"{ID_PREFIX}-table-form-program-alert", "children"),
    Output(f"{ID_PREFIX}-table-form-program-alert", "is_open"),
    Output(f"{ID_PREFIX}-programs-table-container", "data", allow_duplicate=True),
    Output(
        f"{ID_PREFIX}-table-form-program-last-save", "children", allow_duplicate=True
    ),
    Output(f"{ID_PREFIX}-table-form-program-modal", "is_open", allow_duplicate=True),
    Output(f"{ID_PREFIX}-program-duplicate-alert", "is_open"),
    Input(f"{ID_PREFIX}-table-form-program-submit", "n_clicks"),
    State(f"{ID_PREFIX}-table-form-program-input", "value"),
    State(f"{ID_PREFIX}-samples-table-container", "selected_rows"),
    State(f"{ID_PREFIX}-table-input", "active_page"),
    State("data-store", "data"),
    State(f"{ID_PREFIX}-samples-table-container", "data"),
    State(f"{ID_PREFIX}-programs-table-container", "data"),
    State(f"{ID_PREFIX}-programs-table-container", "columns"),
    prevent_initial_call=True,
)
def submit_program(
    submit, program, selected_rows, table_id, data_store, samples_data, data, columns
):
    modal_open = True
    duplicate_alert_open = False
    question_folder = os.path.join(data_store["labels_path"], ID_PREFIX, "")
    table_name = Path(data_store["table_paths"][table_id - 1]).stem
    sample_df_path = os.path.join(
        question_folder, f"{table_name}_programs_labels.jsonl"
    )

    if os.path.exists(sample_df_path):
        last_save = time.ctime(os.path.getmtime(sample_df_path))
    else:
        last_save = " "

    if (program is not None) & (program != ""):
        if ctx.triggered_id == f"{ID_PREFIX}-table-form-program-submit":
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
                return (
                    f"Answer: {answer}",
                    " ",
                    False,
                    data,
                    last_save,
                    modal_open,
                    duplicate_alert_open,
                )
            except Exception as e:
                return (
                    f"Answer: Error!",
                    f"Error: {e}",
                    True,
                    data,
                    last_save,
                    modal_open,
                    duplicate_alert_open,
                )

    return " ", " ", False, data, last_save, modal_open, duplicate_alert_open


# ! Display recorded programs
@callback(
    Output(f"{ID_PREFIX}-programs-table-container", "data", allow_duplicate=True),
    Output(
        f"{ID_PREFIX}-table-form-program-last-save", "children", allow_duplicate=True
    ),
    Input(f"{ID_PREFIX}-table-input", "active_page"),
    State("data-store", "data"),
    State(f"{ID_PREFIX}-programs-table-container", "data"),
    prevent_initial_call=True,
)
def display_recorded_programs(table_id, data_store, data):
    if (table_id is not None) & (table_id!=""):
        if len(data_store["table_paths"])>0:
            question_folder = os.path.join(data_store["labels_path"], ID_PREFIX, "")
            table_name = Path(data_store["table_paths"][table_id - 1]).stem
            sample_df_path = os.path.join(
                question_folder, f"{table_name}_programs_labels.jsonl"
            )
            if os.path.exists(sample_df_path):
                with jsonlines.open(sample_df_path, "r") as jsonl_f:
                    data = []
                    for sample in [i for i in jsonl_f]:
                        sample["Sample IDs"] = str(sample["Sample IDs"])
                        data.append(sample)
                last_save = time.ctime(os.path.getmtime(sample_df_path))
            else:
                data = []
                last_save = " "
            return data, last_save
    raise PreventUpdate


# ! Display upon sample deletion
@callback(
    Output(f"{ID_PREFIX}-programs-table-container", "data", allow_duplicate=True),
    Output(
        f"{ID_PREFIX}-table-form-program-last-save", "children", allow_duplicate=True
    ),
    State(f"{ID_PREFIX}-table-input", "active_page"),
    State(f"{ID_PREFIX}-programs-table-container", "data_previous"),
    Input(f"{ID_PREFIX}-programs-table-container", "data_timestamp"),
    State(f"{ID_PREFIX}-programs-table-container", "data"),
    State("data-store", "data"),
    prevent_initial_call=True,
)
def delete_program_sample(table_id, data_previous, edit_time, data_current, data_store):
    if data_previous is not None:
        question_folder = os.path.join(data_store["labels_path"], ID_PREFIX, "")
        table_name = Path(data_store["table_paths"][table_id - 1]).stem
        sample_df_path = os.path.join(
            question_folder, f"{table_name}_programs_labels.jsonl"
        )
        disk_last_save = os.path.getmtime(sample_df_path)

        if edit_time > disk_last_save:
            samples = []
            for sample in data_current:
                values = {k: v for k, v in sample.items() if "Sample IDs" not in k}
                sample_ids = {
                    k: eval(v) for k, v in sample.items() if k == "Sample IDs"
                }
                sample = {**values, **sample_ids}
                samples.append(sample)
            with jsonlines.open(sample_df_path, mode="w") as writer:
                writer.write_all(samples)

        last_save = time.ctime(os.path.getmtime(sample_df_path))
        return data_current, last_save
    else:
        raise PreventUpdate


# ! Display recorded program details
@callback(
    Output(f"{ID_PREFIX}-table-form-program-details-modal-body", "children"),
    Output(f"{ID_PREFIX}-table-form-program-details-modal", "is_open"),
    Input(f"{ID_PREFIX}-programs-table-container", "active_cell"),
    State(f"{ID_PREFIX}-samples-table-container", "data"),
    State(f"{ID_PREFIX}-programs-table-container", "data"),
    State("data-store", "data"),
    State(f"{ID_PREFIX}-table-input", "active_page"),
    prevent_initial_call=True,
)
def display_program_details_modal(active_cell, samples, programs, data_store, table_id):
    if active_cell:
        row_id = active_cell["row"]
        program_row = programs[row_id]
        sample_ids = eval(program_row["Sample IDs"])
        program_samples = [i for i in samples if i["Sample ID"] in sample_ids]
        program_samples = [eval(i["Value Evidence"]) for i in program_samples]
        program_samples = [i for j in program_samples for i in j]
        program_samples = [i[1] for i in program_samples]

        # * Load the sample data
        table_paths = data_store["table_paths"]

        if table_paths is []:
            raise PreventUpdate

        table_path = table_paths[table_id - 1]

        header, footer, rowData, columnDefs, getRowStyle = rebuild_table(table_path)
        # Highlight value evidence cells
        color = "#daeace"
        for coldef in columnDefs:
            coldef["cellStyle"] = {}
            cellStyle = []
            for e in program_samples:
                if e[0] == "header":
                    header_style = {"background-color": color}
                elif e[0] == "footer":
                    footer_style = {"background-color": color}
                else:
                    row_ind, col_ind = e[0], e[1]
                    if str(col_ind) == coldef["field"]:
                        cell_style = {
                            "condition": f"params.rowIndex==={row_ind}",
                            "style": {"backgroundColor": color, "color": "black"},
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

        modal_body = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Div(children=header, style=header_style),
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
                                            style={"height": 400, "width": "100%"},
                                        ),
                                        html.Div(children=footer, style=footer_style),
                                    ]
                                )
                            ]
                        )
                    ]
                ),
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


#! Check if the evidence of 'Value' contains only 1 evidence
#! Check if the evidence of 'Value' has been recorded
@callback(
    Output(f"{ID_PREFIX}-{MAIN_EVIDENCE_ID}-input", "valid"),
    Output(f"{ID_PREFIX}-{MAIN_EVIDENCE_ID}-alert", "is_open"),
    Output(
        f"{ID_PREFIX}-{MAIN_EVIDENCE_ID}-duplicate-alert",
        "is_open",
        allow_duplicate=True,
    ),
    Input(f"{ID_PREFIX}-{MAIN_EVIDENCE_ID}-evidence", "value"),
    State(f"{ID_PREFIX}-{MAIN_EVIDENCE_ID}-input", "value"),
    State("data-store", "data"),
    State(f"{ID_PREFIX}-table-input", "active_page"),
    prevent_initial_call=True,
)
def valid_value_evidence(evidence, value, data_store, table_id):
    valid = True
    if (evidence is not None) & (evidence != ""):
        # if (value is not None) & (value != ""):
        # 1. check if we have more than 1 evidence
        num = len(evidence.split(", "))
        valid = False if num > 1 else True
        more_alert = False if valid else True
        if valid is False:
            print("Value check - More than 1 evidence detected")

        # 2. check if we have duplicated samples
        labels_path = data_store["labels_path"]
        question_folder = os.path.join(labels_path, ID_PREFIX, "")
        table_name = Path(data_store["table_paths"][table_id - 1]).stem
        sample_df_path = os.path.join(question_folder, f"{table_name}_labels.jsonl")

        if os.path.exists(sample_df_path):
            with jsonlines.open(sample_df_path, "r") as jsonl_f:
                value_evidence = []
                for obj in jsonl_f:
                    ve = obj["Value Evidence"]
                    if isinstance(ve, str):
                        ve = eval(ve)
                    value_evidence.append(ve)
            value_evidence = [i for j in value_evidence for i in j]
            value_evidence = [
                "-".join([str(j) for j in i[1]]) for i in value_evidence
            ]

            # valid = False if evidence in value_evidence else valid
            duplicate_alert = True if evidence in value_evidence else False
            if not valid:
                print("Value check - duplicated samples detected")
        else:
            duplicate_alert = False

        return valid, more_alert, duplicate_alert
    return False, False, False
