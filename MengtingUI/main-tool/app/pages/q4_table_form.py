import dash
from dash import html, Input, Output, callback, ctx, State, ctx, no_update
from dash.exceptions import PreventUpdate
import jsonlines
import time

import os

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
from src.table_form_utils import create_layout, create_callbacks, check_evidence
from src.inputs import q4_inputs as inputs
from pathlib import Path
import re
from uuid import uuid4
import json
import pandas as pd

ID_PREFIX = "q4"

# Register the page
dash.register_page(__name__, path=f"/{ID_PREFIX}_table_form")

# * --------------------------------- Create Layout -----------------------------------
common_layout, samples_table, modal = create_layout(
    id_prefix=ID_PREFIX, inputs=inputs, n_col=1, main_evidence_id="year"
)

layout = html.Div(common_layout + [samples_table, modal])

# * ---------------------------------------------------------------------------------------
# *                                     Callbacks
# * ---------------------------------------------------------------------------------------
create_callbacks(
    id_prefix=ID_PREFIX,
    inputs=inputs,
    main_evidence_cell="Year",
    main_evidence_id="year",
)


# ! Enable the form submit button
@callback(
    Output(f"{ID_PREFIX}-form-submit", "disabled"),
    [Input(f"{ID_PREFIX}-{i['id']}-input", "value") for i in inputs]
    + [Input(f"{ID_PREFIX}-{i['id']}-evidence", "value") for i in inputs],
    Input(f"{ID_PREFIX}-table", "rowData"),
)
def disable_submit(year, scope, year_evidence, scope_evidence, table):
    print("----------------------------------------")
    # * If a table is not selected, then cannot submit
    if (table is None) | (table == "") | (table == []):
        return True
    else:
        disabled = False
        # * If any of the value is not filled, then cannot submit
        for i in [year, scope]:
            if (i is None) | (i == ""):
                print(f"Check failed - value not filled {i}")
                disabled = True

        # * If any evidence is not filled, then cannot submit
        values = [year, scope]
        evidence = [year_evidence, scope_evidence]
        for val, evi in zip(values, evidence):
            valid = check_evidence(val, evi)
            if not valid:
                print(f"Check failed - evidence not filled {val}")
                disabled = True

    return disabled


# ! Display upon form submission
@callback(
    Output(f"{ID_PREFIX}-samples-table-container", "data", allow_duplicate=True),
    Output(f"{ID_PREFIX}-table-form-last-save", "children", allow_duplicate=True),
    Output(f"{ID_PREFIX}-year-duplicate-alert", "is_open", allow_duplicate=True),
    Output(f"{ID_PREFIX}-year-duplicate-alert", "children", allow_duplicate=True),
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
    year_evidence,
    scope_evidence,
):
    if ctx.triggered_id == f"{ID_PREFIX}-form-submit":
        children = "This sample has been recorded before. Please record another sample."
        question_folder = os.path.join(data_store["labels_path"], ID_PREFIX, "")
        table_path = data_store["table_paths"][table_ind - 1]
        table_name = Path(table_path).stem
        table_id = int(re.search(r".+_table(\d+)", table_name).group(1))
        sample_df_path = os.path.join(question_folder, f"{table_name}_labels.jsonl")

        evidences = [year_evidence, scope_evidence]
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
            recorded_samples = pd.DataFrame(samples_current)
            same_samples = recorded_samples[
                (recorded_samples["Year Evidence"] == str(final_evidences[0]))
                & (recorded_samples["Scope Evidence"] == str(final_evidences[1]))
                & (recorded_samples["Year"] == year)
                & (recorded_samples["Scope"] == scope)
            ]
            if same_samples.shape[0] > 0:
                return no_update, no_update, True, children

        value_row = {c["id"]: r for c, r in zip(columns, [year, scope])}
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

        return samples_current, last_save, False, children
    else:
        raise PreventUpdate
