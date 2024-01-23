import dash
from dash import html, Input, Output, callback, ctx, State, no_update
from dash.exceptions import PreventUpdate
import jsonlines
import time

import os
import json

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
from src.table_form_utils import create_layout, create_callbacks, check_evidence
from src.inputs import q32_inputs as inputs
from pathlib import Path
import re
from uuid import uuid4

ID_PREFIX = "q32"
MAIN_EVIDENCE_ID = "exclusion-answer"

# Register the page
dash.register_page(__name__, path=f"/{ID_PREFIX}_table_form")

# * --------------------------------- Create Layout -----------------------------------
common_layout, samples_table, modal = create_layout(
    id_prefix=ID_PREFIX, inputs=inputs, n_col=1, main_evidence_id="exclusion-answer"
)

layout = html.Div(common_layout + [samples_table, modal])

# * ---------------------------------------------------------------------------------------
# *                                     Callbacks
# * ---------------------------------------------------------------------------------------
create_callbacks(
    id_prefix=ID_PREFIX,
    inputs=inputs,
    main_evidence_cell="Answer",
    main_evidence_id=MAIN_EVIDENCE_ID,
)


# ! Enable the form submit button
@callback(
    Output(f"{ID_PREFIX}-form-submit", "disabled"),
    [Input(f"{ID_PREFIX}-{i['id']}-input", "value") for i in inputs]
    + [Input(f"{ID_PREFIX}-{i['id']}-evidence", "value") for i in inputs],
    Input(f"{ID_PREFIX}-table", "rowData"),
    Input(f"{ID_PREFIX}-exclusion-answer-input", "valid"),
)
def disable_submit(
    ei, ei_answer, ei_evidence, ei_answer_evidence, table, valid_ei_answer_evidence
):
    print("----------------------------------------")
    # * If a table is not selected, then cannot submit
    if (table is None) | (table == "") | (table == []):
        return True
    else:
        disabled = False
        # * If any of the value is not filled, then cannot submit
        for i in [ei, ei_answer]:
            if (i is None) | (i == ""):
                print(f"Check failed - value not filled {i}")
                disabled = True

        # * If any evidence is not filled, then cannot submit
        values = [ei_answer]
        evidence = [ei_answer_evidence]
        for val, evi in zip(values, evidence):
            valid = check_evidence(val, evi)
            if not valid:
                print(f"Check failed - evidence not filled {val}")
                disabled = True

        # * If any of the validation didn't pass, then cannot submit
        specific_checks = {"ei_answer": valid_ei_answer_evidence}
        for name, vld in specific_checks.items():
            if not vld:
                print(f"Check failed - specific check for {name} failed ")
                disabled = True

    return disabled


# ! Display upon form submission
@callback(
    Output(f"{ID_PREFIX}-samples-table-container", "data", allow_duplicate=True),
    Output(f"{ID_PREFIX}-table-form-last-save", "children", allow_duplicate=True),
    Output(
        f"{ID_PREFIX}-exclusion-answer-duplicate-alert", "is_open", allow_duplicate=True
    ),
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
    ei,
    ei_answer,
    ei_evidence,
    ei_answer_evidence,
):
    if ctx.triggered_id == f"{ID_PREFIX}-form-submit":
        question_folder = os.path.join(data_store["labels_path"], ID_PREFIX, "")
        table_path = data_store["table_paths"][table_ind - 1]
        table_name = Path(table_path).stem
        table_id = int(re.search(r".+_table(\d+)", table_name).group(1))
        sample_df_path = os.path.join(question_folder, f"{table_name}_labels.jsonl")

        evidences = [ei_evidence, ei_answer_evidence]
        final_evidences = []
        for e in evidences:
            if e == "":
                es = None
            elif e is not None:
                e = e.split(", ")
                es = []
                for eei in e:
                    ee = eei.split("-")
                    try:
                        ee = [int(i) for i in ee]
                    except:
                        pass
                    es.append([table_id, ee])
            else:
                es = None
            final_evidences.append(es)
        # print(final_evidences)

        # Avoid adding the same sample again
        if samples_current != []:
            str_value_evidence = str(final_evidences[1])
            if str_value_evidence in [i["Answer Evidence"] for i in samples_current]:
                return no_update, no_update, True

        value_row = {c["id"]: r for c, r in zip(columns, [ei, ei_answer])}
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

        return samples_current, last_save, False
    else:
        raise PreventUpdate


#! Check if the evidence of 'Value' contains only 1 evidence
#! Check if the evidence of 'Value' has been recorded
@callback(
    Output(f"{ID_PREFIX}-{MAIN_EVIDENCE_ID}-input", "valid"),
    Output(f"{ID_PREFIX}-{MAIN_EVIDENCE_ID}-alert", "is_open"),
    Output(f"{ID_PREFIX}-{MAIN_EVIDENCE_ID}-alert", "children"),
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
    children = "Answer should only have 1 evidence. Please generate 1 sample per Answer evidence."
    if (evidence is not None) & (evidence != ""):
        if (value is not None) & (value != ""):
            # 1. check if we have more than 1 evidence
            num = len(evidence.split(", "))
            valid = False if num > 1 else True
            more_alert = False if valid else True
            if valid is False:
                print("Answer check - More than 1 evidence detected")

            # 2. check if we have duplicated samples
            labels_path = data_store["labels_path"]
            question_folder = os.path.join(labels_path, ID_PREFIX, "")
            table_name = Path(data_store["table_paths"][table_id - 1]).stem
            sample_df_path = os.path.join(question_folder, f"{table_name}_labels.jsonl")

            if os.path.exists(sample_df_path):
                with jsonlines.open(sample_df_path, "r") as jsonl_f:
                    value_evidence = []
                    for obj in jsonl_f:
                        ve = obj["Answer Evidence"]
                        if isinstance(ve, str):
                            ve = eval(ve)
                        value_evidence.append(ve)
                value_evidence = [i for j in value_evidence for i in j]
                value_evidence = [
                    "-".join([str(j) for j in i[1]]) for i in value_evidence
                ]

                valid = False if evidence in value_evidence else valid
                duplicate_alert = True if evidence in value_evidence else False
                if not valid:
                    print("Answer check - duplicated samples detected")
            else:
                duplicate_alert = False

            return valid, more_alert, children, duplicate_alert
    return True, False, False, no_update
