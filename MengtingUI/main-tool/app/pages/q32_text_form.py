import os
import time
from uuid import uuid4
import dash
from dash.exceptions import PreventUpdate
from dash import html, Input, Output, callback, ctx, State
import dash_bootstrap_components as dbc
from src.text_form_utils import create_layout, create_callbacks
from src.inputs import q32_inputs as inputs
import jsonlines
from src.commons import get_page_num

ID_PREFIX = "q32"

# Register the page
dash.register_page(__name__, path=f"/{ID_PREFIX}_text_form")

# * --------------------------------- Create Layout -----------------------------------
common_layout, samples_table, modal = create_layout(
    id_prefix=ID_PREFIX, inputs=inputs, n_col=1
)

layout = dbc.Container(
    common_layout + [samples_table, modal],
    fluid=True,
)


# * ---------------------------------------------------------------------------------------
# *                                     Callbacks
# * ---------------------------------------------------------------------------------------
create_callbacks(
    id_prefix=ID_PREFIX,
    inputs=inputs,
)


# ! Enable the form submit button
@callback(
    Output(f"{ID_PREFIX}-text-form-submit", "disabled"),
    [
        Input(f"{ID_PREFIX}-{i['id']}-text-input", "value")
        for i in inputs
        if i["required"] == True
    ],
    Input(f"{ID_PREFIX}-text-evidence-input", "children"),
)
def disable_submit(ei, ei_answer, text_evidence):
    print("----------------------------------------")

    disabled = False
    # * If any of the value is not filled, then cannot submit
    for i in [ei, ei_answer]:
        if (i is None) | (i == ""):
            print(f"Check failed - value not filled {i}")
            disabled = True

    # * If evidence is not filled, then cannot submit
    if (text_evidence is None) | (text_evidence == "") | (text_evidence == []):
        print(f"Check failed - evidence not filled")
        disabled = True

    return disabled


# * -------------------------------------- Recorded Samples ------------------------------------------
# Display samples
@callback(
    Output(f"{ID_PREFIX}-samples-text-container", "data"),
    Output(f"{ID_PREFIX}-text-form-last-save", "children"),
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
    ei,
    ei_answer,
    report_data,
):
    if data_store!=[]:
        # Get samples path
        question_folder = os.path.join(data_store["labels_path"], ID_PREFIX, "")
        sample_df_path = os.path.join(question_folder, "text_labels.jsonl")

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
            print()

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
                    [ei, ei_answer],
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
            print("Row was deleted")
            print("These are the current rows:")
            samples_cleaned = []
            for sample in samples_current:
                values = {k: v for k, v in sample.items()}
                print(values["Text Evidence"])
                values["Text Evidence"] = eval(values["Text Evidence"])
                values["Bbox Evidence"] = eval(values["Bbox Evidence"])
                samples_cleaned.append(values)
            print()

            with jsonlines.open(sample_df_path, mode="w") as writer:
                writer.write_all(samples_cleaned)
                last_save = time.ctime(os.path.getmtime(sample_df_path))

            saved_samples = samples_current

        return saved_samples, f"Last saved: {last_save}"
    raise dash.exceptions.PreventUpdate
