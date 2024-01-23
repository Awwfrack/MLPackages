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
import re


# Opening JSON file
keywords_path = os.path.join(os.path.dirname(__file__), "keywords.json")
keywords_file = open(keywords_path)
keywords_dict = json.load(keywords_file)
keywords_file.close()


# * ---------------------------------------------------------------------------------------
# *                                     Layout
# * ---------------------------------------------------------------------------------------


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
            id=f"{id_prefix}-{id}-text-input",
        )
    elif comp_type == "input":
        user_input = dbc.Input(
            **comp_kwargs,
            required=required,
            autoFocus=False,
            persistence=True,
            persistence_type="session",
            id=f"{id_prefix}-{id}-text-input",
        )
    elif comp_type == "text-area":
        user_input = dbc.Textarea(
            required=required,
            autoFocus=False,
            persistence=True,
            persistence_type="session",
            id=f"{id_prefix}-{id}-text-input",
        )

    input_group = dbc.InputGroup(
        [
            dbc.InputGroupText(
                name, id=f"{id_prefix}-text-{id}-name", style={"width": 230}
            ),
            user_input,
            dbc.Tooltip(tooltip, target=f"{id_prefix}-text-{id}-name", placement="top"),
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

    evidence_inputs = dbc.Card(
        [
            dbc.InputGroup(
                [
                    dbc.InputGroupText(
                        "Text Evidence",
                        id=f"{id_prefix}-text-evidence-name",
                        style={"width": "85%", "background-color": "secondary"},
                    ),
                    dbc.Tooltip(
                        "Select all text evidences needed to support the above facts.",
                        target=f"{id_prefix}-text-evidence-name",
                        placement="top",
                    ),
                    dbc.Button(
                        "Clear",
                        color="primary",
                        outline=True,
                        id=f"{id_prefix}-text-form-evidence-clear",
                        n_clicks=0,
                        style={"width": "15%"},
                    ),
                ]
            ),
            html.Br(),
            html.Ul(
                id=f"{id_prefix}-text-evidence-input",
                children=[],
                style={"width": "100%"},
            ),
        ],
        className="mb-3",
    )

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
                            id=f"{id_prefix}-text-form-clear",
                            n_clicks=0,
                            className="w-50",
                        ),
                        dbc.Button(
                            "Submit",
                            color="primary",
                            id=f"{id_prefix}-text-form-submit",
                            n_clicks=0,
                            className="w-50",
                        ),
                    ],
                    className="gap-2 d-md-flex",
                )
            )
        ],
        id=f"{id_prefix}-text-form",
        className="g-3",
    )

    return form


def create_layout(id_prefix: str, inputs: List[dict], n_col: int = 1):
    
    # * -------------------------------------- Text Carousel --------------------------------------
    text_carousel = dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(dcc.Markdown(id=f"{id_prefix}-text-form-carousel-page-number")),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [html.H5("PDF View")],
                                id=f"{id_prefix}-text-form-page-view",
                            )
                        ),
                        dbc.Col(
                            [
                                html.H5("Extracted Sentences"),
                                dbc.Card(
                                    id=f"{id_prefix}-text-form-carousel-sentences-form"
                                ),
                            ],
                            style={"width": "50%"},
                        ),
                    ]
                ),
            ]
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
        "Bbox Evidence",
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
                        id=f"{id_prefix}-samples-text-container",
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
                        className="text-center", id=f"{id_prefix}-text-form-last-save"
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
            dbc.ModalBody(id=f"{id_prefix}-text-form-modal-body"),
            dbc.ModalFooter(
                dbc.Button(
                    "Edit",
                    id=f"{id_prefix}-text-form-modal-edit",
                    className="ms-auto",
                    n_clicks=0,
                )
            ),
        ],
        size="xl",
        scrollable=True,
        is_open=False,
        fullscreen=True,
        id=f"{id_prefix}-text-form-modal",
    )

    edit_alert_modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Editting Disabled")),
            dbc.ModalBody(id=f"{id_prefix}-text-form-alert-modal-body"),
        ],
        is_open=False,
        id=f"{id_prefix}-text-form-alert-modal",
    )

    # * ----------------------------------------- Manual Check ----------------------------------------
    manual_check_button = dbc.Button(
        "Extraction Text Check",
        id=f"{id_prefix}-text-form-extraction-text-check-button",
        color="secondary",
        n_clicks=0,
    )
    manual_check_button_tooltip = dbc.Tooltip(
        "View all extracted text from PDF to validate if the extraction engine is working properly.", 
        target=f"{id_prefix}-text-form-extraction-text-check-button", 
        placement="top"
    )
    manual_check_modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Extracted Text Check")),
            dbc.Alert(
                "There is no extracted text. This means the pdf is not searchable. Please continue to manual tagging page.",
                color="danger", 
                id=f"{id_prefix}-text-form-no-extracted-text-alert", 
                dismissable=True, is_open=False
            ),
            dbc.ModalBody(id=f"{id_prefix}-text-form-extracted-text-modal-body"),
            dbc.ModalFooter(
                dbc.Button(
                    "Manual Text Form",
                    color="primary",
                    class_name="ms-auto",
                    id=f"{id_prefix}-text-form-link-to-manual-form",
                    href=f"/{id_prefix}_manual_text_form"
                )
            )
        ],
        size="xl",
        scrollable=True,
        is_open=False,
        fullscreen=False,
        id=f"{id_prefix}-text-form-extracted-text-modal",
    )

    # * ----------------------------------------- Layout ----------------------------------------
    alert = dbc.Alert(
        "There is no text to label. Please click the Extraction Text Check to check if extraction engine is working properly.",
        color="info", 
        id=f"{id_prefix}-text-form-notext-alert", 
        dismissable=True, is_open=False
    )
    layout_header = dbc.Row(
        [
            dbc.Col(
                html.H1(f"{id_prefix.upper()} Text Evidence")
            ),
            dbc.Col(
                [manual_check_button,manual_check_modal, manual_check_button_tooltip],
                width=3,
                className="d-grid justify-content-md-end",
            ),
        ]
    )

    layout = [
        alert,
        html.Br(),
        layout_header,
        html.Br(),
        dbc.Progress(id=f"{id_prefix}-text-form-progress", striped=True),
        html.Br(),
        dbc.Pagination(
            id=f"{id_prefix}-text-form-pagination",
            max_value=1,
            fully_expanded=False,
            previous_next=True,
            active_page=1,
            className="justify-content-center",
        ),
        text_carousel,
        html.Br(),
        form_card,
        html.Br(),
        dmc.Divider(variant="dotted"),
        html.Br(),
        dcc.Store(
            id=f"{id_prefix}-text-form-report-extracted-data-store",
            data=[],
            storage_type="session",
        ),
        dcc.Store(
            id=f"{id_prefix}-text-form-random-report-extracted-data-store",
            data=[],
            storage_type="session",
        ),
        edit_alert_modal
    ]

    return layout, samples_table, modal


# * ---------------------------------------------------------------------------------------
# *                                     Callbacks
# * ---------------------------------------------------------------------------------------


def create_callbacks(id_prefix: str, inputs: List[dict]):

    # * -------------------------------------- Extracted Text ------------------------------------------
    # Get filtered data from report
    @callback(
        Output(f"{id_prefix}-text-form-report-extracted-data-store", "data"),
        Output(f"{id_prefix}-text-form-pagination", "max_value"),
        Output(f"{id_prefix}-text-form-notext-alert", "is_open"),
        Input("data-store", "data"),
    )
    def get_text_data(data_store):
        if (data_store is None) | (data_store == []):
            return [], 1, True
        else:
            report_data_path = data_store["report_data_path"]
            with jsonlines.open(report_data_path, "r") as jsonl_f:
                report_data = [obj for obj in jsonl_f]

            # Get block sentences
            all_block_sentences = []
            for i, block_info in enumerate(report_data):
                block_sentences = split_sentence(block_info["text"])
                for sentence in block_sentences:
                    if not sentence.startswith("<image:"):
                        all_block_sentences.append(
                            {
                                "text": sentence,
                                "bbox": block_info["bbox"],
                                "page_num": block_info["page_num"],
                            }
                        )

            # filter at sentence level
            filtered_data = {}
            for i, sentence_info in enumerate(all_block_sentences):
                sentence_page = sentence_info["page_num"]
                if filter_text_for_keywords(
                    sentence_info["text"], keywords_dict[id_prefix]
                ):
                    if id_prefix=="q31":
                        if re.search("\d+", sentence_info["text"]):
                            if i != 0:
                                previous_sentence = all_block_sentences[i - 1]
                            else:
                                previous_sentence = {"text": "", "bbox": [], "page_num": None}

                            if i < len(all_block_sentences) - 1:
                                next_sentence = all_block_sentences[i + 1]
                            else:
                                next_sentence = {"text": "", "bbox": [], "page_num": None}

                            if sentence_page not in filtered_data:
                                filtered_data[sentence_page] = []

                            filtered_data[sentence_page].append(
                                {
                                    **sentence_info,
                                    "previous": previous_sentence,
                                    "next": next_sentence,
                                }
                            )
                    else:
                        if i != 0:
                            previous_sentence = all_block_sentences[i - 1]
                        else:
                            previous_sentence = {"text": "", "bbox": [], "page_num": None}

                        if i < len(all_block_sentences) - 1:
                            next_sentence = all_block_sentences[i + 1]
                        else:
                            next_sentence = {"text": "", "bbox": [], "page_num": None}

                        if sentence_page not in filtered_data:
                            filtered_data[sentence_page] = []

                        filtered_data[sentence_page].append(
                            {
                                **sentence_info,
                                "previous": previous_sentence,
                                "next": next_sentence,
                            }
                        )


            filtered_data = list(filtered_data.items())

            if len(filtered_data) == 0:
                show_alert = True
            else:
                show_alert = False

            print(len(filtered_data))
            return filtered_data, len(filtered_data), show_alert

    # Get the caroussel text
    @callback(
        Output(f"{id_prefix}-text-form-carousel-page-number", "children"),
        Output(f"{id_prefix}-text-form-carousel-sentences-form", "children"),
        Output(f"{id_prefix}-text-form-page-view", "children"),
        Input(f"{id_prefix}-text-form-pagination", "active_page"),
        Input(f"{id_prefix}-text-form-report-extracted-data-store", "data"),
        State("data-store", "data"),
    )
    @cache.memoize(3600)
    def update_carousel_pages_text(active_page, report_data, data_store):
        if active_page and report_data != []:
            page_num = report_data[active_page - 1][0]
            page_num_sentence = f"**Page Number**: {page_num}"
            page_sentences = report_data[active_page - 1][1]

            sentences_list = []
            for i, s in enumerate(page_sentences):
                button = dbc.Button(
                    "Select",
                    color="primary",
                    id={
                        "type": f"{id_prefix}-text-select-sentence-evidence",
                        "index": i,
                    },
                    n_clicks=0,
                )

                sentence_row = []
                if len(s["previous"]) != 0:
                    previous_button = dbc.Button(
                        s["previous"]["text"],
                        color="secondary",
                        id={
                            "type": f"{id_prefix}-text-select-previous-sentence-evidence",
                            "index": i,
                        },
                        n_clicks=0,
                        class_name="mb-2 mt-1",
                        size="sm",
                        disabled=True,
                    )
                    sentence_row.append(dbc.Row(previous_button))

                sentence_row.append(
                    dbc.Row(
                        [
                            dbc.Col(
                                html.P(
                                    s["text"],
                                    id={
                                        "type": f"{id_prefix}-text-form-carousel-sentence",
                                        "index": i,
                                    },
                                )
                            ),
                            dbc.Col(
                                button,
                                width=2,
                                className="d-grid justify-content-md-end",
                            ),
                        ]
                    )
                )

                if len(s["next"]) != 0:
                    next_button = dbc.Button(
                        s["next"]["text"],
                        color="secondary",
                        id={
                            "type": f"{id_prefix}-text-select-next-sentence-evidence",
                            "index": i,
                        },
                        n_clicks=0,
                        class_name="mt-2 mb-1",
                        size="sm",
                        disabled=True,
                    )
                    sentence_row.append(dbc.Row(next_button))

                sentences_list.append(dbc.ListGroupItem(sentence_row))

            sentences_layout = dbc.ListGroup(sentences_list, flush=True)

            bboxes = []
            for s in page_sentences:
                bboxes.append(s["bbox"])

            report_path = os.path.join(
                data_store["local_store"],
                data_store["company"],
                f"{data_store['report']}.pdf",
            )
            page_image = display_pdf_page(report_path, page_num, bboxes, dpi=200)
            page_image = DashCanvas(
                image_content=page_image,
                tool="select",
                hide_buttons=["pan", "line", "pencil", "rectangle", "undo", "select"],
                goButtonTitle=" ",
                width=600,
            )

            div_body = [
                html.H5("PDF View"),
                page_image,
            ]

            return page_num_sentence, sentences_layout, div_body
        else:
            return "", [], []

    # ! Display the progress
    @callback(
        Output(f"{id_prefix}-text-form-progress", "value"),
        Input(f"{id_prefix}-text-form-pagination", "active_page"),
        Input(f"{id_prefix}-text-form-pagination", "max_value"),
    )
    def display_progress(active_page, max_value):
        if active_page is not None:
            pct = (active_page / max_value) * 100
            return pct
        else:
            return 0

    # Disable previous and next button
    @callback(
        Output(
            {
                "type": f"{id_prefix}-text-select-previous-sentence-evidence",
                "index": MATCH,
            },
            "disabled",
            allow_duplicate=True,
        ),
        Output(
            {"type": f"{id_prefix}-text-select-next-sentence-evidence", "index": MATCH},
            "disabled",
            allow_duplicate=True,
        ),
        Input(
            {"type": f"{id_prefix}-text-select-sentence-evidence", "index": MATCH},
            "n_clicks",
        ),
        prevent_initial_call=True,
    )
    def enable_previous_next(sent_clicked):
        if ctx.triggered[0]["prop_id"] == ".":
            raise PreventUpdate
        try:
            button_index = eval(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
            button_id = eval(ctx.triggered[0]["prop_id"].split(".")[0])["type"]

            if button_id == f"{id_prefix}-text-select-sentence-evidence":
                if sent_clicked == 0:
                    return True, True
                else:
                    return False, False

        except:
            raise PreventUpdate

    # Select sentence evidence to fill Text Evidence
    @callback(
        Output(f"{id_prefix}-text-evidence-input", "children", allow_duplicate=True),
        Input(
            {"type": f"{id_prefix}-text-select-sentence-evidence", "index": ALL},
            "n_clicks",
        ),
        Input(
            {
                "type": f"{id_prefix}-text-select-previous-sentence-evidence",
                "index": ALL,
            },
            "n_clicks",
        ),
        Input(
            {"type": f"{id_prefix}-text-select-next-sentence-evidence", "index": ALL},
            "n_clicks",
        ),
        State(
            {"type": f"{id_prefix}-text-form-carousel-sentence", "index": ALL},
            "children",
        ),
        State(
            {
                "type": f"{id_prefix}-text-select-previous-sentence-evidence",
                "index": ALL,
            },
            "children",
        ),
        State(
            {"type": f"{id_prefix}-text-select-next-sentence-evidence", "index": ALL},
            "children",
        ),
        State(f"{id_prefix}-text-evidence-input", "children"),
        prevent_initial_call=True,
    )
    def auto_fill_text_evidence(
        sent_clicked,
        previous_clicked,
        next_clicked,
        sent_value,
        previous_value,
        next_value,
        text_evidence_list,
    ):
        print(text_evidence_list)
        already_selected_texts = []
        for i in text_evidence_list:
            already_selected_texts.append(i["props"]["children"])

        if ctx.triggered[0]["prop_id"] == ".":
            raise PreventUpdate
        try:
            button_index = eval(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
            button_id = eval(ctx.triggered[0]["prop_id"].split(".")[0])["type"]

            if button_id == f"{id_prefix}-text-select-sentence-evidence":
                text_evidence = sent_value[button_index]
                if not text_evidence in already_selected_texts:
                    text_evidence_list.append(html.Li(text_evidence))

            if button_id == f"{id_prefix}-text-select-previous-sentence-evidence":
                actual_value = sent_value[button_index]
                previous_text_evidence = previous_value[button_index]

                text_evidence_list = []
                for i in already_selected_texts:
                    if i == actual_value and (
                        not previous_text_evidence in already_selected_texts
                    ):
                        text_evidence_list.append(html.Li(previous_text_evidence))
                    text_evidence_list.append(html.Li(i))

            if button_id == f"{id_prefix}-text-select-next-sentence-evidence":
                actual_value = sent_value[button_index]
                next_text_evidence = next_value[button_index]

                text_evidence_list = []
                for i in already_selected_texts:
                    text_evidence_list.append(html.Li(i))
                    if i == actual_value and (
                        not next_text_evidence in already_selected_texts
                    ):
                        text_evidence_list.append(html.Li(next_text_evidence))

        except:
            raise PreventUpdate

        return text_evidence_list

    # * ----------------------------------------- Manual Check ----------------------------------------
    @cache.memoize(3600)
    def get_random_blocks(report_data):
        # Eliminate images from extracted text
        no_image_texts = []
        for i in report_data:
            if not i["text"].startswith("<image:"):
                no_image_texts.append(i)

        return random.sample(no_image_texts, 20)

    @callback(
        Output(f"{id_prefix}-text-form-extracted-text-modal-body", "children"),
        Output(f"{id_prefix}-text-form-extracted-text-modal", "is_open"),
        Output(f"{id_prefix}-text-form-no-extracted-text-alert", "is_open"),
        Output(f"{id_prefix}-text-form-random-report-extracted-data-store", "data"),
        Input(f"{id_prefix}-text-form-extraction-text-check-button", "n_clicks"),
        State("data-store", "data"),
    )
    def open_extracted_text_check(n_clicks, data_store):
        if ctx.triggered_id == f"{id_prefix}-text-form-extraction-text-check-button":
            if (data_store is None) | (data_store == []):
                print("data store is empty")
                print(data_store)
                return [], False, False, []
            else:
                report_data_path = data_store["report_data_path"]
                with jsonlines.open(report_data_path, "r") as jsonl_f:
                    report_data = [obj for obj in jsonl_f]

                if report_data == []:
                    print("Report data is empty")
                    return [], True, True
                
                random_extracted_texts = get_random_blocks(report_data)
                
                pagination = dbc.Pagination(
                    id=f"{id_prefix}-text-form-extracted-text-pagination",
                    max_value=len(random_extracted_texts),
                    fully_expanded=False,
                    previous_next=True,
                    active_page=1,
                    className="justify-content-center",
                )

                
                page_image = html.Img(
                    id=f"{id_prefix}-text-form-extracted-block-page-image",
                    style={"width": "100%", "align":"left"}
                )
                
                modal_body = [
                        dcc.Markdown("Please check that the extracted text (to the right) matches the text highlighted in green in the PDF page (to the left). If it looks good, you may close this modal. If not, continue by clicking in the **Manual Text Form** button."),
                        dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H5("Randomly Selected Extracted Blocks"),
                                    dbc.Row(pagination),
                                    dbc.Row(dcc.Markdown(id=f"{id_prefix}-text-form-extracted-text-page-num")),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                page_image, 
                                            ),
                                            dbc.Col(
                                                dbc.Card(
                                                    id=f"{id_prefix}-text-form-extracted-block-card"
                                                )
                                            )
                                        ]
                                        
                                    )
                                ]
                            )
                        ]
                    )
                ]
                
                return modal_body, True, False, random_extracted_texts
            
        else:
            raise PreventUpdate


    @callback(
        Output(f"{id_prefix}-text-form-extracted-block-card", "children"),
        Output(f"{id_prefix}-text-form-extracted-block-page-image", "src"),
        Output(f"{id_prefix}-text-form-extracted-text-page-num", "children"),
        Input(f"{id_prefix}-text-form-extracted-text-pagination", "active_page"),
        State("data-store", "data"),
        State(f"{id_prefix}-text-form-random-report-extracted-data-store", "data"),
    )
    def update_extracted_text_carousel(active_page, data_store, report_data):
        if active_page:

            report_path = os.path.join(
                data_store["local_store"],
                data_store["company"],
                f"{data_store['report']}.pdf",
            )

            block_info = report_data[active_page-1]

            page_num = block_info["page_num"]
            
            
            t1 = time.time()
            page_image = display_pdf_page(report_path, page_num, block_info["bbox"], dpi=200)
            t2 = time.time()
            print("Get page image:", t2-t1)
            block_text = block_info["text"]

            return dbc.CardBody(block_text), page_image, page_num
        
        else:
            raise PreventUpdate
    

    # * -------------------------------------- Form ------------------------------------------
    # ! Clear form
    @callback(
        [
            Output(f"{id_prefix}-{i['id']}-text-input", "value", allow_duplicate=True)
            for i in inputs
        ]
        + [Output(f"{id_prefix}-text-evidence-input", "children", allow_duplicate=True)]
        + [
            Output(
                {
                    "type": f"{id_prefix}-text-select-previous-sentence-evidence",
                    "index": ALL,
                },
                "disabled",
                allow_duplicate=True,
            ),
            Output(
                {
                    "type": f"{id_prefix}-text-select-next-sentence-evidence",
                    "index": ALL,
                },
                "disabled",
                allow_duplicate=True,
            ),
        ],
        Input(f"{id_prefix}-text-form-clear", "n_clicks"),
        Input(f"{id_prefix}-text-form-submit", "n_clicks"),
        State(
            {
                "type": f"{id_prefix}-text-select-previous-sentence-evidence",
                "index": ALL,
            },
            "disabled",
        ),
        State(
            {"type": f"{id_prefix}-text-select-next-sentence-evidence", "index": ALL},
            "disabled",
        ),
        prevent_initial_call=True,
    )
    def clear_form(n_clicks, submit, previous_disabled, next_disabled):
        if ctx.triggered_id in [f"{id_prefix}-text-form-clear", f"{id_prefix}-text-form-submit"]:
            return (
                [None] * (len(inputs)) + [[]]
                + [[True for i in range(len(previous_disabled))]]
                + [[True for i in range(len(next_disabled))]]
            )
        else:
            raise PreventUpdate

    @callback(
        Output(f"{id_prefix}-text-evidence-input", "children", allow_duplicate=True),
        Output(
            {
                "type": f"{id_prefix}-text-select-previous-sentence-evidence",
                "index": ALL,
            },
            "disabled",
            allow_duplicate=True,
        ),
        Output(
            {"type": f"{id_prefix}-text-select-next-sentence-evidence", "index": ALL},
            "disabled",
            allow_duplicate=True,
        ),
        Input(f"{id_prefix}-text-form-evidence-clear", "n_clicks"),
        State(
            {
                "type": f"{id_prefix}-text-select-previous-sentence-evidence",
                "index": ALL,
            },
            "disabled",
        ),
        State(
            {"type": f"{id_prefix}-text-select-next-sentence-evidence", "index": ALL},
            "disabled",
        ),
        prevent_initial_call=True,
    )
    def clear_text_evidence(clear, previous_disabled, next_disabled):
        if ctx.triggered_id  == f"{id_prefix}-text-form-evidence-clear":
            return (
                [],
                [True for i in range(len(previous_disabled))],
                [True for i in range(len(next_disabled))],
            )

    # * ---------------------------------- Sample Modal ---------------------------------------
    # ! Display recorded sample details
    @callback(
        Output(f"{id_prefix}-text-form-modal-body", "children"),
        Output(f"{id_prefix}-text-form-modal", "is_open"),
        Input(f"{id_prefix}-samples-text-container", "active_cell"),
        State(f"{id_prefix}-samples-text-container", "derived_viewport_data"),
        State("data-store", "data"),
    )
    def display_modal(active_cell, table_data, data_store):
        if active_cell:
            row_id = active_cell["row"]
            print(table_data[row_id])
            row = table_data[row_id]

            # Load sample data
            question_folder = os.path.join(data_store["labels_path"], id_prefix, "")
            sample_df_path = os.path.join(question_folder, "text_labels.jsonl")

            # Load report path
            report_path = os.path.join(
                data_store["local_store"],
                data_store["company"],
                f"{data_store['report']}.pdf",
            )


            page_num = row["Page Number Evidence"]
            bbox = eval(row["Bbox Evidence"])

            page_image = display_pdf_page(report_path, page_num, bbox, dpi=200)
            page_image = html.Img(
                src=page_image,
                style={"height": "100%", "width": "100%", "align": "left"},
            )

            # Get filled form
            form_with_answers = []
            for i in inputs:
                name = i["name"]
                row_input_group = dbc.InputGroup(
                    [
                        dbc.InputGroupText(name, style={"width": 180}),
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

            modal_form = dbc.Form(form_with_answers)

            text_evidence = eval(row["Text Evidence"])
            modal_text_evidence = html.Ul([html.Li(i) for i in text_evidence])

            modal_body = html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div(f"Page Number: {page_num}"),
                                    page_image,
                                ]
                            ),
                            dbc.Col(
                                [
                                    html.H5("Recorded Sample", className="text-center"),
                                    modal_form,
                                    html.Br(),
                                    html.H5("Text Evidence", className="text-center"),
                                    dbc.Card(modal_text_evidence, body=True),
                                ],
                                width=4,
                            ),
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
                Output(f"{id_prefix}-{i['id']}-text-input", "value", allow_duplicate=True)
                for i in inputs
            ] +[
                Output(f"{id_prefix}-text-evidence-input", "children", allow_duplicate=True)
            ] +
            [
                Output(f"{id_prefix}-text-form-modal", "is_open", allow_duplicate=True),
                Output(f"{id_prefix}-text-form-alert-modal", "is_open", allow_duplicate=True)
            ] + 
            [
                Output(f"{id_prefix}-samples-text-container", "data", allow_duplicate=True),
                Output(f"{id_prefix}-text-form-last-save", "children", allow_duplicate=True)
            ] + [
                Output(f"{id_prefix}-text-form-alert-modal-body", "children", allow_duplicate=True)
            ] + [
                Output(f"{id_prefix}-text-form-pagination", "active_page", allow_duplicate=True),
            ],
            Input(f"{id_prefix}-text-form-modal-edit", "n_clicks"),
            State(f"{id_prefix}-samples-text-container", "active_cell"),
            State("data-store", "data"),
            State(f"{id_prefix}-samples-text-container", "derived_viewport_data"),
            State(f"{id_prefix}-text-form-report-extracted-data-store", "data"),
            prevent_initial_call=True
        )
        def edit_sample(n_clicks, active_cell, data_store, viewport_data, report_data):
            if ctx.triggered_id == f"{id_prefix}-text-form-modal-edit":
                print("EDITTING")
                question_folder = os.path.join(data_store["labels_path"], id_prefix, "")
                sample_df_path = os.path.join(question_folder, "text_labels.jsonl")

                row_id = active_cell["row"]
                print(active_cell)
                print(viewport_data[row_id])

                # * Load the sample data
                row = viewport_data[row_id]
                
                # CHECK IF PROGRAM WITH SAMPLE EXISTS:
                programs_df = os.path.join(
                    question_folder, "text_programs_labels.jsonl"
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
                    form_evidences = [None]

                    last_save = os.path.getmtime(sample_df_path)

                else:
                    row = viewport_data.pop(row_id)
                    open_alert_modal = False
                    modal_body = []

                    form_values = [row[i["name"]] for i in inputs]
                    form_evidences = [[html.Li(i) for i in eval(row["Text Evidence"])]]

                    print(form_values)
                    print(form_evidences)

                    # Delete sample
                    samples = []
                    for sample in viewport_data:
                        values = {k: v for k, v in sample.items()}
                        values["Text Evidence"] = eval(values["Text Evidence"])
                        values["Bbox Evidence"] = eval(values["Bbox Evidence"])
                        samples.append(values)

                    with jsonlines.open(sample_df_path, mode="w") as writer:
                        writer.write_all(samples)
                    last_save = time.ctime(os.path.getmtime(sample_df_path))

                # Find active page
                page_num_evidence = row["Page Number Evidence"]
                for i, page_data in enumerate(report_data):
                    if page_data[0] == page_num_evidence:
                        active_page = i+1
                        break

                return form_values + form_evidences + [False, open_alert_modal] + [viewport_data, last_save] + [modal_body] + [active_page]
            
            else:
                PreventUpdate


    else:
        @callback(
            [
                Output(f"{id_prefix}-{i['id']}-text-input", "value", allow_duplicate=True)
                for i in inputs
            ] +[
                Output(f"{id_prefix}-text-evidence-input", "children", allow_duplicate=True)
            ] +
            [
                Output(f"{id_prefix}-text-form-modal", "is_open", allow_duplicate=True),
            ] + 
            [
                Output(f"{id_prefix}-samples-text-container", "data", allow_duplicate=True),
                Output(f"{id_prefix}-text-form-last-save", "children", allow_duplicate=True)
            ] + [
                Output(f"{id_prefix}-text-form-pagination", "active_page", allow_duplicate=True),
            ],
            Input(f"{id_prefix}-text-form-modal-edit", "n_clicks"),
            State(f"{id_prefix}-samples-text-container", "active_cell"),
            State("data-store", "data"),
            State(f"{id_prefix}-samples-text-container", "derived_viewport_data"),
            State(f"{id_prefix}-text-form-report-extracted-data-store", "data"),
            prevent_initial_call=True
        )
        def edit_sample(n_clicks, active_cell, data_store, viewport_data, report_data):
            if ctx.triggered_id == f"{id_prefix}-text-form-modal-edit":
                print("EDITTING")
                question_folder = os.path.join(data_store["labels_path"], id_prefix, "")
                sample_df_path = os.path.join(question_folder, "text_labels.jsonl")

                row_id = active_cell["row"]

                # * Load the sample data
            
                row = viewport_data.pop(row_id)

                form_values = [row[i["name"]] for i in inputs]
                form_evidences = [[html.Li(i) for i in eval(row["Text Evidence"])]]

                print(form_values)
                print(form_evidences)

                # Delete sample
                samples = []
                for sample in viewport_data:
                    values = {k: v for k, v in sample.items()}
                    values["Text Evidence"] = eval(values["Text Evidence"])
                    values["Bbox Evidence"] = eval(values["Bbox Evidence"])
                    samples.append(values)

                with jsonlines.open(sample_df_path, mode="w") as writer:
                    writer.write_all(samples)
                last_save = time.ctime(os.path.getmtime(sample_df_path))

                # Find active page
                page_num_evidence = row["Page Number Evidence"]
                for i, page_data in enumerate(report_data):
                    if page_data[0] == page_num_evidence:
                        active_page = i+1
                        break

                return form_values + form_evidences + [False] + [viewport_data, last_save] + [active_page]
            
            else:
                PreventUpdate





