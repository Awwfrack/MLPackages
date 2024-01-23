"""Common Functions"""

import os
import io
import fitz
from PIL import Image
from typing import Optional, Union
from google.cloud import storage
import re
import dash_bootstrap_components as dbc
import numpy as np
from dash import html
import json
from unidecode import unidecode
import string
from nltk import tokenize

storage_client = storage.Client()


def open_pdf(filepath: str) -> fitz.Document:
    """
    The open_pdf function takes a filepath as input and returns a fitz.Document object.
    If the filepath is in Google Cloud Storage, it will download the PDF to memory first before opening it with PyMuPDF.

    :param filepath: str: Specify the path to the pdf file
    :return: A fitz Document
    """
    # if it is in bucket, load to memory first, then open
    if filepath.startswith("gs://"):
        bucket_name = filepath.split("/")[2]
        fpath = os.path.join(*filepath.split("/")[3:])
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.get_blob(fpath)
        doc = fitz.open("pdf", blob.download_as_string())
    # otherwise, open from local
    else:
        doc = fitz.open(filepath)

    return doc


def extract_image(
    pdf_path: str, page_num: int, dpi: Optional[int] = 200
) -> Union[Image.Image, fitz.Document]:
    """
    The extract_image function takes a PDF file and returns an image of the page.

    :param pdf_path:str: Specify the path to a pdf file
    :param page_num:int: Specify the page number of the pdf to extract
    :param dpi:Optional[int]: Set the resolution of the image
    :return: A tuple of the image and the document
    """
    doc = open_pdf(pdf_path)

    page = doc[page_num - 1]
    if "dpi" in page.get_pixmap.__code__.co_varnames:
        pix = page.get_pixmap(dpi=dpi)
    else:
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    mode = "RGBA" if pix.alpha else "RGB"
    image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    return image, doc


def save_json(output_dict: dict, output_dir: str) -> None:
    """
    The save_json function takes a dictionary and saves it as a json file.
    The function is compatible with both GCP Cloud Storage and local saving.

    :param output_dict: dict: Specify the dictionary that will be outputted
    :param output_dir: str: Specify the output directory
    """
    if output_dir.startswith("gs://"):
        bucket_name = output_dir.split("/")[2]
        bucket = storage_client.get_bucket(bucket_name)

        fpath = os.path.join(*output_dir.split("/")[3:])

        blob = bucket.blob(fpath)
        blob.upload_from_string(
            data=json.dumps(output_dict), content_type="application/json"
        )
    else:
        dir_ = os.path.dirname(output_dir)
        if not os.path.exists(dir_):
            os.makedirs(dir_)

        with open(output_dir, "w") as f:
            json.dump(output_dict, f)


def glob_re(folder, pattern=r".+_table\d+.*.json"):
    paths = []
    for p in os.listdir(folder):
        path = re.fullmatch(pattern, p)
        if path is not None:
            paths.append(os.path.join(folder, path[0]))
    return paths


class Box:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def to_array(self):
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def apply_scale(self, scale):
        return Box(
            self.xmin * scale[0],
            self.ymin * scale[1],
            self.xmax * scale[0],
            self.ymax * scale[1],
        )


def create_form(user_inputs: list, n_col: int, id_prefix: str):
    if n_col > len(user_inputs):
        raise ValueError(
            f"n_col must be smaller than the number of user_inputs which is {len(user_inputs)}"
        )
    m = np.arange(len(user_inputs)).reshape(-1, n_col)

    rows = []
    for irow in m:
        row = dbc.Row([dbc.Col(user_inputs[icol]) for icol in irow], className="g-3")
        rows.append(row)

    form = dbc.Form(
        rows
        + [
            dbc.Row(
                html.Div(
                    [
                        dbc.Button(
                            "Clear",
                            color="primary",
                            outline=True,
                            id=f"{id_prefix}-form-clear",
                            n_clicks=0,
                            className="w-50",
                        ),
                        dbc.Button(
                            "Submit",
                            color="primary",
                            id=f"{id_prefix}-form-submit",
                            n_clicks=0,
                            className="w-50",
                        ),
                    ],
                    className="gap-2 d-md-flex",
                )
            )
        ],
        id=f"{id_prefix}-table-form",
    )

    return form


def remove_punctuation(text):
    # Unidecode
    text = unidecode(text)

    # Remove line breaks
    for line_break in [r"\n\n\n", r"\n\n", r"\n"]:
        text = re.sub(line_break, " ", text)

    # Remove puncuations except .
    translator = str.maketrans(
        "", "", (string.punctuation + "•").replace(".", "").replace("-", "")
    )
    text = text.translate(translator)

    # Remove . but not when it represents decimals in numbers (e.g., 3.25)
    regex = r"(?<!\d)\.|\.(?!\d)"
    text = re.sub(regex, "", text, 0)

    # Remove - but not when it connects two words (e.g., esg-related)
    regex = r"-(?!\w)|(?<!\w)-"
    text = re.sub(regex, "", text, 0)

    return text


def filter_for_keywords(doc_data, keywords):
    keywords = [i.lower() for i in keywords]
    selected_data = []
    for block_data in doc_data:
        for sword in keywords:  # loop through search list
            block_text = " ".join(block_data["text"].lower().split())
            if sword in block_text:  # occurs as one of the words
                selected_data.append(block_data)
                break

    # unselect images
    final_selection = []
    for block_data in selected_data:
        if not block_data["text"].startswith("<image:"):
            final_selection.append(block_data)

    return final_selection


def filter_text_for_keywords(text, keywords):
    if text.startswith("<image:"):
        return False

    for sword in keywords:
        text = " ".join(text.lower().split())
        if sword in text:
            return True

    return False



def get_page_num(report_data, active_page):
    return report_data[active_page - 1][0]


def get_bbox(report_data, active_page):
    return report_data[active_page - 1]["bbox"]


def split_sentence(text):
    return tokenize.sent_tokenize(text)


def combine_blocks_per_page(report_data):
    page_level_data = {}
    for block in report_data:
        page_num = block["page_num"]
        block_text = block["text"]
        # unselect images
        if not block_text.startswith("<image:"):
            if page_num in page_level_data:
                page_level_data[page_num]["text"] += " " + block_text
                page_level_data[page_num]["blocks"].append(block)

            else:
                page_level_data[page_num] = {"text": block["text"], "blocks": [block]}

    return page_level_data


def get_sentence_level_per_page(report_data):
    # Combine at page level
    page_level_data = combine_blocks_per_page(report_data)

    # split sentences
    missing_s = 0
    # Iterate per page
    for page_num, page_info in page_level_data.items():
        # Split page sentences
        sentences = split_sentence(page_info["text"])
        candidate_blocks = page_info["blocks"]
        sentences_dict = []

        # * -------------- For each sentence find corresponding blocks --------------
        for s in sentences:
            bboxes = []
            text_found = ""
            missing_text = s.lstrip()
            found_block = False

            # * -------------- Iterate At Block Level --------------
            for block in candidate_blocks:
                # If found, stop search
                if found_block:
                    break

                block_text = block["text"]
                block_bbox = block["bbox"]

                # If full sentence in Block text, FOUND
                if missing_text in block_text:
                    bboxes.append(block_bbox)
                    found_block = True
                    break

                # If not, check if block sentence in missing text
                else:
                    # Get sentences in block
                    block_sentences = split_sentence(block_text)
                    for block_s in block_sentences:
                        if block_s in missing_text[: len(block_s)]:
                            bboxes.append(block["bbox"])
                            text_found = " ".join([text_found, block_s]).lstrip()
                            missing_text = missing_text[len(block_s) :].lstrip()

                        # If no missing text left, break
                        if missing_text == "":
                            found_block = True
                            break

            # * -------------- ALTERNATIVE: Search for combining two sentences within a block --------------
            if not found_block:
                bboxes = []
                text_found = ""
                missing_text = s.lstrip()

                # Iterate per block
                for block in candidate_blocks:
                    # If found, stop search
                    if found_block:
                        break

                    block_text = block["text"]
                    block_bbox = block["bbox"]

                    block_sentences = split_sentence(block_text)
                    block_sentences.append("")

                    # Iterate per sentence combination
                    skip_next = False
                    for i in range(len(block_sentences) - 1):
                        # If current sentence was used in previous combination we don't look at the next combination
                        if skip_next:
                            skip_next = False
                            continue

                        # Combine both sentences
                        combined_block_s = " ".join(
                            [block_sentences[i], block_sentences[i + 1]]
                        )

                        # If missing text starts with combination, add to evidence
                        if combined_block_s in missing_text[: len(combined_block_s)]:
                            bboxes.append(block["bbox"])
                            text_found = " ".join(
                                [text_found, combined_block_s]
                            ).lstrip()
                            missing_text = missing_text[
                                len(combined_block_s) :
                            ].lstrip()

                            # Don't use next combination
                            skip_next = True

                        # If no missing text, all is found
                        elif missing_text == "":
                            found_block = True
                            break

                        # Elif next combined text starts with missing text, all is found
                        elif missing_text in combined_block_s[: len(missing_text)]:
                            bboxes.append(block["bbox"])
                            found_block = True
                            break

            # If after within block combination not found, do across block combinations
            if not found_block:
                bboxes = []
                text_found = ""
                missing_text = s.lstrip()

                new_candidates = candidate_blocks + [{"bbox": [], "text": ""}]

                last_sentence_being_used = False
                for i in range(len(new_candidates) - 1):
                    # If found, stop search
                    if found_block:
                        break

                    # If last sentence of previous block was used, we skip this combination
                    if last_sentence_being_used:
                        skip_next = False
                        last_sentence_being_used = False
                        continue

                    block_0 = new_candidates[i]
                    block_1 = new_candidates[i + 1]

                    block_0_sentences = split_sentence(block_0["text"])
                    block_1_sentences = split_sentence(block_1["text"])

                    blocks_sentences = block_0_sentences + block_1_sentences

                    block_sentences_bboxs = [block_0["bbox"]] * len(
                        block_0_sentences
                    ) + [block_1["bbox"]] * len(block_1_sentences)

                    if len(blocks_sentences) == 1:
                        blocks_sentences.append("")
                        block_sentences_bboxs.append([])

                    skip_next = False
                    for j in range(len(blocks_sentences) - 1):
                        # If current sentence was used in previous combination we don't look at the next combination
                        if skip_next:
                            skip_next = False
                            continue

                        combined_block_s = " ".join(
                            [blocks_sentences[j], blocks_sentences[j + 1]]
                        )
                        if combined_block_s in missing_text[: len(combined_block_s)]:
                            bboxes.append(block_sentences_bboxs[j])
                            bboxes.append(block_sentences_bboxs[j + 1])

                            text_found = " ".join(
                                [text_found, combined_block_s]
                            ).lstrip()
                            missing_text = missing_text[
                                len(combined_block_s) :
                            ].lstrip()

                            # Don't use next combination
                            skip_next = False

                            if block_1_sentences[-1] in combined_block_s:
                                last_sentence_being_used = True

                        elif missing_text == "":
                            found_block = True
                            break

                        elif missing_text in combined_block_s[: len(missing_text)]:
                            bboxes.append(block_sentences_bboxs[j])
                            if len(missing_text) > len(block_0_sentences):
                                bboxes.append(block_sentences_bboxs[j + 1])

                            found_block = True
                            break

            if not found_block:
                missing_s += 1
                print("page: ", page_num)
                print("block not found for ", s)
                print()
                sentences_dict.append({"text": s, "bbox": []})

            else:
                clean_bboxes = []
                for bbox in bboxes:
                    if bbox not in clean_bboxes and bbox != []:
                        clean_bboxes.append(bbox)
                sentences_dict.append({"text": s, "bbox": clean_bboxes})

        page_level_data[page_num]["sentences"] = sentences_dict

    page_level_data_list = []
    for page_num, page_info in page_level_data.items():
        page_level_data_list.append({"page_num": page_num, **page_info})

    return page_level_data_list
