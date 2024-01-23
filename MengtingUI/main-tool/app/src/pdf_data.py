"""PDF Document Text Extraction"""

import fire
import pathlib
from typing import Union, List, Optional, Callable
from tqdm import tqdm
from enum import Enum
from more_itertools import windowed
import concurrent

# sys.path.insert(0, project_path)
from src.commons import (
    open_pdf,
    save_json,
)
from src import cache
import cv2
import PIL
from PIL import Image
import pytesseract
import numpy as np
import pandas as pd
from nltk import tokenize
import fitz
from google.cloud import storage
from IPython.display import display
import re
import string
from autocorrect import Speller
from unidecode import unidecode
import unicodedata

storage_client = storage.Client()


def postprocess_func(
    text: str,
    spelling_check: bool = False,
    remove_punctuations: bool = False,
    lower: bool = False,
) -> str:
    """
    The postprocess_func function is used to clean up the text extracted from the PDF.
    It does a few things:
    - Converts all characters to lowercase (e.g., &quot;Hello&quot; -&gt; &quot;hello&quot;)
    - Removes line breaks (e.g., &quot;\n&quot; -&gt; &quot;&quot;)
    - Removes punctuation except for periods (&quot;.&quot;), which are needed for decimals in numbers (e.g., 3,25) and abbreviations like U.S.)
    - Spell checks the text using Speller, an open source spell checker written by @makcedward

    :param text:str: Pass in the text to be processed
    :return: The cleaned text
    """
    # Unidecode

    def remove_control_and_symbol_characters(s):
        chars = []
        for ch in s:
            ch_category = unicodedata.category(ch)
            if ch_category[0] == "C" or ch_category == "So":
                chars.append(" ")
            else:
                chars.append(ch)
        return "".join(chars)

    text = remove_control_and_symbol_characters(text)

    text = unidecode(text)

    # Lowercase
    if lower:
        text = text.lower()

    # Remove line breaks
    for line_break in [r"\n\n\n", r"\n\n", r"\n"]:
        text = re.sub(line_break, " ", text)

    if remove_punctuations:
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

    # Spell check - especially needed when OCR is used
    if spelling_check:
        spell = Speller(lang="en")
        text = spell(text)

    # Remove punctuation after etc.
    text = text.replace("etc.", "etc")

    # Remove multi spcaes
    text = " ".join(text.split())

    return text


class Level(Enum):
    """Defines the level of granularity of text extraction"""

    page = 1
    blocks = 2
    paragraphs = 3
    lines = 4
    words = 5
    sentences = 6
    tokens = 7


class DocumentLoader:
    """
    A class for text extraction.

    Example:
        ---------------------------------------------------------------------------------------
        # Load a document
        doc = DocumentLoader(document = document_path)

        # Annotate
        image_bboxes = [
            [158.0488, 700.1564, 1519.7644, 1973.0702],
            [155.0654, 342.6681, 1519.822, 457.9712]
            ]

        page_num = 75

        doc.apply_annotation(page_num, image_bboxes, ["table1", "table2"], bbox_padding_pct=10, verbose=False)

        # Extract
        _ = doc.get_text([10, page_num], level="tokens", token_length=10, token_overlap_size=0, metadata=True, verbose=True)

        # Save
        doc.save(output_filename="./output.json")
        -------------------------------------------------------------------------------------------
    """

    def __init__(
        self, document: Union[str, np.ndarray, PIL.Image.Image]
    ) -> NotImplemented:
        """
        Load in a document.

        :param document: Union[str, np.ndarray, PIL.Image.Image]: Specify the image/pdf file to be read. Can either be a path, image array or PIL.image.image.
        """

        # * =========================== Load document ===========================
        self.document = document
        self.is_image = True
        self.require_masking = False
        self.bboxes_configs = None
        self.bbox_padding = 0.0
        self.bbox_padding_pct = 0.0

        # load if it is a string path
        if isinstance(self.document, str):
            file_extension = pathlib.Path(self.document).suffix
            # pdf
            if file_extension == ".pdf":
                self.document = open_pdf(self.document)
                self.is_image = False
            # image
            else:
                self.document = cv2.cvtColor(
                    cv2.imread(self.document), cv2.COLOR_BGR2RGB
                )

        # image only: if it is a numpy array, do nothing
        elif isinstance(self.document, np.ndarray):
            self.document = [self.document]

        # image only: if it is a PIL.Image.Image, then convert to numpy array
        elif isinstance(self.document, PIL.Image.Image):
            self.document = [np.array(self.document)]

        # else, raise error
        else:
            raise ValueError(
                "`document` can only take in a string path of an image or a pdf document or numpy ndarray of an image."
            )

    def __getitem__(self, page_num: int) -> PIL.Image.Image:
        """
        Specify instance[page_num] to display the page.
            - If the document is an image, then page_num is ignore.
            - If the document is a pdf file, then specify page_num to display a specific page.

        :param page_num:int: Specify the page number of the pdf document that we want to display. This argument is ignore when the document is an image.
        :return: The image of the page number that is passed to it
        """

        if not self.is_image:
            page = self.document[page_num]

        image = self.display_page(page)

        return image

    @staticmethod
    def _get_pixmap(page: fitz.fitz.Page, dpi: int = 200):
        if "dpi" in page.get_pixmap.__code__.co_varnames:
            pix = page.get_pixmap(dpi=dpi)
        else:
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        return pix

    @classmethod
    def _image_to_pdf_bbox(
        cls,
        page: fitz.fitz.Page,
        bbox: List[float],
        dpi: Optional[int] = 200,
        pix: fitz.fitz.Pixmap = None,
    ) -> fitz.fitz.Rect:
        """
        Transforms image bounding boxes to pdf bounding boxes.

        :param page: fitz.fitz.Page: The page of the pdf file
        :param bbox: List[float]: Specify the coordinates of the bounding box in image coordinates
        :param dpi: Optional[int]: Set the resolution of the image
        :return: A rectangle in pdf coordinates
        """
        # * Calculate the transformation matrix
        if pix is None:
            pix = cls._get_pixmap(page, dpi)
        try:
            mat = page.rect.torect(pix.irect)
            # * Create a fitz rectangle object
            r = fitz.Rect(bbox)
            # * Transform to pdf coordinates
            rect = r * ~mat
        except AttributeError:
            mode = "RGBA" if pix.alpha else "RGB"
            mediabox = page.mediabox
            image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            scale = (image.size[0] / mediabox[2], image.size[1] / mediabox[3])
            rect = fitz.Rect(
                [
                    bbox[0] / scale[0],
                    bbox[1] / scale[1],
                    bbox[2] / scale[0],
                    bbox[3] / scale[1],
                ]
            )

        return rect

    @classmethod
    def _pdf_to_image_bbox(
        cls,
        page: fitz.fitz.Page,
        bbox: List[float],
        dpi: Optional[int] = 200,
        pix: fitz.fitz.Pixmap = None,
    ) -> fitz.fitz.Rect:
        """
        Transforms image bounding boxes to image bounding boxes.

        :param page: fitz.fitz.Page: The page of the pdf file
        :param bbox: List[float]: Specify the coordinates of the bounding box in image coordinates
        :param dpi: Optional[int]: Set the resolution of the image
        :return: A rectangle in pdf coordinates
        """
        # * Calculate the transformation matrix
        if pix is None:
            pix = cls._get_pixmap(page, dpi)

        try:
            mat = page.rect.torect(pix.irect)
            # * Create a fitz rectangle object
            r = fitz.Rect(bbox)
            # * Transform to image coordinates
            rect = r * mat
        except AttributeError:
            mode = "RGBA" if pix.alpha else "RGB"
            mediabox = page.mediabox
            image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            scale = (image.size[0] / mediabox[2], image.size[1] / mediabox[3])
            rect = fitz.Rect(
                [
                    bbox[0] * scale[0],
                    bbox[1] * scale[1],
                    bbox[2] * scale[0],
                    bbox[3] * scale[1],
                ]
            )

        return rect

    @classmethod
    def display_page(cls, page: Union[np.ndarray, fitz.fitz.Page]) -> PIL.Image.Image:
        """
        The display_page function takes in a page from the pdf and returns an image.
            If the input is a numpy array, it will be converted to an Image object.
            Otherwise, if it is a fitz Page object, then we get its pixmap and convert that to an Image object.

        :param page: Union[np.ndarray: Specify that the page parameter can be either a numpy array or a fitz Page.
        :return: An image object
        """
        if isinstance(page, np.ndarray):
            image = Image.fromarray(page)
        else:
            image = cls._page_to_image(page, 200)

        return image

    @classmethod
    def _page_to_image(
        cls, page: fitz.fitz.Page, dpi: int, pix: fitz.fitz.Pixmap = None
    ) -> PIL.Image.Image:
        """
        The _page_to_image function takes a page from the pdf and converts it to an image.

        :param page: fitz.fitz.Page: Specify the page to be converted
        :param dpi: int: Image resolution.
        :return: A PIL Image
        """
        if pix is None:
            pix = cls._get_pixmap(page, dpi)

        mode = "RGBA" if pix.alpha else "RGB"
        image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        return image

    @staticmethod
    def _add_padding(
        bbox: List[float],
        bbox_padding_pct: Optional[float] = None,
        bbox_padding: Optional[float] = None,
    ) -> List[float]:
        """
        The _add_padding function adds padding to a bounding box.

        :param bbox: List[float]: Pass the bounding box coordinates
        :param bbox_padding_pct: Optional[float]: Calculate the padding based on a proportion of the bounding box's width and height
        :param bbox_padding: Optional[float]: Add a static padding to the bounding box

            bbox_padding_pct and bbox_padding cannot be both specified.

        :return: A new bounding box with padding
        """
        # * Input value check
        if bbox_padding is not None and bbox_padding_pct is not None:
            raise ValueError(
                "Please give a value for bbox_padding or bbox_padding_pct (but not both)"
            )

        # * Calculate padding based on the proportion of bounding box width and height
        if bbox_padding_pct is not None:
            height_padding = (bbox[3] - bbox[1]) * bbox_padding_pct
            weight_padding = (bbox[2] - bbox[0]) * bbox_padding_pct
        # * Calculate static padding values
        elif bbox_padding is not None:
            height_padding = bbox_padding
            weight_padding = bbox_padding
        # * Otherwise no padding
        else:
            height_padding = 0.0
            weight_padding = 0.0

        # * Calculate the new bounding box
        bbox = [
            bbox[0] - weight_padding,
            bbox[1] - height_padding,
            bbox[2] + weight_padding,
            bbox[3] + height_padding,
        ]

        return bbox

    @staticmethod
    def _write_text(
        page: fitz.fitz.Page, bbox: fitz.fitz.Rect, text: str, fontsize: int
    ) -> fitz.fitz.Page:
        """
        The _write_text function is a helper function that writes text on the fitz page.

        :param page: fitz.fitz.Page: Add the redaction annotation to a specific page
        :param bbox: fitz.fitz.Rect: Define the area where the text will be written
        :param text: str: Set the text that will be written on the page
        :param fontsize: int: Set the font size of the text to be written. Set to "auto" to automatically find the fontsize that will fit the bounding box.
        :return: A page with the text written on it
        """
        if len(text) > 0:
            fontname = "helvetica"

            # * Auto find fontsize
            if fontsize == "auto":
                fontsize = (
                    bbox.width / len(text) * 2
                )  # *2 just because 1pt is too small for a char. It mantains a good ratio for rect's width with larger text, but behaviour is not assured.

            # * Add annotation
            try:
                _ = page.add_redact_annot(
                    bbox,
                    text=text,
                    align=fitz.TEXT_ALIGN_LEFT,
                    fontsize=fontsize,
                    fontname=fontname,
                )
            except AttributeError:
                _ = page.addRedactAnnot(
                    bbox,
                    text=text,
                    align=fitz.TEXT_ALIGN_LEFT,
                    fontsize=fontsize,
                    fontname=fontname,
                )

            # * Apply the annotation
            page.apply_redactions()

            return page

    @staticmethod
    def _plot_bboxes(
        bboxes: List[List[float]], image: Union[PIL.Image.Image, np.array]
    ) -> NotImplementedError:
        """
        The _plot_bboxes function takes in a list of bounding boxes and an image.
        It then plots the bounding boxes on top of the image, then displays the PIL Image.

        :param bboxes: List[List[float]]: Pass in a list of bounding boxes.
        :param image: Union[PIL.Image.Image, np.array]: Specify the image.
        :return: None
        """

        if isinstance(image, PIL.Image.Image):
            page_arr = np.asarray(image).copy()
        else:
            page_arr = image.copy()

        if not isinstance(bboxes[0], list):
            bboxes = [bboxes]

        for bbox in bboxes:
            img = cv2.rectangle(
                page_arr,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                2,
            )

        image = PIL.Image.fromarray(img)
        return image

    def apply_annotation(
        self,
        page_num: Optional[int] = None,
        bboxes: Union[List[float], List[List[float]]] = None,
        texts: Union[Optional[str], List[str]] = None,
        bboxes_configs: Optional[dict] = None,
        fontsize: Optional[Union[str, int]] = "auto",
        is_image_bbox: Optional[bool] = False,
        dpi: Optional[int] = 200,
        bbox_padding: Optional[float] = None,
        bbox_padding_pct: Optional[float] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        """
        The apply_annotation function is used to annotate a document with bounding boxes and text.

        :param page_num: Optional[int]: Specify the page number of the document. Only be used when processing a single page. If multiple pages needs to be annotated, please specify page_num in bbox_configs.
        :param bboxes: Union[List[float], List[List[float]]]: Specify the bounding boxes to be masked. Only be used when processing a single page. If multiple pages needs to be annotated, please specify bboxes in bbox_configs.
        :param texts: Union[Optional[str], List[str]]: Specify the text that will be written inside the bounding box. Default to None. Only be used when processing a single page. If multiple pages needs to be annotated, please specify texts in bbox_configs.
        :param bboxes_configs: Optional[dict]: Used when annotating multiple pages. Pass in a list of dictionaries of the following structure:
                                [
                            {
                                "page_num": 1,
                                "bboxes": [
                                    {
                                        "text": text,
                                        "bbox": [xmin, ymin, xmax, ymax]
                                    },
                                    ...
                                ]
                            },
                            ...
                        ]
        :param fontsize: Optional[Union[str, int]]: Specify the font size of the text. Set to "auto" to automatically find the most appropriate fontsize.
        :param is_image_bbox: Optional[bool]: Determine whether the bounding boxes are given for images or pdfs
        :param dpi: Optional[int]: Specify the resolution of the image. Can be ignored if is_image_bbox is False.
        :param bbox_padding: Optional[float]: Add padding to the bounding box
        :param bbox_padding_pct: Optional[float]: Add padding based on % of bbox's width and height
        :param verbose: Optional[bool]: Whether to display the page after each annotation
        """
        # * ----------------------------- Input Checks ---------------------------------

        # Extract bounding boxes configs - Only be used when multiple pages are passed in
        if bboxes_configs is not None:
            bboxes = [
                [
                    bbox_configs["bboxes"][i]["bbox"]
                    for i in range(len(bbox_configs["bboxes"]))
                ]
                for bbox_configs in bboxes_configs
            ]

            texts = [
                [
                    bbox_configs["bboxes"][i]["text"]
                    for i in range(len(bbox_configs["bboxes"]))
                ]
                for bbox_configs in bboxes_configs
            ]

            page_nums = [bbox_configs["page_num"] for bbox_configs in bboxes_configs]

        # Otherwise, extract configs from page_num, bboxes and text arguments
        else:
            page_nums = [page_num]

            if bboxes is None:
                return "Nothing to annotate."

            if not isinstance(bboxes[0], list):
                bboxes = [bboxes]

            if len(bboxes[0]) == 0:
                raise ValueError("Does not accept empty bounding box.")

            if texts is not None:
                assert len(bboxes) == len(texts), ValueError(
                    "The length of bboxes and texts does not match."
                )
            else:
                texts = [""] * len(bboxes)

            bboxes, texts = [bboxes], [texts]

        # dpi
        if is_image_bbox:
            assert dpi is not None, ValueError(
                "dpi must be specified when the bounding boxes are given for images."
            )

        # * ================================ ANNOTATION ==================================
        for page_num, pbboxes, ptexts in zip(page_nums, bboxes, texts):
            # Apply padding to the bounding box
            pbboxes = [
                self._add_padding(b, bbox_padding, bbox_padding_pct) for b in pbboxes
            ]

            # * Images
            if self.is_image:
                for t, box in zip(pbboxes, ptexts):
                    # Specify coordinates
                    xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]

                    # Mask tables - fill with white pixels
                    self.document[
                        int(np.floor(ymin)) : int(np.ceil(ymax)),
                        int(np.floor(xmin)) : int(np.ceil(xmax)),
                        :,
                    ] = 255

                    # Write text inside the bounding box
                    # auto calculate fontsize
                    if fontsize == "auto":
                        center = (xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2)
                        text_face = cv2.FONT_HERSHEY_SIMPLEX
                        text_scale = 1
                        text_thickness = 1

                        text_size, _ = cv2.getTextSize(
                            t, text_face, text_scale, text_thickness
                        )
                    else:
                        text_scale = fontsize

                    text_origin = (
                        int(center[0] - text_size[0] / 2),
                        int(center[1] + text_size[1] / 2),
                    )
                    self.document = cv2.putText(
                        self.document,
                        t,
                        text_origin,
                        text_face,
                        text_scale,
                        (0, 0, 0),
                        text_thickness,
                        cv2.LINE_AA,
                    )

                    if verbose:
                        display(self.display_page(self.document))

            # * PDFs
            else:
                page = self.document[page_num]

                # Transform the image bounding box to pdf bounding box
                pix = DocumentLoader._get_pixmap(page, dpi)

                if is_image_bbox:
                    pbboxes = [
                        self._image_to_pdf_bbox(page=page, bbox=b, dpi=dpi, pix=pix)
                        for b in pbboxes
                    ]
                else:
                    pbboxes = [fitz.Rect(b[0], b[1], b[2], b[3]) for b in pbboxes]

                # Write text inside the bounding box
                if ptexts is not None:
                    for ind in range(len(pbboxes)):
                        if ptexts[ind] == "":
                            bt = " "
                        else:
                            bt = ptexts[ind]
                        try:
                            page = self._write_text(page, pbboxes[ind], bt, fontsize)
                        except RuntimeError:
                            self.require_masking = True
                            self.bboxes_configs = bboxes_configs
                            self.bbox_padding = bbox_padding
                            self.bbox_padding_pct = bbox_padding_pct
                            print("Annotation failed. Try to mask out texts that overlap with the bounding boxes.")

                if verbose:
                    display(self.display_page(page))

    @staticmethod
    def _split_tokens(
        words: List[str], token_length: int, token_overlap_size: int
    ) -> List[str]:
        """
        The _split_tokens function splits the text into tokens of a specified length.

        :param words: List[str]: Specify the list of words that we want to split into chunks
        :param token_length: int: Specify the length of each chunk
        :param token_overlap_size: int: Specify the number of words that overlap between two chunks
        :return: A list of strings
        """
        # Input Checks
        assert (
            token_length is not None
        ), "Please make sure token_length is specified when level is set to 'tokens'."

        assert (
            token_overlap_size is not None
        ), "Please make sure token_overlap_size is specified when level is set to 'tokens'."

        # Split into chunks
        step = token_length - token_overlap_size
        words_chunks = list(windowed(words, n=token_length, step=step, fillvalue=""))
        extracted_text = [" ".join(words_chunk).strip() for words_chunk in words_chunks]

        return extracted_text

    @staticmethod
    def _split_sentences(text: List[str]) -> List[str]:
        """
        The _split_sentences function takes in a list of strings and returns a list of sentences.

        :param text: List[str]: Pass in the text that is to be split into sentences
        :return: A list of sentences
        """
        extracted_text = tokenize.sent_tokenize(text[0])
        return extracted_text

    @classmethod
    def _apply_text_splitter(
        cls,
        level: str,
        document_data: List[dict],
        metadata: bool,
        token_length: Optional[int] = None,
        token_overlap_size: Optional[int] = None,
    ) -> List[dict]:
        """
        The _apply_text_splitter function is a helper function that splits the text into tokens or sentences.

        :param level: str: Specify the level of text splitting. Either "tokens" or "sentences".
        :param document_data: List[dict]: Pass in the list of dictionaries that contains the metadata and text extracted from each page
        :param metadata: bool: Specify whether the metadata should be included in the output
        :param token_length: Optional[int]: Specify the length of each chunk
        :param token_overlap_size: Optional[int]: Specify the number of tokens to overlap between two chunks
        :return: A list of dictionaries
        """

        assert level in ["tokens", "sentences"], ValueError(
            "level can only be 'tokens' or 'sentences'."
        )

        if level == "tokens":
            assert (token_length is not None) & (
                token_overlap_size is not None
            ), ValueError(
                "Please make sure that token_length and token_overlap_size is specified when level is set to 'tokens'."
            )

        new_document_data = []

        # * Token-level split
        if level == "tokens":
            for page_num in set([data["page_num"] for data in document_data]):
                words = []
                for data in document_data:
                    if data["page_num"] == page_num:
                        words.append(data["text"])

                extracted_text = cls._split_tokens(
                    words, token_length, token_overlap_size
                )
                if metadata:
                    new_document_data += [
                        {
                            "page_num": page_num,
                            "table_id": "-1",
                            "cell_id": "-1",
                            "level": level,
                            "bbox": "-1",
                            "text": et,
                        }
                        for et in extracted_text
                    ]
                else:
                    new_document_data += [{"text": et} for et in extracted_text]

        # * Sentence-level split
        if level == "sentences":
            for page_num in set([data["page_num"] for data in document_data]):
                page_text = []
                for data in document_data:
                    if data["page_num"] == page_num:
                        page_text.append(data["text"])

                extracted_text = cls._split_sentences(page_text)
                if metadata:
                    new_document_data += [
                        {
                            "page_num": page_num,
                            "table_id": "-1",
                            "cell_id": "-1",
                            "level": level,
                            "bbox": "-1",
                            "text": et,
                        }
                        for et in extracted_text
                    ]
                else:
                    new_document_data += [{"text": et} for et in extracted_text]

        return new_document_data

    def get_text(
        self,
        page_nums: Optional[List[int]] = None,
        level: Optional[str] = "blocks",
        dpi: Optional[int] = 200,
        token_length: Optional[int] = 100,
        token_overlap_size: Optional[int] = 50,
        metadata: Optional[bool] = True,
        verbose: Optional[bool] = False,
        n_jobs: int = 1,
    ) -> List[dict]:
        """
        The get_text function extracts text from a PDF file.

        :param page_nums: Optional[List[int]]: Specify the page numbers to extract text from. None if to extract all pages.
        :param level: Optional[str]: Specify the level of extraction.
                        - "page"
                        - "blocks"
                        - "sentences"
                        - "words"
                        - "tokens"
        :param dpi:Optional[int]: Set the dpi of the image
        :param token_length: Optional[int]: Specify the length of each token chunk
        :param token_overlap_size: Optional[int]: Determine the number of overlapping tokens between chunks
        :param metadata:Optional[bool]: Determine whether the metadata should be returned or not
        :param verbose: Optional[bool]: Plot the bounding boxes of each text block
        :param n_jobs: int: If greater than 1, use multi-threading. Set to -1 to use all available threads.
        :return: A list of dictionaries of the extracted texts.
        """

        if self.is_image:
            raise ValueError(
                "get_text can only run on pdf files. Please use get_text_ocr instead."
            )

        # Process all pages of a document if page_nums is not specified
        if page_nums is None:
            page_nums = list(range(len(self.document)))

        if not isinstance(page_nums, list):
            page_nums = [page_nums]

        if level in ["page", "sentences"]:
            extract_level = "text"
        elif level == "tokens":
            extract_level = "words"
        else:
            extract_level = level

        def _get_text(page_num: int):
            page = self.document[page_num]

            if extract_level in ["text", "blocks", "words"]:
                try:
                    extracted_text = page.get_text(extract_level, sort=True)
                except TypeError:
                    extracted_text = page.get_text(extract_level)

                if not isinstance(extracted_text, list):
                    extracted_text = [extracted_text]

                # if extract_level == "text":
                pix = DocumentLoader._get_pixmap(page, dpi)
                bbox = [0, 0, pix.width, pix.height]

                if verbose:
                    bboxes = [
                        list(et[:4]) if extract_level in ["words", "blocks"] else bbox
                        for et in extracted_text
                    ]
                    bboxes = [
                        list(self._pdf_to_image_bbox(page, b, dpi, pix)) for b in bboxes
                    ]
                    display(
                        self._plot_bboxes(bboxes, self._page_to_image(page, dpi, pix))
                    )
                
                all_text = [
                        {
                            "page_num": page_num,
                            "table_id": "-1",
                            "cell_id": "-1",
                            "level": level,
                            "bbox": list(
                                self._pdf_to_image_bbox(
                                    page,
                                    list(et[:4])
                                    if level in ["words", "blocks"]
                                    else bbox,
                                    dpi,
                                    pix,
                                )
                            ),
                            "text": et[4] if extract_level != "text" else et,
                        }
                        for et in extracted_text
                    ]
                
                if self.require_masking:
                    bboxes = [
                        [
                            bbox_configs["bboxes"][i]["bbox"]
                            for i in range(len(bbox_configs["bboxes"]))
                        ]
                        for bbox_configs in self.bboxes_configs
                    ]

                    texts = [
                        [
                            bbox_configs["bboxes"][i]["text"]
                            for i in range(len(bbox_configs["bboxes"]))
                        ]
                        for bbox_configs in self.bboxes_configs
                    ]

                    page_nums = [bbox_configs["page_num"] for bbox_configs in self.bboxes_configs]
                    
                    for pn, pbboxes, ptexts in zip(page_nums, bboxes, texts):
                        # Apply padding to the bounding box
                        pbboxes = [
                            self._add_padding(b, self.bbox_padding, self.bbox_padding_pct) for b in pbboxes
                        ]
                        
                        if pn==page_num:
                            print(f"Page {pn}: masking in progress")
                            for it in all_text[:]:
                                for pbbox in pbboxes:
                                    pct = self.get_overlap_pct(it["bbox"], pbbox)
                                    if pct>0.8:
                                        try:
                                            all_text.remove(it)
                                        except:
                                            pass

                if metadata:
                    return all_text
                else:
                    return [{"text": et["text"]} for et in extracted_text]
            else:
                raise ValueError(
                    f"get_text do not support level={level}. Please try get_text_ocr instead."
                )

        self.document_data = []

        if n_jobs == 1:
            print("Extracting text...")
            self.document_data = [_get_text(page_num) for page_num in tqdm(page_nums)]
            self.document_data = [j for i in self.document_data for j in i]
        else:
            if n_jobs == -1:
                max_workers = None
            else:
                max_workers = n_jobs

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                results = [
                    executor.submit(_get_text, page_num) for page_num in page_nums
                ]
                for future in concurrent.futures.as_completed(results):
                    try:
                        extracted_page_data = future.result()
                        self.document_data.extend(extracted_page_data)
                    except Exception as e:
                        print(f"An error occurred: {e}")

        # * Text Splitter
        if level in ["tokens", "sentences"]:
            self.document_data = self._apply_text_splitter(
                document_data=self.document_data,
                level=level,
                token_length=token_length,
                token_overlap_size=token_overlap_size,
                metadata=metadata,
            )

        return self.document_data

    @staticmethod
    def get_overlap_pct(bb1, bb2):

        assert bb1[0] < bb1[2]
        assert bb1[1] < bb1[3]
        assert bb2[0] < bb2[2]
        assert bb2[1] < bb2[3]
        
        # determine the coordinates of the intersection rectangle
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        
        overlap_pct = intersection_area/bb1_area
        
        return overlap_pct
    
    @staticmethod
    def _preprocess_image(image: np.ndarray) -> np.ndarray:
        """
        The _preprocess_image function takes in an image and returns a preprocessed version of the image.
        The function first turns the image into grayscale, then removes the background color from it.
        It does this by binarizing (turning it into a binary image) where pixels are either 0 or 1,
        and then removing noise.

        :param image: Pass in the image that we want to preprocess
        :return: An array of image
        """
        # turn image into grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # remove bg color
        # TODO: check if this function can generalise to other images
        # thresh = cv2.threshold(page_arr,127, 255, cv2.THRESH_BINARY_INV)[1]
        # thresh = 255 - thresh

        # bg removal: binarize the image (turn it into a binary image) where pixels are either 0 or 1
        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # removing noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        image = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        return image

    def get_text_ocr(
        self,
        page_nums: Optional[int] = None,
        level: Optional[str] = "blocks",
        token_length: Optional[int] = 100,
        token_overlap_size: Optional[int] = 50,
        metadata: Optional[bool] = True,
        dpi: Optional[int] = 200,
        verbose: Optional[bool] = False,
    ) -> List[dict]:
        """
        The get_text_ocr function takes in a image or pdf file and returns the text from it applying OCR.

        :param page_nums: Optional[int]: Specify the page numbers to be extracted. None if to extract all pages.
        :param level: Optional[str]: Specify the level of text extraction
        :param token_length: Optional[int]: Set the length of each token chunk
        :param token_overlap_size: Optional[int]: Define the number of overlapping tokens between chunks
        :param metadata:Optional[bool]: Determine whether to return the metadata of each text
        :param dpi:Optional[int]: Set the dpi of the image
        :param verbose: Optional[bool]: Plot the bounding boxes of each text element
        :param : Specify the level of text extraction. Pick from 'page', 'blocks', 'lines', 'sentences', 'words', 'tokens'.
        :return: A list of dictionaries, each dictionary containing the page number, table id (none), level (extraction level), bounding box coordinates and text
        """

        if not self.is_image:
            if page_nums is None:
                page_nums = len(self.document)
            elif not isinstance(page_nums, list):
                page_nums = [page_nums]

            page_arrs = []
            # convert every page to an image
            for page_num in page_nums:
                page = self.document[page_num]
                image = self._page_to_image(page, dpi)
                page_arrs.append(np.asarray(image))
        else:
            page_arrs = [self.document]
            page_nums = [1]

        page_arrs = [self._preprocess_image(page_arr) for page_arr in page_arrs]

        if level == "sentences":
            extract_level = "page"
        elif level == "tokens":
            extract_level = "words"
        else:
            extract_level = level

        try:
            extract_level = Level[extract_level]
        except KeyError:
            raise ValueError(
                "Please make sure that level is in ['page', 'blocks', 'lines', 'sentences', 'words', 'tokens']."
            )

        extract_level = extract_level.value

        self.document_data = []
        for page_arr, page_num in zip(page_arrs, page_nums):
            # text extraction
            d = pytesseract.image_to_data(page_arr, output_type=pytesseract.Output.DICT)
            d_df = pd.DataFrame.from_dict(d)

            # generate ids for each row
            cols = ["page_num", "block_num", "par_num", "line_num", "word_num"]
            d_df = d_df.astype({col: "str" for col in cols})
            digits = 6
            for col in cols:
                d_df[col] = d_df[col].str.zfill(digits)
            d_df["id"] = d_df[cols].agg("".join, axis=1)

            # extract
            if extract_level <= 5:
                extracted_text = []
                level_ids = (
                    d_df[d_df["level"] == extract_level]["id"]
                    .str.slice(start=0, stop=extract_level * digits)
                    .values
                )
                for i in level_ids:
                    words = d_df[
                        (d_df["level"] == 5)
                        & (d_df["id"].str.startswith(i))  # word level
                    ]["text"].values
                    text = " ".join(words).strip()
                    if text != "":
                        extracted_text.append(text)

            else:
                extracted_text = [
                    " ".join(d_df[d_df["level"] == 5]["text"].values).strip()
                ]

            leveldf = d_df[d_df["level"] == extract_level].copy(deep=True)
            leveldf["xmin"] = leveldf["left"]
            leveldf["xmax"] = leveldf["xmin"] + leveldf["width"]
            leveldf["ymin"] = leveldf["top"]
            leveldf["ymax"] = leveldf["ymin"] + leveldf["height"]
            bboxes = [
                [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
                for _, row in leveldf.iterrows()
            ]

            if verbose:
                self._plot_bboxes(bboxes, page_arr)

            if metadata:
                text_dict = [
                    {
                        "page_num": page_num,
                        "table_id": "-1",
                        "cell_id": "-1",
                        "level": Level(extract_level).name,
                        "bbox": bboxes[i],
                        "text": extracted_text[i],
                    }
                    for i in range(len(extracted_text))
                ]
            else:
                text_dict = [
                    {"text": extracted_text[i]} for i in range(len(extracted_text))
                ]

            self.document_data = self.document_data + text_dict

        # * Text Splitter
        if level in ["tokens", "sentences"]:
            self.document_data = self._apply_text_splitter(
                document_data=self.document_data,
                level=level,
                token_length=token_length,
                token_overlap_size=token_overlap_size,
                metadata=metadata,
            )

        return self.document_data

    def save(
        self,
        postprocess_func: Optional[
            Callable[[str, List[str]], Union[str, List[str]]]
        ] = postprocess_func,
        output_filename: Optional[str] = None,
    ) -> List[dict]:
        """
        The save function takes in a postprocess_func and an output_filename.
        The postprocess function is used to clean up the text extracted from the PDF,
        and can be any function that takes in a string and returns a string. The output filename is where you want to save your JSON file.

        :param postprocess_func: Optional[Callable[[str, List[str]]: Specify a function that will be used to postprocess the text extracted from each document
        :param output_filename: Optional[str]: Specify the patha and name of the json file to save.
        :return: None
        """

        # postprocess text extracted
        if postprocess_func is not None:
            for i in self.document_data:
                i["text"] = postprocess_func(i["text"])

        # save
        if output_filename is not None:
            save_json(self.document_data, output_filename)

        return self.document_data


if __name__ == "__main__":
    import fire
    import time
    from itertools import repeat
    from multiprocessing import Pool, cpu_count, freeze_support
    from tqdm import tqdm

    def per_document_extraction(
        document_path: str,
        page_nums: List[int] = None,
        bboxes_configs=None,
        dpi: int = 200,
        bbox_padding: int = None,
        bbox_padding_pct: float = None,
        level: str = "sentences",
        ocr: bool = False,
        output_filename: str = None,
    ):
        doc = DocumentLoader(document_path)
        if bboxes_configs is not None:
            doc.apply_annotation(
                bboxes_configs=bboxes_configs,
                dpi=200,
                bbox_padding=None,
                bbox_padding_pct=None,
            )
        if not ocr:
            _ = doc.get_text(page_nums=page_nums, level=level, dpi=dpi)
        else:
            _ = doc.get_text(page_nums=page_nums, level=level, dpi=dpi)

        doc.save(output_filename=output_filename)

    # TODO: Change the following code to process real documents
    N = 30
    document_paths = [
        "/home/jupyter/workbench/question_answering/data/raw/documents/Allianz_Group_Sustainability_Report_2022-web.pdf"
    ] * N
    output_filenames = [
        f"gs://esg-satelite-data-warehouse/teq-automation/qa/text_extraction/file{i}.josn"
        for i in range(N)
    ]

    def execute(
        document_paths: str = document_paths,
        page_nums: List[int] = None,
        bboxes_configs=None,
        dpi: int = 200,
        bbox_padding: int = None,
        bbox_padding_pct: float = None,
        level: str = "sentences",
        ocr: bool = False,
        output_filenames: str = output_filenames,
    ):
        freeze_support()

        def custom_error_callback(error):
            print(f"Got error: {error}")

        if bboxes_configs is None:
            bboxes_configs = repeat(bboxes_configs)

        t0 = time.perf_counter()
        with Pool() as pool:
            _ = pool.starmap(
                per_document_extraction,
                tqdm(
                    zip(
                        document_paths,
                        bboxes_configs,
                        repeat(page_nums),
                        repeat(dpi),
                        repeat(bbox_padding),
                        repeat(bbox_padding_pct),
                        repeat(level),
                        repeat(ocr),
                        output_filenames,
                    ),
                    total=len(document_paths),
                ),
                # error_callback=custom_error_callback
            )
        t1 = time.perf_counter()
        print("Total time %g seconds" % round(t1 - t0, 2))

    fire.Fire(execute)

@cache.memoize(3600)
def display_pdf_page(report_path, page_num, bbox, dpi=200):
    doc_loader = DocumentLoader(report_path)
    # document = doc_loader.document
    print(f"text displaying page: {page_num}")
    if not isinstance(bbox[0], list):
        bbox = [bbox]
    page_image = doc_loader._plot_bboxes(
        bbox, get_page_image(report_path, page_num, dpi)
    )

    return page_image

@cache.memoize(3600)
def get_page_image(report_path, page_num, dpi=200):
    doc_loader = DocumentLoader(report_path)
    document = doc_loader.document

    return doc_loader._page_to_image(document[page_num], dpi)
