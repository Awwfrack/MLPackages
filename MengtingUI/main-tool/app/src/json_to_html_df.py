import json
import os
from typing import List, Tuple
from IPython.display import display, HTML
from bs4 import BeautifulSoup
import fitz
from PIL import Image
import numpy as np
from pathlib import Path
import re
import cv2
import matplotlib.pyplot as plt
import string
import pytesseract
from pytesseract import Output
import cv2
import pandas as pd
from unidecode import unidecode


class Interval:
    def __init__(self, _min, _max):
        self.min = _min
        self.max = _max

    def __lt__(self, other):
        return self.min < other.min

    def get(self, is_min):
        return self.min if is_min else self.max

    def get_len(self):
        return self.max - self.min

    def apply_transform(self, offset_before=0.0, scale=1.0, offset_after=0.0):
        return Interval(
            offset_after + (self.min + offset_before) * scale,
            offset_after + (self.max + offset_before) * scale,
        )

class Box:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    @staticmethod
    def from_itvs(x_itv, y_itv):
        return Box(x_itv.min, y_itv.min, x_itv.max, y_itv.max)

    def get_x_itv(self):
        return Interval(self.xmin, self.xmax)

    def get_y_itv(self):
        return Interval(self.ymin, self.ymax)

    def get_itv(self, is_x):
        return self.get_x_itv() if is_x else self.get_y_itv()

    def get_area(self):
        if self.xmin < self.xmax and self.ymin < self.ymax:
            return (self.xmax - self.xmin) * (self.ymax - self.ymin)
        return 0.0

    def intersect(self, other):
        return Box(
            max(self.xmin, other.xmin),
            max(self.ymin, other.ymin),
            min(self.xmax, other.xmax),
            min(self.ymax, other.ymax),
        )

    @staticmethod
    def hull(boxes):
        return Box(
            min(b.xmin for b in boxes),
            min(b.ymin for b in boxes),
            max(b.xmax for b in boxes),
            max(b.ymax for b in boxes),
        )

    def to_array(self):
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def apply_offset(self, offset):
        return Box(
            self.xmin + offset[0],
            self.ymin + offset[1],
            self.xmax + offset[0],
            self.ymax + offset[1],
        )

    def apply_scale(self, scale):
        return Box(
            self.xmin * scale[0],
            self.ymin * scale[1],
            self.xmax * scale[0],
            self.ymax * scale[1],
        )

class TableCell:
    def __init__(
        self,
        _start_pos: Tuple[int, int],
        _end_pos: Tuple[int, int],
        _span: Tuple[int, int],
        _text: str,
    ):
        self.start_pos = _start_pos
        self.end_pos = _end_pos
        self.span = _span
        self.text = _text

    def get_info(self):
        return {
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "span": self.span,
            "text": self.text,
        }


class JsonToHTML:
    separators = [" ", "  ", "\n", "\t", "�"]

    superscript_map = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹"
    }

    subscript_map = {
        "0": "₀",
        "1": "₁",
        "2": "₂",
        "3": "₃",
        "4": "₄",
        "5": "₅",
        "6": "₆",
        "7": "₇",
        "8": "₈",
        "9": "₉"
    }
    
    ghg_units_patterns = {
        "C\ *O\ *₂\ *e\ ":"CO2e ", 
        "C\ *O\ *₂":"CO2", 
        "C\ *H\ *₄":"CH4", 
        "N\ *₂\ *O":"N2O", 
        "S\ *F\ *₆":"SF6", 
        "N\ *F\ *₃":"NF3"
        }

    trans = str.maketrans(
        "".join(superscript_map.keys()), "".join(superscript_map.values())
    )

    sub_trans = str.maketrans(
        "".join(subscript_map.keys()), "".join(subscript_map.values())
    )

    def __init__(self, table_path: str, dpi:int=200, ocr:bool=True):
        self.dpi = dpi
        
        # load data
        self.table_path = table_path
        with open(self.table_path) as f:
            self.table_json = json.load(f)
            
        # describe the table
        self.ncol, self.nrow, self.n_header_cells = self.describe() 
        
        # OCR
        self.ocr = ocr
        
        # load pdf page
        self.page_num = self.table_json["page_no"]
        report_path = Path(table_path).parent.parent
        report_path = os.path.join(report_path, self.table_json["document_id"])
        self.doc = fitz.open(report_path)

        self.image = self.get_image(dpi=self.dpi)
        self.page = self.doc[self.page_num]
        self.image_page_ratio = max(self.image.size[0]/self.page.mediabox[2], self.image.size[1]/self.page.mediabox[3])
        
    def transform(self, verbose: bool = False):
        #* ------------------ PostProcess --------------------------
        # calculate cell row and col based on span
        cells = self.process_cells()
        self.header_indexes = list(range(self.n_header_cells))
        self.subheader_indexes = [
            table_cell["start_pos"][0]
            for table_cell in cells
            if table_cell["span"] == (1, self.ncol)
        ]
        
        #* ----------------- Extract Text -------------------------
        # - Cells
        cells = self.populate_text_from_pdf(cells)
        # - Header/Footer
        self.process_header_footer(ocr=False)
        
        #* ----------------- Extract Text OCR -------------------------
        data = [cell["text"] for cell in cells]
        data = [i==[[]] for i in data for j in i]
        if all(data):
            table_is_image = True
        else:
            table_is_image = False
        if table_is_image & self.ocr:
            print(f"OCR table...")
            cells = self.populate_text_from_pdf_ocr(cells)
        
        # - Header/Footer: if empty, run ocr
        for hf in ["table_header", "table_footer"]:
            if (len(self.table_json[hf]["bounding_box"])>0) & (self.table_json[hf]["text"]==""):
                header_footer_is_image = True
            else:
                header_footer_is_image = False
        if header_footer_is_image & self.ocr:
            print(f"OCR footer/header...")
            self.process_header_footer(ocr=True)
        
        #* -------------------------- HTML Table -------------------------------
        html_string = self.get_html_with_text(cells, self.n_header_cells, verbose)

        # display html table
        self.display_html(cells, html_string, verbose)

        return html_string, self.table_json
    
#* ----------------------------------- General Utils -------------------------------------------
    def describe(self) -> Tuple[int, int, int]:
        ncol = self.table_json["col_count"]
        nrow = self.table_json["row_count"]

        header_row_index = []
        for table_cell in self.table_json["cells"]:
            if table_cell["is_header"]:
                header_row_index.append(table_cell["row_end"])

        print(header_row_index)
        if header_row_index:
            if len(header_row_index) > 0:
                n_header_cells = max(header_row_index) + 1
        else:
            n_header_cells = 0
        return ncol, nrow, n_header_cells
    
    def get_image(self, verbose: bool = False, dpi:int=200):
        pix = self._get_pixmap(self.doc[self.page_num], dpi=dpi)
        mode = "RGBA" if pix.alpha else "RGB"
        image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        if verbose:
            display(image)
        return image
    
    @classmethod
    def _postprocess_text(cls, text: List[List[str]]):
        cell_texts = []
        for text_row in text:
            for d in cls.separators:
                text = [i for j in [t.split(d) for t in text_row] for i in j]
            cell_text = " ".join([i for i in text if i.strip() != ""])
            cell_texts.append(cell_text)
        cell_text = '\n'.join(cell_texts)
        
        for k, v in cls.ghg_units_patterns.items():
            cell_text=re.sub(k, v, cell_text)
        
        return cell_text
    
    def _add_padding(self, pad_pct: float, box: List[float]):
        box = [i for j in box for i in j]
        pad_pct = pad_pct / 100
        width = box[2] - box[0]
        height = box[3] - box[1]
        image_size = self.image.size
        padded_box = [
            max(box[0] - pad_pct * width, 0),
            max(box[1] - pad_pct * height, 0),
            min(box[2] + pad_pct * width, image_size[0]),
            min(box[3] + pad_pct * height, image_size[1]),
        ]
        padded_box = [int(i) for i in padded_box]
        return padded_box
    
    def convert_bbox(self, box):
        if not len(box) == 0:
            box = Box(*box).apply_scale(self.image.size).to_array()
            xmin, ymin, xmax, ymax = box
            return (int(xmin), int(ymin)), (int(xmax), int(ymax))
        else:
            return False
    
    @staticmethod
    def _get_pixmap(page: fitz.fitz.Page, dpi: int = 200):
        if "dpi" in page.get_pixmap.__code__.co_varnames:
            pix = page.get_pixmap(dpi=dpi)
        else:
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        return pix

    # @staticmethod
    # def _box_pdf_to_image(box: Box, image: Image, page: fitz.Page):
    #     return box.apply_scale(
    #         [image.size[0] / page.mediabox[2], image.size[1] / page.mediabox[3]]
    #     )
    
    @staticmethod
    def _box_pdf_to_image(box: Box, ratio:float):
        return box.apply_scale((ratio, ratio))

    def display_html(self, cells, html_string:str, verbose:bool=True):
        # header_indexes = list(range(self.n_header_cells))
        # subheader_indexes = [
        #     table_cell["start_pos"][0]
        #     for table_cell in cells
        #     if table_cell["span"] == (1, self.ncol)
        # ]

        if verbose:
            htmll = BeautifulSoup(html_string, features="html.parser")
            for row in self.header_indexes:
                htmll.find_all("tr")[row].attrs["style"] = "background-color:#483D8B"
            for row in self.subheader_indexes:
                htmll.find_all("tr")[row].attrs["style"] = "background-color:#6495ED"
            display(HTML(str(htmll)))
    
    
#* -------------------------------------- 1. Processing ---------------------------------------------
    def process_cells(self) -> List[TableCell]:
        """
        Turn json cells into the expected cell format to run the subsequent processes
        """
        table_cells = self.table_json["cells"]
        
        cells = []
        for table_cell in table_cells:
            start_pos = (table_cell["row_start"], table_cell["col_start"])
            end_pos = (table_cell["row_end"], table_cell["col_end"])
            span = (
                table_cell["row_end"] - table_cell["row_start"] + 1,
                table_cell["col_end"] - table_cell["col_start"] + 1,
            )

            bbox = Box(*table_cell["bounding_box"]).apply_scale(self.image.size)

            table_cell = {
                "start_pos": start_pos,
                "end_pos": end_pos,
                "span": span,
                "box": bbox,
                "text_box": [[]],
                "text": [[]],
                "size": [[]],
            }

            cells.append(table_cell)
        return cells

    def process_header_footer(self, ocr:bool=False):
        """Extract and save text to json"""
        
        header_footer_cells = []

        for hf in ["table_header", "table_footer"]:
            if self.table_json[hf]["bounding_box"]:
                bbox = Box(*self.table_json[hf]["bounding_box"]).apply_scale(self.image.size)
                header_footer_cells.append(
                    {"type": hf, "box": bbox, "text_box": [[]], "text": [[]], "size": [[]]}
                )

        if header_footer_cells:
            if not ocr:
                header_footer_cells = self.populate_text_from_pdf(header_footer_cells, is_header_footer=True)
            else:
                header_footer_cells = self.populate_text_from_pdf_ocr(header_footer_cells, is_header_footer=True)
            
            for hf_t in header_footer_cells:
                cell_text = hf_t["text"]
                self.table_json[hf_t["type"]]["text"] = cell_text
        
        self.header = self.table_json["table_header"]["text"]
        self.footer = self.table_json["table_footer"]["text"]
    
#* --------------------------------------- Text Extraction -------------------------------------------
    @classmethod
    def _add_word(cls, word:str, box:Box, cells:List[dict], size:float) -> List[dict]:
        # get intersection area for the text bbox with all the table cells
        areas = [box.intersect(cell['box']).get_area() for cell in cells]
    
        # get the cell index with the largest area intersection
        cell_index = np.argmax(areas)
    
        # check the area intersected is larger than 50% of the text bbox
        if areas[cell_index] > box.get_area() * 0.5:
        
            # check if there are already text added to the table cell
            if cells[cell_index]['text_box'][-1]:
                # find the ymins, ymaxs of the texts in the current row
                cell_text_bbox = np.array([[bbox.ymin, bbox.ymax] for bbox in cells[cell_index]['text_box'][-1]])
                # find their average ymin and ymax value
                avg_cell_text_ybbox = np.mean(cell_text_bbox,axis=0)
            
                # form a rectangle from all the previous texts in the table cell
                prev_text_box = [cells[cell_index]['box'].xmin, avg_cell_text_ybbox[0],
                                cells[cell_index]['box'].xmax, avg_cell_text_ybbox[1]]
            
                # calculate the intersection of the new word with the bbox of all previous text
                intersect_area = box.intersect(Box(*prev_text_box)).get_area()
            
                if intersect_area < box.get_area() * 0.3:
                    # next row
                    cells[cell_index]['text'].insert(len(cells[cell_index]['text']), [word])
                    cells[cell_index]['size'].insert(len(cells[cell_index]['size']), [size])
                    cells[cell_index]['text_box'].insert(len(cells[cell_index]['text_box']), [box])
                else:
                    # current row
                    # if the word size smaller than 70% of all other texts in the same table cell
                    # superscipt or subscript
                    if cells[cell_index]['size'][-1] and size < 0.7*np.average(cells[cell_index]['size'][-1]):
    
                        # find the average ymin/ymax of the text already belonging to the table cell
                        cell_text_bbox = np.array([[bbox.ymin, bbox.ymax] for bbox in cells[cell_index]['text_box'][-1]])
                        avg_cell_text_ybbox = np.mean(cell_text_bbox,axis=0)
                    
                        # get the top half of the table cell text bbox
                        cell_box_top = [cells[cell_index]['box'].xmin, avg_cell_text_ybbox[0],
                            cells[cell_index]['box'].xmax, (avg_cell_text_ybbox[0] + avg_cell_text_ybbox[1])/2]
    
                        # get the bottom half of the table cell text bbox
                        cell_box_bottom = [cells[cell_index]['box'].xmin, (avg_cell_text_ybbox[0] + avg_cell_text_ybbox[1])/2,
                            cells[cell_index]['box'].xmax, avg_cell_text_ybbox[1]]
                    
                        # check if the new text bbox has more intersection with the top half or bottom half
                        if box.intersect(Box(*cell_box_top)).get_area() > box.intersect(Box(*cell_box_bottom)).get_area():
                            # superscipt
                            cells[cell_index]['text'][-1].append(word.translate(cls.trans))
                        else:
                            # subscript
                            word = word.translate(cls.sub_trans)
                            cells[cell_index]['text'][-1].append(word)
                    else:
                        # not superscipt or subscript
                        cells[cell_index]['text'][-1].append(word)
                        cells[cell_index]['size'][-1].append(size)
                        cells[cell_index]['text_box'][-1].append(box)
            else:
                # add first item
                cells[cell_index]['text'][-1].append(word)
                cells[cell_index]['size'][-1].append(size)
                cells[cell_index]['text_box'][-1].append(box)
            
        return cells

    def populate_text_from_pdf(self, cells, is_header_footer:bool=False):
        page = self.doc[self.page_num]
        for span in (span
            for block in page.get_text('rawdict')['blocks'] if 'lines' in block
            for line in block['lines']
            for span in line['spans']):
            size = span['size']
            char_boxes = []
            current_text = ''
            for char in span['chars'] + [{'c': ' '}]:
                if char['c'].isspace():
                    if len(current_text) > 0:
                        cells = self._add_word(current_text,
                                self._box_pdf_to_image(Box.hull(char_boxes), self.image_page_ratio),
                                cells, size)
                        current_text = ''
                        char_boxes = []
                else:
                    char_boxes.append(Box(*char['bbox']))
                    current_text += char['c']
                    
        for table_cell in cells:
            cell_text = self._postprocess_text(table_cell["text"])
            table_cell["text"] = "" if cell_text==[[]] else cell_text
        
        if not is_header_footer:
            for table_cell in cells:
                for cell in self.table_json["cells"]:
                    if (
                        (cell["row_start"]==list(table_cell["start_pos"])[0])
                        &(cell["row_end"]==list(table_cell["end_pos"])[0])
                        &(cell["col_start"]==list(table_cell["start_pos"])[1])
                        &(cell["col_end"]==list(table_cell["end_pos"])[1])
                    ):
                        cell["text"] = table_cell["text"]
        
        return cells

#* --------------------------------------- OCR Text Extraction -------------------------------------------
    @classmethod
    def _add_word_ocr(cls, page_arr, prep_option, plot=False):
        # transfer image of pdf_file into array
        removed = page_arr.copy()
        
        # turn image into grayscale
        page_arr = cv2.cvtColor(page_arr, cv2.COLOR_BGR2GRAY)
        
        if prep_option == 1:
            thresh = cv2.threshold(page_arr,105, 255, cv2.THRESH_BINARY_INV)[1]
            thresh = 255 - thresh
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            page_arr = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        elif prep_option == 2:
            page_arr = cv2.adaptiveThreshold(page_arr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                            cv2.THRESH_BINARY, 11, 11)
        
        elif prep_option == 3:
            page_arr = cv2.adaptiveThreshold(page_arr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                            cv2.THRESH_BINARY, 11, 11)
            
            thresh = 255 - page_arr

            # Remove vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
            remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                cv2.drawContours(removed, [c], -1, (255,255,255), 15)

            # Remove horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
            remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                cv2.drawContours(removed, [c], -1, (255,255,255), 5)

            # Repair kernel
            repair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            removed = 255 - removed
            dilate = cv2.dilate(removed, repair_kernel, iterations=5)
            dilate = cv2.cvtColor(dilate, cv2.COLOR_BGR2GRAY)
            pre_result = cv2.bitwise_and(dilate, thresh)

            result = cv2.morphologyEx(pre_result, cv2.MORPH_CLOSE, repair_kernel, iterations=5)
            page_arr = cv2.bitwise_and(result, thresh)

            page_arr = 255 - page_arr
        
        d = pytesseract.image_to_data(page_arr, output_type=Output.DICT)
        d_df = pd.DataFrame.from_dict(d)
        d_df = d_df[d_df['level']==5]
        
        if plot:
            display(d_df)
            plt.figure(figsize=(32, 20), dpi=72)
            plt.imshow(page_arr, cmap="gray")
            ax = plt.gca()
            for index, row in d_df.iterrows():
                # col = colors[np.random.randint(len(colors))]
                # print("the text: " + row['text'], "the level: " + str(row['level']))
                (x, y, w, h) = (row['left'], row['top'], row['width'], row['height'])
                ax.add_patch(plt.Rectangle((x, y), w, h,
                                        fill=False, linewidth=1))

            plt.axis('off')
            plt.show()

        text = cls._postprocess_text([d_df['text']])
        return text

    def populate_text_from_pdf_ocr(self, cells, is_header_footer:bool=False):
        image = np.array(self.image)
        for cell in cells:
            cell_bbox = cell['box'].to_array()
            cell_image = image[
                int(np.floor(cell_bbox[1])) : int(np.ceil(cell_bbox[3])), 
                int(np.floor(cell_bbox[0])) : int(np.ceil(cell_bbox[2]))
                ]
            cell['text'] = self._add_word_ocr(cell_image, prep_option=3, plot=False)
        
        if not is_header_footer:
            for table_cell in cells:
                # cell_text = self._postprocess_text(table_cell["text"])
                cell_text = unidecode(table_cell["text"])
                table_cell["text"] = "" if cell_text==[] else cell_text
                
                for cell in self.table_json["cells"]:
                    if (
                        (cell["row_start"]==list(table_cell["start_pos"])[0])
                        &(cell["row_end"]==list(table_cell["end_pos"])[0])
                        &(cell["col_start"]==list(table_cell["start_pos"])[1])
                        &(cell["col_end"]==list(table_cell["end_pos"])[1])
                    ):
                        cell["text"] = table_cell["text"]
        
        return cells

#* ------------------------------------------ HTML ------------------------------------------------
    def get_html_with_text(self, cells, header_rows, verbose: bool = False):
        
        spans = self._process_spans(cells)
        
        html_string = """<html><table>%s</table></html>""" % (
            "".join(self._get_html_tokens(spans, header_rows))
        )
        cell_nodes = list(re.finditer(r"(<td[^<>]*>)(</td>)", html_string))

        offset = 0
        for n, cell in zip(cell_nodes, cells):
            cell_text = "" if cell["text"]==[[]] else cell["text"]
            
            html_string = (
                html_string[: n.end(1) + offset]
                + cell_text
                + html_string[n.start(2) + offset :]
            )
            offset += len(cell_text)

        html_string = html_string.replace("\t", " ")

        if verbose:
            display(HTML(html_string))

        return html_string
    
    def _process_spans(self, cells) -> List:
        spans = [[[1, 1] for _ in range(self.ncol)] for _ in range(self.nrow)]

        for table_cell in cells:
            if table_cell["span"] != (1, 1):
                for i in range(
                    table_cell["start_pos"][0], table_cell["end_pos"][0] + 1
                ):
                    for j in range(
                        table_cell["start_pos"][1], table_cell["end_pos"][1] + 1
                    ):
                        spans[i][j] = None

                spans[table_cell["start_pos"][0]][table_cell["start_pos"][1]] = [
                    table_cell["span"][0],
                    table_cell["span"][1],
                ]

        return spans

    @staticmethod
    def _get_html_tokens(spans, header_rows):
        tokens = ["<thead>"]
        for i, ts_row in enumerate(spans):
            if i == header_rows:
                if tokens[-1] == "<thead>":
                    tokens[-1] = "<tbody>"
                else:
                    tokens += ["</thead>", "<tbody>"]
            tokens.append("<tr>")
            for ts_cell in ts_row:
                if ts_cell is not None:
                    if ts_cell[:2] == [1, 1]:
                        tokens.append("<td>")
                    else:
                        tokens.append("<td")
                        if ts_cell[0] > 1:
                            tokens.append(f' rowspan="{ts_cell[0]}"')
                        if ts_cell[1] > 1:
                            tokens.append(f' colspan="{ts_cell[1]}"')
                        tokens.append(">")
                    tokens.append("</td>")
            tokens.append("</tr>")
        return tokens

#* ----------------------------------------- Plotting --------------------------------------
    def plot_table(self, pad_pct: float = 20.0, verbose: bool = False):
        def plot_tagged_table(table, cells, padded_box, header, footer, image):
            image = np.array(image.convert("RGB"))
            # plot cells
            for cell in cells:
                image = cv2.rectangle(
                    image, pt1=cell[0], pt2=cell[1], color=(0, 0, 255), thickness=2
                )
            # plot table
            image = cv2.rectangle(
                image, pt1=table[0], pt2=table[1], color=(255, 0, 0), thickness=3
            )
            # plot header
            if header:
                image = cv2.rectangle(
                    image,
                    pt1=header[0],
                    pt2=header[1],
                    color=(0, 255, 255),
                    thickness=2,
                )
            # plot header
            if footer:
                image = cv2.rectangle(
                    image,
                    pt1=footer[0],
                    pt2=footer[1],
                    color=(0, 255, 255),
                    thickness=2,
                )
            # crop
            image = image[padded_box[1] : padded_box[3], padded_box[0] : padded_box[2]]
            return image

        overall_bbox = self.table_json["bounding_box"]
        header_bbox = self.table_json["table_header"]["bounding_box"]
        footer_bbox = self.table_json["table_footer"]["bounding_box"]

        if len(header_bbox) > 0:
            overall_bbox = [
                min(overall_bbox[0], header_bbox[0]),
                min(overall_bbox[1], header_bbox[1]),
                max(overall_bbox[2], header_bbox[2]),
                max(overall_bbox[3], header_bbox[3]),
            ]

        if len(footer_bbox) > 0:
            overall_bbox = [
                min(overall_bbox[0], footer_bbox[0]),
                min(overall_bbox[1], footer_bbox[1]),
                max(overall_bbox[2], footer_bbox[2]),
                max(overall_bbox[3], footer_bbox[3]),
            ]

        table_bbox = self.convert_bbox(self.table_json["bounding_box"])
        padded_box = self._add_padding(pad_pct, self.convert_bbox(overall_bbox))
        cell_bboxes = [
            self.convert_bbox(cell["bounding_box"])
            for cell in self.table_json["cells"]
        ]
        header_bbox = self.convert_bbox(
            self.table_json["table_header"]["bounding_box"]
        )
        footer_bbox = self.convert_bbox(
            self.table_json["table_footer"]["bounding_box"]
        )
        table_image = plot_tagged_table(
            table_bbox, cell_bboxes, padded_box, header_bbox, footer_bbox, self.image
        )

        if verbose:
            plt.imshow(table_image)

        return table_image