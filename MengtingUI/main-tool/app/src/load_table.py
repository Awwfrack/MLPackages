from .json_to_html_df import JsonToHTML
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
from .commons import glob_re
import os
import json
import numpy as np
import re
from copy import deepcopy
from IPython.display import display

def postprocess_tables(tables_folder: str):
    # * Table Postprocessing
    table_paths = glob_re(tables_folder, r".+_table\d+.json")

    if len(table_paths) > 0:
        for table_path in table_paths:
            print(f"Table Processing: {table_path}")
            df, soup, header_inds, table_json = table_postprocess_func(table_path)
            table_json["df"] = df.to_dict()
            table_json["html"] = str(soup)
            table_json["header_inds"] = header_inds

            with open(table_path, "w") as f:
                json.dump(table_json, f)


def table_postprocess_func(json_path: str, verbose=False):
    """Get the html string, pandas dataframe, header row indexes of the the table json"""

    # * json to HTML
    processor = JsonToHTML(json_path)
    html_string, json_data = processor.transform(verbose=verbose)
    soup = BeautifulSoup(html_string, "html.parser")
    # table_image = processor.plot_table(verbose=verbose)
    
    # ! For image tables where no text is extracted
    cells = json_data["cells"]
    data = [cell["text"] for cell in cells]
    unique_data = set(data)
    # replace with "No Text"
    if (len(unique_data)==1) & (list(unique_data)[0]==""):
        for cell in cells:
            cell["text"] = "No Text"
    
    # * Build df
    row_count = json_data["row_count"]
    col_count = json_data["col_count"]
    df = pd.DataFrame(".", index=range(row_count), columns=range(col_count))
    cells = json_data["cells"]
    for cell in cells:
        row, col = cell["row_start"], cell["col_start"]
        df.iloc[row, col] = cell["text"]
    
    # * Get number of header rows
    head = soup.thead
    if head is not None:
        nhead = len(head.find_all("tr"))
    else:
        nhead = 0
    head_inds = list(range(nhead))

    return df, soup, head_inds, json_data

def load_table_image(json_path: str, verbose=False, dpi:int=200):
    # * json to HTML
    processor = JsonToHTML(json_path, dpi=dpi)
    html_string = processor.transform(verbose=verbose)

    # * Table image
    table_image = processor.plot_table(verbose=True)

    return table_image, processor.table_json["page_no"]


def concatenate_tables(tables_folder):
    """Concatenate and save to a new json file"""

    report_name = Path(tables_folder).stem.replace("_json", "")
    
    # * Get all tables paths
    table_paths = glob_re(tables_folder, r".+_table\d+.json")
    if len(table_paths)>0:
        table_ids = [re.match(".*_(table\d+).json", table_path).group(1) for table_path in table_paths]
        
        # * Table Overview
        table_overview_path = os.path.join(
            tables_folder, f"table_overview.json"
        )

        # * Concatenate
        if not os.path.exists(table_overview_path):
            raise FileExistsError(f"{tables_folder} does not have a table_overview.json")
        else:
            with open(table_overview_path, "r") as f:
                conts = json.load(f)
                if len(conts) > 0:
                    for cont_sets in conts:
                        assert len(cont_sets) > 1, ValueError(
                            "There is only 1 table id in the list. Nothing to concatenate."
                        )

                        # --------------- Main Table ----------------
                        main_table_id = cont_sets[0]
                        table1_path = [i for i,j in zip(table_paths, table_ids) if j==main_table_id][0]
                        main_table_name = Path(table1_path).stem
                        with open(table1_path, "r") as f:
                            table1_json = json.load(f)
                        df1 = pd.DataFrame(table1_json["df"])
                        header_inds1 = table1_json["header_inds"]
                        document_id1 = table1_json["document_id"]
                        category1 = table1_json["category"]
                        page_num1 = table1_json["page_no"]
                        header = table1_json["table_header"]["text"]
                        footer = table1_json["table_footer"]["text"]
                        bounding_box1 = table1_json["bounding_box"]
                        assert len(header_inds1) > 0, ValueError(
                            f"Table {table1_path} is the main table, however, it does not have a header row. Please double check."
                        )

                        # --------------- Remaining Tables ----------------
                        all_dfs = [df1]
                        document_ids = [document_id1]
                        categories = [category1]
                        page_nums = [page_num1]
                        bounding_boxes = [bounding_box1]

                        concat_axis = 0
                        for table_id in cont_sets[1:]:
                            table_path = [i for i,j in zip(table_paths, table_ids) if j==table_id][0]
                            # table_path = os.path.join(tables_folder, f"{table_name}.json")
                            with open(table_path, "r") as f:
                                table_json = json.load(f)
                            df = pd.DataFrame(table_json["df"])
                            header_inds = table_json["header_inds"]
                            # vertical concatenation if the table also has a header
                            concat_axis = 1 if len(header_inds) > 0 else concat_axis
                            all_dfs.append(df)

                            document_id = table_json["document_id"]
                            document_ids.append(document_id)

                            category = table_json["category"]
                            categories.append(category)

                            page_num = table_json["page_no"]
                            page_nums.append(page_num)

                            header += f"\n{table_json['table_header']['text']}"
                            footer += f"\n{table_json['table_footer']['text']}"

                            bounding_box = table_json["bounding_box"]
                            bounding_boxes.append(bounding_box)

                        # ----------------- Concatenate ----------------
                        if concat_axis == 0:
                            final_df = pd.DataFrame(
                                np.concatenate([d.values for d in all_dfs]),
                                columns=df1.columns,
                            )
                        elif concat_axis == 1:
                            final_df = pd.concat(all_dfs, ignore_index=True, axis=1)

                        # -------------------- Save ----------------
                        document_ids = list(set(document_ids))
                        assert len(document_ids) == 1, ValueError(
                            f"Got different document ids {document_ids}. Something is wrong."
                        )

                        #! We only process Emissions tables
                        if not "Emission" in categories:
                            continue
                        else:
                            table_json_names = [Path([i for i,j in zip(table_paths, table_ids) if j==table_id][0]).stem for table_id in cont_sets]
                            final_json = {
                                "table_id": table_json_names,
                                "bounding_box": bounding_boxes,
                                "df": final_df.to_dict(),
                                "table_header": {
                                    "bounding_box": [],
                                    "text": header.strip(),
                                },
                                "table_footer": {
                                    "bounding_box": [],
                                    "text": footer.strip(),
                                },
                                "header_inds": header_inds1,
                                "document_id": document_ids[0],
                                "page_no": list(set(page_nums)),
                                "category": "Emissions",
                            }
                            with open(
                                os.path.join(tables_folder, f"{main_table_name}_cont.json"),
                                "w",
                            ) as f:
                                json.dump(final_json, f)


def get_table_paths(tables_folder, table_category="Emissions"):
    # get table paths
    report_name = Path(tables_folder).stem.replace("_json", "")
    table_paths = glob_re(tables_folder, r".+_table\d+.*.json")
    table_ids = [re.match(".*_(table\d+.*).json", table_path).group(1) for table_path in table_paths]
    table_paths_ids = deepcopy(table_ids)
    
    # remove continuous tables
    table_overview_path = os.path.join(
        tables_folder, f"table_overview.json"
    )

    if not os.path.exists(table_overview_path):
        raise FileExistsError(f"{tables_folder} does not have a table_overview.json")
    else:
        with open(table_overview_path, "r") as f:
            conts = json.load(f)

        table_overview_ids = [i for j in conts for i in j]
        
        for table_overview_id in table_overview_ids:
            try:
                table_paths_ids.remove(table_overview_id)
            except:
                raise FileExistsError(
                    f"Table {table_overview_path} does not exist but is in the table_overview.json. Please check."
                )
                
    table_paths = [table_path for table_path in table_paths if re.match(".*_(table\d+.*).json", table_path).group(1) in table_paths_ids]

    # ! Only load emissions tables
    if isinstance(table_category, str):
        table_category = [table_category]

    new_table_paths = []
    for table_path in table_paths:
        with open(table_path, "r") as f:
            table_json = json.load(f)

        if table_json["category"] in table_category:
            new_table_paths.append(table_path)

    return new_table_paths
