import pandas as pd
from pathlib import Path
import os

parent_dir = Path(__file__).parent
units = pd.read_csv(os.path.join(parent_dir, "units.csv"))["Standardized Unit"].tolist()

q31_inputs = [
    {
        "name": "Year",
        "id": "year",
        "comp_type": "dropdown",
        "comp_kwargs": {
            "options": ["not specified"] + [str(y) for y in range(2010, 2081)]
        },
        "tooltip": "The year of emission. If no year is specified, select 'not specified'.",
        "color": "MistyRose",
        "required": True,
    },
    {
        "name": "Scope",
        "id": "scope",
        "comp_type": "dropdown",
        "comp_kwargs": {
            "options": [
                "not specified",
                "1",
                "2",
                "3",
                "1 & 2",
                "1 & 3",
                "2 & 3",
                "1 & 2 & 3",
            ]
        },
        "tooltip": "The scope of emission. If no scope is specified, select 'not specified'.",
        "color": "NavajoWhite",
        "required": True,
    },
    {
        "name": "Absolute or Intensity",
        "id": "absolute",
        "comp_type": "dropdown",
        "comp_kwargs": {"options": ["absolute", "intensity"]},
        "tooltip": "Whether it is an absolute emission or emission intensity. If it is not directly specified, please select based on the unit.",
        "color": "LightGoldenRodYellow",
        "required": True,
    },
    {
        "name": "Value",
        "id": "value",
        "comp_type": "input",
        "comp_kwargs": {
            "type": "number",
            "inputmode": "numeric",
            "min": 0.0,
        },
        "tooltip": "The emission value.",
        "color": "HoneyDew",
        "required": True,
    },
    {
        "name": "Raw Unit",
        "id": "raw-unit",
        "comp_type": "input",
        "comp_kwargs": {"type": "text"},
        "tooltip": "The unit of the emission value. Please copy the text from the table directly. If no unit is specified, leave empty",
        "color": "LightCyan",
        "required": True,
    },
    {
        "name": "Standardized Unit",
        "id": "std-unit",
        "comp_type": "dropdown",
        "comp_kwargs": {
            "options": ["not specified", "not found"] + units,
        },
        "tooltip": "Select what the raw unit should map to. If no mapping is found, then select 'not found'. If raw unit is empty, select 'not specified'",
        "color": "LightBlue",
        "required": True,
    },
    {
        "name": "Market or Location based",
        "id": "market",
        "comp_type": "dropdown",
        "comp_kwargs": {"options": ["not specified", "market", "location"]},
        "tooltip": "Whether a scope 2 emission (both absolute and intensity) is market-based or location-based. This entry only applies to scope 2 emissions. If not determined, please select 'not specified'.",
        "color": "Lavender",
        "required": True,
    },
    {
        "name": "Category Name",
        "id": "category",
        "comp_type": "input",
        "comp_kwargs": {"type": "text"},
        "tooltip": "If emission is reported by categories, please specify the category name here. If the emission is not at the category-level, leave empty.",
        "color": "LightGray",
        "required": False,
    },
    {
        "name": "Additional Information",
        "id": "additional",
        "comp_type": "input",
        "comp_kwargs": {"type": "text"},
        "tooltip": "Any additional information that is associated with the emission. For example, whether the emission is equity or control. Leave blank if there is no additional information",
        "color": "SkyBlue",
        "required": False,
    },
    {
        "name": "Percentage Type",
        "id": "percentage-type",
        "comp_type": "dropdown",
        "comp_kwargs": {"options": ["not specified", "proportion", "increase", "decrease"], "disabled": True},
        "tooltip": "Only applies when the unit is %. Whether the percentage change refers to a proportion of emissions or emission increase/decrease from a base year.",
        "color": "DarkKhaki",
        "required": False,
    },
    {
        "name": "Base Year",
        "id": "base-year",
        "comp_type": "dropdown",
        "comp_kwargs": {
            "options": ["not specified"] + [str(y) for y in range(2010, 2081)],
            "disabled": True
        },
        "tooltip": "Only applies when the unit is %. The base year of emission.",
        "color": "Thistle",
        "required": False,
    },
    {
        "name": "Base Scope",
        "id": "base-scope",
        "comp_type": "dropdown",
        "comp_kwargs": {
            "options": [
                "not specified",
                "1",
                "2",
                "3",
                "1 & 2",
                "1 & 3",
                "2 & 3",
                "1 & 2 & 3",
            ],
            "disabled": True
        },
        "tooltip": "Only applies when the unit is %. The scope of the base year emission. If no scope is specified, select 'not specified'.",
        "color": "SandyBrown",
        "required": False,
    },
]

q32_inputs = [
    {
        "name": "Exclusion or Inclusion",
        "id": "exclusion",
        "comp_type": "dropdown",
        "comp_kwargs": {"options": ["exclusion", "inclusion"]},
        "tooltip": "Whether the answer determines what has been excluded from the emissions calculation or what has been included in the calculation.",
        "color": "SkyBlue",
        "required": True,
    },
    {
        "name": "Answer",
        "id": "exclusion-answer",
        "comp_type": "text-area",
        "comp_kwargs": {"type": "text"},
        "tooltip": "Specify what has been excluded/included from the emissions calculation",
        "color": "NavajoWhite",
        "required": True,
    },
]

q4_inputs = [
    {
        "name": "Year",
        "id": "year",
        "comp_type": "dropdown",
        "comp_kwargs": {
            "options": ["not specified"] + [str(y) for y in range(2010, 2081)]
        },
        "tooltip": "The year of the net-zero target. If no year is specified, select 'not specified'.",
        "color": "SkyBlue",
        "required": True,
    },
    {
        "name": "Scope",
        "id": "scope",
        "comp_type": "dropdown",
        "comp_kwargs": {
            "options": [
                "not specified",
                "1",
                "2",
                "3",
                "1 & 2",
                "1 & 3",
                "2 & 3",
                "1 & 2 & 3",
            ]
        },
        "tooltip": "The scope of the net-zero target. If no scope is specified, select 'not specified'.",
        "color": "NavajoWhite",
        "required": True,
    },
]
