#!/usr/bin/env python3
"""Audit released MINTNN feature archives without extracting them."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

MOF_FEATURES = [
    ("CA", "facet"),
    ("FPRC", "forman"),
    ("PH", "homology"),
    ("PL", "lap"),
]

LD50_FEATURES = [
    ("CA/facet", ["data/ld50/topology_features/CA/facet/{split}/"]),
    (
        "EIC/single_direction/curvature",
        [
            "data/ld50/topology_features/EIC/single_direction/curvature/{split}/",
            "data/ld50/topology_features/EIC/curvature/{split}/",
        ],
    ),
    (
        "EIC/bidirectional/curvature",
        [
            "data/ld50/topology_features/EIC/bidirectional/curvature/{split}/",
            "data/ld50/topology_features/EIC_BI/curvature/{split}/",
        ],
    ),
    ("FPRC/forman", ["data/ld50/topology_features/FPRC/forman/{split}/"]),
    ("PH/homology", ["data/ld50/topology_features/PH/homology/{split}/"]),
    ("PL/lap", ["data/ld50/topology_features/PL/lap/{split}/"]),
]


def column_name(cell_ref: str) -> str:
    return "".join(ch for ch in cell_ref if ch.isalpha())


def cell_value(cell, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//a:t", XLSX_NS))
    value_node = cell.find("a:v", XLSX_NS)
    if value_node is None:
        return ""
    if cell_type == "s":
        return shared_strings[int(value_node.text)]
    return value_node.text or ""


def read_xlsx_rows_from_archive(archive: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    workbook_bytes = archive.read(member)
    with zipfile.ZipFile(io.BytesIO(workbook_bytes)) as workbook:
        shared_strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", XLSX_NS):
                shared_strings.append("".join(text.text or "" for text in item.findall(".//a:t", XLSX_NS)))

        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        raw_rows = []
        for row in sheet.findall(".//a:sheetData/a:row", XLSX_NS):
            raw_rows.append({
                column_name(cell.attrib.get("r", "")): cell_value(cell, shared_strings)
                for cell in row.findall("a:c", XLSX_NS)
            })

    headers = raw_rows[0]
    col_to_name = {col: name for col, name in headers.items() if name}
    return [
        {name: raw.get(col, "") for col, name in col_to_name.items()}
        for raw in raw_rows[1:]
    ]


def normalized_cas_key(name: str):
    name = name.strip()
    slash_match = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", name)
    if slash_match:
        month, day, year = slash_match.groups()
        return (str(int(year)), str(int(month)), str(int(day)))
    dash_match = re.fullmatch(r"(\d+)-(\d+)-(\d+)", name)
    if dash_match:
        first, second, third = dash_match.groups()
        return (str(int(first)), str(int(second)), str(int(third)))
    return None


def build_cas_index(paths: list[str]) -> dict[tuple[str, str, str], list[str]]:
    index: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for path in paths:
        stem = path.rsplit("/", 1)[-1][:-4]
        key = normalized_cas_key(stem)
        if key is not None:
            index[key].append(stem)
    return index


def audit_mof(mof_zip: Path) -> dict:
    with zipfile.ZipFile(mof_zip) as archive:
        names = set(archive.namelist())
        results = {}
        for prop in ["O2uptakemolkg", "N2uptakemolkg"]:
            rows = read_xlsx_rows_from_archive(archive, f"data/mof/2STD/{prop}.xlsx")
            ids = []
            for row in rows:
                mofid = row.get("MOFRefcodes", "")
                value = row.get(prop, "")
                if not mofid or value in ("", None):
                    continue
                try:
                    float(value)
                except ValueError:
                    continue
                ids.append(str(mofid))

            feature_results = {}
            for family, topology in MOF_FEATURES:
                missing = [
                    mofid
                    for mofid in ids
                    if f"data/mof/features/{family}/{topology}/{prop}/{mofid}.npy" not in names
                ]
                feature_results[f"{family}/{topology}"] = {
                    "labels": len(ids),
                    "missing": len(missing),
                    "missing_examples": missing[:5],
                }
            results[prop] = feature_results
    return results


def audit_ld50(ld50_zip: Path) -> dict:
    with zipfile.ZipFile(ld50_zip) as archive:
        names = set(archive.namelist())
        results = {}
        for split in ["train", "test"]:
            csv_text = archive.read(f"data/ld50/LD50_{split}.csv").decode("utf-8-sig")
            rows = list(csv.DictReader(io.StringIO(csv_text)))
            ids = [row["filename"].strip() for row in rows if row.get("filename", "").strip()]

            feature_results = {}
            for feature_name, prefix_templates in LD50_FEATURES:
                prefixes = [template.format(split=split) for template in prefix_templates]
                feature_files = [
                    path
                    for path in names
                    if any(path.startswith(prefix) for prefix in prefixes) and path.endswith(".npy")
                ]
                index = build_cas_index(feature_files)
                missing = []
                ambiguous = []
                for name in ids:
                    exact_matches = [f"{prefix}{name}.npy" for prefix in prefixes]
                    if any(exact in names for exact in exact_matches):
                        continue
                    key = normalized_cas_key(name)
                    candidates = index.get(key, []) if key is not None else []
                    if len(candidates) == 1:
                        continue
                    if len(candidates) > 1:
                        ambiguous.append({"name": name, "candidates": candidates[:5]})
                    else:
                        missing.append(name)
                source_prefixes = [
                    prefix
                    for prefix in prefixes
                    if any(path.startswith(prefix) and path.endswith(".npy") for path in names)
                ]
                feature_results[feature_name] = {
                    "labels": len(ids),
                    "missing": len(missing),
                    "ambiguous": len(ambiguous),
                    "source_prefix": source_prefixes[0] if source_prefixes else None,
                    "missing_examples": missing[:5],
                    "ambiguous_examples": ambiguous[:2],
                }
            results[split] = feature_results
    return results


def has_failures(report: dict) -> bool:
    for app_report in report.values():
        for split_report in app_report.values():
            for item in split_report.values():
                if item.get("missing", 0) or item.get("ambiguous", 0):
                    return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mof-zip", required=True)
    parser.add_argument("--ld50-zip", required=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    report = {
        "mof": audit_mof(Path(args.mof_zip)),
        "ld50": audit_ld50(Path(args.ld50_zip)),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    return 1 if has_failures(report) else 0


if __name__ == "__main__":
    raise SystemExit(main())
