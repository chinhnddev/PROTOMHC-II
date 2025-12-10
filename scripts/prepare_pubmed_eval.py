"""Clean NeoAntigen PubMed dataset (positives + negatives) into a simple CSV.

The provided XLSX has two sheets (immunogenicity / non-immunogenicity) but is
malformed for typical Excel readers. We parse it directly from the XLSX ZIP and
export a tidy file with columns: sequence,label.

Usage:
python scripts/prepare_pubmed_eval.py \
    --input data/processed/NeoAntigen-PubData-2024-2025.xlsx \
    --output data/processed/neoantigen_pubmed_eval.csv \
    --min-len 9 --max-len 13
"""
import argparse
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

NS = {"main": "http://purl.oclc.org/ooxml/spreadsheetml/main"}
VALID_AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


def load_shared_strings(z: zipfile.ZipFile):
    data = z.read("xl/sharedStrings.xml")
    root = ET.fromstring(data)
    strings = []
    for si in root:
        # concatenate all <t> within the si element
        text = "".join(
            node.text or "" for node in si.iter("{http://purl.oclc.org/ooxml/spreadsheetml/main}t")
        )
        strings.append(text)
    return strings


def sheet_to_dataframe(z: zipfile.ZipFile, sheet_xml: str, shared_strings):
    xml_bytes = z.read(sheet_xml)
    root = ET.fromstring(xml_bytes)
    rows = []
    for row in root.findall(".//main:row", NS):
        cells = []
        for c in row.findall("main:c", NS):
            t = c.attrib.get("t")
            v = c.find("main:v", NS)
            if v is None:
                cells.append("")
                continue
            val = v.text or ""
            if t == "s":  # shared string
                cells.append(shared_strings[int(val)])
            else:
                cells.append(val)
        rows.append(cells)
    if not rows:
        return pd.DataFrame()
    header, data = rows[0], rows[1:]
    return pd.DataFrame(data, columns=header)


def load_pubmed(path: Path):
    with zipfile.ZipFile(path) as z:
        shared_strings = load_shared_strings(z)
        pos_df = sheet_to_dataframe(z, "xl/worksheets/sheet1.xml", shared_strings)
        neg_df = sheet_to_dataframe(z, "xl/worksheets/sheet2.xml", shared_strings)

    # Normalize headers and keep only peptide + label
    for df in (pos_df, neg_df):
        df.columns = [c.strip() for c in df.columns]
        df.rename(columns={"Peptide": "sequence", "immunogenicity": "label"}, inplace=True)
        df["sequence"] = df["sequence"].astype(str).str.strip()

    pos_df["label"] = 1
    neg_df["label"] = 0
    combined = pd.concat([pos_df[["sequence", "label"]], neg_df[["sequence", "label"]]], ignore_index=True)
    return combined


def filter_and_dedup(df: pd.DataFrame, min_len=None, max_len=None):
    df = df.copy()
    df = df[df["sequence"].apply(lambda s: bool(VALID_AA_RE.match(s)))]

    if min_len or max_len:
        df = df[df["sequence"].str.len().between(min_len or 0, max_len or 10**9)]

    # Drop exact duplicate rows; if conflicting labels existed, you could resolve here.
    df = df.drop_duplicates(subset=["sequence", "label"])
    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare PubMed neoantigen dataset for eval.")
    parser.add_argument("--input", type=Path, required=True, help="Path to NeoAntigen-PubData-2024-2025.xlsx")
    parser.add_argument("--output", type=Path, required=True, help="Where to save the cleaned CSV")
    parser.add_argument("--min-len", type=int, default=9, help="Minimum peptide length (default 9)")
    parser.add_argument("--max-len", type=int, default=13, help="Maximum peptide length (default 13)")
    args = parser.parse_args()

    df = load_pubmed(args.input)
    before = len(df)
    df = filter_and_dedup(df, min_len=args.min_len, max_len=args.max_len)
    after = len(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Loaded {before} rows; kept {after} after filtering.")
    print(f"Output: {args.output}")
    print(f"Positives: {df[df.label==1].shape[0]}  Negatives: {df[df.label==0].shape[0]}")
    lengths = df['sequence'].str.len()
    print(f"Length range: {lengths.min()}–{lengths.max()}, top lengths: {lengths.value_counts().head().to_dict()}")


if __name__ == "__main__":
    main()
