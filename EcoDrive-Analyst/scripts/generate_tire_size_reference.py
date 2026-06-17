from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


DEFAULT_DYNAMIC_FACTOR = 0.97
DEFAULT_OUTPUT = Path("data/reference/tire_size_reference.csv")
SIZE_PATTERN = re.compile(r"^\s*(?P<width>\d{3})/(?P<aspect>\d{2,3})R(?P<rim>\d{2})\s*$", re.IGNORECASE)


def parse_size_code(size_code: str) -> dict:
    match = SIZE_PATTERN.match(str(size_code or "").strip())
    if not match:
        raise ValueError(
            f"Unsupported tire size format: {size_code!r}. Expected forms like '205/55R16'."
        )
    width_mm = float(match.group("width"))
    aspect_ratio_pct = float(match.group("aspect"))
    rim_in = float(match.group("rim"))
    return {
        "size_code": str(size_code).strip().upper(),
        "width_mm": width_mm,
        "aspect_ratio_pct": aspect_ratio_pct,
        "rim_in": rim_in,
    }


def build_reference_row(size_code: str, dynamic_factor: float = DEFAULT_DYNAMIC_FACTOR) -> dict:
    parsed = parse_size_code(size_code)
    sidewall_height_mm = parsed["width_mm"] * parsed["aspect_ratio_pct"] / 100.0
    rim_diameter_mm = parsed["rim_in"] * 25.4
    unloaded_diameter_mm = rim_diameter_mm + 2.0 * sidewall_height_mm
    unloaded_radius_mm = unloaded_diameter_mm / 2.0
    unloaded_circumference_mm = math.pi * unloaded_diameter_mm
    expected_rolling_radius_mm = unloaded_radius_mm * dynamic_factor
    expected_effective_circumference_mm = unloaded_circumference_mm * dynamic_factor

    return {
        **parsed,
        "unloaded_diameter_mm": round(unloaded_diameter_mm, 3),
        "unloaded_radius_mm": round(unloaded_radius_mm, 3),
        "unloaded_circumference_mm": round(unloaded_circumference_mm, 3),
        "dynamic_factor": round(dynamic_factor, 5),
        "expected_rolling_radius_mm": round(expected_rolling_radius_mm, 3),
        "expected_effective_circumference_mm": round(expected_effective_circumference_mm, 3),
        "source": "geometric_estimate",
        "notes": "Nominal geometry estimate with explicit dynamic factor; not measured or certified tire data.",
    }


def read_size_codes(args) -> list[str]:
    if args.sizes:
        return [str(s).strip() for s in args.sizes if str(s).strip()]
    if args.input_csv:
        path = Path(args.input_csv)
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                raise ValueError("Input CSV must contain a header row.")

            fieldnames = set(reader.fieldnames)
            candidate_columns = ("size_code", "SizeCode", "size", "Size", "Medida")
            column_name = next((name for name in candidate_columns if name in fieldnames), None)
            if not column_name:
                raise ValueError(
                    "Input CSV must contain one of these columns: "
                    "'size_code', 'size', or 'Medida'."
                )
            return [str(row[column_name]).strip() for row in reader if str(row.get(column_name, "")).strip()]
    raise ValueError("Provide at least one --size value or an --input-csv with a size_code column.")


def write_reference(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "size_code",
        "width_mm",
        "aspect_ratio_pct",
        "rim_in",
        "unloaded_diameter_mm",
        "unloaded_radius_mm",
        "unloaded_circumference_mm",
        "dynamic_factor",
        "expected_rolling_radius_mm",
        "expected_effective_circumference_mm",
        "source",
        "notes",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate geometric tire size reference rows for EcoDrive tire roadload lookup."
    )
    parser.add_argument("--size", dest="sizes", action="append", help="Nominal tire size code, e.g. 205/55R16")
    parser.add_argument("--input-csv", help="CSV with a size_code column to batch-generate rows.")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT), help="Output CSV path.")
    parser.add_argument("--dynamic-factor", type=float, default=DEFAULT_DYNAMIC_FACTOR, help="Dynamic approximation factor.")
    args = parser.parse_args()

    size_codes = read_size_codes(args)
    rows = [build_reference_row(size_code, dynamic_factor=float(args.dynamic_factor)) for size_code in size_codes]
    rows = sorted(rows, key=lambda row: row["size_code"])
    write_reference(rows, Path(args.output_csv))


if __name__ == "__main__":
    main()
