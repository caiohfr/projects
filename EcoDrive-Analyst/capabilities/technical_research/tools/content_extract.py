from __future__ import annotations

from io import BytesIO
import re
import zipfile
from xml.etree import ElementTree

from bs4 import BeautifulSoup
from pypdf import PdfReader


def extract_document_text(content: bytes, content_type: str) -> str:
    lowered = content_type.lower()
    if "spreadsheetml" in lowered:
        return _extract_xlsx_text(content)
    if "pdf" in lowered:
        reader = PdfReader(BytesIO(content))
        return "\n\n".join(
            f"--- PAGE {index} ---\n{(page.extract_text() or '').strip()}"
            for index, page in enumerate(reader.pages, start=1)
        ).strip()
    decoded = content.decode("utf-8", errors="replace")
    if "html" in lowered:
        soup = BeautifulSoup(decoded, "html.parser")
        # Modern OEM specification pages frequently render their visible table
        # from embedded JSON. Preserve only data-bearing script blocks with
        # component-research terms; executable instructions remain untrusted
        # source text and are handled as data by the extraction boundary.
        technical_terms = (
            "transmission", "gear ratio", "final drive", "drag coefficient",
            "frontal area", "tire size", "tyre size", "\"tires\"",
        )
        embedded_blocks: list[str] = []
        embedded_chars = 0
        for node in soup.find_all("script"):
            payload = node.string or node.get_text(" ", strip=True)
            if not payload or not any(term in payload.lower() for term in technical_terms):
                continue
            lowered_payload = payload.lower()
            windows: list[tuple[int, int]] = []
            for term in technical_terms:
                for match in list(re.finditer(re.escape(term), lowered_payload))[:8]:
                    windows.append((max(0, match.start() - 1_300), min(len(payload), match.end() + 1_300)))
            merged: list[tuple[int, int]] = []
            for start, end in sorted(windows):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            for start, end in merged:
                remaining = 500_000 - embedded_chars
                if remaining <= 0:
                    break
                excerpt = payload[start:min(end, start + remaining)]
                embedded_blocks.append(excerpt)
                embedded_chars += len(excerpt)
        for node in soup(["script", "style", "noscript"]):
            node.decompose()
        visible = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
        if embedded_blocks:
            return visible + "\n\n--- EMBEDDED STRUCTURED PAGE DATA ---\n" + "\n".join(embedded_blocks)
        return visible
    return decoded


def _extract_xlsx_text(content: bytes) -> str:
    """Read cell values from XLSX with the standard library only."""
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(BytesIO(content)) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.text or "" for node in item.findall(".//x:t", namespace)) for item in root.findall("x:si", namespace)]
        lines: list[str] = []
        sheets = sorted(name for name in archive.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name))
        for sheet in sheets:
            root = ElementTree.fromstring(archive.read(sheet))
            lines.append(f"--- XLSX {sheet.rsplit('/', 1)[-1]} ---")
            for row in root.findall(".//x:row", namespace):
                values: list[str] = []
                for cell in row.findall("x:c", namespace):
                    kind = cell.get("t", "")
                    value_node = cell.find("x:v", namespace)
                    inline = cell.find("x:is", namespace)
                    value = ""
                    if kind == "inlineStr" and inline is not None:
                        value = "".join(node.text or "" for node in inline.findall(".//x:t", namespace))
                    elif value_node is not None:
                        value = value_node.text or ""
                        if kind == "s" and value.isdigit() and int(value) < len(shared):
                            value = shared[int(value)]
                    values.append(value)
                if any(value.strip() for value in values):
                    lines.append("\t".join(values))
        return "\n".join(lines)
