from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    start_character: int
    end_character: int


def chunk_text(content: str, *, max_characters: int = 1800, overlap: int = 200) -> tuple[TextChunk, ...]:
    if max_characters < 200:
        raise ValueError("max_characters must be at least 200")
    if overlap < 0 or overlap >= max_characters:
        raise ValueError("overlap must be non-negative and smaller than max_characters")
    stripped = content.strip()
    if not stripped:
        return ()
    chunks: list[TextChunk] = []
    start = 0
    while start < len(stripped):
        provisional_end = min(start + max_characters, len(stripped))
        end = provisional_end
        if provisional_end < len(stripped):
            boundary = stripped.rfind("\n", start, provisional_end)
            if boundary <= start:
                boundary = stripped.rfind(" ", start, provisional_end)
            if boundary > start + max_characters // 2:
                end = boundary
        text = stripped[start:end].strip()
        if text:
            chunks.append(TextChunk(len(chunks), text, start, end))
        if end >= len(stripped):
            break
        start = max(end - overlap, start + 1)
    return tuple(chunks)

