from __future__ import annotations


_MOJIBAKE_REPLACEMENTS = {
    "\u00e2\u0080\u0099": "'",
    "\u00e2\u0080\u0098": "'",
    "\u00e2\u0080\u009c": '"',
    "\u00e2\u0080\u009d": '"',
    "\u00e2\u0080\u0093": "-",
    "\u00e2\u0080\u0094": "-",
    "\u00e2\u0080\u00b2": "\u2032",
    "\u00e2\u0080\u00b3": "\u2033",
    "\u00e2\u0086\u0092": "\u2192",
    "\u00e2\u0088\u0092": "\u2212",
    "\u00e2\u0088\u0091": "\u2211",
    "\u00e2\u0088\u0097": "\u2217",
    "\u00e2\u0088\u009a": "\u221a",
    "\u00e2\u0088\u009e": "\u221e",
    "\u00e2\u0088\u00ab": "\u222b",
    "\u00e2\u0088\u00b4": "\u2234",
    "\u00e2\u0088\u0088": "\u2208",
    "\u00e2\u0088\u0089": "\u2209",
    "\u00e2\u0088\u0086": "\u2206",
    "\u00e2\u0088\u0082": "\u2202",
    "\u00e2\u0089\u00a0": "\u2260",
    "\u00e2\u0089\u00a4": "\u2264",
    "\u00e2\u0089\u00a5": "\u2265",
    "\u00e2\u0089\u0088": "\u2248",
    "\u00e2\u008a\u0086": "\u2286",
    "\u00e2\u008c\u008a": "\u230a",
    "\u00e2\u008c\u008b": "\u230b",
    "\u00c3\u0097": "\u00d7",
    "\u00c2\u00b7": "\u00b7",
    "\u00c2\u00b5": "\u03bc",
    "\u00c2\u00a0": " ",
    "\u00ce\u00b1": "\u03b1",
    "\u00ce\u00b2": "\u03b2",
    "\u00ce\u00b3": "\u03b3",
    "\u00ce\u00b4": "\u03b4",
    "\u00ce\u00b5": "\u03b5",
    "\u00ce\u00ba": "\u03ba",
    "\u00ce\u00bb": "\u03bb",
    "\u00ce\u00bc": "\u03bc",
    "\u00ce\u00be": "\u03be",
    "\u00cf\u0081": "\u03c1",
    "\u00cf\u0087": "\u03c7",
    "\u00cf\u0089": "\u03c9",
}


def clean_markdown_text(text: str) -> str:
    cleaned = str(text or "")
    for broken, fixed in _MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(broken, fixed)
    return cleaned
