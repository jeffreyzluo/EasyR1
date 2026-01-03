"""
Keyphrases dictionary v0 for CtrlG processing.
"""

CONSTRAINTS_DICT = {
    # --- VISUAL EXTRACTION & VERIFICATION (VLM Specific) ---
    "VisualGrounding": [[
        " notation",
        " symbols",
        " title",
        " alignment",
        " coordinates",
        " layout",
        " scale",
        " shape",
        " image",
        " describe",
        " see",
        " identify",
        " observe",
        " inspect"
    ]],

    # --- LOGICAL REASONING (From Original List) ---
    "General": [[
        " backwards",
        " reverse",
        " recall",
        " imagine",
        " alternatively",
        " maybe",
        " small example",
        " pattern"
    ]],
    "Reflection": [[
        " Wait",
        " double-check",
        " doube check"
        " verify",
        " Re-examine",
        " re-examine",
        " missed"
    ]]
}
