"""
Keyphrases dictionary v0 for CtrlG processing.
"""

CONSTRAINTS_DICT = {
    # --- VISUAL EXTRACTION & VERIFICATION (VLM Specific) ---
    "SymbolVerification": [[
        " Verify the numbers",
        " Check the signs",
        " Re-read the text",
        " Check notation",
        " Inspect the symbols"
    ]],
    "GeometricGrounding": [[
        " Check alignment",
        " Trace the lines",
        " Verify coordinates",
        " Analyze the layout",
        " Inspect the shape"
    ]],
    "VisualReinspection": [[
        " Look closer",
        " Re-examine the image",
        " Check the scale",
        " Verify units",
        " Any missed details?"
    ]],

    # --- LOGICAL REASONING (From Original List) ---
    "Backwarding": [[
        " Working backwards",
        " work backwards",
        " Thinking in reverse",
        " think in reverse",
    ]],
    "Backtracking": [[
        " Let me go back",
        " Going back",
        " Undo the last step",
        " undo the last step",
        " try another way"
    ]],
    "Induction": [[
        " try a small example",
        " Trying a small example",
        " test with simple numbers",
        " look for a pattern"
    ]],
    "Counterfactual": [[
        " What if I",
        " what if I",
        " imagine if I",
        " Alternatively, I could",
    ]],
    "OverthinkingAwareness": [[
        " is getting too long",
        " overcomplicating",
        " going in circles",
        " overthink",
        " sufficient to answer",
    ]]
}
