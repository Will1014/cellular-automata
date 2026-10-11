"""Wolfram's classes for the 88 distinct elementary rules.

Source: G. J. Martinez (2013), "A Note on Elementary Cellular Automata
Classification", Journal of Cellular Automata 8:233-259, Table 2,
arXiv:1306.5577. Each list holds the representative (smallest-numbered)
rule of each mirror/complement equivalence class. Class assignments for a
few rules (notably 41 and 106 in class IV) are contested in the literature.
"""

CLASSES = {
    1: [0, 8, 32, 40, 128, 136, 160, 168],
    2: [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 19, 23, 24, 25, 26, 27, 28, 29,
        33, 34, 35, 36, 37, 38, 42, 43, 44, 46, 50, 51, 56, 57, 58, 62, 72, 73, 74, 76,
        77, 78, 94, 104, 108, 130, 132, 134, 138, 140, 142, 152, 154, 156, 162, 164, 170,
        172, 178, 184, 200, 204, 232],
    3: [18, 22, 30, 45, 60, 90, 105, 122, 126, 146, 150],
    4: [41, 54, 106, 110],
}

CLASS_OF = {rule: c for c, rules in CLASSES.items() for rule in rules}
