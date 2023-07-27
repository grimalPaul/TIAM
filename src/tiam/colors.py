import colour

C = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]


def munsell_to_lab(hue, brightness, chroma):
    XYZ = munsell_to_XYZ(hue, brightness, chroma)
    XYZ_LAB = XYZ_to_lab(XYZ)
    return XYZ_LAB


def munsell_to_XYZ(hue, brightness, chroma):
    MRS_c = f"{hue} {brightness}/{chroma}"
    try:
        xyY = colour.munsell_colour_to_xyY(MRS_c)
    except:
        raise ValueError(f"error with {MRS_c}")
    return colour.xyY_to_XYZ(xyY)


def XYZ_to_lab(color):
    return colour.XYZ_to_Lab(color, C)


def sRGB_to_XYZ(sRGB):
    return colour.sRGB_to_XYZ(sRGB, C)


def sRGB_to_lab(sRGB):
    xyz = sRGB_to_XYZ(sRGB)
    return XYZ_to_lab(xyz)


METHODS = {
    "lab": munsell_to_lab,
}

COLORS = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
]


def interpolate_best_example(best_examples_per_color, methods=METHODS):
    one_best_example = {}
    for color, values in best_examples_per_color.items():
        one_best_example[color] = {k: None for k in methods.keys()}
        for key, method in methods.items():
            for v in values:
                if one_best_example[color][key] is None:
                    one_best_example[color][key] = method(
                        HUE[v[1]], BRIGHTNESS[v[0]], CHROMA[v[0]][v[1]]
                    )
                else:
                    one_best_example[color][key] += method(
                        HUE[v[1]], BRIGHTNESS[v[0]], CHROMA[v[0]][v[1]]
                    )
            one_best_example[color][key] /= len(values)
    return one_best_example


# XYZ color space representation of the 'd65' dark and white references illuminant
XYZ_dark_black = [0.0, 0.0, 0.0]
XYZ_white = [0.95047, 1.0, 1.08883]

############## Munsell array ##################
# vertical axis brightness/lightness/value
BRIGHTNESS = list(range(9, 1, -1))

# horizontal axis hue
hue_values = ["2.5", "5", "7.5", "10"]
hue_colors = ["R", "YR", "Y", "GY", "G", "BG", "B", "PB", "P", "RP"]
HUE = [h + c for c in hue_colors for h in hue_values]

# chroma
# Find chroma values https://link.springer.com/referenceworkentry/10.1007/978-1-4419-8071-7_113
row1 = [2] * 8 + [4] + [6] * 4 + [4] + [2] * 26
row2 = (
    [6] * 6
    + [8] * 1
    + [14]
    + [10]
    + [14]
    + [12] * 3
    + [10] * 2
    + [8] * 2
    + [6] * 4
    + [4] * 7
    + [6] * 2
    + [4] * 4
    + [6] * 6
)
row3 = (
    [8] * 2
    + [10] * 3
    + [14] * 3
    + [12] * 6
    + [10] * 3
    + [8] * 5
    + [6] * 5
    + [8] * 3
    + [6] * 4
    + [8] * 2
    + [10] * 2
    + [8] * 2
)
row4 = (
    [12] * 3
    + [14]
    + [10]
    + [12] * 3
    + [10] * 6
    + [12] * 2
    + [10] * 4
    + [8] * 7
    + [10] * 3
    + [8] * 4
    + [10] * 4
    + [12] * 2
)
row5 = (
    [14] * 3
    + [10]
    + [14]
    + [12]
    + [10] * 2
    + [8] * 6
    + [10]
    + [12] * 2
    + [10] * 4
    + [8] * 6
    + [10]
    + [12] * 2
    + [10] * 5
    + [12] * 3
    + [14] * 2
)
row6 = (
    [14] * 4
    + [10]
    + [8] * 2
    + [6] * 7
    + [8] * 2
    + [10] * 4
    + [8] * 4
    + [6] * 2
    + [8] * 2
    + [10] * 2
    + [12]
    + [10] * 9
)
row7 = (
    [10] * 2
    + [12]
    + [10]
    + [8]
    + [6] * 3
    + [4] * 6
    + [6] * 2
    + [8] * 2
    + [10]
    + [8]
    + [6] * 7
    + [8]
    + [10] * 2
    + [12]
    + [10] * 9
)
row8 = (
    [8] * 3
    + [6]
    + [4] * 3
    + [2] * 7
    + [4] * 3
    + [6] * 3
    + [4] * 6
    + [6] * 3
    + [8]
    + [10]
    + [8] * 3
    + [6] * 2
    + [8] * 4
)

CHROMA = [row1, row2, row3, row4, row5, row6, row7, row8]

##### Berlin and Kay (1969) Colorimetry #####

best_example_Berlin_Kay_string = {
    "pink": [["5RP", 7], ["5RP", 6]],
    "red": [["7.5R", 4], ["7.5R", 3]],
    "orange": [["10R", 5]],
    "yellow": [["2.5Y", 8]],
    "brown": [["2.5YR", 2], ["5YR", 2], ["7.5YR", 2]],
    "green": [["2.5G", 4], ["2.5G", 5]],
    "blue": [
        ["7.5B", 4],
        ["10B", 4],
        ["2.5PB", 4],
        ["7.5B", 5],
        ["10B", 5],
        ["2.5PB", 5],
    ],
    "purple": [["5P", 2], ["5P", 3]],
}

# best_example
# position brightness value
best_example_Berlin_Kay = {
    key: [[BRIGHTNESS.index(bv), HUE.index(h)] for h, bv in value]
    for key, value in best_example_Berlin_Kay_string.items()
}

best_example_Berlin_Kay_coord2label = {
    (i, j): label
    for label, coords in best_example_Berlin_Kay.items()
    for i, j in coords
}

# Compute one best example for each color
# one example in Lab

one_best_example_Berlin_Kay = interpolate_best_example(
    best_examples_per_color=best_example_Berlin_Kay
)

# define cielbab_berlin_kay
cielab_Berlin_Kay = {
    color: v["lab"] for color, v in one_best_example_Berlin_Kay.items()
}


comparator_Berlin_Kay_lab = cielab_Berlin_Kay.copy()
comparator_Berlin_Kay_lab["black"] = XYZ_to_lab(XYZ_dark_black)
comparator_Berlin_Kay_lab["white"] = XYZ_to_lab(XYZ_white)
comparator_Berlin_Kay_lab.pop("orange")
comparator_Berlin_Kay_lab.pop("brown")
