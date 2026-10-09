"""Class definitions and the label bit layout used by the packed dataset.

Each tile's label is stored as one uint16 PNG. Bit ``i`` (1..8) is set when the
pixel carries Agriculture-Vision class ``AV_CLASSES[i]``; several bits can be set
at once because the official labels overlap. Bit 15 marks pixels outside the
field boundary or outside the valid-image mask.
"""

AV_CLASSES = [
    "background",
    "double_plant",
    "drydown",
    "endrow",
    "nutrient_deficiency",
    "planter_skip",
    "water",
    "waterway",
    "weed_cluster",
]
AV_ID = {name: i for i, name in enumerate(AV_CLASSES)}
LABEL_CLASSES = AV_CLASSES[1:]           # folders under labels/
IGNORED_LABELS = ("storm_damage",)       # present in the archive, not evaluated since 2021
INVALID_BIT = 15
INVALID = 1 << INVALID_BIT

# The three classes this project is about.
TARGET_CLASSES = ("drydown", "nutrient_deficiency", "weed_cluster")

# When labels overlap, the rarer class wins the single-label target.
# Order from the 2021 foreground pixel shares (rarest first).
DEFAULT_PRIORITY = (
    "planter_skip",
    "double_plant",
    "endrow",
    "waterway",
    "water",
    "weed_cluster",
    "nutrient_deficiency",
    "drydown",
)

# Model class sets. Index 0 is always background.
CLASS_SETS = {
    "all9": tuple(AV_CLASSES),
    "targets": ("background",) + TARGET_CLASSES,
}

PALETTE = {
    "background": (40, 40, 40),
    "double_plant": (31, 119, 180),
    "drydown": (230, 159, 0),
    "endrow": (148, 103, 189),
    "nutrient_deficiency": (220, 50, 47),
    "planter_skip": (23, 190, 207),
    "water": (0, 90, 220),
    "waterway": (120, 180, 255),
    "weed_cluster": (60, 180, 75),
}


def bit(name):
    return 1 << AV_ID[name]


def resolve_classes(classes):
    """Accept a class-set name ('all9', 'targets') or an explicit list."""
    if isinstance(classes, str):
        classes = CLASS_SETS[classes]
    classes = tuple(classes)
    if classes[0] != "background":
        raise ValueError("the first class must be 'background'")
    unknown = [c for c in classes if c not in AV_ID]
    if unknown:
        raise ValueError(f"unknown classes: {unknown}")
    return classes
