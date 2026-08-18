import os
import csv
import random
import argparse

import numpy as np
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

SEARCH_SIZE = 1000

# Search image:
# 10 nm / pixel
SEARCH_NM_PER_PIXEL = 10

# Reference image:
# 1 nm / pixel
REFERENCE_NM_PER_PIXEL = 1

# 100 x 100 Search pixels =
# 1000 nm x 1000 nm = 1 um x 1 um
CROP_SIZE = 100

# Reference image size
REFERENCE_SIZE = 1000

# Number of References generated from one Search image
NUMBER_OF_REFERENCES = 15

# Reference rotation range
MIN_ROTATION_DEG = -1.7
MAX_ROTATION_DEG = 1.8

SEED = 42


# ============================================================
# LOAD SEARCH IMAGE
# ============================================================

def load_search_image(path):

    image = Image.open(path).convert("L")

    if image.size != (SEARCH_SIZE, SEARCH_SIZE):
        raise ValueError(
            f"Search image must be 1000x1000 pixels. "
            f"Found: {image.size}"
        )

    return np.array(image)


# ============================================================
# CALCULATE STRUCTURAL / EDGE SCORE
# ============================================================

def edge_score(crop):
    """
    Calculates structural variation inside a 100x100 crop.

    Higher score generally indicates stronger:
    - edges
    - boundaries
    - transitions
    - structural features
    """

    crop = crop.astype(np.float32)

    horizontal = np.abs(
        crop[:, 1:] - crop[:, :-1]
    )

    vertical = np.abs(
        crop[1:, :] - crop[:-1, :]
    )

    return float(
        np.mean(horizontal) +
        np.mean(vertical)
    )


# ============================================================
# GENERATE POSSIBLE LOCATIONS
# ============================================================

def get_candidates(search):

    candidates = []

    # Check possible locations every 10 pixels.
    STEP = 10

    for y in range(
        0,
        SEARCH_SIZE - CROP_SIZE + 1,
        STEP
    ):

        for x in range(
            0,
            SEARCH_SIZE - CROP_SIZE + 1,
            STEP
        ):

            crop = search[
                y:y + CROP_SIZE,
                x:x + CROP_SIZE
            ]

            score = edge_score(crop)

            candidates.append(
                (score, x, y)
            )

    return candidates


# ============================================================
# CHECK WHETHER TWO SITES ARE TOO CLOSE
# ============================================================

def sufficiently_far(
    x,
    y,
    selected,
    minimum_distance=120
):

    current_center_x = x + 49.5
    current_center_y = y + 49.5

    for _, old_x, old_y in selected:

        old_center_x = old_x + 49.5
        old_center_y = old_y + 49.5

        distance = np.sqrt(
            (current_center_x - old_center_x) ** 2 +
            (current_center_y - old_center_y) ** 2
        )

        if distance < minimum_distance:
            return False

    return True


# ============================================================
# SELECT 15 DIFFERENT LOCATIONS
# ============================================================

def select_locations(search):

    random.seed(SEED)

    candidates = get_candidates(search)

    # Strongest structural regions first
    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    selected = []

    # --------------------------------------------------------
    # STEP 1:
    # Select 8 edge/border-rich regions
    # --------------------------------------------------------

    for score, x, y in candidates:

        if sufficiently_far(
            x,
            y,
            selected,
            minimum_distance=130
        ):

            selected.append(
                ("border", x, y)
            )

        if len(selected) == 8:
            break

    # --------------------------------------------------------
    # STEP 2:
    # Select remaining random regions
    # --------------------------------------------------------

    attempts = 0

    while len(selected) < NUMBER_OF_REFERENCES:

        attempts += 1

        if attempts > 50000:
            raise RuntimeError(
                "Unable to find 15 sufficiently "
                "different locations."
            )

        x = random.randint(
            0,
            SEARCH_SIZE - CROP_SIZE
        )

        y = random.randint(
            0,
            SEARCH_SIZE - CROP_SIZE
        )

        if sufficiently_far(
            x,
            y,
            selected,
            minimum_distance=100
        ):

            selected.append(
                ("random", x, y)
            )

    return selected


# ============================================================
# CREATE REFERENCE IMAGE
# ============================================================

def create_reference(crop):
    """
    Creates the Reference image from a 100x100 Search crop.

    Search:
        100 x 100 pixels
        10 nm/pixel

    Reference:
        1000 x 1000 pixels
        1 nm/pixel

    The Reference is then rotated randomly between
    -1 and +1 degrees.

    IMPORTANT:
    Rotation changes ONLY the Reference image.

    The ground-truth coordinates remain the coordinates
    of the original crop in the Search image.
    """

    crop_image = Image.fromarray(crop)

    # --------------------------------------------------------
    # STEP 1:
    # Convert 100x100 Search crop
    # to 1000x1000 Reference
    # --------------------------------------------------------

    reference = crop_image.resize(
        (
            REFERENCE_SIZE,
            REFERENCE_SIZE
        ),
        Image.Resampling.NEAREST
    )

    # --------------------------------------------------------
    # STEP 2:
    # Generate random rotation
    # between -1 and +1 degrees
    # --------------------------------------------------------

    rotation_deg = random.uniform(
        MIN_ROTATION_DEG,
        MAX_ROTATION_DEG
    )

    # --------------------------------------------------------
    # STEP 3:
    # Rotate Reference
    #
    # expand=False keeps the image exactly
    # 1000x1000 pixels.
    #
    # BICUBIC gives smoother interpolation
    # than NEAREST during rotation.
    # --------------------------------------------------------

    reference = reference.rotate(
        rotation_deg,
        resample=Image.Resampling.BICUBIC,
        expand=False
    )

    return reference, rotation_deg


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_dataset(
    search_path,
    output_dir
):

    # Create output directory
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print("FINFET REFERENCE GENERATOR")
    print("=" * 70)

    print(
        f"Search image : {search_path}"
    )

    print(
        "Search size  : 1000 x 1000"
    )

    print(
        "Search scale : 10 nm/pixel"
    )

    print(
        "Reference    : 1000 x 1000"
    )

    print(
        "Reference    : 1 nm/pixel"
    )

    print(
        "References   : 15"
    )

    print(
        "Rotation     : -1 to +1 degrees"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load Search image
    # --------------------------------------------------------

    search = load_search_image(
        search_path
    )

    # --------------------------------------------------------
    # Select 15 different locations
    # --------------------------------------------------------

    locations = select_locations(
        search
    )

    # --------------------------------------------------------
    # CSV path
    # --------------------------------------------------------

    csv_path = os.path.join(
        output_dir,
        "ground_truth.csv"
    )

    # --------------------------------------------------------
    # Create CSV
    # --------------------------------------------------------

    with open(
        csv_path,
        "w",
        newline=""
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        # EXACT CSV FORMAT
        writer.writerow([
            "reference_id",
            "reference_filename",
            "search_image",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
            "search_nm_per_pixel",
            "reference_nm_per_pixel",
            "reference_fov_nm",
            "search_fov_nm",
            "rotation_deg"
        ])

        # ----------------------------------------------------
        # Generate 15 References
        # ----------------------------------------------------

        for reference_id, (
            selection_type,
            x,
            y
        ) in enumerate(
            locations,
            start=211
        ):

            # -----------------------------------------------
            # Extract original 100x100 Search region
            # -----------------------------------------------

            crop = search[
                y:y + CROP_SIZE,
                x:x + CROP_SIZE
            ]

            # -----------------------------------------------
            # Create Reference + rotation
            # -----------------------------------------------

            reference, rotation_deg = create_reference(
                crop
            )

            # -----------------------------------------------
            # Filename
            # -----------------------------------------------

            filename = (
                f"reference_{reference_id:03d}.png"
            )

            reference_path = os.path.join(
                output_dir,
                filename
            )

            reference.save(
                reference_path
            )

            # -----------------------------------------------
            # Ground-truth bounding box
            # -----------------------------------------------

            x1 = x
            y1 = y

            x2 = x + CROP_SIZE - 1
            y2 = y + CROP_SIZE - 1

            # -----------------------------------------------
            # Exact geometric center
            #
            # Example:
            # x1 = 600
            # x2 = 699
            #
            # center_x = (600 + 699) / 2
            #          = 649.5
            # -----------------------------------------------

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # -----------------------------------------------
            # FOV
            # -----------------------------------------------

            reference_fov_nm = (
                REFERENCE_SIZE *
                REFERENCE_NM_PER_PIXEL
            )

            search_fov_nm = (
                SEARCH_SIZE *
                SEARCH_NM_PER_PIXEL
            )

            # -----------------------------------------------
            # Write CSV row
            # -----------------------------------------------

            writer.writerow([
                reference_id,
                filename,
                os.path.basename(search_path),
                x1,
                y1,
                x2,
                y2,
                center_x,
                center_y,
                SEARCH_NM_PER_PIXEL,
                REFERENCE_NM_PER_PIXEL,
                reference_fov_nm,
                search_fov_nm,
                round(rotation_deg, 4)
            ])

            # -----------------------------------------------
            # Terminal information
            # -----------------------------------------------

            print(
                f"Reference {reference_id:02d} | "
                f"location=({x1},{y1})-({x2},{y2}) | "
                f"center=({center_x},{center_y}) | "
                f"rotation={rotation_deg:.4f}° | "
                f"type={selection_type}"
            )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"15 Reference images saved to:"
    )

    print(
        f"  {output_dir}"
    )

    print()

    print(
        f"Ground truth saved to:"
    )

    print(
        f"  {csv_path}"
    )

    print("=" * 70)


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate 15 FinFET Reference images "
            "from different 100x100 regions of "
            "a 1000x1000 Search image."
        )
    )

    parser.add_argument(
        "--search",
        default="input/search15.png",
        help="Path to Search image"
    )

    parser.add_argument(
        "--output",
        default="output",
        help="Output directory"
    )

    args = parser.parse_args()

    generate_dataset(
        search_path=args.search,
        output_dir=args.output
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()