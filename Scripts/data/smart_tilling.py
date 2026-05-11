import cv2
import numpy as np
import random
from pathlib import Path
import uuid
import shutil


# Set deterministic randomness for reproducibility
random.seed(42)


# =========================
# CONFIGURATION
# =========================

# Valid image extensions
DEFAULT_IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Minimum object area (in pixels) required after tiling
# Used to remove extremely small fragmented detections
MIN_AREA_PX = 3000

# Temporary directories used during tiling generation
TMP_DIR = Path("tiles_tmp")
TMP_IMG = TMP_DIR / "images"
TMP_LBL = TMP_DIR / "labels"


# =========================
# YOLO UTILITIES
# =========================

def label_is_empty(label_path: Path):
    """
    Check whether a YOLO label file is empty or missing.

    Parameters
    ----------
    label_path : Path
        Path to YOLO annotation file.

    Returns
    -------
    bool
        True if label file does not exist or is empty.
    """
    return (not label_path.exists()) or label_path.stat().st_size == 0


def yolo_to_xyxy(xc, yc, bw, bh, W, H):
    """
    Convert YOLO normalized bounding box format to absolute XYXY format.

    Parameters
    ----------
    xc, yc : float
        Normalized center coordinates.

    bw, bh : float
        Normalized width and height.

    W, H : int
        Image width and height.

    Returns
    -------
    tuple
        Bounding box in (x1, y1, x2, y2) format.
    """

    x_c = xc * W
    y_c = yc * H

    box_w = bw * W
    box_h = bh * H

    return (
        x_c - box_w / 2,
        y_c - box_h / 2,
        x_c + box_w / 2,
        y_c + box_h / 2,
    )


def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    """
    Convert absolute XYXY bounding box format into YOLO normalized format.

    Parameters
    ----------
    x1, y1, x2, y2 : float
        Absolute bounding box coordinates.

    W, H : int
        Image width and height.

    Returns
    -------
    tuple
        YOLO format bounding box:
        (x_center, y_center, width, height)
    """

    bw = (x2 - x1) / W
    bh = (y2 - y1) / H

    xc = (x1 + x2) / 2 / W
    yc = (y1 + y2) / 2 / H

    return xc, yc, bw, bh


def intersect(a, b):
    """
    Compute intersection between two bounding boxes.

    Parameters
    ----------
    a, b : tuple
        Bounding boxes in XYXY format.

    Returns
    -------
    tuple or None
        Intersection box or None if no overlap exists.
    """

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])

    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


# =========================
# LABEL UTILITIES
# =========================

def read_yolo_labels(label_path):
    """
    Read YOLO annotation file.

    Parameters
    ----------
    label_path : Path
        Path to YOLO label file.

    Returns
    -------
    list
        List of annotations:
        [(class_id, xc, yc, bw, bh), ...]
    """

    if not label_path.exists():
        return []

    txt = label_path.read_text().strip()

    if not txt:
        return []

    out = []

    for line in txt.splitlines():

        p = line.split()

        if len(p) < 5:
            continue

        out.append((int(p[0]), *map(float, p[1:5])))

    return out


def write_yolo_labels(path, items):
    """
    Save YOLO annotations to disk.

    Parameters
    ----------
    path : Path
        Destination label file.

    items : list
        YOLO annotations.
    """

    lines = [
        f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        for c, x, y, w, h in items
        if w > 0 and h > 0
    ]

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text("\n".join(lines) + "\n")


# =========================
# IMAGE PROCESSING
# =========================

def crop(img, rect, size):
    """
    Extract image crop from a rectangular region.

    If crop exceeds image boundaries, output image is padded with zeros.

    Parameters
    ----------
    img : ndarray
        Input image.

    rect : tuple
        Crop rectangle (x1, y1, x2, y2).

    size : int
        Output tile size.

    Returns
    -------
    ndarray
        Cropped tile image.
    """

    x1, y1, x2, y2 = map(int, rect)

    patch = img[y1:y2, x1:x2]

    out = np.zeros((size, size, 3), dtype=img.dtype)

    h, w = patch.shape[:2]

    out[:h, :w] = patch

    return out


def transform_labels(labels, W, H, rect, size):
    """
    Transform annotations from original image coordinates
    into tile-local coordinates.

    Bounding boxes are clipped to tile boundaries and
    filtered using a minimum area threshold.

    Parameters
    ----------
    labels : list
        YOLO annotations.

    W, H : int
        Original image dimensions.

    rect : tuple
        Tile rectangle.

    size : int
        Tile size.

    Returns
    -------
    list
        Transformed YOLO annotations for tile.
    """

    cx1, cy1, cx2, cy2 = rect

    out = []

    for cls, xc, yc, bw, bh in labels:

        bx1, by1, bx2, by2 = yolo_to_xyxy(xc, yc, bw, bh, W, H)

        inter = intersect(
            (bx1, by1, bx2, by2),
            (cx1, cy1, cx2, cy2)
        )

        if inter is None:
            continue

        fx1, fy1 = inter[0] - cx1, inter[1] - cy1
        fx2, fy2 = inter[2] - cx1, inter[3] - cy1

        nxc, nyc, nbw, nbh = xyxy_to_yolo(
            fx1, fy1, fx2, fy2, size, size
        )

        # Remove extremely small fragmented boxes
        area = (nbw * size) * (nbh * size)

        if area >= MIN_AREA_PX:
            out.append((cls, nxc, nyc, nbw, nbh))

    return out


def tile_one_image(img_path, lbl_path, empty_imgs):
    """
    Generate square tiles centered around annotated objects.

    Positive tiles:
    - Centered on annotated bounding boxes.

    Negative tiles:
    - Random crops extracted from empty images.

    Parameters
    ----------
    img_path : Path
        Path to image.

    lbl_path : Path
        Path to YOLO annotation file.

    empty_imgs : list
        List of images without annotations.

    Returns
    -------
    tuple
        (positive_tile_names, negative_tile_names)
    """

    img = cv2.imread(str(img_path))

    if img is None:
        return [], []

    H, W = img.shape[:2]

    labels = read_yolo_labels(lbl_path)

    pos_tiles = []
    neg_tiles = []

    for cls, xc, yc, bw, bh in labels:

        x1, y1, x2, y2 = yolo_to_xyxy(
            xc, yc, bw, bh, W, H
        )

        # Fixed tile size
        size = 2000

        # Bounding box center
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # Compute tile origin
        sx = int(max(0, min(max(W - size, 0), cx - size / 2)))
        sy = int(max(0, min(max(H - size, 0), cy - size / 2)))

        rect = (sx, sy, sx + size, sy + size)

        # Crop tile
        im = crop(img, rect, size)

        # Transform annotations
        labs = transform_labels(labels, W, H, rect, size)

        if labs:

            name = uuid.uuid4().hex

            cv2.imwrite(str(TMP_IMG / f"{name}.png"), im)

            write_yolo_labels(
                TMP_LBL / f"{name}.txt",
                labs
            )

            pos_tiles.append(name)

    # Generate negative tiles
    if empty_imgs:

        k = max(1, int(len(pos_tiles) * 0.1))

        for _ in range(k):

            empty_img = cv2.imread(
                str(random.choice(empty_imgs))
            )

            if empty_img is None:
                continue

            name = uuid.uuid4().hex

            cv2.imwrite(
                str(TMP_IMG / f"{name}.png"),
                empty_img[:2000, :2000]
            )

            (TMP_LBL / f"{name}.txt").write_text("")

            neg_tiles.append(name)

    return pos_tiles, neg_tiles


# =========================
# MAIN PIPELINE
# =========================

def square_tiling_transform(dataset_root):
    """
    Generate balanced square-tiled YOLO dataset.

    The pipeline:
    - Collects positive and empty images
    - Generates object-centered tiles
    - Generates negative tiles
    - Rebuilds train/validation splits
    - Removes original dataset structure

    Parameters
    ----------
    dataset_root : str or Path
        Root directory of YOLO dataset.

    Returns
    -------
    None
    """

    dataset_root = Path(dataset_root)

    TMP_IMG.mkdir(parents=True, exist_ok=True)
    TMP_LBL.mkdir(parents=True, exist_ok=True)

    all_pos = []
    all_neg = []

    empty_imgs = []

    # Collect empty images
    for split in ["train", "valid"]:

        for img_path in (dataset_root / split / "images").rglob("*"):

            lbl_path = (
                dataset_root
                / split
                / "labels"
                / (img_path.stem + ".txt")
            )

            if label_is_empty(lbl_path):
                empty_imgs.append(img_path)

    # Generate tiles
    for split in ["train", "valid"]:

        img_root = dataset_root / split / "images"
        lbl_root = dataset_root / split / "labels"

        for img_path in img_root.rglob("*"):

            if img_path.suffix.lower() not in DEFAULT_IMG_EXTS:
                continue

            lbl_path = lbl_root / (img_path.stem + ".txt")

            pos, neg = tile_one_image(
                img_path,
                lbl_path,
                empty_imgs
            )

            all_pos.extend(pos)
            all_neg.extend(neg)

    print(
        f"Tiles -> Positives: {len(all_pos)} | "
        f"Negatives: {len(all_neg)}"
    )

    # Remove original dataset
    shutil.rmtree(dataset_root / "train")
    shutil.rmtree(dataset_root / "valid")

    # =========================
    # BALANCED TRAIN/VALID SPLIT
    # =========================

    def split_and_move(pos_list, neg_list, split_ratio=0.7):
        """
        Split positive and negative tiles into
        train/validation sets and move files.

        Parameters
        ----------
        pos_list : list
            Positive tile names.

        neg_list : list
            Negative tile names.

        split_ratio : float
            Training split ratio.

        Returns
        -------
        None
        """

        random.shuffle(pos_list)
        random.shuffle(neg_list)

        def split(lst):
            idx = int(len(lst) * split_ratio)
            return lst[:idx], lst[idx:]

        pos_train, pos_val = split(pos_list)
        neg_train, neg_val = split(neg_list)

        def move(names, dst):

            (dst / "images").mkdir(
                parents=True,
                exist_ok=True
            )

            (dst / "labels").mkdir(
                parents=True,
                exist_ok=True
            )

            for n in names:

                shutil.move(
                    str(TMP_IMG / f"{n}.png"),
                    dst / "images" / f"{n}.png"
                )

                lbl = TMP_LBL / f"{n}.txt"

                if lbl.exists():

                    shutil.move(
                        str(lbl),
                        dst / "labels" / f"{n}.txt"
                    )

                else:
                    (dst / "labels" / f"{n}.txt").write_text("")

        move(pos_train + neg_train, dataset_root / "train")
        move(pos_val + neg_val, dataset_root / "valid")

    split_and_move(all_pos, all_neg)

    # Remove temporary directory
    shutil.rmtree(TMP_DIR)

    print("Final tiled dataset generated successfully")

# =========================
# RUN SCRIPT
# =========================

if __name__ == "__main__":

    BASE_DIR = Path("path/to/dataset")

    square_tiling_transform(BASE_DIR)