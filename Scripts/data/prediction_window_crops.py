import os
from pathlib import Path

try:
    import cv2
except ImportError as e:
    raise ImportError(
        "OpenCV is missing. Install it with: pip install opencv-python"
    ) from e


# =========================
# CONFIG
# =========================

# Base dataset directory
BASE_DATASET_DIR = Path(r"")

# Valid image sizes (Height, Width) and their corresponding crop margins
DEFAULT_MARGINS_BY_SIZE = {
    (6656, 13312): dict(top=3000, bottom=2500, left=1000, right=1000),
    (8192, 16384): dict(top=3686, bottom=3000, left=1500, right=1500),
}

# Allowed image extensions
DEFAULT_IMG_EXTS = {".jpg", ".jpeg", ".png"}


# =========================
# IMAGE UTILITIES
# =========================
def split_image_vertical(img):
    """
    Split an image vertically into two equal halves.

    Parameters
    ----------
    img : numpy.ndarray
        Input image loaded with OpenCV.

    Returns
    -------
    tuple
        A tuple containing:
        - left half image
        - right half image
    """

    H, W = img.shape[:2]

    # Compute middle column
    mid = W // 2

    # Split image into left and right halves
    left = img[:, :mid]
    right = img[:, mid:]

    return left, right


def crop_image_with_margins(img, margins):
    """
    Crop an image using predefined margins.

    Parameters
    ----------
    img : numpy.ndarray
        Input image.

    margins : dict
        Dictionary containing crop margins:
        {
            "top": int,
            "bottom": int,
            "left": int,
            "right": int
        }

    Returns
    -------
    numpy.ndarray or None
        Cropped image if valid, otherwise None.
    """

    H, W = img.shape[:2]

    # Extract crop margins
    top = int(margins["top"])
    bottom = int(margins["bottom"])
    left = int(margins["left"])
    right = int(margins["right"])

    # Ensure crop coordinates remain inside image bounds
    y1 = max(0, min(top, H))
    y2 = max(0, min(H - bottom, H))

    x1 = max(0, min(left, W))
    x2 = max(0, min(W - right, W))

    # Invalid crop check
    if y2 <= y1 or x2 <= x1:
        return None

    # Return cropped region
    return img[y1:y2, x1:x2]


# =========================
# PROCESS SINGLE IMAGE
# =========================
def process_one_image(img_path: Path, margins_by_size):
    """
    Process a single image:
    - Validate image size
    - Split image vertically
    - Crop both halves using margins
    - Save cropped images
    - Delete original image

    Parameters
    ----------
    img_path : Path
        Path to the image file.

    margins_by_size : dict
        Dictionary mapping image sizes to crop margins.

    Returns
    -------
    bool
        True if processing succeeded,
        False if image was invalid or deleted.
    """

    # Load image
    img = cv2.imread(str(img_path))

    # Skip unreadable images
    if img is None:
        return False

    H, W = img.shape[:2]
    key = (H, W)

    # Delete image if size is not allowed
    if key not in margins_by_size:
        img_path.unlink(missing_ok=True)
        return False

    margins = margins_by_size[key]

    # Split image into left and right halves
    left_img, right_img = split_image_vertical(img)

    # Crop both halves
    left_crop = crop_image_with_margins(left_img, margins)
    right_crop = crop_image_with_margins(right_img, margins)

    # Delete image if crop failed
    if left_crop is None or right_crop is None:
        img_path.unlink(missing_ok=True)
        return False

    # Build output filenames
    stem = img_path.stem
    ext = img_path.suffix

    left_img_path = img_path.with_name(f"{stem}_L{ext}")
    right_img_path = img_path.with_name(f"{stem}_R{ext}")

    # Save cropped images
    cv2.imwrite(str(left_img_path), left_crop)
    cv2.imwrite(str(right_img_path), right_crop)

    # Remove original image
    img_path.unlink(missing_ok=True)

    return True


# =========================
# DATASET PROCESSING
# =========================
def window_transform(dataset_path, margins_by_size=None, img_exts=None):
    """
    Process all images inside a dataset directory.

    The function recursively searches for images,
    validates their dimensions, applies vertical splitting
    and cropping transformations, and removes invalid images.

    Parameters
    ----------
    dataset_path : str or Path
        Root dataset directory.

    margins_by_size : dict, optional
        Dictionary mapping image sizes to crop margins.
        If None, DEFAULT_MARGINS_BY_SIZE is used.

    img_exts : set, optional
        Allowed image extensions.
        If None, DEFAULT_IMG_EXTS is used.

    Returns
    -------
    None
    """

    dataset_root = Path(dataset_path)

    margins_by_size = margins_by_size or DEFAULT_MARGINS_BY_SIZE
    img_exts = img_exts or DEFAULT_IMG_EXTS

    # Statistics counters
    count_seen = 0
    count_done = 0
    count_deleted = 0

    # Recursively iterate through all files
    for img_path in dataset_root.rglob("*"):

        # Skip non-image files
        if img_path.suffix.lower() not in img_exts:
            continue

        count_seen += 1

        # Process image
        result = process_one_image(img_path, margins_by_size)

        if result:
            count_done += 1
        else:
            count_deleted += 1

    # Print processing summary
    print(f"Images scanned: {count_seen}")
    print(f"Images transformed: {count_done}")
    print(f"Images deleted (invalid size): {count_deleted}")


# =========================
# RUN SCRIPT
# =========================
if __name__ == "__main__":
    window_transform(BASE_DATASET_DIR)