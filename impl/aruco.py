import numpy as np
import cv2
from random import random, randint

# DICT_4X4_250: 250 markers (IDs 0-249) + 2 custom (250=black, 251=white)
_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
_MARKER_SIZE = 4
_N_BITS = 16
_N_REAL_MARKERS = 250


def _generate_opencv_marker(marker_id: int, side_pixels: int) -> np.ndarray:
    """Génère un marqueur au format standard OpenCV (DICT_4X4_250)."""
    try:
        return cv2.aruco.generateImageMarker(_ARUCO_DICT, marker_id, side_pixels)
    except AttributeError:
        return cv2.aruco.drawMarker(_ARUCO_DICT, marker_id, side_pixels)


def _extract_bits_from_opencv_marker(img: np.ndarray) -> np.ndarray:
    """
    Extrait les 16 bits du marqueur OpenCV (ordre standard).
    img: sortie de generateImageMarker, structure 6x6 cellules (borderBits=1).
    """
    h, w = img.shape
    cell_sz = h // 6
    bits = []
    for row in range(4):
        for col in range(4):
            y = (1 + row) * cell_sz + cell_sz // 2
            x = (1 + col) * cell_sz + cell_sz // 2
            val = img[min(y, h - 1), min(x, w - 1)]
            bits.append(1.0 if val > 128 else 0.0)
    return np.array(bits, dtype=np.float32)


def _build_ids_as_bits():
    """Liste des 250 motifs de bits (format OpenCV) pour find_id."""
    return [_extract_bits_from_opencv_marker(_generate_opencv_marker(i, 24)) for i in range(_N_REAL_MARKERS)]


ids_as_bits = _build_ids_as_bits()

# Marqueurs custom pour build_dataset (faux marqueurs)
FAKE_BLACK_BITS = np.zeros(_N_BITS, dtype=np.float32)
FAKE_WHITE_BITS = np.ones(_N_BITS, dtype=np.float32)


def id_to_bits(id):
    """Retourne les bits du marqueur (pour compatibilité)."""
    if id < _N_REAL_MARKERS:
        return ids_as_bits[id].tolist()
    elif id == 250:
        return FAKE_BLACK_BITS.tolist()
    else:  # 251
        return FAKE_WHITE_BITS.tolist()


def get_marker(id, size=512, border_width=1.0):
    """
    Génère un marqueur ArUco au format standard OpenCV (DICT_4X4_250).
    Retourne (canvas RGBA, corners).
    """
    wo_border = int(size - 2 * (size / 10))
    center = size // 2
    bg_size = int(size - 2 * (size / 10) * (1 - border_width))
    y0 = (size - wo_border) // 2
    x0 = (size - wo_border) // 2

    if id < _N_REAL_MARKERS:
        # Marqueurs réels : utiliser OpenCV (format standard)
        marker_gray = _generate_opencv_marker(id, wo_border)
        marker_bgra = cv2.cvtColor(marker_gray, cv2.COLOR_GRAY2BGRA)
    elif id == 250:
        # Tout noir (faux marqueur)
        marker_bgra = np.zeros((wo_border, wo_border, 4), dtype=np.uint8)
        marker_bgra[:, :, 3] = 255
    else:  # 251
        # Tout blanc (base pour designs)
        marker_bgra = np.full((wo_border, wo_border, 4), 255, dtype=np.uint8)

    canvas = np.ones((size, size, 4), dtype=np.uint8) * 255
    canvas[:, :, 3] = 0
    canvas[
        center - bg_size // 2 : center + bg_size // 2,
        center - bg_size // 2 : center + bg_size // 2,
        3,
    ] = 255
    canvas[y0 : y0 + wo_border, x0 : x0 + wo_border] = marker_bgra

    corners = [
        [x0, y0],
        [x0, y0 + wo_border - 1],
        [x0 + wo_border - 1, y0 + wo_border - 1],
        [x0 + wo_border - 1, y0],
    ]

    return canvas, corners


def find_id(bits):
    """Associe les bits prédits au dictionnaire. Retourne (id, distance)."""
    bits_flat = np.array(bits).flatten()
    if bits_flat.size != _N_BITS:
        bits_flat = bits_flat[:_N_BITS]

    rot0 = bits_flat
    bits_4x4 = bits_flat.reshape(4, 4)
    rot90 = np.rot90(bits_4x4, 1).flatten()
    rot180 = np.rot90(bits_4x4, 2).flatten()
    rot270 = np.rot90(bits_4x4, 3).flatten()

    distances = [
        int(
            np.min(
                [
                    np.sum(np.abs(rot0 - cb)),
                    np.sum(np.abs(rot90 - cb)),
                    np.sum(np.abs(rot180 - cb)),
                    np.sum(np.abs(rot270 - cb)),
                ]
            )
        )
        for cb in ids_as_bits
    ]

    best_id = int(np.argmin(distances))
    return (best_id, distances[best_id])


if __name__ == "__main__":
    marker, corners = get_marker(randint(0, 251), border_width=random())
    for c in corners:
        cv2.circle(marker, (int(c[0]), int(c[1])), 5, (0, 255, 0, 255), -1, lineType=cv2.LINE_AA)
    img = np.where(marker[:, :, 3:4] > 0, marker[:, :, :3], 255).astype(np.uint8)
    cv2.imwrite("test_aruco.png", img)
