from typing import Tuple, Optional
import time

import numpy as np
from scipy.ndimage import sobel

WIDTH_FIRST = "width-first"
HEIGHT_FIRST = "height-first"
VALID_ORDERS = (WIDTH_FIRST, HEIGHT_FIRST)

FORWARD_ENERGY = "forward"
BACKWARD_ENERGY = "backward"
VALID_ENERGY_MODES = (FORWARD_ENERGY, BACKWARD_ENERGY)

FROM_LEFT = True
FROM_TOP = False

DROP_MASK_ENERGY = 1e5
KEEP_MASK_ENERGY = 1e3


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def interpolate(subBin, LU, RU, LB, RB, subX, subY):
    subImage = np.zeros(subBin.shape)
    num = subX * subY
    for i in range(subX):
        inverseI = subX - i
        for j in range(subY):
            inverseJ = subY - j
            val = subBin[i, j].astype(int)
            subImage[i, j] = np.floor(
                (
                    inverseI * (inverseJ * LU[val] + j * RU[val])
                    + i * (inverseJ * LB[val] + j * RB[val])
                )
                / float(num)
            )
    return subImage


def clahe(img, clipLimit, nrBins=128, nrX=0, nrY=0):
    """img - Input image
    clipLimit - Normalized clipLimit. Higher value gives more contrast
    nrBins - Number of graylevel bins for histogram("dynamic range")
    nrX - Number of contextial regions in X direction
    nrY - Number of Contextial regions in Y direction"""
    h, w = img.shape

    img = (img * 255).astype(np.uint8)

    if clipLimit == 1:
        return
    nrBins = max(nrBins, 128)
    if nrX == 0:
        # Taking dimensions of each contextial region to be a square of 32X32
        xsz = 32
        ysz = 32
        nrX = np.ceil(h / xsz).astype(int)  # 240
        # Excess number of pixels to get an integer value of nrX and nrY
        excX = int(xsz * (nrX - h / xsz))
        nrY = np.ceil(w / ysz).astype(int)  # 320
        excY = int(ysz * (nrY - w / ysz))
        # Pad that number of pixels to the image
        if excX != 0:
            img = np.append(img, np.zeros((excX, img.shape[1])).astype(int), axis=0)
        if excY != 0:
            img = np.append(img, np.zeros((img.shape[0], excY)).astype(int), axis=1)
    else:
        xsz = round(h / nrX)
        ysz = round(w / nrY)

    nrPixels = xsz * ysz
    xsz2 = round(xsz / 2)
    ysz2 = round(ysz / 2)
    claheimg = np.zeros(img.shape)

    if clipLimit > 0:
        clipLimit = max(1, clipLimit * xsz * ysz / nrBins)
    else:
        clipLimit = 50

    # makeLUT
    minVal = 0  # np.min(img)
    maxVal = 255  # np.max(img)

    # maxVal1 = maxVal + np.maximum(np.array([0]),minVal) - minVal
    # minVal1 = np.maximum(np.array([0]),minVal)

    binSz = np.floor(1 + (maxVal - minVal) / float(nrBins))
    LUT = np.floor((np.arange(minVal, maxVal + 1) - minVal) / float(binSz))

    # BACK TO CLAHE
    bins = LUT[img]
    # makeHistogram
    hist = np.zeros((nrX, nrY, nrBins))
    for i in range(nrX):
        for j in range(nrY):
            bin_ = bins[i * xsz : (i + 1) * xsz, j * ysz : (j + 1) * ysz].astype(int)
            for i1 in range(xsz):
                for j1 in range(ysz):
                    hist[i, j, bin_[i1, j1]] += 1

    # clipHistogram
    if clipLimit > 0:
        for i in range(nrX):
            for j in range(nrY):
                nrExcess = 0
                for nr in range(nrBins):
                    excess = hist[i, j, nr] - clipLimit
                    if excess > 0:
                        nrExcess += excess

                binIncr = nrExcess / nrBins
                upper = clipLimit - binIncr
                for nr in range(nrBins):
                    if hist[i, j, nr] > clipLimit:
                        hist[i, j, nr] = clipLimit
                    else:
                        if hist[i, j, nr] > upper:
                            nrExcess += upper - hist[i, j, nr]
                            hist[i, j, nr] = clipLimit
                        else:
                            nrExcess -= binIncr
                            hist[i, j, nr] += binIncr

                if nrExcess > 0:
                    stepSz = max(1, np.floor(1 + nrExcess / nrBins))
                    for nr in range(nrBins):
                        nrExcess -= stepSz
                        hist[i, j, nr] += stepSz
                        if nrExcess < 1:
                            break

    # mapHistogram
    map_ = np.zeros((nrX, nrY, nrBins))
    scale = (maxVal - minVal) / float(nrPixels)
    for i in range(nrX):
        for j in range(nrY):
            sum_ = 0
            for nr in range(nrBins):
                sum_ += hist[i, j, nr]
                map_[i, j, nr] = np.floor(min(minVal + sum_ * scale, maxVal))

    # BACK TO CLAHE
    # INTERPOLATION
    xI = 0
    for i in range(nrX + 1):
        if i == 0:
            subX = int(xsz / 2)
            xU = 0
            xB = 0
        elif i == nrX:
            subX = int(xsz / 2)
            xU = nrX - 1
            xB = nrX - 1
        else:
            subX = xsz
            xU = i - 1
            xB = i

        yI = 0
        for j in range(nrY + 1):
            if j == 0:
                subY = int(ysz / 2)
                yL = 0
                yR = 0
            elif j == nrY:
                subY = int(ysz / 2)
                yL = nrY - 1
                yR = nrY - 1
            else:
                subY = ysz
                yL = j - 1
                yR = j
            UL = map_[xU, yL, :]
            UR = map_[xU, yR, :]
            BL = map_[xB, yL, :]
            BR = map_[xB, yR, :]
            # print("CLAHE vals...")
            subBin = bins[xI : xI + subX, yI : yI + subY]
            # print("clahe subBin shape: ",subBin.shape)
            subImage = interpolate(subBin, UL, UR, BL, BR, subX, subY)
            claheimg[xI : xI + subX, yI : yI + subY] = subImage
            yI += subY
        xI += subX

    if excX == 0 and excY != 0:
        return claheimg[:, :-excY] / 255.0
    elif excX != 0 and excY == 0:
        return claheimg[:-excX, :] / 255.0
    elif excX != 0 and excY != 0:
        return claheimg[:-excX, :-excY] / 255.0
    else:
        return claheimg / 255.0


# @timeit
def histogram_equalization(img):
    # 1
    (N, M) = img.shape

    img = (img * 255).astype(np.uint8)

    G = 256  # gray levels
    H = np.zeros(G)  # initialize an array Histogram

    # 2
    for g in img.ravel():
        H[g] += 1

    g_min = np.min(np.nonzero(H))

    # 3
    H_c = np.zeros_like(H)  # cumulative image histogram
    H_c[0] = H[0]
    for g in range(1, G):
        H_c[g] = H_c[g - 1] + H[g]

    H_min = H_c[g_min]

    # 4
    T = np.round((H_c - H_min) / (M * N - H_min) * (G - 1))

    # 5
    result = np.zeros_like(img)
    for n in range(N):
        for m in range(M):
            result[n, m] = T[img[n, m]]

    result = result / 255.0

    return result, T


def _rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to a grayscale image"""
    coeffs = np.array([0.2125, 0.7154, 0.0721], dtype=np.float32)
    gray = (rgb @ coeffs).astype(rgb.dtype)

    # hist_eq_gray, _ = histogram_equalization(gray)
    hist_eq_gray = clahe(gray, 8, 256, 0)

    return hist_eq_gray  # (rgb @ coeffs).astype(rgb.dtype)


# @timeit
def _get_seam_mask(src: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Convert a list of seam column indices to a mask"""
    return ~np.eye(src.shape[1], dtype=np.bool)[seam]


# @timeit
def _remove_seam_mask(src: np.ndarray, seam_mask: np.ndarray) -> np.ndarray:
    """Remove a seam from the source image according to the given seam_mask"""
    if src.ndim == 3:
        h, w, c = src.shape
        seam_mask = np.dstack([seam_mask] * c)
        dst = src[seam_mask].reshape((h, w - 1, c))
    else:
        h, w = src.shape
        dst = src[seam_mask].reshape((h, w - 1))
    return dst


# @timeit
def _remove_seam(src: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Remove a seam from the source image, given a list of seam columns"""
    seam_mask = _get_seam_mask(src, seam)
    dst = _remove_seam_mask(src, seam_mask)
    return dst


# @timeit
def _get_energy(gray: np.ndarray) -> np.ndarray:
    """Get backward energy map from the source image"""
    assert gray.ndim == 2

    gray = gray.astype(np.float32)
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    energy = np.abs(grad_x) + np.abs(grad_y)
    return energy


# @timeit
def _get_backward_seam(energy: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute the minimum vertical seam from the backward energy map"""
    assert energy.size > 0 and energy.ndim == 2
    h, w = energy.shape
    cost = energy[0]
    parent = np.empty((h, w), dtype=np.int32)
    base_idx = np.arange(-1, w - 1, dtype=np.int32)

    for r in range(1, h):
        left_shift = np.hstack((cost[1:], np.inf))
        right_shift = np.hstack((np.inf, cost[:-1]))
        min_idx = np.argmin([right_shift, cost, left_shift], axis=0) + base_idx
        parent[r] = min_idx
        cost = cost[min_idx] + energy[r]

    c = np.argmin(cost)
    total_cost = cost[c]
    seam = np.empty(h, dtype=np.int32)

    for r in range(h - 1, -1, -1):
        seam[r] = c
        c = parent[r, c]

    return seam, total_cost


# @timeit
def _get_backward_seams(
    gray: np.ndarray, num_seams: int, keep_mask: Optional[np.ndarray]
) -> np.ndarray:
    """Compute the minimum N vertical seams using backward energy"""
    h, w = gray.shape
    seams_mask = np.zeros((h, w), dtype=np.bool)
    rows = np.arange(0, h, dtype=np.int32)
    idx_map = np.tile(np.arange(0, w, dtype=np.int32), h).reshape((h, w))
    energy = _get_energy(gray)
    for _ in range(num_seams):
        if keep_mask is not None:
            energy[keep_mask] += KEEP_MASK_ENERGY
        seam, _ = _get_backward_seam(energy)
        seams_mask[rows, idx_map[rows, seam]] = True

        seam_mask = _get_seam_mask(gray, seam)
        gray = _remove_seam_mask(gray, seam_mask)
        idx_map = _remove_seam_mask(idx_map, seam_mask)
        if keep_mask is not None:
            keep_mask = _remove_seam_mask(keep_mask, seam_mask)

        _, cur_w = energy.shape
        lo = max(0, np.min(seam) - 1)
        hi = min(cur_w, np.max(seam) + 1)
        pad_lo = 1 if lo > 0 else 0
        pad_hi = 1 if hi < cur_w - 1 else 0
        mid_block = gray[:, lo - pad_lo : hi + pad_hi]
        _, mid_w = mid_block.shape
        mid_energy = _get_energy(mid_block)[:, pad_lo : mid_w - pad_hi]
        energy = np.hstack((energy[:, :lo], mid_energy, energy[:, hi + 1 :]))

    return seams_mask


# @timeit
def _get_forward_seam(
    gray: np.ndarray, keep_mask: Optional[np.ndarray]
) -> Tuple[np.ndarray, float]:
    """Compute the minimum vertical seam using forward energy"""
    assert gray.size > 0 and gray.ndim == 2
    gray = gray.astype(np.float32)
    h, w = gray.shape

    top_row = gray[0]
    top_row_lshift = np.hstack((top_row[1:], top_row[-1]))
    top_row_rshift = np.hstack((top_row[0], top_row[:-1]))
    dp = np.abs(top_row_lshift - top_row_rshift)

    parent = np.zeros(gray.shape, dtype=np.int32)
    base_idx = np.arange(-1, w - 1, dtype=np.int32)

    for r in range(1, h):
        curr_row = gray[r]
        curr_lshift = np.hstack((curr_row[1:], curr_row[-1]))
        curr_rshift = np.hstack((curr_row[0], curr_row[:-1]))
        cost_top = np.abs(curr_lshift - curr_rshift)
        if keep_mask is not None:
            cost_top[keep_mask[r]] += KEEP_MASK_ENERGY

        prev_row = gray[r - 1]
        cost_left = cost_top + np.abs(prev_row - curr_rshift)
        cost_right = cost_top + np.abs(prev_row - curr_lshift)

        dp_left = np.hstack((np.inf, dp[:-1]))
        dp_right = np.hstack((dp[1:], np.inf))

        choices = np.vstack([cost_left + dp_left, cost_top + dp, cost_right + dp_right])
        dp = np.min(choices, axis=0)
        parent[r] = np.argmin(choices, axis=0) + base_idx

    c = np.argmin(dp)
    total_cost = dp[c]

    seam = np.empty(h, dtype=np.int32)
    for r in range(h - 1, -1, -1):
        seam[r] = c
        c = parent[r, c]

    return seam, total_cost


# @timeit
def _get_forward_seams(
    gray: np.ndarray, num_seams: int, keep_mask: Optional[np.ndarray]
) -> np.ndarray:
    """Compute minimum N vertical seams using forward energy"""
    h, w = gray.shape
    seams_mask = np.zeros((h, w), dtype=np.bool)
    rows = np.arange(0, h, dtype=np.int32)
    idx_map = np.tile(np.arange(0, w, dtype=np.int32), h).reshape((h, w))
    for _ in range(num_seams):
        seam, _ = _get_forward_seam(gray, keep_mask)
        seams_mask[rows, idx_map[rows, seam]] = True
        seam_mask = _get_seam_mask(gray, seam)
        gray = _remove_seam_mask(gray, seam_mask)
        idx_map = _remove_seam_mask(idx_map, seam_mask)
        if keep_mask is not None:
            keep_mask = _remove_seam_mask(keep_mask, seam_mask)

    return seams_mask


# @timeit
def _get_seams(
    gray: np.ndarray, num_seams: int, energy_mode: str, keep_mask: Optional[np.ndarray]
) -> np.ndarray:
    """Get the minimum N seams from the grayscale image"""
    assert energy_mode in VALID_ENERGY_MODES
    if energy_mode == BACKWARD_ENERGY:
        return _get_backward_seams(gray, num_seams, keep_mask)
    else:
        return _get_forward_seams(gray, num_seams, keep_mask)


# @timeit
def _reduce_width(
    src: np.ndarray, delta_width: int, energy_mode: str, keep_mask: Optional[np.ndarray]
) -> np.ndarray:
    """Reduce the width of image by delta_width pixels"""
    assert src.ndim in (2, 3) and delta_width >= 0
    if src.ndim == 2:
        gray = src
        src_h, src_w = src.shape
        dst_shape = (src_h, src_w - delta_width)
    else:
        gray = _rgb2gray(src)
        src_h, src_w, src_c = src.shape
        dst_shape = (src_h, src_w - delta_width, src_c)

    seams_mask = _get_seams(gray, delta_width, energy_mode, keep_mask)
    dst = src[~seams_mask].reshape(dst_shape)
    return dst


# @timeit
def _expand_width(
    src: np.ndarray, delta_width: int, energy_mode: str, keep_mask: Optional[np.ndarray]
) -> np.ndarray:
    """Expand the width of image by delta_width pixels"""
    assert src.ndim in (2, 3) and delta_width >= 0
    if src.ndim == 2:
        gray = src
        src_h, src_w = src.shape
        dst_shape = (src_h, src_w + delta_width)
    else:
        gray = _rgb2gray(src)
        src_h, src_w, src_c = src.shape
        dst_shape = (src_h, src_w + delta_width, src_c)

    seams_mask = _get_seams(gray, delta_width, energy_mode, keep_mask)
    dst = np.empty(dst_shape, dtype=np.uint8)

    for row in range(src_h):
        dst_col = 0
        for src_col in range(src_w):
            if seams_mask[row, src_col]:
                lo = max(0, src_col - 1)
                hi = src_col + 1
                dst[row, dst_col] = src[row, lo:hi].mean(axis=0)
                dst_col += 1
            dst[row, dst_col] = src[row, src_col]
            dst_col += 1
        assert dst_col == src_w + delta_width

    return dst


# @timeit
def _resize_width(
    src: np.ndarray, width: int, energy_mode: str, keep_mask: Optional[np.ndarray]
) -> np.ndarray:
    """Resize the width of image by removing vertical seams"""
    assert src.size > 0 and src.ndim in (2, 3)
    assert width > 0
    assert energy_mode in VALID_ENERGY_MODES

    src_w = src.shape[1]
    if src_w < width:
        dst = _expand_width(src, width - src_w, energy_mode, keep_mask)
    else:
        dst = _reduce_width(src, src_w - width, energy_mode, keep_mask)
    return dst


# @timeit
def _resize_height(
    src: np.ndarray, height: int, energy_mode: str, keep_mask: Optional[np.ndarray]
) -> np.ndarray:
    """Resize the height of image by removing horizontal seams"""
    assert src.ndim in (2, 3) and height > 0
    if src.ndim == 3:
        src = _resize_width(
            src.transpose((1, 0, 2)), height, energy_mode, keep_mask
        ).transpose((1, 0, 2))
    else:
        src = _resize_width(src.T, height, energy_mode, keep_mask).T
    return src


# @timeit
def _check_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Ensure the mask to be a 2D grayscale map of specific shape"""
    mask = np.asarray(mask, dtype=np.bool)
    if mask.ndim != 2:
        raise ValueError(
            "Invalid mask of shape {}: expected to be a 2D "
            "binary map".format(mask.shape)
        )
    if mask.shape != shape:
        raise ValueError(
            "The shape of mask must match the image: expected {}, "
            "got {}".format(shape, mask.shape)
        )
    return mask


# @timeit
def _check_src(src: np.ndarray) -> np.ndarray:
    """Ensure the source to be RGB or grayscale"""
    src = np.asarray(src, dtype=np.uint8)
    if src.size == 0 or src.ndim not in (2, 3):
        raise ValueError(
            "Invalid src of shape {}: expected an 3D RGB image or "
            "a 2D grayscale image".format(src.shape)
        )
    return src


# @timeit
def resize(
    src: np.ndarray,
    size: Tuple[int, int],
    energy_mode: str = "backward",
    order: str = "width-first",
    keep_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Resize the image using the content-aware seam-carving algorithm.

    :param src: A source image in RGB or grayscale format.
    :param size: The target size in pixels, as a 2-tuple (width, height)
    :param energy_mode: Policy to compute energy for the source image. Could be
        one of ``backward`` or ``forward``. If ``backward``, compute the energy
        as the gradient at each pixel. If ``forward``, compute the energy as the
        distances between adjacent pixels after each pixel is removed.
    :param order: The order to remove horizontal and vertical seams. Could be
        one of ``width-first`` or ``height-first``. In ``width-first`` mode, we
        remove or insert all vertical seams first, then the horizontal ones,
        while ``height-first`` is the opposite.
    :param keep_mask: An optional mask where the foreground is protected from
        seam removal. If not specify, no area will be protected.
    :return: A resized copy of the source image.
    """
    src = _check_src(src)
    src_h, src_w = src.shape[:2]

    width, height = size
    width = int(round(width))
    height = int(round(height))
    if width <= 0 or height <= 0:
        raise ValueError("Invalid size {}: expected > 0".format(size))
    if width >= 2 * src_w:
        raise ValueError(
            "Invalid target width {}: expected less than twice "
            "the source width (< {})".format(width, 2 * src_w)
        )
    if height >= 2 * src_h:
        raise ValueError(
            "Invalid target height {}: expected less than twice "
            "the source height (< {})".format(height, 2 * src_h)
        )

    if order not in VALID_ORDERS:
        raise ValueError("Invalid order {}: expected {}".format(order, VALID_ORDERS))

    if energy_mode not in VALID_ENERGY_MODES:
        raise ValueError(
            "Invalid energy mode {}: expected {}".format(
                energy_mode, VALID_ENERGY_MODES
            )
        )

    if keep_mask is not None:
        keep_mask = _check_mask(keep_mask, (src_h, src_w))

    if order == WIDTH_FIRST:
        src = _resize_width(src, width, energy_mode, keep_mask)
        src = _resize_height(src, height, energy_mode, keep_mask)
    else:
        src = _resize_height(src, height, energy_mode, keep_mask)
        src = _resize_width(src, width, energy_mode, keep_mask)

    return src


# @timeit
def remove_object(
    src: np.ndarray, drop_mask: np.ndarray, keep_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Remove an object on the source image.

    :param src: A source image in RGB or grayscale format.
    :param drop_mask: A binary object mask to remove.
    :param keep_mask: An optional binary object mask to be protected from
        removal. If not specified, no area is protected.
    :return: A copy of the source image where the drop_mask is removed.
    """
    src = _check_src(src)

    drop_mask = _check_mask(drop_mask, src.shape[:2])

    if keep_mask is not None:
        keep_mask = _check_mask(keep_mask, src.shape[:2])

    gray = src if src.ndim == 2 else _rgb2gray(src)

    while drop_mask.any():
        energy = _get_energy(gray)
        energy[drop_mask] -= DROP_MASK_ENERGY
        if keep_mask is not None:
            energy[keep_mask] += KEEP_MASK_ENERGY
        seam, _ = _get_backward_seam(energy)
        seam_mask = _get_seam_mask(src, seam)
        gray = _remove_seam_mask(gray, seam_mask)
        drop_mask = _remove_seam_mask(drop_mask, seam_mask)
        src = _remove_seam_mask(src, seam_mask)
        if keep_mask is not None:
            keep_mask = _remove_seam_mask(keep_mask, seam_mask)

    return src


if __name__ == "__main__":
    from PIL import Image

    img = np.array(
        Image.open("E:/imagenet-mini/val/n01440764/ILSVRC2012_val_00009111.JPEG")
    )
    img_resized = resize(img, size=(224, 224))
