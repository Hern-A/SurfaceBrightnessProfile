# utils_bkg_tuner.py
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any, Optional

import numpy as np
from astropy.stats import SigmaClip
from photutils.background import Background2D, SExtractorBackground, MADStdBackgroundRMS
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint  # optional for round dilation


# ---------- Core background call (same engine as your wrapper) ----------
def _run_bkg(
    data: np.ndarray,
    bw: int,
    bh: int,
    fw: int,
    fh: int,
    sigma: float,
    maxiters: int,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    sigma_clip = SigmaClip(sigma=sigma, maxiters=maxiters)
    bkg2d = Background2D(
        data,
        box_size=(bw, bh),
        filter_size=(fw, fh),
        bkg_estimator=SExtractorBackground(),
        bkgrms_estimator=MADStdBackgroundRMS(),
        sigma_clip=sigma_clip,
        mask=mask,
    )
    return bkg2d.background, bkg2d.background_rms


# ---------- Simple auto-mask via segmentation (Photutils ≥ 2.2) ----------
def make_auto_mask(
    data: np.ndarray,
    *,
    nsigma: float = 2.5,
    npixels: int = 20,
    dilate: int = 3,
    circular: bool = False,
    clip_sigma: float = 3.0,
) -> np.ndarray:
    """
    Build a boolean mask of sources to exclude from background estimation.

    Returns
    -------
    mask : 2D bool ndarray
        True where pixels are masked (ignored by Background2D).
    """
    sigma_clip = SigmaClip(clip_sigma)
    # auto threshold from background estimate of the data itself
    thr = detect_threshold(data, nsigma=nsigma, sigma_clip=sigma_clip)
    segm = detect_sources(data, threshold=thr, npixels=npixels)

    if segm is None:
        # no detections; return no mask
        return np.zeros_like(data, dtype=bool)

    if circular:
        fp = circular_footprint(radius=max(1, dilate))
        mask = segm.make_source_mask(footprint=fp)
    else:
        mask = segm.make_source_mask(size=max(1, dilate))  # square is fastest
    return mask


# ---------- Scoring ----------
@dataclass
class BkgScore:
    params: Dict[str, Any]
    med: float
    mad: float
    smooth_penalty: float
    score: float


def _score_solution(
    data: np.ndarray,
    bkg: np.ndarray,
    mask: Optional[np.ndarray],
    *,
    smooth_weight: float = 0.25,
) -> BkgScore:
    """
    Score a background solution using:
      - |median(residual)| (target ~ 0)
      - MAD(residual)     (smaller is better)
      - smoothness penalty on bkg map (discourage high-frequency structure)
    """
    resid = data - bkg
    if mask is not None:
        valid = (~mask) & np.isfinite(resid)
    else:
        valid = np.isfinite(resid)

    # Robust stats on background-only pixels
    vals = resid[valid]
    if vals.size == 0:
        med = np.inf
        mad = np.inf
    else:
        med = float(np.nanmedian(vals))
        mad = float(np.nanmedian(np.abs(vals - med)))

    # Simple smoothness metric: RMS of Laplacian-like filter on bkg map
    # (penalizes checkerboard or boxy patterns)
    # Use a tiny finite-difference kernel without external deps
    k = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]], dtype=float)
    # valid region for bkg
    b = bkg
    if not np.all(np.isfinite(b)):
        b = np.nan_to_num(b, copy=False)

    # cheap valid convolution (no padding): compute on inner region
    smooth_area = 0.0
    if b.shape[0] > 2 and b.shape[1] > 2:
        conv = (
            k[0, 1] * b[:-2, 1:-1] + k[1, 0] * b[1:-1, :-2] +
            k[1, 1] * b[1:-1, 1:-1] +
            k[1, 2] * b[1:-1, 2:] + k[2, 1] * b[2:, 1:-1]
        )
        smooth_area = float(np.sqrt(np.nanmean(conv**2)))
    smooth_penalty = smooth_weight * smooth_area

    # Final scalar score (lower is better)
    score = abs(med) + mad + smooth_penalty

    return med, mad, smooth_penalty, score  # packed below by caller


# ---------- Parameter tuner ----------
def tune_photutils_bkg(
    data: np.ndarray,
    *,
    # parameter grids (edit these to taste)
    bw_list: Iterable[int] = (64, 96, 128),
    bh_list: Iterable[int] = (64, 96, 128),
    fw_list: Iterable[int] = (3, 5),
    fh_list: Iterable[int] = (3, 5),
    sigma_list: Iterable[float] = (2.5, 3.0, 3.5),
    maxiters_list: Iterable[int] = (3, 5),
    # masking
    auto_mask: bool = True,
    nsigma: float = 2.5,
    npixels: int = 20,
    dilate: int = 3,
    circular_mask: bool = False,
    user_mask: Optional[np.ndarray] = None,
    # scoring
    smooth_weight: float = 0.25,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, List[BkgScore]]:
    """
    Sweep a small grid of parameters and pick the best by residual stats.

    Returns
    -------
    best_params : dict
    best_bkg    : 2D array
    best_rms    : 2D array
    report      : list[BkgScore] sorted by increasing score
    """
    # Compose mask
    mask = None
    if auto_mask:
        auto = make_auto_mask(
            data,
            nsigma=nsigma,
            npixels=npixels,
            dilate=dilate,
            circular=circular_mask,
        )
        mask = auto if user_mask is None else (auto | user_mask)
    else:
        mask = user_mask

    results: List[BkgScore] = []
    best = dict(score=np.inf)
    best_bkg = None
    best_rms = None

    for bw, bh, fw, fh, sigma, maxiters in itertools.product(
        bw_list, bh_list, fw_list, fh_list, sigma_list, maxiters_list
    ):
        # enforce odd filter size (Background2D expects odd integers)
        if fw % 2 == 0 or fh % 2 == 0:
            continue

        try:
            bkg_map, bkg_rms = _run_bkg(
                data, bw, bh, fw, fh, sigma, maxiters, mask=mask
            )
            med, mad, smooth_pen, score = _score_solution(
                data, bkg_map, mask, smooth_weight=smooth_weight
            )
        except Exception as e:
            # if any combo fails (e.g., too-small boxes), skip it
            # print(f"Skip combo {(bw,bh,fw,fh,sigma,maxiters)}: {e}")
            continue

        params = dict(bw=bw, bh=bh, fw=fw, fh=fh, sigma=sigma, maxiters=maxiters)
        entry = BkgScore(params=params, med=med, mad=mad,
                         smooth_penalty=smooth_pen, score=score)
        results.append(entry)

        if score < best["score"]:
            best = dict(**params, score=score, med=med, mad=mad, smooth=smooth_pen)
            best_bkg = bkg_map
            best_rms = bkg_rms

    # sort report by score (ascending)
    results_sorted = sorted(results, key=lambda r: r.score)

    if best_bkg is None:
        raise RuntimeError("No valid background solution found. "
                           "Try different grids or disable auto_mask.")

    return best, best_bkg, best_rms, results_sorted
