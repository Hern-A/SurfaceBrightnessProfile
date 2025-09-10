# utils_background.py
from __future__ import annotations

import numpy as np
from astropy.stats import SigmaClip
from photutils.background import (
    Background2D,
    SExtractorBackground,
    MADStdBackgroundRMS,
)

def photutils_bkg(
    data: np.ndarray,
    bw: int = 64,
    bh: int = 64,
    fw: int = 3,
    fh: int = 3,
    *,
    mask: np.ndarray | None = None,
    sigma: float = 3.0,
    maxiters: int = 5,
):
    """
    SEP-like background & RMS using Photutils.
    Returns (background_map, background_rms_map).

    Parameters
    ----------
    data : 2D ndarray
        Image data.
    bw, bh : int
        Background mesh (box) size in x and y (like sep bw/bh).
    fw, fh : int
        Median filter size in x and y (like sep fw/fh). Use small odd ints.
    mask : 2D bool ndarray, optional
        True where pixels should be ignored (e.g., sources, bad pixels).
    sigma : float
        Sigma for sigma-clipping.
    maxiters : int
        Max iters for sigma-clipping.
    edge_method : {'pad','crop'}
        How to handle partial boxes at edges.

    Notes
    -----
    Uses SExtractorBackground (robust mode/median) and
    MADStdBackgroundRMS (robust RMS).
    """
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
