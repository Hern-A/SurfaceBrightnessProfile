# utils_bkg_tuner.py
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any, Optional

import numpy as np
from astropy.io import fits
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
    Returns True where pixels are masked (ignored by Background2D).
    """
    sigma_clip = SigmaClip(clip_sigma)
    thr = detect_threshold(data, nsigma=nsigma, sigma_clip=sigma_clip)
    segm = detect_sources(data, threshold=thr, npixels=npixels)
    if segm is None:
        return np.zeros_like(data, dtype=bool)

    if circular:
        fp = circular_footprint(radius=max(1, dilate))
        mask = segm.make_source_mask(footprint=fp)
    else:
        mask = segm.make_source_mask(size=max(1, dilate))  # square = fastest
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
) -> Tuple[float, float, float, float]:
    """
    Score a background solution using:
      - |median(residual)| (target ~ 0)
      - MAD(residual)     (smaller is better)
      - smoothness penalty on bkg map (discourage high-frequency structure)
    """
    resid = data - bkg
    valid = np.isfinite(resid)
    if mask is not None:
        valid &= ~mask

    vals = resid[valid]
    if vals.size == 0:
        med = np.inf
        mad = np.inf
    else:
        med = float(np.nanmedian(vals))
        mad = float(np.nanmedian(np.abs(vals - med)))

    # Simple Laplacian RMS as smoothness penalty
    k = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]], dtype=float)
    b = np.nan_to_num(bkg, copy=False)
    smooth_area = 0.0
    if b.shape[0] > 2 and b.shape[1] > 2:
        conv = (
            k[0, 1] * b[:-2, 1:-1] + k[1, 0] * b[1:-1, :-2] +
            k[1, 1] * b[1:-1, 1:-1] +
            k[1, 2] * b[1:-1, 2:] + k[2, 1] * b[2:, 1:-1]
        )
        smooth_area = float(np.sqrt(np.nanmean(conv**2)))
    smooth_penalty = smooth_weight * smooth_area

    score = abs(med) + mad + smooth_penalty
    return med, mad, smooth_penalty, score


# ---------- Parameter tuner ----------
def tune_photutils_bkg(
    data: np.ndarray,
    *,
    bw_list: Iterable[int] = (64, 96, 128),
    bh_list: Iterable[int] = (64, 96, 128),
    fw_list: Iterable[int] = (3, 5),
    fh_list: Iterable[int] = (3, 5),
    sigma_list: Iterable[float] = (2.5, 3.0, 3.5),
    maxiters_list: Iterable[int] = (3, 5),
    auto_mask: bool = True,
    nsigma: float = 2.5,
    npixels: int = 20,
    dilate: int = 3,
    circular_mask: bool = False,
    user_mask: Optional[np.ndarray] = None,
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
        # enforce odd filter size
        if fw % 2 == 0 or fh % 2 == 0:
            continue

        try:
            bkg_map, bkg_rms = _run_bkg(data, bw, bh, fw, fh, sigma, maxiters, mask=mask)
            med, mad, smooth_pen, score = _score_solution(
                data, bkg_map, mask, smooth_weight=smooth_weight
            )
        except Exception:
            continue

        params = dict(bw=bw, bh=bh, fw=fw, fh=fh, sigma=sigma, maxiters=maxiters)
        results.append(BkgScore(params=params, med=med, mad=mad,
                                smooth_penalty=smooth_pen, score=score))

        if score < best["score"]:
            best = dict(**params, score=score, med=med, mad=mad, smooth=smooth_pen)
            best_bkg = bkg_map
            best_rms = bkg_rms

    results_sorted = sorted(results, key=lambda r: r.score)

    if best_bkg is None:
        raise RuntimeError("No valid background solution found. Try different grids or disable auto_mask.")

    return best, best_bkg, best_rms, results_sorted


# ---------- CLI helpers ----------
def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]

def _parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]

def _write_fits(array: np.ndarray, like_header: fits.Header | None, path: str, overwrite: bool):
    hdu = fits.PrimaryHDU(array, header=like_header)
    hdu.writeto(path, overwrite=overwrite)


def main():
    p = argparse.ArgumentParser(
        description="Tune Photutils Background2D parameters and optionally save results."
    )
    p.add_argument("fits_path", help="Input FITS image (2D).")
    p.add_argument("--ext", type=int, default=0, help="FITS HDU extension (default 0).")

    # grids
    p.add_argument("--bw", default="64,96,128", help="Comma-separated list for box width (e.g., 64,96,128)")
    p.add_argument("--bh", default="64,96,128", help="Comma-separated list for box height (e.g., 64,96,128)")
    p.add_argument("--fw", default="3,5", help="Comma-separated list for filter width (odd ints)")
    p.add_argument("--fh", default="3,5", help="Comma-separated list for filter height (odd ints)")
    p.add_argument("--sigma", default="2.5,3.0,3.5", help="Comma-separated list for sigma-clip sigma")
    p.add_argument("--maxiters", default="3,5", help="Comma-separated list for sigma-clip maxiters")

    # masking
    p.add_argument("--no-auto-mask", action="store_true", help="Disable auto source masking")
    p.add_argument("--nsigma", type=float, default=2.5, help="Detection nsigma for auto-mask")
    p.add_argument("--npixels", type=int, default=20, help="Min pixels per source for auto-mask")
    p.add_argument("--dilate", type=int, default=3, help="Dilation size/radius for mask growth")
    p.add_argument("--circular-mask", action="store_true", help="Use circular dilation footprint")

    # scoring
    p.add_argument("--smooth-weight", type=float, default=0.25, help="Weight of background smoothness in score")

    # outputs
    p.add_argument("--write-prefix", default=None,
                   help="If set, write <prefix>_bkg.fits, <prefix>_rms.fits, <prefix>_resid.fits")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output FITS")
    p.add_argument("--plot", default=None,
                   help="If set, save a quick PNG tri-panel to this path (requires matplotlib).")
    p.add_argument("--topk", type=int, default=5, help="Print top-K parameter sets by score")

    args = p.parse_args()

    # load data
    with fits.open(args.fits_path, memmap=True) as hdul:
        data = hdul[args.ext].data.astype(float, copy=False)
        hdr = hdul[args.ext].header

    # parse grids
    bw_list = _parse_int_list(args.bw)
    bh_list = _parse_int_list(args.bh)
    fw_list = _parse_int_list(args.fw)
    fh_list = _parse_int_list(args.fh)
    sigma_list = _parse_float_list(args.sigma)
    maxiters_list = _parse_int_list(args.maxiters)

    # run tuner
    best, bkg_map, bkg_rms, report = tune_photutils_bkg(
        data,
        bw_list=bw_list,
        bh_list=bh_list,
        fw_list=fw_list,
        fh_list=fh_list,
        sigma_list=sigma_list,
        maxiters_list=maxiters_list,
        auto_mask=not args.no_auto_mask,
        nsigma=args.nsigma,
        npixels=args.npixels,
        dilate=args.dilate,
        circular_mask=args.circular_mask,
        user_mask=None,
        smooth_weight=args.smooth_weight,
    )

    # print results
    print("\nBest params:")
    for k in ["bw", "bh", "fw", "fh", "sigma", "maxiters"]:
        print(f"  {k} = {best[k]}")
    print(f"Score = {best['score']:.5g}  (med={best['med']:.4g}, MAD={best['mad']:.4g}, smooth={best['smooth']:.4g})\n")

    topk = max(1, min(args.topk, len(report)))
    print(f"Top {topk} candidates:")
    for r in report[:topk]:
        pstr = ", ".join(f"{k}={v}" for k, v in r.params.items())
        print(f"  [{pstr}]  score={r.score:.5g}  med={r.med:.4g}  MAD={r.mad:.4g}  smooth={r.smooth_penalty:.4g}")

    # write outputs
    if args.write_prefix:
        _write_fits(bkg_map, hdr, f"{args.write_prefix}_bkg.fits", overwrite=args.overwrite)
        _write_fits(bkg_rms, hdr, f"{args.write_prefix}_rms.fits", overwrite=args.overwrite)
        resid = data - bkg_map
        _write_fits(resid, hdr, f"{args.write_prefix}_resid.fits", overwrite=args.overwrite)
        print(f"\nWrote FITS:\n  {args.write_prefix}_bkg.fits\n  {args.write_prefix}_rms.fits\n  {args.write_prefix}_resid.fits")

    # optional plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
            for a in ax: a.set_xticks([]); a.set_yticks([])
            ax[0].imshow(data, origin="lower", cmap="gray")
            ax[0].set_title("Data")
            ax[1].imshow(bkg_map, origin="lower", cmap="gray")
            ax[1].set_title("Background")
            ax[2].imshow(data - bkg_map, origin="lower", cmap="gray")
            ax[2].set_title("Residual")
            fig.savefig(args.plot, dpi=150)
            plt.close(fig)
            print(f"Saved plot: {args.plot}")
        except Exception as e:
            print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
