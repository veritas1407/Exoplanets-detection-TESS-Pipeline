"""Stage 1 — Data ingestion.

Download 2-minute cadence SPOC light curves from MAST, stitch all sectors, read the
dilution keywords (CROWDSAP / FLFRCSAP), and optionally fetch a Target Pixel File for the
difference-imaging blend test.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from . import config

warnings.filterwarnings("ignore")


@dataclass
class Star:
    """A stitched, NaN-cleaned light curve plus header metadata for one target."""
    target: str
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    crowdsap: float
    flfrcsap: float
    n_sectors: int
    sectors: list = field(default_factory=list)
    raw_lc: object = None          # the stitched lightkurve object (for plotting)
    pre_flattened: bool = False    # True if flux is already per-sector detrended

    @property
    def baseline_days(self) -> float:
        return float(self.time.max() - self.time.min())


def load_star(target: str, max_sectors: int | None = None,
              quality_bitmask: str = "default",
              flatten_per_sector: bool = False,
              detrend_window: float | None = None) -> Star:
    """Search MAST, download all (or the first ``max_sectors``) SPOC 2-min sectors,
    stitch them, and return a :class:`Star`.

    Parameters
    ----------
    target : str
        e.g. ``"TIC 307210830"``.
    max_sectors : int, optional
        Limit the number of sectors (speeds up dev / blind search). ``None`` = all.
    quality_bitmask : str
        Passed to lightkurve. ``"hard"`` removes momentum-dump / scattered-light cadences
        that otherwise create strong short-period systematics in a blind search.
    flatten_per_sector : bool
        If True, detrend each sector *before* stitching (wotan biweight). This removes
        per-sector systematic structure (scattered-light ramps) far better than detrending
        the stitched series across the sector gaps. Recommended for blind search.
    detrend_window : float, optional
        Window (days) for the per-sector flatten (default ``config.DETREND_WINDOW``).
    """
    import lightkurve as lk
    from wotan import flatten as _wflatten

    window = detrend_window or config.DETREND_WINDOW
    search = lk.search_lightcurve(target, author="SPOC", cadence="short")
    if len(search) == 0:
        raise ValueError(f"No SPOC 2-min light curves found for {target!r}")
    if max_sectors is not None:
        search = search[:max_sectors]

    collection = None
    last_err = None
    for attempt in range(4):                    # MAST occasionally drops connections
        try:
            collection = search.download_all(quality_bitmask=quality_bitmask,
                                             download_dir=str(config.CACHE_DIR))
            break
        except Exception as e:
            last_err = e
            import time as _t
            _t.sleep(5 * (attempt + 1))
    if collection is None:
        raise RuntimeError(f"MAST download failed after retries for {target!r}: {last_err}")

    hdr0 = collection[0].meta
    crowdsap = float(hdr0.get("CROWDSAP", np.nan))
    flfrcsap = float(hdr0.get("FLFRCSAP", np.nan))

    if flatten_per_sector:
        # Normalise + biweight-detrend each sector independently, then concatenate.
        times, fluxes, errs = [], [], []
        for lc1 in collection:
            lc1 = lc1.remove_nans().normalize()
            t = np.ascontiguousarray(lc1.time.value, dtype="float64")
            f = np.ascontiguousarray(lc1.flux.value, dtype="float64")
            if t.size < 50:
                continue
            ftr, _ = _wflatten(t, f, window_length=window,
                               method=config.DETREND_METHOD, return_trend=True)
            good = np.isfinite(ftr)
            times.append(t[good]); fluxes.append(ftr[good])
            try:
                e = np.ascontiguousarray(lc1.flux_err.value, dtype="float64")[good]
            except Exception:
                e = np.full(good.sum(), np.nan)
            errs.append(e)
        time = np.concatenate(times)
        flux = np.concatenate(fluxes)
        flux_err = np.concatenate(errs)
        order = np.argsort(time)
        time, flux, flux_err = time[order], flux[order], flux_err[order]
        raw_lc = collection.stitch().remove_nans()   # keep raw for the plot panel
    else:
        lc = collection.stitch().remove_nans()
        time = np.ascontiguousarray(lc.time.value, dtype="float64")
        flux = np.ascontiguousarray(lc.flux.value, dtype="float64")
        try:
            flux_err = np.ascontiguousarray(lc.flux_err.value, dtype="float64")
        except Exception:
            flux_err = np.full_like(flux, np.nan)
        raw_lc = lc

    sectors = []
    try:
        sectors = [int(str(m).split("Sector")[1].strip())
                   for m in search.table["mission"]]
    except Exception:
        pass

    return Star(target=target, time=time, flux=flux, flux_err=flux_err,
                crowdsap=crowdsap, flfrcsap=flfrcsap,
                n_sectors=len(collection), sectors=sectors, raw_lc=raw_lc,
                pre_flattened=flatten_per_sector)


def clean(star: Star) -> Star:
    """Sigma-clip flares/glitches on the upper side aggressively, transits gently.

    Returns a new :class:`Star` with the cleaned arrays (raw_lc preserved).
    """
    import lightkurve as lk

    lc = lk.LightCurve(time=star.time, flux=star.flux)
    clipped = lc.remove_outliers(sigma_upper=config.SIGMA_UPPER,
                                 sigma_lower=config.SIGMA_LOWER)
    keep = np.isin(star.time, clipped.time.value)
    return Star(
        target=star.target,
        time=star.time[keep], flux=star.flux[keep],
        flux_err=star.flux_err[keep] if star.flux_err is not None else None,
        crowdsap=star.crowdsap, flfrcsap=star.flfrcsap,
        n_sectors=star.n_sectors, sectors=star.sectors, raw_lc=star.raw_lc,
        pre_flattened=star.pre_flattened,
    )


# --------------------------------------------------------------------------------------
# Bulk-sector ingest — the dataset PS7 actually asks for
# --------------------------------------------------------------------------------------
def sector_lc_manifest(sector: int | None = None, limit: int | None = None,
                       force: bool = False):
    """Fetch + parse a sector's 2-min LC bulk-download script into a manifest.

    Downloads ``tesscurl_sector_{sector}_lc.sh`` from MAST and parses each ``curl`` line
    into ``{tic, url, lc_file, sector}`` — *without* downloading any FITS yet. The parsed
    manifest is cached to ``data/labels/sector_{sector}_manifest.parquet``.

    Returns a pandas DataFrame (one row per target).
    """
    import re
    import pandas as pd

    sector = int(sector if sector is not None else config.DEFAULT_SECTOR)
    cache = config.LABELS_DIR / f"sector_{sector}_manifest.parquet"
    if cache.exists() and not force:
        df = pd.read_parquet(cache)
        return df.head(limit) if limit else df

    url = config.BULK_SCRIPT_URL.format(sector=sector)
    text = _http_get_text(url)
    rows = []
    # Each data line: curl -C - -L -o <lc_file>.fits <download_url>
    # The 16-digit zero-padded TIC id is the middle field of the SPOC filename:
    #   tess<obsdate>-s00NN-0000000307210830-0125-s_lc.fits
    line_re = re.compile(r"-o\s+(\S+\.fits)\s+(\S+)")
    tic_re = re.compile(r"-s\d{4}-0*(\d+)-")
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("curl"):
            continue
        m = line_re.search(line)
        if not m:
            continue
        lc_file, dl_url = m.group(1), m.group(2)
        tm = tic_re.search(lc_file)
        if not tm:
            continue
        rows.append({"tic": f"TIC {int(tm.group(1))}", "tid": int(tm.group(1)),
                     "lc_file": lc_file, "url": dl_url, "sector": sector})

    df = pd.DataFrame(rows).drop_duplicates(subset="tid").reset_index(drop=True)
    df.to_parquet(cache, index=False)
    return df.head(limit) if limit else df


def _http_get_text(url: str) -> str:
    """GET a text resource with a small retry loop (MAST occasionally drops connections)."""
    import time as _t
    import urllib.request

    last_err = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "exopipeline/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            last_err = e
            _t.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url!r} after retries: {last_err}")


def _star_from_lightcurve(lc, target: str) -> Star:
    """Build a cleaned :class:`Star` from a single lightkurve LightCurve (one sector)."""
    lc = lc.remove_nans()
    time = np.ascontiguousarray(lc.time.value, dtype="float64")
    flux = np.ascontiguousarray(lc.flux.value, dtype="float64")
    try:
        flux_err = np.ascontiguousarray(lc.flux_err.value, dtype="float64")
    except Exception:
        flux_err = np.full_like(flux, np.nan)
    # normalise to ~1.0 so detrend/search thresholds match the stitched path
    med = np.nanmedian(flux)
    if np.isfinite(med) and med != 0:
        flux = flux / med
        flux_err = flux_err / med
    meta = getattr(lc, "meta", {}) or {}
    crowdsap = float(meta.get("CROWDSAP", np.nan))
    flfrcsap = float(meta.get("FLFRCSAP", np.nan))
    sector = meta.get("SECTOR")
    return Star(target=target, time=time, flux=flux, flux_err=flux_err,
                crowdsap=crowdsap, flfrcsap=flfrcsap, n_sectors=1,
                sectors=[int(sector)] if sector is not None else [], raw_lc=lc,
                pre_flattened=False)


def load_lc_from_file(path, target: str | None = None) -> Star:
    """Read one SPOC 2-min LC FITS already on disk into a :class:`Star` (PDCSAP_FLUX)."""
    import lightkurve as lk

    lc = lk.read(str(path))            # SPOC LC -> defaults to PDCSAP_FLUX
    tic = target
    if tic is None:
        tic = f"TIC {lc.meta.get('TICID')}" if lc.meta.get("TICID") else str(path)
    return _star_from_lightcurve(lc, tic)


def load_lc_from_url(url: str, lc_file: str | None = None,
                     target: str | None = None, keep_fits: bool = True) -> Star:
    """Download one SPOC 2-min LC FITS from a MAST file URL and load it as a :class:`Star`.

    Caches the FITS under ``data/cache/``. If ``keep_fits`` is False the file is deleted
    after loading (streaming mode to cap disk footprint on a full-sector run).
    """
    import re
    import time as _t
    import urllib.request

    if lc_file is None:
        m = re.search(r"([^/=:]+\.fits)", url)
        lc_file = m.group(1) if m else "lc.fits"
    dest = config.CACHE_DIR / lc_file

    if not dest.exists():
        last_err = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "exopipeline/1.0"})
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = resp.read()
                dest.write_bytes(data)
                break
            except Exception as e:
                last_err = e
                _t.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"Download failed for {url!r}: {last_err}")

    try:
        star = load_lc_from_file(dest, target=target)
    finally:
        if not keep_fits and dest.exists():
            try:
                dest.unlink()
            except Exception:
                pass
    return star


def load_tpf(target: str, sector: int | None = None):
    """Fetch a SPOC 2-min Target Pixel File for the difference-imaging blend test.

    Returns the first (or the requested sector's) TPF, or ``None`` if unavailable.
    """
    import lightkurve as lk

    search = lk.search_targetpixelfile(target, author="SPOC", cadence="short")
    if len(search) == 0:
        return None
    if sector is not None:
        match = [i for i, m in enumerate(search.table["mission"])
                 if f"Sector {sector}" in str(m)]
        if match:
            search = search[match[0]]
    return search[0].download(download_dir=str(config.CACHE_DIR))
