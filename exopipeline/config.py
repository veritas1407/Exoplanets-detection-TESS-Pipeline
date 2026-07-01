"""Central configuration for the exopipeline transit-vetting pipeline.

All tunable thresholds, physical constants, and paths live here so the modules,
notebooks, and the Streamlit app share one source of truth.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PKG_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
LABELS_DIR = DATA_DIR / "labels"
FEATURES_DIR = DATA_DIR / "features"
CACHE_DIR = DATA_DIR / "cache"          # downloaded light curves / TPFs
for _d in (DATA_DIR, LABELS_DIR, FEATURES_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FEATURE_TABLE = FEATURES_DIR / "features.parquet"
MODEL_PATH = FEATURES_DIR / "classifier.joblib"

# --------------------------------------------------------------------------------------
# Bulk-sector ingest (the dataset PS7 actually asks for: a sector's 2-min light curves)
# --------------------------------------------------------------------------------------
# Per-sector MAST bulk-download shell script of 2-min SPOC light-curve cURL commands.
# NOTE: the PS link archive.stsci.edu/tess/tic_ctl.html is the *star catalog* (metadata,
# no photometry) — NOT the light curves. The light curves live in these per-sector scripts.
BULK_SCRIPT_URL = ("https://archive.stsci.edu/missions/tess/download_scripts/"
                   "sector/tesscurl_sector_{sector}_lc.sh")
# A single FITS is fetched per target from this file-download API (parsed out of the script):
#   https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:TESS/product/<lc_file>.fits
DEFAULT_SECTOR = 5                # contains TOI 700 (TIC 307210830) + TOI-270 (TIC 259377017)
SECTOR_CANDIDATES = [5, 4, 3]     # southern sectors with the anchor targets (auto-pick order)
SLICE_SIZE = 4000                 # representative unbiased slice (first N of ~20k); None = full
SCAN_CHECKPOINT_EVERY = 25        # rows between candidate-CSV checkpoints
# Single-sector scans only span ~27 d, so a period > baseline/2 has <2 transits and is
# undetectable. Cap the blind BLS grid accordingly to keep the per-target search fast.
SCAN_PERIOD_MAX_FRAC = 0.5        # period_max = this * light-curve baseline (days)
SCAN_PERIOD_MAX_CAP = 13.0        # never search longer than this in a single-sector scan
SCAN_N_PERIODS = 12000            # BLS grid points for the (smaller) single-sector range
SCAN_WORKERS = 0                  # 0 = use (cpu_count - 1) processes; >=1 sets it explicitly
PREFETCH_WORKERS = 24             # I/O-bound download threads (network waits release the
                                   # GIL, so this can far exceed cpu_count -- see ingest.py)
SECTOR_CANDIDATES_CSV = FEATURES_DIR / "sector_{sector}_candidates.csv"
SECTOR_MANIFEST = LABELS_DIR / "sector_{sector}_manifest.parquet"

# --------------------------------------------------------------------------------------
# Detection thresholds
# --------------------------------------------------------------------------------------
SDE_THRESHOLD = 9.0       # TLS signal-detection-efficiency floor for a real signal
SNR_THRESHOLD = 7.0       # per-event SNR floor
BLS_POWER_RATIO_MIN = 7.0  # BLS broad-sweep stop criterion (peak / median power)
MAX_PLANETS = 5           # iterative-masking loop cap per star

# --------------------------------------------------------------------------------------
# Search grid
# --------------------------------------------------------------------------------------
PERIOD_MIN = 0.5          # days
PERIOD_MAX = 40.0         # days  (keeps blind search tractable on 2-min cadence)
N_PERIODS = 60000         # BLS broad-sweep grid resolution
# Box durations (days) spanning ~1.7 h .. ~6 h transits
BLS_DURATIONS = [0.04, 0.06, 0.08, 0.10, 0.13, 0.17, 0.22]
TLS_REFINE_HALFWIDTH = 0.05   # +/- days around a BLS peak for the narrow TLS refine

# --------------------------------------------------------------------------------------
# Detrending
# --------------------------------------------------------------------------------------
DETREND_WINDOW = 0.4      # days; ~5x a typical 2 h transit
DETREND_METHOD = "biweight"

# --------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------
SIGMA_UPPER = 4.0         # clip flares/glitches aggressively
SIGMA_LOWER = 6.0         # clip transits gently (never remove the signal)

# --------------------------------------------------------------------------------------
# Limb darkening + MCMC
# --------------------------------------------------------------------------------------
LD_QUADRATIC = [0.35, 0.20]   # M-dwarf-ish, TESS band; override per-star if known
MCMC_NWALKERS = 32
MCMC_NSTEPS = 3000
MCMC_NBURN = 1000
FIT_WINDOW = 0.18             # days either side of mid-transit fitted by MCMC

# --------------------------------------------------------------------------------------
# Stellar reference values (used to convert Rp/R* -> R_Earth in demos)
# --------------------------------------------------------------------------------------
RSUN_REARTH = 109.1           # R_sun / R_earth
DEFAULT_RSTAR_SUN = 0.420     # TOI 700 radius; override per target

# --------------------------------------------------------------------------------------
# Classifier
# --------------------------------------------------------------------------------------
CLASSES = ["transit", "eclipsing_binary", "blend", "other"]
FEATURE_COLUMNS = [
    "period", "depth_ppm", "duration_hr", "sde", "snr", "log_fap",
    "odd_even_diff_ppm", "odd_even_sigma", "secondary_ppm", "secondary_snr",
    "flatness", "vshape_ratio", "n_transits", "snr_per_transit", "transit_snr",
    "depth_consistency", "phase_coverage", "dur_over_period",
    "rho_ratio", "rp_rs", "crowdsap",
]

# --------------------------------------------------------------------------------------
# Default demo targets
# --------------------------------------------------------------------------------------
DEMO_PLANET = "TIC 307210830"     # TOI 700 d — focused characterisation (literature match)
DEMO_BLIND = "TIC 259377017"      # TOI-270 — clean multi-planet blind-search demo (c, d)
# TOI 700 is a known-hard *blind* target: a dominant ~3.69 d instrumental systematic
# (SDE ~ 128, robust to detrending) buries its shallow planets (local SDE ~ 10-20). We
# therefore characterise TOI 700 d with a focused search and reserve blind discovery for
# cleaner targets — exactly how real vetting pipelines operate.
