from __future__ import annotations
"""Dynamic‑mechanical‑analysis (DMA) log parser.

This version adds a **dual analysis path** controlled by the ``analysis_mode``
attribute:

* ``"fft"`` (default) – lock‑in / Fourier first‑harmonic projection with THD &
  S/N metrics.
* ``"fit"`` – legacy least‑squares single‑sine fit preserved for full
  backward compatibility.

Both paths return the same high‑level quantities (G', G" in **GPa**, tan δ,
phase, etc.) so downstream jobs remain agnostic.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.fft  # lock‑in projection & THD
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

from jobflow import Maker, job, Response

# ---- atomate2 schema imports -------------------------------------------------
from atomate2.dmax.schemas.task import (
    DmaxDmaParserDocument,
    DmaxLammpsInputDocument,
    DmaxLammpsRunDocument,
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _first_harmonic_projection(t: np.ndarray, y: np.ndarray, omega: float) -> Tuple[float, float]:
    """Return the Fourier coefficients (A₁, B₁) of *y(t)* at `omega`.

    A₁ multiplies **sin(ωt)** → in‑phase (storage) component
    B₁ multiplies **cos(ωt)** → out‑of‑phase (loss) component
    """
    # Numerical lock‑in: ⟨y·sin⟩ and ⟨y·cos⟩ over an **integer** number of periods.
    sin_ref = np.sin(omega * t)
    cos_ref = np.cos(omega * t)
    A1 = 2.0 * np.dot(y, sin_ref) / len(t)
    B1 = 2.0 * np.dot(y, cos_ref) / len(t)
    return A1, B1


def _harmonic_magnitudes(y: np.ndarray, n_harmonics: int = 5) -> dict[int, float]:
    """Return |Hₙ| for odd harmonics 1,3,5,… up to *n_harmonics*."""
    Y = np.abs(scipy.fft.rfft(y))  # magnitude spectrum (no zero padding)
    return {k: Y[k] for k in range(1, n_harmonics + 1, 2)}


def _fit_sinusoid(t: np.ndarray, y: np.ndarray, omega: float) -> Tuple[float, float]:
    """Legacy single‑sine least‑squares fit returning amplitude & phase."""

    def sin_func(tt, A, phi):  # local to avoid *global* omega capture
        return A * np.sin(omega * tt + phi)

    A0 = (y.max() - y.min()) / 2
    phi0 = 0.0
    popt, _ = curve_fit(sin_func, t, y, p0=[A0, phi0])
    return tuple(popt)  # (A_fit, phi_fit)


# -----------------------------------------------------------------------------
# Parser Maker
# -----------------------------------------------------------------------------

@dataclass
class DmaParserMaker(Maker):
    """Parse LAMMPS DMA run and extract mechanical properties.

    Parameters
    ----------
    analysis_mode
        * "fft"  (default) → lock‑in / Fourier projection.
        * "fit"            → legacy least‑squares sinusoid fitting.
    """

    name: str = "dma_parser"
    analysis_mode: Literal["fft", "fit"] = "fft"

    # ---------------------------------------------------------------------
    # Public Job entry point
    # ---------------------------------------------------------------------

    @job(output_schema=DmaxDmaParserDocument)
    def make(
        self,
        input_doc: DmaxLammpsInputDocument,
        run_doc: DmaxLammpsRunDocument,
    ) -> Response:
        workdir = Path(input_doc.input_dir)
        log_file = next(workdir.glob("*.log"), None)
        if log_file is None:
            raise FileNotFoundError("No LAMMPS *.log found in " + str(workdir))

        # -----------------------------------------------------------------
        # 1. Parse input script for amplitude, period, axis, timestep
        # -----------------------------------------------------------------
        in_script = workdir / input_doc.input_file
        amp_pc, period, axis, dt_fs = _parse_input_script(in_script)
        dt_s = dt_fs * 1e-15
        omega_fs = 2 * np.pi / period  # rad/fs
        omega_s = omega_fs / 1e-15     # rad/s for plots/metadata

        # -----------------------------------------------------------------
        # 2. Thermo data → DataFrame (slice after "NPT Dynamic Mechanical Analysis")
        # -----------------------------------------------------------------
        thermo_df = _read_thermo_after_dma(log_file, in_script)
        t = thermo_df["time"].to_numpy()             # fs
        stress_col = f"p{axis}{axis}"
        if stress_col not in thermo_df.columns:
            raise KeyError(f"{stress_col} not found in thermo columns")
        p_raw = thermo_df[stress_col].to_numpy()
        p = p_raw - p_raw.mean()                      # de‑mean (helps orthogonality)

        # -----------------------------------------------------------------
        # 3. Analysis paths
        # -----------------------------------------------------------------
        if self.analysis_mode == "fft":
            A1, B1 = _first_harmonic_projection(t, p, omega_fs)
            fit_curve = A1 * np.sin(omega_fs * t) + B1 * np.cos(omega_fs * t)
            fit_r2 = r2_score(p, fit_curve)
            fit_rmse = np.sqrt(mean_squared_error(p, fit_curve))
            extras = {
                "harmonics": _harmonic_magnitudes(p),
                "THD": _harmonic_magnitudes(p)[3] / _harmonic_magnitudes(p)[1],
                "SNR": _harmonic_magnitudes(p)[1] / p.std(),
            }
            amp_pa = np.hypot(A1, B1)
            phase_rad = np.arctan2(B1, A1)
        else:  # "fit"
            A_fit, phi_fit = _fit_sinusoid(t, p, omega_fs)
            fit_curve = A_fit * np.sin(omega_fs * t + phi_fit)
            fit_r2 = r2_score(p, fit_curve)
            fit_rmse = np.sqrt(mean_squared_error(p, fit_curve))
            extras = {}
            amp_pa = abs(A_fit)
            phase_rad = phi_fit

        # -----------------------------------------------------------------
        # 4. Moduli (convert to **GPa**)
        # -----------------------------------------------------------------
        amp_mpa = amp_pa * 0.101325  # atm → MPa
        gamma0 = amp_pc             # engineering strain amplitude (fraction)
        G_storage = (amp_mpa / gamma0 * np.cos(phase_rad)) / 1e3  # → GPa
        G_loss    = (amp_mpa / gamma0 * np.sin(phase_rad)) / 1e3
        tan_delta = G_loss / G_storage if G_storage != 0 else None

        # -----------------------------------------------------------------
        # 5. Diagnostic plots (pressure vs time & fit)
        # -----------------------------------------------------------------
        _plot_pressure_fit(t, p, fit_curve, workdir / "dma_pressure_fit.png")

        # Stress‑strain ellipse kept unchanged – can be added here if needed.

        # -----------------------------------------------------------------
        # 6. Assemble document
        # -----------------------------------------------------------------
        doc = DmaxDmaParserDocument(
            storage_modulus=G_storage,
            loss_modulus=G_loss,
            tan_delta=tan_delta,
            phase_angle_rad=phase_rad,
            fit_r2=fit_r2,
            fit_rmse=fit_rmse,
            analysis_mode=self.analysis_mode,
            **extras,
        )
        return Response(output=doc)

# -----------------------------------------------------------------------------
# Helper functions (internal use only)
# -----------------------------------------------------------------------------

def _parse_input_script(path: Path) -> Tuple[float, int, str, float]:
    """Return (amp_fraction, period_steps, axis, dt_fs)."""
    amp_pc = period = axis = dt_fs = None
    with path.open() as fh:
        for line in fh:
            parts = line.split()
            if parts[:3] == ["variable", "oap", "equal"]:
                # e.g. 0.01*15  or 0.15
                raw = parts[-1]
                if "*" in raw:
                    amp_pc = float(raw.split("*")[0])
                else:
                    amp_pc = float(raw)
            elif parts[:3] == ["variable", "period", "equal"]:
                period = int(parts[-1])
            elif parts[:3] == ["variable", "timestep", "equal"]:
                dt_fs = float(parts[-1])
            elif parts and parts[0] == "fix" and "deform" in parts:
                idx = parts.index("deform")
                if len(parts) > idx + 2:
                    axis = parts[idx + 2]
    if None in (amp_pc, period, axis, dt_fs):
        raise RuntimeError("Failed to parse input script " + str(path))
    if amp_pc > 1:
        amp_pc /= 100.0
    return amp_pc, period, axis, dt_fs


def _read_thermo_after_dma(log_file: Path, in_script: Path) -> pd.DataFrame:
    """Return thermo DataFrame sliced after the DMA banner."""
    # first get thermo column labels from script
    with in_script.open() as fh:
        for line in fh:
            if line.strip().startswith("thermo_style"):
                parts = line.split()
                labels = parts[2:] if parts[1] == "custom" else parts[1:]
                break
        else:
            raise RuntimeError("thermo_style not found in " + str(in_script))

    lines = log_file.read_text(errors="ignore").splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if "NPT Dynamic Mechanical Analysis" in l)
    except StopIteration:
        start = 0
    data: list[list[float]] = []
    for line in lines[start:]:
        if line.strip() and line.lstrip()[0].isdigit():
            vals = line.split()
            if len(vals) == len(labels):
                data.append([float(v) for v in vals])
    return pd.DataFrame(data, columns=labels)


def _plot_pressure_fit(t: np.ndarray, p: np.ndarray, fit: np.ndarray, out: Path) -> None:
    """Scatter raw pressure (atm) and overlay fitted curve (atm)."""
    atm_to_mpa = 0.101325
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(t, -p * atm_to_mpa, s=4, alpha=0.6, label="data")
    ax.plot(t, -fit * atm_to_mpa, "r", lw=1.0, label="fit")
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("pressure (MPa)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
