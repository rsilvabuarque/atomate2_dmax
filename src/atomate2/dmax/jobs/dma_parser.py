"""dma_parser.py
================
Parser–maker for dynamic mechanical analysis (DMA) jobs
------------------------------------------------------

**Key features**
~~~~~~~~~~~~~~~~
* **Dual analysis** controlled by ``analysis_mode`` (``"fft"`` default or
  ``"fit"``).
* Calculates storage/loss moduli **G′, G″** (in MPa), tan δ, Poisson ratio,
  THD and S/N (FFT mode), or R²/RMSE (fit mode).
* Generates four PNGs per run: pressure‑time overlay, magnitude spectrum
  (FFT), stress–strain ellipse, and Poisson–time.
* Returns a **DmaxDmaParserDocument** (defined in ``task.py``) so that no other
  downstream code changes are needed.

"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Any

import numpy as np
import pandas as pd
import scipy.fft as sp_fft
import scipy.optimize as opt
import scipy.signal as sig
import matplotlib.pyplot as plt

from jobflow import Maker, Flow
from monty.json import MSONable

from task import DmaxDmaParserDocument  # <- existing schema

########################################################################################
# ---------------------------- helper math utilities --------------------------------- #
########################################################################################

def _fundamental_projection(t: np.ndarray, y: np.ndarray, omega: float) -> tuple[float, float]:
    """Return the sine (A1) and cosine (B1) coefficients of *y* at *omega*.

    A1 multiplies ``sin(ωt)`` (in‑phase with strain for shear); B1 multiplies
    ``cos(ωt)`` (90° out‑of‑phase).
    """
    sin_part = np.sin(omega * t)
    cos_part = np.cos(omega * t)
    # Orthogonal projection → ⟨y, φ(t)⟩ / ⟨φ, φ⟩ ; with uniform sampling ⟨φ,φ⟩ = N/2
    n = len(t)
    a1 = 2.0 / n * np.dot(y, sin_part)
    b1 = 2.0 / n * np.dot(y, cos_part)
    return a1, b1


def _fft_metrics(y: np.ndarray, sampling_hz: float, n_harmonics: int = 5) -> dict[str, Any]:
    """Return THD, S/N, and a magnitude dictionary of the first *n_harmonics*.

    The first element (f₁) is treated as the signal; noise power is total
    minus f₁ and its harmonics.
    """
    yf = sp_fft.rfft(y) / len(y)  # normalised complex spectrum
    mag = np.abs(yf)
    thd = mag[3] / mag[1] if mag[1] != 0 else np.nan  # f₃ / f₁
    noise_power = np.sum(mag**2) - np.sum(mag[1 : n_harmonics + 1] ** 2)
    snr = mag[1] / np.sqrt(noise_power) if noise_power > 0 else np.inf
    harmonics = {k: mag[k] for k in range(1, n_harmonics + 1)}
    return {"THD": thd, "SNR": snr, "harmonics": harmonics}


########################################################################################
# ------------------------- plotting helper functions -------------------------------- #
########################################################################################

def _plot_pressure_overlay(t: np.ndarray, p: np.ndarray, p_rec: np.ndarray, out: Path):
    plt.figure()
    plt.scatter(t, p, s=4, alpha=0.5, label="raw")
    plt.plot(t, p_rec, lw=1.6, label="fit/FFT", zorder=10)
    plt.xlabel("Time (fs)")
    plt.ylabel("Pressure (MPa)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def _plot_spectrum(y: np.ndarray, sampling_hz: float, out: Path):
    yf = sp_fft.rfft(y) / len(y)
    mag = np.abs(yf)
    freq = sp_fft.rfftfreq(len(y), d=1 / sampling_hz)
    plt.figure()
    plt.stem(freq, mag, basefmt=" ")
    plt.xlim(0, 5 * freq[1])  # show first 5 harmonics
    plt.xlabel("Frequency (THz)")
    plt.ylabel("|P(f)| (MPa)")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def _plot_stress_strain(strain: np.ndarray, stress: np.ndarray, out: Path):
    plt.figure()
    plt.plot(strain, stress, lw=1.2)
    plt.xlabel("Strain (unitless)")
    plt.ylabel("Stress (MPa)")
    plt.title("Stress–Strain Lissajous")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def _plot_poisson(time: np.ndarray, pratio: np.ndarray, out: Path):
    plt.figure()
    plt.plot(time, pratio, lw=1.2)
    plt.xlabel("Time (fs)")
    plt.ylabel("Poisson ratio")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

########################################################################################
# --------------------------- Maker implementation ----------------------------------- #
########################################################################################

class DmaParserMaker(Maker, MSONable):
    """Jobflow maker to parse a single DMA run log."""

    name: str = "dma-parser"
    analysis_mode: Literal["fft", "fit"] = "fft"
    sampling_hz: float | None = None  # computed if None from timestep
    skip_initial_cycles: int = 1      # mitigate equil‑to‑DMA transient

    def make(self, log_file: str | Path, strain_amp: float, period_fs: float) -> DmaxDmaParserDocument:  # noqa: C901
        log_file = Path(log_file)
        workdir = log_file.parent
        outdir = workdir / "plots"
        outdir.mkdir(exist_ok=True)

        ##############################################################################
        # 1) read thermo block AFTER the DMA banner
        ##############################################################################
        pressure_col = []
        lz_col = []
        step_col = []
        with open(log_file, "r", encoding="utf8") as fh:
            in_dma = False
            header = None
            for line in fh:
                if "NPT Dynamic Mechanical Analysis" in line or "NVT Dynamic Mechanical Analysis" in line:
                    in_dma = True
                    continue
                if not in_dma:
                    continue
                if "Step" in line and "Press" in line:
                    header = line.split()
                    continue
                if header and line.strip() and line[0].isdigit():
                    parts = line.split()
                    rec = dict(zip(header, parts))
                    step_col.append(int(rec["Step"]))
                    pressure_col.append(float(rec["Press"]))
                    lz_col.append(float(rec["Lz"]))
        if not pressure_col:
            raise RuntimeError("No DMA section found in log file – cannot parse")

        step = np.array(step_col)
        p_raw = np.array(pressure_col)  # already in MPa per script
        lz = np.array(lz_col)

        ##############################################################################
        # 2) time axis & detrend
        ##############################################################################
        timestep_fs = 1.0  # real units → variable timestep is typically 1 fs
        if self.sampling_hz is None:
            sampling_hz = 1_000_000.0 / timestep_fs  # convert fs -> THz (1/fs)
        else:
            sampling_hz = self.sampling_hz
        time_fs = step * timestep_fs

        # cut off first cycles to suppress transient
        samples_per_cycle = int(period_fs / timestep_fs)
        start_idx = self.skip_initial_cycles * samples_per_cycle
        time_fs = time_fs[start_idx:]
        p_raw = p_raw[start_idx:]
        lz = lz[start_idx:]

        p_detr = sig.detrend(p_raw, type="constant")

        ##############################################################################
        # 3) choose analysis path
        ##############################################################################
        omega = 2.0 * np.pi / period_fs  # rad/fs

        if self.analysis_mode == "fit":
            # legacy least‑squares single‑sine fit P(t) = A·sin(ωt)+B·cos(ωt)+C
            def _model(t, A, B, C):
                return A * np.sin(omega * t) + B * np.cos(omega * t) + C

            popt, pcov = opt.curve_fit(_model, time_fs, p_raw, p0=[1, 0, 0])
            A, B, C = popt
            p_rec = _model(time_fs, *popt)
            residual = p_raw - p_rec
            r2 = 1 - np.var(residual) / np.var(p_raw)
            rmse = float(np.sqrt(np.mean(residual**2)))
            a1, b1 = A, B  # for moduli calc
            metrics = {"fit_r2": r2, "fit_rmse": rmse, "THD": None, "SNR": None}
        else:
            # lock‑in: fundamental projection
            a1, b1 = _fundamental_projection(time_fs, p_detr, omega)
            p_rec = a1 * np.sin(omega * time_fs) + b1 * np.cos(omega * time_fs)
            # pseudo‑RMSE for info
            residual = p_detr - p_rec
            rmse = float(np.sqrt(np.mean(residual**2)))
            thd_metrics = _fft_metrics(p_detr, sampling_hz)
            metrics = {"fit_r2": np.nan, "fit_rmse": rmse, "THD": thd_metrics["THD"], "SNR": thd_metrics["SNR"]}

        ##############################################################################
        # 4) mechanical properties (MPa → return GPa later if needed)
        ##############################################################################
        g_storage = a1 / strain_amp  # MPa
        g_loss = b1 / strain_amp     # MPa
        tan_delta = g_loss / g_storage if g_storage != 0 else np.nan
        phase_rad = np.arctan2(g_loss, g_storage)
        phase_deg = np.degrees(phase_rad)
        elastic_modulus = g_storage  # same for linear small‑strain shear

        ##############################################################################
        # 5) Poisson ratio (simple amplitude ratio of box lengths)
        ##############################################################################
        lz_mean = np.mean(lz)
        strain_z = (lz - lz_mean) / lz_mean  # engineering strain in z
        poisson_ratio_series = -strain_z / strain_amp
        poisson_ratio = float(np.mean(poisson_ratio_series))

        ##############################################################################
        # 6) plots
        ##############################################################################
        pressure_plot = outdir / "pressure_overlay.png"
        _plot_pressure_overlay(time_fs, p_raw, p_rec, pressure_plot)

        ellipse_plot = outdir / "stress_strain_ellipse.png"
        strain_series = strain_amp * np.sin(omega * time_fs)
        _plot_stress_strain(strain_series, p_raw, ellipse_plot)

        poisson_plot = outdir / "poisson_vs_time.png"
        _plot_poisson(time_fs, poisson_ratio_series, poisson_plot)

        if self.analysis_mode == "fft":
            spectrum_plot = outdir / "spectrum.png"
            _plot_spectrum(p_detr, sampling_hz, spectrum_plot)
        else:
            spectrum_plot = ""

        ##############################################################################
        # 7) pack in TaskDocument
        ##############################################################################
        doc = DmaxDmaParserDocument(
            storage_modulus=g_storage,
            loss_modulus=g_loss,
            tan_delta=tan_delta,
            elastic_modulus=elastic_modulus,
            poisson_ratio=poisson_ratio,
            phase_angle_rad=phase_rad,
            phase_angle_deg=phase_deg,
            fit_r2=metrics["fit_r2"],
            fit_rmse=metrics["fit_rmse"],
            amplitude=float(np.hypot(a1, b1)),
            pressure_plot=str(pressure_plot),
            stress_strain_plot=str(ellipse_plot),
        )
        # Attach optional spectrum plot if FFT path
        if spectrum_plot:
            doc_dict = doc.dict()
            doc_dict["spectrum_plot"] = str(spectrum_plot)
            doc = DmaxDmaParserDocument(**doc_dict)

        return doc
