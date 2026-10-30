"""Adaptive RF equalisation based on a measurement of the playback channel.

The amplitude response of the playback path (tape, heads, preamp and capture device)
differs from tape to tape and deck to deck, while the RF filters in the format
definitions are tuned for one typical response. This module estimates the actual
response from the RF capture and derives a correction relative to the response
the fixed filters were tuned for.

Measurement: the luma is frequency modulated with a constant amplitude on tape, so
the phase of the band-limited RF gives a reference signal s(t) = cos(phase(t)) that
is what the channel input should look like. The channel response is then estimated
with the usual cross-spectral (Welch) estimator H(f) = Psx(f) / Pss(f), together
with the coherence which tells where the estimate can be trusted (color-under,
hifi carriers and the noise floor lower it).

The measured response is smoothed with a low-order polynomial fit in dB. Only the
slope (dB/MHz around the carrier) of that fit is used for the correction: an
attenuation that is symmetric around the carrier reduces both FM sidebands the same
way, which looks the same as video with less high frequency content, so the
curvature can not be measured blindly (it is underestimated considerably). The
asymmetric part shows up as amplitude modulation correlated with the instantaneous
frequency and is measured accurately. It is also the part that varies most between
tapes and decks, as the spacing loss between head and tape is a straight line in dB
over frequency.
"""

from dataclasses import dataclass

import numpy as np
import scipy.signal as sps


@dataclass
class ChannelEstimate:
    freqs: np.ndarray
    response_db: np.ndarray
    coherence: np.ndarray
    # Polynomial (np.polyval order) of the response in dB over (f - ref_freq) in MHz.
    model: np.ndarray
    fit_low: float
    fit_high: float
    ref_freq: float

    @property
    def slope(self) -> float:
        """Slope of the response in dB/MHz at ref_freq."""
        return float(self.model[-2])

    def model_db(self, freqs: np.ndarray) -> np.ndarray:
        return eval_slope_db(self.slope, freqs, self.fit_low, self.fit_high, self.ref_freq)


def eval_slope_db(slope, freqs, fit_low, fit_high, ref_freq):
    """Evaluate a straight line in dB over MHz, held constant outside the fitted range
    so it is never extrapolated past what was measured."""
    f = np.clip(np.abs(freqs), fit_low, fit_high)
    return slope * (f - ref_freq) / 1e6


def estimate_channel_response(
    rf_data: np.ndarray,
    freq_hz: float,
    fm_band: tuple,
    ref_freq: float,
    nperseg: int = 4096,
) -> tuple:
    """Estimate the channel magnitude response (dB, 0 dB at ref_freq) and the coherence."""
    x = np.asarray(rf_data, dtype=np.float64)
    x = x - np.mean(x)
    sos = sps.butter(
        6, [fm_band[0] / (freq_hz / 2), fm_band[1] / (freq_hz / 2)], "bandpass", output="sos"
    )
    y = sps.sosfiltfilt(sos, x)
    s = np.cos(np.angle(sps.hilbert(y)))
    del y

    freqs, p_ss = sps.welch(s, freq_hz, nperseg=nperseg)
    _, p_sx = sps.csd(s, x, freq_hz, nperseg=nperseg)
    _, p_xx = sps.welch(x, freq_hz, nperseg=nperseg)

    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.abs(p_sx) / p_ss
        coherence = np.abs(p_sx) ** 2 / (p_ss * p_xx)
    mag = np.nan_to_num(mag)
    coherence = np.nan_to_num(coherence)

    ref = mag[np.argmin(np.abs(freqs - ref_freq))]
    response_db = 20 * np.log10(np.maximum(mag, 1e-12) / max(ref, 1e-12))
    return freqs, response_db, coherence


def fit_channel_model(
    freqs, response_db, coherence, fit_band, ref_freq, min_coherence=0.7, order=2
):
    """Fit a polynomial in dB to the well-measured part of the response.
    Returns (model, fit_low, fit_high) or None if there is not enough usable data."""
    use = (freqs >= fit_band[0]) & (freqs <= fit_band[1]) & (coherence >= min_coherence)
    if np.count_nonzero(use) < 20:
        return None
    f_use = freqs[use]
    # Weight by the signal-to-noise ratio implied by the coherence.
    weights = np.sqrt(coherence[use] / np.maximum(1 - coherence[use], 1e-3))
    model = np.polyfit((f_use - ref_freq) / 1e6, response_db[use], order, w=weights)
    # Re-reference so the model is 0 dB at ref_freq.
    model[-1] = 0.0
    return model, float(f_use.min()), float(f_use.max())


def measure_channel(rf_data, freq_hz, params) -> ChannelEstimate:
    """Measure the channel using the "rf_eq_*" entries of the format parameters."""
    ref_freq = params["rf_eq_ref_freq"]
    freqs, response_db, coherence = estimate_channel_response(
        rf_data, freq_hz, params["rf_eq_fm_band"], ref_freq
    )
    fit = fit_channel_model(
        freqs, response_db, coherence, params["rf_eq_fit_band"], ref_freq
    )
    if fit is None:
        return None
    model, fit_low, fit_high = fit
    return ChannelEstimate(
        freqs, response_db, coherence, model, fit_low, fit_high, ref_freq
    )


def gen_correction_filter(estimate: ChannelEstimate, params, freq_hz, block_len):
    """Generate a real-valued, full (two-sided) fft-length magnitude correction:
    reference response / measured response, limited to the configured range."""
    freqs = np.fft.fftfreq(block_len, 1 / freq_hz)
    correction_db = np.clip(
        eval_slope_db(
            params["rf_eq_reference_slope"] - estimate.slope,
            freqs,
            estimate.fit_low,
            estimate.fit_high,
            estimate.ref_freq,
        ),
        -params["rf_eq_max_cut_db"],
        params["rf_eq_max_boost_db"],
    )
    return 10 ** (correction_db / 20)


def describe(estimate: ChannelEstimate, params) -> str:
    return (
        "measured channel slope %.2f dB/MHz around %.1f MHz (filters tuned for %.2f dB/MHz),"
        " curvature %.2f dB/MHz^2, fit range %.2f-%.2f MHz"
        % (
            estimate.slope,
            estimate.ref_freq / 1e6,
            params["rf_eq_reference_slope"],
            estimate.model[0],
            estimate.fit_low / 1e6,
            estimate.fit_high / 1e6,
        )
    )
