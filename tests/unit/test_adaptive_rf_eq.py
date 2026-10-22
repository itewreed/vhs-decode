import numpy as np
import scipy.signal as sps

import vhsdecode.process as process
from vhsdecode.addons import adaptive_rf_eq

FREQ_HZ = 40e6


def gen_fm_signal(length, seed=0):
    """PAL VHS-like FM signal: 3.8-4.8 MHz carrier, modulated by band limited noise."""
    rng = np.random.default_rng(seed)
    video = rng.standard_normal(length)
    video = sps.sosfilt(sps.butter(4, 3e6 / (FREQ_HZ / 2), output="sos"), video)
    video = 0.5 + 0.5 * np.clip(video / (3 * np.std(video)), -1, 1)
    inst_freq = 3.8e6 + video * 1.0e6
    phase = 2 * np.pi * np.cumsum(inst_freq) / FREQ_HZ
    return np.cos(phase)


def apply_channel_db(signal, model):
    """Pass a signal through a channel with a zero-phase magnitude response given as
    polynomial in dB over MHz from 4.3 MHz."""
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1 / FREQ_HZ)
    gain_db = np.polyval(model, (freqs - 4.3e6) / 1e6)
    return np.fft.irfft(spectrum * 10 ** (gain_db / 20), len(signal))


def get_params():
    return process.VHSRFDecode(inputfreq=40, system="PAL").DecoderParams


def test_estimate_recovers_channel():
    params = get_params()
    channel = np.array([-0.8, -3.5, 0.0])
    rf = apply_channel_db(gen_fm_signal(1 << 20), channel)
    rf += np.random.default_rng(1).standard_normal(len(rf)) * 0.01

    estimate = adaptive_rf_eq.measure_channel(rf, FREQ_HZ, params)
    assert estimate is not None

    # Only the slope can be measured reliably (see module docstring).
    assert abs(estimate.slope - channel[-2]) < 0.1


def test_correction_follows_slope_difference():
    """A channel that loses 1 dB/MHz more than the reference should get that back,
    regardless of the curvature."""
    params = get_params()
    signal = gen_fm_signal(1 << 20)
    reference = adaptive_rf_eq.measure_channel(
        apply_channel_db(signal, np.array([-0.6, -2.5, 0.0])), FREQ_HZ, params
    )
    params = {**params, "rf_eq_reference_slope": reference.slope}
    estimate = adaptive_rf_eq.measure_channel(
        apply_channel_db(signal, np.array([-1.2, -3.5, 0.0])), FREQ_HZ, params
    )
    correction = adaptive_rf_eq.gen_correction_filter(estimate, params, FREQ_HZ, 32768)
    freqs = np.fft.fftfreq(32768, 1 / FREQ_HZ)
    band = (freqs > 2.5e6) & (freqs < 5.5e6)
    np.testing.assert_allclose(
        20 * np.log10(correction[band]), (freqs[band] - 4.3e6) / 1e6, atol=0.2
    )


def test_rf_decode_applies_correction():
    decoder = process.VHSRFDecode(inputfreq=40, system="PAL")
    fixed = decoder.Filters["RFVideo"].copy()
    rf = apply_channel_db(gen_fm_signal(1 << 20), np.array([-0.6, -4.5, 0.0]))
    assert decoder.adapt_rf_eq(rf) is not None
    freqs = np.fft.fftfreq(decoder.blocklen, 1 / FREQ_HZ)
    upper = np.argmin(np.abs(freqs - 5.5e6))
    lower = np.argmin(np.abs(freqs - 3.0e6))
    # Steeper loss than the reference: the upper side should get boosted relative to the lower.
    gain = decoder.Filters["RFVideo"] / np.where(fixed > 0, fixed, 1)
    assert gain[upper] / gain[lower] > 10 ** (2 / 20)
