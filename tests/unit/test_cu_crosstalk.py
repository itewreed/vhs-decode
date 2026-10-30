"""Tests for the colour-under crosstalk canceller used on the demodulated luma."""

import numpy as np

from vhsdecode.demod import ColorUnderCrosstalkCanceller

FREQ_HZ = 40e6
BLOCKLEN = 32 * 1024
PAL_VHS_CU = 625 * 25 * 40 + 1953


def _chroma_block(rng):
    t = np.arange(BLOCKLEN)
    raw = np.zeros(BLOCKLEN)
    for offset, amp, phase in ((0, 1000, 0.3), (50e3, 300, 1.0), (-80e3, 200, 2.0)):
        raw += amp * np.cos(2 * np.pi * (PAL_VHS_CU + offset) * t / FREQ_HZ + phase)
    return raw + rng.normal(0, 50, BLOCKLEN)


def _estimate(canceller, demod, raw_fft):
    spec = np.zeros(BLOCKLEN // 2 + 1, dtype=np.complex128)
    spec[canceller.bins] = canceller.crosstalk_spectrum(
        np.fft.rfft(demod), raw_fft[canceller.ref_bins]
    )
    return np.fft.irfft(spec, BLOCKLEN)


def test_removes_crosstalk_predictable_from_color_under():
    rng = np.random.default_rng(1)
    canceller = ColorUnderCrosstalkCanceller(PAL_VHS_CU, FREQ_HZ, BLOCKLEN)
    raw_fft = np.fft.fft(_chroma_block(rng))

    # Build the crosstalk the canceller models: colour-under, its derivative,
    # 2nd harmonic and its derivative, each with an arbitrary complex gain.
    ref_fft = np.zeros(BLOCKLEN, dtype=np.complex128)
    ref_fft[canceller.ref_bins] = 2 * raw_fft[canceller.ref_bins] * canceller.ref_mask
    k = np.fft.fftfreq(BLOCKLEN) * BLOCKLEN
    z = np.fft.ifft(ref_fft)
    dz = np.fft.ifft(ref_fft * 2j * np.pi * k / BLOCKLEN)
    crosstalk = (
        (0.7 - 0.2j) * z + (3 + 1j) * dz + (1e-4 + 2e-4j) * z * z
        + (5e-4 - 1e-4j) * 2 * z * dz
    ).real
    demod = 4e6 + crosstalk + rng.normal(0, 30, BLOCKLEN)

    estimate = _estimate(canceller, demod, raw_fft)
    mid = slice(2000, -2000)
    residual = (crosstalk - estimate)[mid].std()
    assert residual < 0.02 * crosstalk[mid].std()


def test_leaves_uncorrelated_signal_alone():
    rng = np.random.default_rng(2)
    canceller = ColorUnderCrosstalkCanceller(PAL_VHS_CU, FREQ_HZ, BLOCKLEN)
    raw_fft = np.fft.fft(_chroma_block(rng))
    demod = 4e6 + rng.normal(0, 100e3, BLOCKLEN)

    estimate = _estimate(canceller, demod, raw_fft)
    # Only the noise inside the fitted bands can leak into the estimate.
    assert estimate[2000:-2000].std() < 0.03 * 100e3
