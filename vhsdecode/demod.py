from numba import njit
import numpy as np

import vhsd_rust


@njit(cache=True, nogil=True)
def replace_spikes(demod, demod_diffed, max_value, replace_start=8, replace_end=30):
    """Go through and replace spikes and some samples after them with data
    from the diff demod pass"""
    assert len(demod) == len(
        demod_diffed
    ), "diff demod length doesn't match demod length"
    too_high = max_value
    to_fix = np.where(demod > too_high)[0]

    for i in to_fix:
        start = max(i - replace_start, 0)
        end = min(i + replace_end, len(demod_diffed) - 1)
        # Only replace if it seems to help
        if max(demod_diffed[start:end]) < max(demod[start:end]):
            demod[start:end] = demod_diffed[start:end]

    return demod


@njit(cache=True, nogil=True)
def smooth_spikes(demod, max_value):
    """Go through spikes above max value and replace with the average of the neighbours."""
    too_high = max_value
    # Note - optimization, avoid first/last value so we don't have to check
    # array bounds later.
    to_fix = np.where(demod[1:-1] > too_high)[0]

    for i in to_fix:
        demod[i + 1] = (demod[i] + demod[i + 2]) / 2

    return demod


def unwrap_hilbert(hilbert, freq_hz):
    # return hilbert_test.unwrap_hilbert(hilbert, freq_hz)
    return vhsd_rust.unwrap_hilbert(hilbert, freq_hz)


@njit(cache=True, nogil=True)
def _solve_small(a, b):
    """Solve a small linear system with gaussian elimination, unsolvable
    (all zero) directions get a zero coefficient."""
    a = a.copy()
    x = b.copy()
    n = len(x)
    scale = max(np.abs(np.diag(a)).max(), 1e-300)
    for col in range(n):
        pivot = col + np.argmax(np.abs(a[col:, col]))
        if np.abs(a[pivot, col]) < 1e-12 * scale:
            a[col, :] = 0
            a[:, col] = 0
            a[col, col] = 1
            x[col] = 0
            continue
        if pivot != col:
            for j in range(n):
                a[col, j], a[pivot, j] = a[pivot, j], a[col, j]
            x[col], x[pivot] = x[pivot], x[col]
        for row in range(n):
            if row != col:
                f = a[row, col] / a[col, col]
                for j in range(n):
                    a[row, j] -= f * a[col, j]
                x[row] -= f * x[col]
    for i in range(n):
        x[i] /= a[i, i]
    return x


class ColorUnderCrosstalkCanceller:
    """Remove colour-under crosstalk from the demodulated luma signal.

    Strong colour-under chroma (saturated red/cyan/orange) leaves a phase
    modulation at the colour-under carrier and its 2nd harmonic on the FM
    luma carrier. RF filtering can't remove it since it sits on the FM
    carrier itself, and after demodulation it shows up as a diagonal mesh
    pattern in the luma (the line to line phase of the colour-under
    differs per head, so the two fields get opposite slopes).

    The crosstalk is linearly predictable from the colour-under signal, so
    each block is fitted with least squares against the colour-under signal,
    its square (2nd harmonic) and their derivatives (frequency dependent
    response), and the fit is subtracted.
    The reference only uses a narrow band around the colour-under carrier so
    it doesn't correlate with the lower luma FM sidebands. Fit and correction
    are done directly on the spectrum, limited to the bins the references
    occupy, so luma outside those bands isn't touched.
    """

    def __init__(self, color_under_hz, freq_hz, blocklen, ref_bw_hz=150e3):
        n = blocklen
        self.blocklen = n
        k0 = int(round(color_under_hz * n / freq_hz))
        half = int(np.ceil(1.7 * ref_bw_hz * n / freq_hz))
        self.ref_bins = np.arange(k0 - half, k0 + half + 1)
        offset = (self.ref_bins * freq_hz / n - color_under_hz) / ref_bw_hz
        self.ref_mask = np.exp(-(offset**8))

        # The square of the reference is computed on a small baseband grid,
        # it has to hold 4 * half bins.
        self.small_len = 1 << int(np.ceil(np.log2(8 * half)))
        self.ref_small_idx = self.ref_bins - k0
        sq_bins = np.arange(2 * k0 - 2 * half, 2 * k0 + 2 * half + 1)
        self.sq_small_idx = sq_bins - 2 * k0

        # All bins either reference touches.
        self.bins = np.arange(self.ref_bins[0], sq_bins[-1] + 1)
        self.ref_pos = self.ref_bins - self.bins[0]
        self.sq_pos = sq_bins - self.bins[0]
        self.deriv = 1j * self.bins / n

    def crosstalk_spectrum(self, demod_fft, ref_band_fft):
        """Return the estimated crosstalk as rfft bins (see self.bins) of demod.
        ref_band_fft is the fft of the raw rf data of the same block at ref_bins."""
        ref_spec = ref_band_fft * self.ref_mask

        small = np.zeros(self.small_len, dtype=np.complex128)
        small[self.ref_small_idx] = ref_spec
        zb = np.fft.ifft(small)
        sq_spec = np.fft.fft(zb * zb)[self.sq_small_idx]

        z_full = np.zeros(len(self.bins), dtype=np.complex128)
        z_full[self.ref_pos] = ref_spec
        sq_full = np.zeros(len(self.bins), dtype=np.complex128)
        sq_full[self.sq_pos] = sq_spec
        # The regressors are analytic, so a real fit in time is a complex
        # least squares fit over the positive frequency bins.
        regressors = np.stack(
            (z_full, z_full * self.deriv, sq_full, sq_full * self.deriv), axis=1
        )
        target = demod_fft[self.bins]
        # Normal equations via einsum and a small solver of our own, BLAS/LAPACK
        # calls from the decoder threads contend for the BLAS lock and slow
        # down decoding noticeably.
        gram = np.einsum("ki,kj->ij", regressors.conj(), regressors)
        rhs = np.einsum("ki,k->i", regressors.conj(), target)
        coef = _solve_small(gram, rhs)
        return z_full * (coef[0] + coef[1] * self.deriv) + sq_full * (
            coef[2] + coef[3] * self.deriv
        )
