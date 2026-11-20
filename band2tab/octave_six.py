"""Pitch-blind, in-band harmonic ablation of the octave guard."""
import numpy as np

SCHEMA = 'upper_octave_six_harmonic_v1'


def summarize(x, meta):
    if not len(x):
        return np.empty((0, 35), np.float64)
    # All six partials fit the CQT range for every eligible lower note.
    # Renormalize after slicing so out-of-band high partials cannot set scale.
    x = np.asarray(x[:, :, :6], np.float64).copy()
    x /= np.maximum(1e-5, np.percentile(x, 90, axis=(1, 2)))[:, None, None]
    x = np.clip(x, 0, 8)
    mean = x.mean(1)
    std = x.std(1)
    change = x[:, 16:].mean(1) - x[:, :8].mean(1)
    z = x - x.mean(1, keepdims=True)

    def corr(a, b):
        return (z[:, :, a] * z[:, :, b]).sum(1) / np.sqrt(
            (z[:, :, a] ** 2).sum(1) * (z[:, :, b] ** 2).sum(1) + 1e-8)

    correlations = np.stack([corr(0, j) for j in range(1, 6)] +
                            [corr(1, j) for j in (3, 5)], 1)
    ratios = np.clip(np.log((mean[:, 1:] + 1e-4) / (mean[:, :1] + 1e-4)), -8, 8)
    shapes = np.stack([mean[:, 1::2].mean(1) - mean[:, ::2].mean(1),
                       mean[:, :3].mean(1) - mean[:, 3:].mean(1)], 1)
    out = np.concatenate([mean, std, change, correlations, ratios, shapes,
                          meta[:, 1:]], 1)
    if out.shape[1] != 35 or not np.isfinite(out).all():
        raise ValueError('Invalid six-harmonic summary')
    return out


def probabilities(state, x, meta):
    if state['schema'] != SCHEMA:
        raise ValueError('Unknown six-harmonic schema')
    z = np.clip((summarize(x, meta) - np.asarray(state['mean'])) /
                np.asarray(state['std']), -8, 8)
    raw = z @ np.asarray(state['coef']) + state['intercept']
    return 1 / (1 + np.exp(-np.clip(raw, -60, 60)))
