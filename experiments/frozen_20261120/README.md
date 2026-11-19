# Initial Mu pilot source snapshot

This is the exact initial `scripts/a_mu_adapt.py` used by `mu_adapt_v1`.
SHA256: `78f18950883ab4f654d846e1e508f5c26c4b953ddd938f4ea894891e3375081c`.
It used PyTorch GradScaler's initial value 65536. The current training core uses
512, as do `mu_native_v2` and the matched `mu_canonical_v3` controls.

A final audit found that a local change occurred after initial remote staging.
The executed original protocol was recovered from the independently verified
Azure backup and bound to the saved checkpoints by each config's protocol hash.
The unused local draft was preserved separately. Nothing was silently relabelled
as an isolated program-target comparison. Two canonical controls were rerun with
512; the paper's matched comparison is canonical_v3 versus native_v2.

This file is a historical source snapshot, not a new app or recommended entry
point. Reproducing the initial pilot requires an isolated checkout with this
snapshot restored to `scripts/a_mu_adapt.py`, the executed original protocol and
licensed training caches recovered from the private experiment archive. Keep
all original runs; the training code rejects overwriting them. Dataset audio,
reference notes, credentials and model weights are intentionally absent here.
