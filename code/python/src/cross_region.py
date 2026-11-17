"""S-B: do neuron ensembles stay inside a region / probe, or span the brain? Awake vs isoflurane.

Per session and state: every well-isolated grey-matter unit (>= 0.5 Hz awake), rates on the session's state
seconds (from locality.run_session). Ensembles from spiking alone, two ways (both allow unassigned neurons):
correlation communities (population mean removed; ensembles.corr_membership) and getClust.
Per ensemble: purity = share of members in its most common region (and probe); span = number of regions.
Null: region (probe) labels permuted across the session's units, ensemble memberships kept (1,000 permutations);
excess purity = observed - null mean. Positive = ensembles are more anatomically confined than chance.
"""
import numpy as np
import pandas as pd

from . import ensembles as en
from . import locality as lc


def purity(labels_of_members):
    _, c = np.unique(labels_of_members, return_counts=True)
    return c.max() / c.sum(), len(c)


def ensemble_rows(lab, region, probe, rng, n_perm=1000):
    rows = []
    groups = [np.flatnonzero(lab == k) for k in np.unique(lab[lab > 0])]
    groups = [g for g in groups if g.size >= 3]
    for key, labs in (("region", region), ("probe", probe)):
        real = np.array([purity(labs[g])[0] for g in groups]) if groups else np.array([])
        null = np.array([[purity(p[g])[0] for g in groups] for p in (rng.permutation(labs) for _ in range(n_perm))])
        for i, g in enumerate(groups):
            rows.append({"ensemble": i + 1, "size": g.size, "by": key, "purity": real[i], "null_purity": null[:, i].mean(),
                         "p": float((1 + (null[:, i] >= real[i]).sum()) / (n_perm + 1)),
                         "n_distinct": purity(labs[g])[1]})
    return rows


def run_session(session, state, rng):
    S = lc.run_session(session, np.ones(10) / np.sqrt(10), state)    # the LFP side is not used here
    if S is None:
        return pd.DataFrame()
    U = S["units"][S["keep"]].reset_index(drop=True)
    R = S["rate"][:, S["keep"]]
    region, probe = U.region.to_numpy(), U.probe.to_numpy()
    out = []
    lab, _, _ = en.corr_membership(R, rng)
    out += [dict(r, method="corr") for r in ensemble_rows(lab, region, probe, rng)]
    lab = en.getclust_membership(R)
    out += [dict(r, method="getclust") for r in ensemble_rows(lab, region, probe, rng)]
    return pd.DataFrame(out).assign(session=session, state="awake" if state == 0 else "isoflurane",
                                    n_units=len(U), n_regions=len(set(region)), n_probes=len(set(probe)))
