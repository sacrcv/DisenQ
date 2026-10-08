"""CMC and mAP for the activity-biometrics evaluation protocols.

Section 4 / Appendix E of the paper defines the protocol axes:

activity
    ``same``   -- all activities appear in both probe and gallery.
    ``cross``  -- activities in one set are absent from the other. This is a
                  property of the split, but ``cross_activity_runtime=True``
                  can also enforce it at match time by discarding gallery
                  tracklets whose activity matches the probe's.

view (NTU RGB-AB only, which records viewpoint)
    ``include`` -- every viewpoint is available in the gallery.
    ``exclude`` -- the probe's own viewpoint is removed from the gallery.

MEVID uses the standard video re-ID rule instead: drop gallery tracklets that
share both the identity and the camera with the probe. That is
``junk_rule="same_pid_same_view"``.
"""

from typing import Dict, Optional

import numpy as np

#: Gallery-filtering rules.
JUNK_RULES = ("none", "same_view", "same_pid_same_view")


def compute_cmc_map(
    distmat: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_views: Optional[np.ndarray] = None,
    gallery_views: Optional[np.ndarray] = None,
    query_actions: Optional[np.ndarray] = None,
    gallery_actions: Optional[np.ndarray] = None,
    max_rank: int = 50,
    junk_rule: str = "none",
    cross_activity_runtime: bool = False,
) -> Dict[str, object]:
    """Rank-k accuracies and mean average precision.

    Args:
        distmat: ``[num_query, num_gallery]`` distances (smaller = more similar).
        query_pids / gallery_pids: identity labels.
        query_views / gallery_views: camera or viewpoint ids, needed for any
            ``junk_rule`` other than ``"none"``.
        query_actions / gallery_actions: activity ids, needed when
            ``cross_activity_runtime`` is set.
        junk_rule: one of :data:`JUNK_RULES`.
        cross_activity_runtime: also discard same-activity gallery tracklets.

    Returns:
        ``{"cmc": array[max_rank], "mAP": float, "num_valid_query": int}``.
    """
    if junk_rule not in JUNK_RULES:
        raise ValueError(f"junk_rule must be one of {JUNK_RULES}, got '{junk_rule}'")
    if junk_rule != "none" and (query_views is None or gallery_views is None):
        raise ValueError(f"junk_rule='{junk_rule}' needs query_views and gallery_views")
    if cross_activity_runtime and (query_actions is None or gallery_actions is None):
        raise ValueError("cross_activity_runtime needs query_actions and gallery_actions")

    num_query, num_gallery = distmat.shape
    max_rank = min(max_rank, num_gallery)

    indices = np.argsort(distmat, axis=1)
    matches = (gallery_pids[indices] == query_pids[:, None]).astype(np.int32)

    all_cmc, all_ap = [], []
    num_valid_query = 0

    for q in range(num_query):
        order = indices[q]
        remove = np.zeros(num_gallery, dtype=bool)

        if junk_rule == "same_view":
            remove |= gallery_views == query_views[q]
        elif junk_rule == "same_pid_same_view":
            remove |= (gallery_pids == query_pids[q]) & (gallery_views == query_views[q])

        if cross_activity_runtime:
            remove |= gallery_actions == query_actions[q]

        keep = ~remove[order]
        row_matches = matches[q][keep]
        if not row_matches.any():
            # No ground-truth match survives the filter: this probe is skipped,
            # exactly as in the standard Market-1501 / MARS evaluation.
            continue

        cmc = row_matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        num_relevant = row_matches.sum()
        cumulative = row_matches.cumsum()
        precision = cumulative / (np.arange(len(row_matches)) + 1.0)
        all_ap.append((precision * row_matches).sum() / num_relevant)
        num_valid_query += 1

    if num_valid_query == 0:
        raise RuntimeError(
            "no valid query remained after gallery filtering; check that query and "
            "gallery share identities and that the junk_rule is appropriate"
        )

    padded = np.zeros((len(all_cmc), max_rank), dtype=np.float32)
    for i, cmc in enumerate(all_cmc):
        padded[i, : len(cmc)] = cmc
        if len(cmc) < max_rank:
            padded[i, len(cmc) :] = cmc[-1]

    return {
        "cmc": padded.mean(axis=0),
        "mAP": float(np.mean(all_ap)),
        "num_valid_query": num_valid_query,
    }


def format_results(results: Dict[str, object], ranks=(1, 5, 10, 20)) -> str:
    cmc = results["cmc"]
    parts = [f"mAP: {100 * results['mAP']:.2f}"]
    for rank in ranks:
        if rank <= len(cmc):
            parts.append(f"R@{rank}: {100 * cmc[rank - 1]:.2f}")
    parts.append(f"valid_query: {results['num_valid_query']}")
    return "  ".join(parts)


def summary_dict(results: Dict[str, object], ranks=(1, 5, 10, 20)) -> Dict[str, float]:
    cmc = results["cmc"]
    out = {"mAP": 100 * float(results["mAP"])}
    for rank in ranks:
        if rank <= len(cmc):
            out[f"R@{rank}"] = 100 * float(cmc[rank - 1])
    return out
