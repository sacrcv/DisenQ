"""Feature extraction and retrieval evaluation."""

from typing import Dict, List, Optional

import numpy as np
import torch

from disenq.evaluation.metrics import compute_cmc_map, format_results, summary_dict
from disenq.utils.dist import all_gather_tensor, get_world_size
from disenq.utils.logger import MetricLogger, get_logger
from disenq.utils.misc import unwrap_model

#: Protocol variants reported per dataset.
#: ``junk_rule`` filters the gallery; ``cross_activity`` additionally removes
#: same-activity gallery tracklets at match time.
PROTOCOLS = {
    # NTU RGB-AB: viewpoint is recorded.
    "same_activity_include_view": {"junk_rule": "none", "cross_activity": False},
    "same_activity_exclude_view": {"junk_rule": "same_view", "cross_activity": False},
    "cross_activity_include_view": {"junk_rule": "none", "cross_activity": True},
    "cross_activity_exclude_view": {"junk_rule": "same_view", "cross_activity": True},
    # Charades-AB: no viewpoint data.
    "same_activity": {"junk_rule": "none", "cross_activity": False},
    "cross_activity": {"junk_rule": "none", "cross_activity": True},
    # MEVID and other traditional video re-ID benchmarks.
    "general": {"junk_rule": "same_pid_same_view", "cross_activity": False},
}


@torch.no_grad()
def extract_split_features(
    model,
    loader,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
    print_freq: int = 50,
    header: str = "extract",
) -> Dict[str, np.ndarray]:
    """Run the inference path over a loader and gather features across ranks.

    Returns numpy arrays for ``biometrics``, ``motion``, ``non_biometrics``,
    ``pid``, ``view`` and ``action``, ordered by dataset index with any
    distributed padding removed.
    """
    net = unwrap_model(model)
    net.eval()

    chunks: Dict[str, List[torch.Tensor]] = {
        key: [] for key in
        ("biometrics", "motion", "non_biometrics", "pid", "view", "action", "index")
    }

    metric_logger = MetricLogger()
    for _, batch in metric_logger.log_every(loader, print_freq, header=header):
        frames = batch.get("frames")
        vit_features = batch.get("vit_features")
        if frames is not None:
            frames = frames.to(device, non_blocking=True)
        if vit_features is not None:
            vit_features = vit_features.to(device, non_blocking=True)
        frame_mask = batch["frame_mask"].to(device, non_blocking=True)

        autocast = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if amp_dtype is not None and device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with autocast:
            features = net.extract_features(
                frames=frames, vit_features=vit_features,
                frame_mask=frame_mask, normalize=True,
            )

        for key in ("biometrics", "motion", "non_biometrics"):
            chunks[key].append(features[key].float())
        chunks["pid"].append(batch["raw_pid"].to(device))
        chunks["view"].append(batch["view"].to(device))
        chunks["action"].append(batch["action_id"].to(device))
        chunks["index"].append(batch["index"].to(device))

    merged = {key: torch.cat(values, dim=0) for key, values in chunks.items()}

    if get_world_size() > 1:
        merged = {key: all_gather_tensor(value) for key, value in merged.items()}

    # Undo the InferenceSampler padding and restore dataset order.
    index = merged.pop("index").cpu().numpy()
    _, unique_positions = np.unique(index, return_index=True)
    order = unique_positions[np.argsort(index[unique_positions])]

    out = {}
    for key, value in merged.items():
        array = value.cpu().numpy()
        out[key] = array[order]
    return out


def compute_distance_matrix(
    model,
    query: Dict[str, np.ndarray],
    gallery: Dict[str, np.ndarray],
    device: torch.device,
    use_motion: bool = True,
    adaptive: bool = True,
    fixed_alpha: float = 0.5,
    chunk_size: int = 512,
) -> np.ndarray:
    """Eq. 8 similarity, returned as a distance matrix (``1 - similarity``)."""
    net = unwrap_model(model)
    query_t = {
        key: torch.from_numpy(query[key]).to(device)
        for key in ("biometrics", "motion")
    }
    gallery_t = {
        key: torch.from_numpy(gallery[key]).to(device)
        for key in ("biometrics", "motion")
    }
    with torch.no_grad():
        similarity = net.similarity_matrix(
            query_t, gallery_t,
            use_motion=use_motion, adaptive=adaptive,
            fixed_alpha=fixed_alpha, chunk_size=chunk_size,
        )
    return (1.0 - similarity).cpu().numpy()


def evaluate(
    model,
    query_loader,
    gallery_loader,
    device: torch.device,
    protocols: List[str],
    amp_dtype: Optional[torch.dtype] = None,
    use_motion: bool = True,
    adaptive: bool = True,
    fixed_alpha: float = 0.5,
    max_rank: int = 50,
    print_freq: int = 50,
    feature: str = "fused",
) -> Dict[str, Dict[str, float]]:
    """Extract both splits and report every requested protocol.

    Args:
        protocols: keys of :data:`PROTOCOLS`.
        feature: ``fused`` (Eq. 8), ``biometrics``, ``motion`` or
            ``non_biometrics`` -- the latter three reproduce the per-feature
            ablation in Table 3 (bottom).
    """
    logger = get_logger()
    unknown = [p for p in protocols if p not in PROTOCOLS]
    if unknown:
        raise ValueError(f"unknown protocol(s) {unknown}; available: {sorted(PROTOCOLS)}")

    query = extract_split_features(
        model, query_loader, device, amp_dtype, print_freq, header="query  "
    )
    gallery = extract_split_features(
        model, gallery_loader, device, amp_dtype, print_freq, header="gallery"
    )
    logger.info(
        f"extracted features: query={query['biometrics'].shape}, "
        f"gallery={gallery['biometrics'].shape}"
    )

    if feature == "fused":
        distmat = compute_distance_matrix(
            model, query, gallery, device,
            use_motion=use_motion, adaptive=adaptive, fixed_alpha=fixed_alpha,
        )
    elif feature in ("biometrics", "motion", "non_biometrics"):
        distmat = 1.0 - query[feature] @ gallery[feature].T
    else:
        raise ValueError(
            f"feature must be 'fused', 'biometrics', 'motion' or 'non_biometrics', "
            f"got '{feature}'"
        )

    results: Dict[str, Dict[str, float]] = {}
    for name in protocols:
        spec = PROTOCOLS[name]
        raw = compute_cmc_map(
            distmat=distmat,
            query_pids=query["pid"],
            gallery_pids=gallery["pid"],
            query_views=query["view"],
            gallery_views=gallery["view"],
            query_actions=query["action"],
            gallery_actions=gallery["action"],
            max_rank=max_rank,
            junk_rule=spec["junk_rule"],
            cross_activity_runtime=spec["cross_activity"],
        )
        logger.info(f"[{feature}] {name:>30}  {format_results(raw)}")
        results[name] = summary_dict(raw)
    return results
