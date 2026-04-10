from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # pragma: no cover - optional dependency
    GaussianHMM = None

try:
    import hdbscan as hdbscan_pkg
except ImportError:  # pragma: no cover - optional dependency
    hdbscan_pkg = None

try:
    from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
except ImportError:  # pragma: no cover - optional dependency
    SklearnHDBSCAN = None

try:
    import ruptures as rpt
except ImportError:  # pragma: no cover - optional dependency
    rpt = None


ArrayLike = np.ndarray


@dataclass
class RegimeDetectionResult:
    name: str
    model: Any
    labels: ArrayLike
    X_raw: ArrayLike
    X_scaled: ArrayLike
    scaler: StandardScaler
    feature_names: List[str]
    probs: Optional[ArrayLike] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def single_snapshot_features(snap: Mapping[str, ArrayLike], top_k: int = 5) -> ArrayLike:
    best_bid = float(snap["bid_prices"][0])
    best_ask = float(snap["ask_prices"][0])
    mid_price = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid

    bid_vol = np.asarray(snap["bid_vol"], dtype=float)
    ask_vol = np.asarray(snap["ask_vol"], dtype=float)
    top_k = max(1, min(top_k, len(bid_vol), len(ask_vol)))

    total_bid_vol = float(np.sum(bid_vol))
    total_ask_vol = float(np.sum(ask_vol))
    top_bid_vol = float(np.sum(bid_vol[:top_k]))
    top_ask_vol = float(np.sum(ask_vol[:top_k]))

    imbalance = (top_bid_vol - top_ask_vol) / (top_bid_vol + top_ask_vol + 1e-8)
    book_pressure = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-8)

    return np.array(
        [
            mid_price,
            spread,
            total_bid_vol,
            total_ask_vol,
            top_bid_vol,
            top_ask_vol,
            imbalance,
            book_pressure,
        ],
        dtype=float,
    )


def regime_features(book: Sequence[Mapping[str, ArrayLike]], top_k: int = 5) -> ArrayLike:
    return np.vstack([single_snapshot_features(snap, top_k=top_k) for snap in book])


def _feature_names() -> List[str]:
    return [
        "mid_price",
        "spread",
        "total_bid_vol",
        "total_ask_vol",
        "top_bid_vol",
        "top_ask_vol",
        "imbalance",
        "book_pressure",
    ]


def _scale_features(X: ArrayLike) -> Tuple[ArrayLike, StandardScaler]:
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def _compute_centroids(X_scaled: ArrayLike, labels: ArrayLike) -> Dict[int, ArrayLike]:
    centroids: Dict[int, ArrayLike] = {}
    for label in sorted(set(int(x) for x in np.unique(labels))):
        mask = labels == label
        if np.any(mask):
            centroids[label] = X_scaled[mask].mean(axis=0)
    return centroids


def train_kmeans_regime_model(
    book: Sequence[Mapping[str, ArrayLike]],
    n_clusters: int = 3,
    top_k: int = 5,
    random_state: int = 0,
) -> RegimeDetectionResult:
    X = regime_features(book, top_k=top_k)
    X_scaled, scaler = _scale_features(X)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_scaled)

    return RegimeDetectionResult(
        name="kmeans",
        model=model,
        labels=labels,
        X_raw=X,
        X_scaled=X_scaled,
        scaler=scaler,
        feature_names=_feature_names(),
        metadata={"centroids": _compute_centroids(X_scaled, labels)},
    )


def train_gmm_regime_model(
    book: Sequence[Mapping[str, ArrayLike]],
    n_components: int = 3,
    top_k: int = 5,
    covariance_type: str = "full",
    random_state: int = 0,
) -> RegimeDetectionResult:
    X = regime_features(book, top_k=top_k)
    X_scaled, scaler = _scale_features(X)

    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    model.fit(X_scaled)
    labels = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)

    return RegimeDetectionResult(
        name="gmm",
        model=model,
        labels=labels,
        X_raw=X,
        X_scaled=X_scaled,
        scaler=scaler,
        feature_names=_feature_names(),
        probs=probs,
        metadata={"centroids": _compute_centroids(X_scaled, labels)},
    )


def train_hmm_regime_model(
    book: Sequence[Mapping[str, ArrayLike]],
    n_components: int = 3,
    top_k: int = 5,
    covariance_type: str = "diag",
    random_state: int = 0,
    n_iter: int = 200,
) -> RegimeDetectionResult:
    if GaussianHMM is None:
        raise ImportError("hmmlearn is required for HMM regime detection.")

    X = regime_features(book, top_k=top_k)
    X_scaled, scaler = _scale_features(X)

    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        n_iter=n_iter,
    )
    model.fit(X_scaled)
    labels = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)

    return RegimeDetectionResult(
        name="hmm",
        model=model,
        labels=labels,
        X_raw=X,
        X_scaled=X_scaled,
        scaler=scaler,
        feature_names=_feature_names(),
        probs=probs,
        metadata={"centroids": _compute_centroids(X_scaled, labels)},
    )


def train_hdbscan_regime_model(
    book: Sequence[Mapping[str, ArrayLike]],
    top_k: int = 5,
    min_cluster_size: int = 50,
    min_samples: Optional[int] = None,
) -> RegimeDetectionResult:
    X = regime_features(book, top_k=top_k)
    X_scaled, scaler = _scale_features(X)

    if hdbscan_pkg is not None:
        model = hdbscan_pkg.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=True,
        )
    elif SklearnHDBSCAN is not None:
        model = SklearnHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            allow_single_cluster=True,
        )
    else:
        raise ImportError("HDBSCAN requires either the `hdbscan` package or sklearn with HDBSCAN support.")

    labels = model.fit_predict(X_scaled)
    probs = getattr(model, "probabilities_", None)

    metadata = {"centroids": _compute_centroids(X_scaled, labels)}
    if hdbscan_pkg is not None and isinstance(model, hdbscan_pkg.HDBSCAN):
        metadata["approx_predict"] = hdbscan_pkg.approximate_predict

    return RegimeDetectionResult(
        name="hdbscan",
        model=model,
        labels=labels,
        X_raw=X,
        X_scaled=X_scaled,
        scaler=scaler,
        feature_names=_feature_names(),
        probs=probs,
        metadata=metadata,
    )


def train_change_point_regime_model(
    book: Sequence[Mapping[str, ArrayLike]],
    top_k: int = 5,
    n_bkps: int = 4,
    min_segment_length: int = 50,
) -> RegimeDetectionResult:
    X = regime_features(book, top_k=top_k)
    X_scaled, scaler = _scale_features(X)

    if rpt is not None:
        signal = PCA(n_components=min(3, X_scaled.shape[1])).fit_transform(X_scaled)
        algo = rpt.Pelt(model="rbf", min_size=min_segment_length).fit(signal)
        change_points = algo.predict(pen=min_segment_length)
    else:
        change_points = _cusum_breakpoints(X_scaled, n_bkps=n_bkps, min_segment_length=min_segment_length)

    labels = np.zeros(len(X_scaled), dtype=int)
    start = 0
    for segment_id, end in enumerate(change_points):
        end = min(end, len(labels))
        labels[start:end] = segment_id
        start = end
    if start < len(labels):
        labels[start:] = int(labels[start - 1] + 1) if start > 0 else 0

    return RegimeDetectionResult(
        name="change_point",
        model=None,
        labels=labels,
        X_raw=X,
        X_scaled=X_scaled,
        scaler=scaler,
        feature_names=_feature_names(),
        metadata={
            "change_points": change_points,
            "centroids": _compute_centroids(X_scaled, labels),
        },
    )


def _cusum_breakpoints(
    X_scaled: ArrayLike,
    n_bkps: int = 4,
    min_segment_length: int = 50,
) -> List[int]:
    if len(X_scaled) <= min_segment_length:
        return [len(X_scaled)]

    signal = PCA(n_components=1).fit_transform(X_scaled).ravel()
    diffs = np.abs(np.diff(signal, prepend=signal[0]))
    candidate_idx = np.argsort(diffs)[::-1]

    breakpoints: List[int] = []
    for idx in candidate_idx:
        if idx < min_segment_length or idx > len(signal) - min_segment_length:
            continue
        if any(abs(idx - bp) < min_segment_length for bp in breakpoints):
            continue
        breakpoints.append(int(idx))
        if len(breakpoints) >= n_bkps:
            break

    breakpoints = sorted(breakpoints)
    breakpoints.append(len(signal))
    return breakpoints


def _transform_features(
    trained: RegimeDetectionResult,
    snap_or_features: Mapping[str, ArrayLike] | Sequence[float] | ArrayLike,
) -> ArrayLike:
    if isinstance(snap_or_features, Mapping):
        features = single_snapshot_features(snap_or_features)
    else:
        features = np.asarray(snap_or_features, dtype=float)
    return trained.scaler.transform(np.atleast_2d(features))


def predict_regime(
    trained: RegimeDetectionResult,
    snap_or_features: Mapping[str, ArrayLike] | Sequence[float] | ArrayLike,
) -> int:
    X_scaled = _transform_features(trained, snap_or_features)

    if trained.name in {"kmeans", "gmm", "hmm"}:
        return int(trained.model.predict(X_scaled)[0])

    if trained.name == "hdbscan":
        approx_predict = trained.metadata.get("approx_predict")
        if approx_predict is not None:
            labels, _ = approx_predict(trained.model, X_scaled)
            return int(labels[0])

    centroids = trained.metadata.get("centroids", {})
    if not centroids:
        raise ValueError(f"No prediction rule available for regime model `{trained.name}`.")

    labels = list(centroids.keys())
    centroid_matrix = np.vstack([centroids[label] for label in labels])
    distances = np.linalg.norm(centroid_matrix - X_scaled[0], axis=1)
    return int(labels[int(np.argmin(distances))])


def regime_probabilities(
    trained: RegimeDetectionResult,
    snap_or_features: Mapping[str, ArrayLike] | Sequence[float] | ArrayLike,
) -> Optional[ArrayLike]:
    X_scaled = _transform_features(trained, snap_or_features)

    if trained.name in {"gmm", "hmm"}:
        return trained.model.predict_proba(X_scaled)[0]

    if trained.name == "kmeans":
        centers = trained.model.cluster_centers_
        distances = np.linalg.norm(centers - X_scaled[0], axis=1)
        inv = 1.0 / (distances + 1e-8)
        return inv / inv.sum()

    return None


def get_mid_price(snap: Mapping[str, ArrayLike]) -> float:
    return float((snap["bid_prices"][0] + snap["ask_prices"][0]) / 2.0)


def distance_from_mid(price: float, mid: float) -> float:
    return float(abs(price - mid))


def get_bucket(value: float, buckets: Sequence[float]) -> int:
    idx = np.digitize([value], buckets, right=False)[0] - 1
    return int(np.clip(idx, 0, len(buckets) - 2))


def build_regime_surfaces(
    book: Sequence[Mapping[str, ArrayLike]],
    regimes: Sequence[int],
    horizon: int,
    depth: int,
    tau: int,
    dist_buckets: Sequence[float],
    vol_buckets: Sequence[float],
) -> Tuple[Dict[int, ArrayLike], Dict[int, ArrayLike]]:
    regimes = np.asarray(regimes)
    bid_surfaces: Dict[int, ArrayLike] = {}
    ask_surfaces: Dict[int, ArrayLike] = {}

    for regime in sorted(set(int(x) for x in np.unique(regimes))):
        counts_bid = np.zeros((len(dist_buckets) - 1, len(vol_buckets) - 1))
        counts_ask = np.zeros((len(dist_buckets) - 1, len(vol_buckets) - 1))
        fills_bid = np.zeros_like(counts_bid)
        fills_ask = np.zeros_like(counts_ask)

        idx = np.where(regimes == regime)[0]
        for t in idx:
            if t + horizon >= len(book):
                continue

            snap = book[t]
            future = book[t + horizon]
            mid = get_mid_price(snap)
            max_depth = min(depth, len(snap["bid_prices"]), len(snap["ask_prices"]))

            for level in range(max_depth):
                bid_price = float(snap["bid_prices"][level])
                bid_vol = float(snap["bid_vol"][level])
                ask_price = float(snap["ask_prices"][level])
                ask_vol = float(snap["ask_vol"][level])

                bid_dist = distance_from_mid(bid_price, mid)
                ask_dist = distance_from_mid(ask_price, mid)
                d_bin_bid = get_bucket(bid_dist, dist_buckets)
                d_bin_ask = get_bucket(ask_dist, dist_buckets)
                v_bin_bid = get_bucket(bid_vol, vol_buckets)
                v_bin_ask = get_bucket(ask_vol, vol_buckets)

                counts_bid[d_bin_bid, v_bin_bid] += 1
                counts_ask[d_bin_ask, v_bin_ask] += 1

                future_asks = np.asarray(future["ask_prices"][:tau], dtype=float)
                future_bids = np.asarray(future["bid_prices"][:tau], dtype=float)

                if np.any(future_asks <= bid_price):
                    fills_bid[d_bin_bid, v_bin_bid] += 1
                if np.any(future_bids >= ask_price):
                    fills_ask[d_bin_ask, v_bin_ask] += 1

        bid_surfaces[regime] = fills_bid / (counts_bid + 1e-8)
        ask_surfaces[regime] = fills_ask / (counts_ask + 1e-8)

    return bid_surfaces, ask_surfaces


def plot_regime_surface(
    surface: ArrayLike,
    regime: int,
    dist_buckets: Sequence[float],
    vol_buckets: Sequence[float],
    side: str = "Bid",
    cmap: str = "viridis",
) -> None:
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        surface,
        cmap=cmap,
        xticklabels=np.round(vol_buckets[:-1], 4),
        yticklabels=np.round(dist_buckets[:-1], 4),
    )
    plt.title(f"{side} Fill Probability Surface - Regime {regime}")
    plt.xlabel("Volume Bucket")
    plt.ylabel("Distance Bucket")
    plt.tight_layout()
    plt.show()


def plot_regime_surfaces(
    bid_surfaces: Mapping[int, ArrayLike],
    ask_surfaces: Mapping[int, ArrayLike],
    dist_buckets: Sequence[float],
    vol_buckets: Sequence[float],
) -> None:
    for regime, surface in bid_surfaces.items():
        plot_regime_surface(surface, regime, dist_buckets, vol_buckets, side="Bid")
    for regime, surface in ask_surfaces.items():
        plot_regime_surface(surface, regime, dist_buckets, vol_buckets, side="Ask")


def plot_regime_projection(
    trained: RegimeDetectionResult,
    title: Optional[str] = None,
    annotate_centroids: bool = True,
) -> None:
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(trained.X_scaled)

    plt.figure(figsize=(10, 7))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=trained.labels, cmap="viridis", s=35, edgecolor="k")

    centroids = trained.metadata.get("centroids", {})
    if annotate_centroids and centroids:
        labels = list(centroids.keys())
        projected = pca.transform(np.vstack([centroids[label] for label in labels]))
        plt.scatter(projected[:, 0], projected[:, 1], c="red", marker="X", s=180, label="Centroids")
        for idx, label in enumerate(labels):
            plt.text(projected[idx, 0], projected[idx, 1], f"R{label}", ha="center", va="center", weight="bold")

    plt.title(title or f"{trained.name.upper()} Regime Projection")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def query_fill_probability(
    price: float,
    volume: float,
    mid: float,
    surfaces: Mapping[int, ArrayLike],
    regime: int,
    dist_buckets: Sequence[float],
    vol_buckets: Sequence[float],
) -> float:
    if regime not in surfaces:
        return 0.0

    dist = distance_from_mid(price, mid)
    d_bin = get_bucket(dist, dist_buckets)
    v_bin = get_bucket(volume, vol_buckets)
    return float(surfaces[regime][d_bin, v_bin])


def estimate_queue_position(snap: Mapping[str, ArrayLike], side: str = "bid", level: int = 0) -> float:
    if side == "bid":
        return float(snap["bid_vol"][level])
    return float(snap["ask_vol"][level])


def p_queue(queue_pos: float, traded_volume: float) -> float:
    if queue_pos <= 0:
        return 1.0
    return float(min(traded_volume / queue_pos, 1.0))


def estimate_traded_volume(
    book: Sequence[Mapping[str, ArrayLike]],
    t: int,
    horizon: int,
    scale: float = 100.0,
) -> float:
    if t + horizon >= len(book):
        return 0.0
    mid_now = get_mid_price(book[t])
    mid_future = get_mid_price(book[t + horizon])
    return float(abs(mid_future - mid_now) * scale)


def compute_p_queue(
    book: Sequence[Mapping[str, ArrayLike]],
    t: int,
    side: str = "bid",
    level: int = 0,
    horizon: int = 1,
    trade_scale: float = 100.0,
) -> float:
    snap = book[t]
    queue_pos = estimate_queue_position(snap, side=side, level=level)
    traded_vol = estimate_traded_volume(book, t, horizon=horizon, scale=trade_scale)
    return p_queue(queue_pos, traded_vol)


def get_bid_ask_probabilities(
    book: Sequence[Mapping[str, ArrayLike]],
    trained: RegimeDetectionResult,
    bid_surfaces: Mapping[int, ArrayLike],
    ask_surfaces: Mapping[int, ArrayLike],
    dist_buckets: Sequence[float],
    vol_buckets: Sequence[float],
    levels: Iterable[int] = (0, 2, 5, 10),
    start: int = 500,
    allowed_regimes: Optional[Iterable[int]] = None,
) -> Dict[int, Dict[str, List[float]]]:
    level_set = tuple(levels)
    allowed = set(allowed_regimes) if allowed_regimes is not None else None

    out: Dict[int, Dict[str, List[float]]] = {
        level: {"bid": [], "ask": [], "regime": []} for level in level_set
    }

    for t in range(start, len(book)):
        snap = book[t]
        mid = get_mid_price(snap)
        regime = predict_regime(trained, snap)

        for level in level_set:
            if level >= len(snap["bid_prices"]) or level >= len(snap["ask_prices"]):
                continue

            if allowed is not None and regime not in allowed:
                out[level]["bid"].append(0.0)
                out[level]["ask"].append(0.0)
                out[level]["regime"].append(regime)
                continue

            bid_price = float(snap["bid_prices"][level])
            bid_vol = float(snap["bid_vol"][level])
            ask_price = float(snap["ask_prices"][level])
            ask_vol = float(snap["ask_vol"][level])

            out[level]["bid"].append(
                query_fill_probability(
                    bid_price,
                    bid_vol,
                    mid,
                    bid_surfaces,
                    regime,
                    dist_buckets,
                    vol_buckets,
                )
            )
            out[level]["ask"].append(
                query_fill_probability(
                    ask_price,
                    ask_vol,
                    mid,
                    ask_surfaces,
                    regime,
                    dist_buckets,
                    vol_buckets,
                )
            )
            out[level]["regime"].append(regime)

    return out


__all__ = [
    "RegimeDetectionResult",
    "build_regime_surfaces",
    "compute_p_queue",
    "distance_from_mid",
    "estimate_queue_position",
    "estimate_traded_volume",
    "get_bid_ask_probabilities",
    "get_bucket",
    "get_mid_price",
    "p_queue",
    "plot_regime_projection",
    "plot_regime_surface",
    "plot_regime_surfaces",
    "predict_regime",
    "query_fill_probability",
    "regime_features",
    "regime_probabilities",
    "single_snapshot_features",
    "train_change_point_regime_model",
    "train_gmm_regime_model",
    "train_hdbscan_regime_model",
    "train_hmm_regime_model",
    "train_kmeans_regime_model",
]
