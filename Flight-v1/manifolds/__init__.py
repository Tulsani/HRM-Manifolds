from __future__ import annotations

from manifolds.base import ManifoldExpert
from manifolds.euclidean import EuclideanExpert
from manifolds.hyperbolic import HyperbolicExpert
from manifolds.product import ProductManifoldExpert


def build_expert(skill: str, geometry_config: dict, backbone_dim: int) -> ManifoldExpert:
    cfg = geometry_config["skills"][skill]
    geometry = cfg["geometry"]
    n_prototypes = int(cfg.get("n_prototypes", 32))
    manifold_dim = int(cfg.get("manifold_dim", cfg.get("h_dim", 0) + cfg.get("e_dim", 0) or cfg.get("e_dim", 32)))
    if geometry == "hyperbolic":
        return HyperbolicExpert(
            skill=skill,
            input_dim=backbone_dim,
            h_dim=cfg["h_dim"],
            curvature=cfg["curvature"],
            n_prototypes=n_prototypes,
        )
    if geometry == "euclidean":
        return EuclideanExpert(
            skill=skill,
            input_dim=backbone_dim,
            e_dim=cfg["e_dim"],
        )
    if geometry == "product":
        return ProductManifoldExpert(
            skill=skill,
            input_dim=backbone_dim,
            h_dim=cfg["h_dim"],
            e_dim=cfg["e_dim"],
            curvature=cfg["curvature"],
            manifold_dim=manifold_dim,
            n_prototypes=n_prototypes,
        )
    raise ValueError(f"Unknown geometry: {geometry}")


__all__ = [
    "ManifoldExpert",
    "HyperbolicExpert",
    "EuclideanExpert",
    "ProductManifoldExpert",
    "build_expert",
]
