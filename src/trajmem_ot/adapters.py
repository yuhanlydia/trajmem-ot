from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class EditableMemory:
    values: torch.Tensor
    mask: torch.Tensor | None
    field: str


def extract_robomme_history(batch: Mapping[str, torch.Tensor]) -> EditableMemory:
    """Select only an official RoboMME history field for optimization.

    Current images/state and instruction tokens are deliberately not accepted as
    editable values. Recurrent video memory is preferred; perceptual memory is
    the fallback supported by the currently released best checkpoint.
    """
    if batch.get("recur_image_emb") is not None:
        return EditableMemory(batch["recur_image_emb"], batch.get("recur_mask"), "recur_image_emb")
    if batch.get("static_image_emb") is not None:
        return EditableMemory(batch["static_image_emb"], batch.get("static_mask"), "static_image_emb")
    raise KeyError("RoboMME batch has neither recurrent nor perceptual history embeddings")


def replace_robomme_history(
    batch: Mapping[str, torch.Tensor], editable: EditableMemory, replacement: torch.Tensor
) -> dict[str, torch.Tensor]:
    if replacement.shape != editable.values.shape:
        raise ValueError(f"replacement shape {replacement.shape} != history shape {editable.values.shape}")
    result = dict(batch)
    result[editable.field] = replacement
    return result

