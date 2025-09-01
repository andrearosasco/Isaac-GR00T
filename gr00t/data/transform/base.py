# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import Any
import copy

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from gr00t.data.schema import DatasetMetadata


class ModalityTransform(BaseModel, ABC):
    """
    Abstract class for transforming data modalities, e.g. video frame augmentation or action normalization.
    """

    apply_to: list[str] = Field(..., description="The keys to apply the transform to.")
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )
    _dataset_metadata: DatasetMetadata | None = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def dataset_metadata(self) -> DatasetMetadata:
        assert (
            self._dataset_metadata is not None
        ), "Dataset metadata is not set. Please call set_metadata() before calling apply()."
        return self._dataset_metadata

    @dataset_metadata.setter
    def dataset_metadata(self, value: DatasetMetadata):
        self._dataset_metadata = value

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """
        Set the dataset metadata. This is useful for transforms that need to know the dataset metadata, e.g. to normalize actions.
        Subclasses can override this method if they need to do something more complex.
        """
        self.dataset_metadata = dataset_metadata

    # Extension hook: transforms that change dimensionality can announce new final dims.
    # Returns mapping full_key (e.g. 'action.eef_rotation_delta') -> new_dim (int).
    # Default: no overrides.
    def shape_overrides(self) -> dict[str, int]:  # pragma: no cover - trivial
        return {}

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply the transformation to the data corresponding to target_keys and return the processed data.

        Args:
            data (dict[str, Any]): The data to transform.
                example: data = {
                    "video.image_side_0": np.ndarray,
                    "action.eef_position": np.ndarray,
                    ...
                }

        Returns:
            dict[str, Any]: The transformed data.
                example: transformed_data = {
                    "video.image_side_0": np.ndarray,
                    "action.eef_position": torch.Tensor,  # Normalized and converted to tensor
                    ...
                }
        """
        return self.apply(data)

    @abstractmethod
    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply the transformation to the data corresponding to keys matching the `apply_to` regular expression and return the processed data."""

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class InvertibleModalityTransform(ModalityTransform):
    @abstractmethod
    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Reverse the transformation to the data corresponding to keys matching the `apply_to` regular expression and return the processed data."""


class ComposedModalityTransform(ModalityTransform):
    """Compose multiple modality transforms."""

    transforms: list[ModalityTransform] = Field(..., description="The transforms to compose.")
    apply_to: list[str] = Field(
        default_factory=list, description="Will be ignored for composed transforms."
    )
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        # We allow earlier transforms to declare dimension changes that later ones (e.g. concat) should see.
        meta = dataset_metadata
        meta_copy = None  # Lazy deep-copy only if an override appears.
        any_override = False
        for transform in self.transforms:
            transform.set_metadata(meta)
            overrides = transform.shape_overrides()
            if overrides:
                if meta_copy is None:
                    meta_copy = copy.deepcopy(meta)
                    meta = meta_copy
                for full_key, new_dim in overrides.items():
                    try:
                        modality, sub = full_key.split(".", 1)
                    except ValueError as e:  # safety
                        raise ValueError(f"Invalid override key '{full_key}': {e}") from e
                    modality_cfg = getattr(meta.modalities, modality, None)
                    assert modality_cfg is not None, f"Modality '{modality}' not in metadata"
                    assert sub in modality_cfg, f"Subkey '{sub}' not in modality '{modality}'"
                    modality_cfg[sub].shape = [new_dim]
                any_override = True
        # Ensure all transforms reference the final (possibly copied) metadata object
        if any_override:
            for transform in self.transforms:
                transform.dataset_metadata = meta

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for i, transform in enumerate(self.transforms):
            try:
                data = transform(data)
            except Exception as e:
                raise ValueError(f"Error applying transform {i} to data: {e}") from e
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for i, transform in enumerate(reversed(self.transforms)):
            if isinstance(transform, InvertibleModalityTransform):
                try:
                    data = transform.unapply(data)
                except Exception as e:
                    step = len(self.transforms) - i - 1
                    raise ValueError(f"Error unapplying transform {step} to data: {e}") from e
        return data

    def train(self):
        for transform in self.transforms:
            transform.train()

    def eval(self):
        for transform in self.transforms:
            transform.eval()
