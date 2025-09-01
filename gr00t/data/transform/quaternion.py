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

import torch
from typing import Any
from pydantic import Field

from .base import ModalityTransform


class QuaternionXYZWToWXYZTransform(ModalityTransform):
    """
    Transform to convert quaternions from XYZW format (scalar last) to WXYZ format (scalar first).
    
    This is needed because some datasets provide quaternions as [qx, qy, qz, qw] but PyTorch3D 
    expects them as [qw, qx, qy, qz]. This transform should be applied before any rotation
    processing transforms (like StateActionTransform with target_rotations).
    
    Args:
        quaternion_keys (list[str]): Keys that contain quaternion data in XYZW format that need
            to be converted to WXYZ format.
    
    Example:
        >>> transform = QuaternionXYZWToWXYZTransform(
        ...     apply_to=[],  # Not used, kept for compatibility
        ...     quaternion_keys=["state.left_arm_orientation", "action.right_arm_orientation"]
        ... )
        >>> data = {"state.left_arm_orientation": torch.tensor([0.1, 0.2, 0.3, 0.9])}  # [x,y,z,w]
        >>> result = transform.apply(data)
        >>> # result["state.left_arm_orientation"] is now [0.9, 0.1, 0.2, 0.3]  # [w,x,y,z]
    """
    
    quaternion_keys: list[str] = Field(
        ..., 
        description="Keys that contain quaternion data in XYZW format that need conversion to WXYZ"
    )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.quaternion_keys:
            if key not in data:
                continue
                
            quat_data = data[key]
            assert isinstance(quat_data, torch.Tensor), f"Expected tensor, got {type(quat_data)}"
            
            # Handle different tensor shapes
            original_shape = quat_data.shape
            
            # Ensure the last dimension is 4 (quaternion components)
            assert original_shape[-1] == 4, f"Expected quaternion (4 components), got shape {original_shape}"
            
            # Convert from XYZW to WXYZ: [x,y,z,w] -> [w,x,y,z]
            if len(original_shape) == 1:
                # Single quaternion: [x,y,z,w] -> [w,x,y,z]
                wxyz_quat = torch.stack([quat_data[3], quat_data[0], quat_data[1], quat_data[2]])
            else:
                # Batch of quaternions: [..., x,y,z,w] -> [..., w,x,y,z]
                wxyz_quat = torch.stack([
                    quat_data[..., 3], quat_data[..., 0], 
                    quat_data[..., 1], quat_data[..., 2]
                ], dim=-1)
            
            data[key] = wxyz_quat
            
        return data
