# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['dynamic_voxelize_forward', 'hard_voxelize_forward'])


class _Voxelization(Function):

    @staticmethod
    def forward(
        ctx: Any,
        points: torch.Tensor,
        voxel_size: Union[tuple, float],
        coors_range: Union[tuple, float],
        max_points: int = 35,
        max_voxels: int = 20000,
        deterministic: bool = True,
        remove_outside_points: bool = True
    ) -> Union[Tuple[torch.Tensor], Tuple]:
        """Convert kitti points(N, >=3) to voxels.

        Args:
            points (torch.Tensor): [N, ndim]. Points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity.
            voxel_size (tuple or float): The size of voxel with the shape of
                [3].
            coors_range (tuple or float): The coordinate range of voxel with
                the shape of [6].
            max_points (int): Maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize. Defaults to 35.
            max_voxels (int): Maximum voxels this function create. For SECOND,
                20000 is a good choice. Users should shuffle points before call
                this function because max_voxels may drop points.
                Defaults to 20000.
            deterministic (bool): Whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
            remove_outside_points (bool): Whether to drop points out of
                coors_range. For voxel-based segmentation, we should set it to
                False. Defaults to True.

        Returns:
            tuple[torch.Tensor]: tuple[torch.Tensor]: A tuple contains three
            elements. The first one is the output voxels with the shape of
            [M, max_points, n_dim], which only contain points and returned
            when max_points != -1. The second is the voxel coordinates with
            shape of [M, 3]. The last is number of point per voxel with the
            shape of [M], which only returned when max_points != -1.
        """
        if max_points == -1 or max_voxels == -1:
            coors = points.new_zeros(size=(points.size(0), 3), dtype=torch.int)
            ext_module.dynamic_voxelize_forward(
                points,
                torch.tensor(voxel_size, dtype=torch.float),
                torch.tensor(coors_range, dtype=torch.float),
                coors,
                NDim=3,
                remove_outside_points=remove_outside_points)
            return coors
        else:
            voxels = points.new_zeros(
                size=(max_voxels, max_points, points.size(1)))
            coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
            num_points_per_voxel = points.new_zeros(
                size=(max_voxels, ), dtype=torch.int)
            voxel_num = torch.zeros(size=(), dtype=torch.long)
            ext_module.hard_voxelize_forward(
                points,
                torch.tensor(voxel_size, dtype=torch.float),
                torch.tensor(coors_range, dtype=torch.float),
                voxels,
                coors,
                num_points_per_voxel,
                voxel_num,
                max_points=max_points,
                max_voxels=max_voxels,
                NDim=3,
                deterministic=deterministic)
            # select the valid voxels
            voxels_out = voxels[:voxel_num]
            coors_out = coors[:voxel_num]
            num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
            return voxels_out, coors_out, num_points_per_voxel_out


voxelization = _Voxelization.apply


class Voxelization(nn.Module):
    """Convert kitti points(N, >=3) to voxels.

    Please refer to `Point-Voxel CNN for Efficient 3D Deep Learning
    <https://arxiv.org/abs/1907.03739>`_ for more details.

    Args:
        point_cloud_range (Sequence[float]): The coordinate range of voxel with
            the shape of [6].
        max_num_points (int): Maximum points contained in a voxel. if
            max_points=-1, it means using dynamic_voxelize.
        voxel_size (Sequence[float], optional): The size of voxel with the
            shape of [3]. Defaults to None.
        grid_size (Sequence[int], optional): The grid size of voxel with the
            shape of [3]. Usually used in voxel-based segmentation.
            Defaults to None.
        max_voxels (int): Maximum voxels this function create. For SECOND,
            20000 is a good choice. Users should shuffle points before call
            this function because max_voxels may drop points.
            Defaults to 20000.
    """

    def __init__(self,
                 point_cloud_range: Sequence[float],
                 max_num_points: int,
                 voxel_size: Optional[Sequence[float]] = None,
                 grid_size: Optional[Sequence[int]] = None,
                 max_voxels: Union[tuple, int] = 20000,
                 deterministic: bool = True) -> None:
        super().__init__()

        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)
        self.deterministic = deterministic

        assert (voxel_size is None and grid_size is not None) or (
            voxel_size is not None and grid_size is None
        ), 'voxel_size and grid_size cannot be specified simultaneously.'

        point_cloud_range = torch.tensor(
            point_cloud_range, dtype=torch.float32)
        if voxel_size:
            self.remove_outside_points = True
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
            grid_size = (
                point_cloud_range[3:] -  # type: ignore
                point_cloud_range[:3]) / voxel_size  # type: ignore
            grid_size = torch.round(grid_size).long()
        elif grid_size:
            assert self.max_num_points == -1, \
                'We only support dynamic_voxelization'
            self.remove_outside_points = False
            grid_size = torch.tensor(grid_size, dtype=torch.int32)
            voxel_size = (
                point_cloud_range[3:] -  # type: ignore
                point_cloud_range[:3]) / (grid_size - 1)  # type: ignore
        else:
            raise ValueError('Both voxel_size and grid_size are None.')
        self.voxel_size = voxel_size.tolist()  # type: ignore
        input_feat_shape = grid_size[:2]  # type: ignore
        self.grid_size = grid_size
        # the origin shape is as [x-len, y-len, z-len]
        # [w, h, d] -> [d, h, w]
        self.pcd_shape = [*input_feat_shape, 1][::-1]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        return voxelization(input, self.voxel_size, self.point_cloud_range,
                            self.max_num_points, max_voxels,
                            self.deterministic, self.remove_outside_points)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'point_cloud_range=' + str(self.point_cloud_range)
        s += ', max_num_points=' + str(self.max_num_points)
        s += ', voxel_size=' + str(self.voxel_size)
        s += ', grid_size=' + str(self.grid_size)
        s += ', max_voxels=' + str(self.max_voxels)
        s += ', deterministic=' + str(self.deterministic)
        s += ')'
        return s
