"""Affine augmentation.

https://github.com/Project-MONAI/MONAI/blob/f6f9e817892090f6c1647f60f51fbc5ed80b74f1/monai/transforms/utils.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
import numpy as np


def get_2d_rotation_matrix(
    rotations: torch.Tensor,
) -> torch.Tensor:
    """Return 2d rotation matrix given radians.

    The affine transformation applies as following:
        [x, = [[* * 0]  * [x,
         y,    [* * 0]     y,
         1]    [0 0 1]]    1]

    Args:
        rotations: tuple of one values, correspond to xy planes.

    Returns:
        Rotation matrix of shape (3, 3).
    """
    sin, cos = torch.sin(rotations[0]), torch.cos(rotations[0])
    return torch.tensor(
        [
            [cos, -sin, 0.0],
            [sin, cos, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=rotations.dtype,
        device=rotations.device,
    )


def get_3d_rotation_matrix(
    rotations: torch.Tensor,
) -> torch.Tensor:
    """Return 3d rotation matrix given radians.

    The affine transformation applies as following:
        [x, = [[* * * 0]  * [x,
         y,    [* * * 0]     y,
         z,    [* * * 0]     z,
         1]    [0 0 0 1]]    1]

    Args:
        rotations: tuple of three values, correspond to yz, xz, xy planes.

    Returns:
        Rotation matrix of shape (4, 4).
    """
    affine = torch.eye(4, dtype=rotations.dtype, device=rotations.device)

    # rotation of yz around x-axis
    sin, cos = torch.sin(rotations[0]), torch.cos(rotations[0])
    affine_ax = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos, -sin, 0.0],
            [0.0, sin, cos, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=rotations.dtype,
        device=rotations.device,
    )
    affine = torch.matmul(affine_ax, affine)

    # rotation of zx around y-axis
    sin, cos = torch.sin(rotations[1]), torch.cos(rotations[1])
    affine_ax = torch.tensor(
        [
            [cos, 0.0, sin, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin, 0.0, cos, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=rotations.dtype,
        device=rotations.device,
    )
    affine = torch.matmul(affine_ax, affine)

    # rotation of xy around z-axis
    sin, cos = torch.sin(rotations[2]), torch.cos(rotations[2])
    affine_ax = torch.tensor(
        [
            [cos, -sin, 0.0, 0.0],
            [sin, cos, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=rotations.dtype,
        device=rotations.device,
    )
    affine = torch.matmul(affine_ax, affine)

    return affine


def get_rotation_matrix(
    rotations: torch.Tensor,
) -> torch.Tensor:
    """Return rotation matrix given radians.

    Rotation is anti-clockwise.

    Args:
        rotations: correspond to rotate around each axis.

    Returns:
        Rotation matrix of shape (n+1, n+1).

    Raises:
        ValueError: if not 2D or 3D.
    """
    if rotations.shape == (1,):
        return get_2d_rotation_matrix(rotations)
    if rotations.shape == (3,):
        return get_3d_rotation_matrix(rotations)
    raise ValueError(f"Only support 2D/3D rotations, got {rotations.shape}.")


def get_translation_matrix(
    shifts: torch.Tensor,
) -> torch.Tensor:
    """Return 3d translation matrix given shifts.

    For example, the 3D affine transformation applies as following:
        [x, = [[1 0 0 *]  * [x,
         y,    [0 1 0 *]     y,
         z,    [0 0 1 *]     z,
         1]    [0 0 0 1]]    1]

    Args:
        shifts: correspond to each axis shift, (n,).

    Returns:
        Translation matrix of shape (n+1, n+1).
    """
    n = shifts.shape[0]
    shifts = torch.cat([shifts, torch.tensor([1.0], dtype=shifts.dtype, device=shifts.device)])
    return torch.cat(
        [
            torch.eye(n + 1, n, dtype=shifts.dtype, device=shifts.device),
            shifts[:, None],
        ],
        dim=1,
    )


def get_scaling_matrix(
    scales: torch.Tensor,
) -> torch.Tensor:
    """Return scaling matrix given scales.

    For example, the 3D affine transformation applies as following:
        [x, = [[* 0 0 0]  * [x,
         y,    [0 * 0 0]     y,
         z,    [0 0 * 0]     z,
         1]    [0 0 0 1]]    1]

    Args:
        scales: correspond to each axis scaling.

    Returns:
        Affine matrix of shape (n+1, n+1).
    """
    scales = torch.cat([scales, torch.tensor([1.0], dtype=scales.dtype, device=scales.device)])
    return torch.diag(scales)


def get_2d_shear_matrix(
    shears: torch.Tensor,
) -> torch.Tensor:
    """Return 2D shear matrix.

    For example, the 2D shear matrix for x-axis applies as following
        [x, = [[1 s 0]  * [x,
         y,    [0 1 0]     y,
         1]    [0 0 1]]    1]
    where s is shear in x direction
    such that x = x + sy

    https://www.mauriciopoppe.com/notes/computer-graphics/transformation-matrices/shearing/

    Args:
        shears: radians, correspond to each plane shear, (2,).

    Returns:
        Affine matrix of shape (3, 3).
    """
    tans = torch.tan(shears)
    shear_x = torch.tensor(
        [
            [1.0, tans[0], 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=shears.dtype,
        device=shears.device,
    )
    shear_y = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [tans[1], 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=shears.dtype,
        device=shears.device,
    )
    return torch.matmul(shear_y, shear_x)


def get_3d_shear_matrix(
    shears: torch.Tensor,
) -> torch.Tensor:
    """Return 3D shear matrix.

    For example, the 3D shear matrix for xy plane applies as following
        [x, = [[1 0 s 0]  * [x,
         y,    [0 1 t 0]     y,
         z,    [0 0 1 0]     z,
         1]    [0 0 0 1]]    1]
    where s is shear in x direction, t is shear in y direction,
    such that x = x + sz, y = y + tz.

    https://www.mauriciopoppe.com/notes/computer-graphics/transformation-matrices/shearing/

    Args:
        shears: radians, correspond to plane yz, xz, xy, (6,).

    Returns:
        Affine matrix of shape (4, 4).
    """
    tans = torch.tan(shears)
    shear_yz = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [tans[0], 1.0, 0.0, 0.0],
            [tans[1], 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=shears.dtype,
        device=shears.device,
    )
    shear_xz = torch.tensor(
        [
            [1.0, tans[2], 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, tans[3], 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=shears.dtype,
        device=shears.device,
    )
    shear_xy = torch.tensor(
        [
            [1.0, 0.0, tans[4], 0.0],
            [0.0, 1.0, tans[5], 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=shears.dtype,
        device=shears.device,
    )
    return torch.matmul(shear_yz, torch.matmul(shear_xz, shear_xy))


def get_shear_matrix(
    shears: torch.Tensor,
) -> torch.Tensor:
    """Return rotation matrix given radians.

    Args:
        shears: correspond to shearing per axis/plane.

    Returns:
        Shear matrix of shape (n+1, n+1).

    Raises:
        ValueError: if not 2D or 3D.
    """
    if shears.shape == (2,):
        return get_2d_shear_matrix(shears)
    if shears.shape == (6,):
        return get_3d_shear_matrix(shears)
    raise ValueError(f"Only support 2D/3D shearing, got {shears.shape}.")


def get_affine_matrix(
    half_image_size_mm: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    shears: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    """Return an affine matrix from parameters.

    The matrix is not squared, as the last row is not needed. For rotation,
    translation, and scaling matrix, they are kept for composition purpose.
    For example, the 3D affine transformation applies as following:
        [x, = [[* * * *]  * [x,
         y,    [* * * *]     y,
         z,    [* * * *]     z,
         1]    [0 0 0 1]]    1]

    As the spacing may not be uniform, pytorch defines grid on [-1, 1] relative to image size.
    We first map [-1, 1] to real image size,
    then apply rotation, scale, shear, and shift, at last scale back to [-1, 1].

    Args:
        half_image_size_mm: half og image size in mm, (n,)
        rotations: correspond to rotate around each axis, (1,) for 2d and (3,) for 3d
        scales: correspond to each axis scaling, (n,).
        shears: correspond to each plane shear, (n*(n-1),).
        shifts: correspond to each axis shift, (n,).

    Returns:
        Affine matrix of shape (n+1, n+1).
    """
    affine_to_iso = get_scaling_matrix(half_image_size_mm)
    affine_rot = get_rotation_matrix(rotations)
    affine_scale = get_scaling_matrix(scales)
    affine_shear = get_shear_matrix(shears)
    affine_shift = get_translation_matrix(shifts)
    affine_from_iso = get_scaling_matrix(1.0 / half_image_size_mm)

    affine = torch.matmul(affine_rot, affine_to_iso)
    affine = torch.matmul(affine_scale, affine)
    affine = torch.matmul(affine_shear, affine)
    affine = torch.matmul(affine_shift, affine)
    affine = torch.matmul(affine_from_iso, affine)

    return affine


def batch_get_random_affine_matrix(
    half_image_size_mm: torch.Tensor,
    max_rotation: torch.Tensor,
    max_zoom: torch.Tensor,
    max_shear: torch.Tensor,
    max_shift: torch.Tensor,
    p: float,
) -> torch.Tensor:
    """Get a batch of random affine matrices.

    Args:
        half_image_size_mm: half of image size in mm, (n,).
        max_rotation: maximum rotation in radians,
            of shape (batch, 1) for 2D, correspond to rotation in xy plane,
            of shape (batch, 3) for 3D, correspond to rotations in yz, xz, xy planes.
        max_zoom: maximum zoom as a fraction of image size,
            of shape (batch, 2) for 2D, correspond to x, y axis,
            of shape (batch, 3) for 3D, correspond to x, y, z axis.
        max_shear: maximum shear in radians,
            of shape (batch, 1) for 2D, correspond to shear in xy plane,
            of shape (batch, 3) for 3D, correspond to shear in yz, xz, xy planes.
        max_shift: maximum shift in mm,
            of shape (batch, 2) for 2D, correspond to x, y axis,
            of shape (batch, 3) for 3D, correspond to x, y, z axis.
        p: probability to activate each transformation independently.

    Returns:
        Affine matrix of shape (batch, n+1, n+1), n is n_spatial_dims.
    """
    rotations = (torch.rand_like(max_rotation) * 2.0 - 1.0) * max_rotation
    rotations = torch.where(
        torch.rand_like(rotations) < p,
        rotations,
        torch.zeros_like(rotations),
    )

    scales = (torch.rand_like(max_zoom) * 2.0 - 1.0) * max_zoom + 1.0
    scales = torch.where(
        torch.rand_like(scales) < p,
        scales,
        torch.ones_like(scales),
    )

    shears = (torch.rand_like(max_shear) * 2.0 - 1.0) * max_shear
    shears = torch.where(
        torch.rand_like(shears) < p,
        shears,
        torch.zeros_like(shears),
    )

    shifts = (torch.rand_like(max_shift) * 2.0 - 1.0) * max_shift
    shifts = torch.where(
        torch.rand_like(shifts) < p,
        shifts,
        torch.zeros_like(shifts),
    )

    batch_size = max_rotation.shape[0]
    affine_matrices = torch.stack(
        [
            get_affine_matrix(half_image_size_mm, rotations[i], scales[i], shears[i], shifts[i])
            for i in range(batch_size)
        ],
        dim=0,
    )

    return affine_matrices


class BatchImageRandomAffine:
    """Random affine transformation for images."""

    def __init__(  # noqa: C901
        self,
        image_size: tuple[int, ...],
        spacing: tuple[float, ...],
        max_rotation: tuple[float, ...],
        max_zoom: tuple[float, ...],
        max_shear: tuple[float, ...],
        max_shift: tuple[float, ...],
        p: float,
        align_corners: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Initialize the transform.

        Args:
            image_size: image size, (d1, ..., dn).
            spacing: correspond to each axis spacing, (n,).
            max_rotation: maximum rotation in degrees,
                of shape (1,) for 2D, correspond to rotation in xy plane,
                of shape (3,) for 3D, correspond to rotations in yz, xz, xy planes.
            max_zoom: maximum zoom as a fraction of image size,
                of shape (2,) for 2D, correspond to x, y axis,
                of shape (3,) for 3D, correspond to x, y, z axis.
            max_shear: maximum shear in degrees,
                of shape (2,) for 2D, correspond to shear in xy plane,
                of shape (6,) for 3D, each two values correspond to shear in yz, xz, xy planes.
            max_shift: maximum shift as a fraction of image size,
                of shape (2,) for 2D, correspond to x, y axis,
                of shape (3,) for 3D, correspond to x, y, z axis.
            p: probability to activate each transformation independently.
            align_corners: align_corners in F.grid_sample and F.affine_grid.
            dtype: data type.
            device: device.
        """
        self.n = len(spacing)
        # sanity checks
        if self.n == 2:
            if len(max_rotation) != 1:
                raise ValueError(f"max_rotation should have 1 value for 2D, got {len(max_rotation)}.")
            if len(max_zoom) != 2:
                raise ValueError(f"max_zoom should have 2 values for 2D, got {len(max_zoom)}.")
            if len(max_shear) != 2:
                raise ValueError(f"max_shear should have 1 value for 2D, got {len(max_shear)}.")
            if len(max_shift) != 2:
                raise ValueError(f"max_shift should have 2 values for 2D, got {len(max_shift)}.")
        elif self.n == 3:
            if len(max_rotation) != 3:
                raise ValueError(f"max_rotation should have 3 values for 3D, got {len(max_rotation)}.")
            if len(max_zoom) != 3:
                raise ValueError(f"max_zoom should have 3 values for 3D, got {len(max_zoom)}.")
            if len(max_shear) != 6:
                raise ValueError(f"max_shear should have 3 values for 3D, got {len(max_shear)}.")
            if len(max_shift) != 3:
                raise ValueError(f"max_shift should have 3 values for 3D, got {len(max_shift)}.")
        else:
            raise ValueError(f"Only support 2D/3D, got {self.n}.")

        # store configs
        self.device = device
        self.dtype = dtype
        self.p = p
        self.align_corners = align_corners
        self.half_image_size_mm = (
            torch.tensor(image_size, dtype=dtype, device=device) * torch.tensor(spacing, dtype=dtype, device=device) / 2
        )  # in mm
        self.max_rotation = torch.deg2rad(torch.tensor(max_rotation, dtype=dtype, device=device))
        self.max_zoom = torch.tensor(max_zoom, dtype=dtype, device=device)
        self.max_shear = torch.deg2rad(torch.tensor(max_shear, dtype=dtype, device=device))
        self.max_shift = torch.tensor(max_shift, dtype=dtype, device=device) * self.half_image_size_mm  # in mm

    def __call__(
        self, image: torch.Tensor, label: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply random affine transformation to the input.

        Args:
            image: (batch, ch, d1, ..., dn, ...), may have additional dimensions.
            label: not used.

        Returns:
            augmented data.
        """
        if image.ndim < self.n + 2:
            raise ValueError(f"Input should have at least {self.n+2} dimensions, got {image.ndim}.")
        if label is not None:
            raise ValueError("Label should not used, please use `BatchImageLabelRandomAffine` for image-label pairs.")
        batch_size = image.shape[0]
        max_rotation = torch.tile(self.max_rotation[None, ...], (batch_size, 1))
        max_zoom = torch.tile(self.max_zoom[None, ...], (batch_size, 1))
        max_shear = torch.tile(self.max_shear[None, ...], (batch_size, 1))
        max_shift = torch.tile(self.max_shift[None, ...], (batch_size, 1))

        # (batch, n+1, n+1)
        affine_matrix = batch_get_random_affine_matrix(
            half_image_size_mm=self.half_image_size_mm,
            max_rotation=max_rotation,
            max_zoom=max_zoom,
            max_shear=max_shear,
            max_shift=max_shift,
            p=self.p,
        )
        # (batch, n, n+1)
        affine_matrix = affine_matrix[:, :-1, :]

        # for 3D, affine_matrix is (4, 4) each row is x, y, z, 1
        # but in F.affine_grid, theta is (N, 3, 4) corresponding to a (x,y,z) * (x,y,z,1) matrix
        # but the size is (N, C, D, H, W), whereas we store in (N, C, H, W, D) or (N, C, H, W, D, T)

        # (batch, d1, ..., dn, n)
        if image.ndim == self.n + 2:
            if self.n == 2:
                # (batch, ch, h, w)
                grid = F.affine_grid(
                    theta=affine_matrix,
                    size=list(image.shape[: self.n + 2]),
                    align_corners=self.align_corners,
                )
                return self.grid_sample_image(image, grid)
            if self.n == 3:
                # (batch, ch, h, w, d) -> (batch, ch, d, h, w)
                image = image.permute(0, 1, 4, 2, 3).contiguous()
                grid = F.affine_grid(
                    theta=affine_matrix,
                    size=list(image.shape[: self.n + 2]),
                    align_corners=self.align_corners,
                )
                image = self.grid_sample_image(image, grid)
                # (batch, ch, d, h, w) -> (batch, ch, h, w, d)
                return image.permute(0, 1, 3, 4, 2).contiguous()
            raise ValueError(f"Only support 2D/3D, got {self.n}.")

        # there are additional dimensions
        extra_dims = image.shape[self.n + 2 :]
        # (batch, ch, d1, ..., dn, -1)
        image = image.reshape((*image.shape[: self.n + 2], -1))
        if self.n == 3:
            # (batch, ch, h, w, d, m) -> (batch, ch, d, h, w, m)
            image = image.permute(0, 1, 4, 2, 3, 5).contiguous()

        grid = F.affine_grid(
            theta=affine_matrix,
            size=list(image.shape[: self.n + 2]),
            align_corners=self.align_corners,
        )
        image = torch.vmap(
            self.grid_sample_image,
            in_dims=(-1, None),
            out_dims=-1,
        )(image, grid)

        if self.n == 3:
            # (batch, ch, d, h, w, m) -> (batch, ch, h, w, d, m)
            image = image.permute(0, 1, 3, 4, 2, 5).contiguous()
        # (batch, ch, d1, ..., dn, ...)
        image = image.reshape((*image.shape[: self.n + 2], *extra_dims))
        return image

    def grid_sample_image(self, image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Apply grid sample to the image."""
        return F.grid_sample(image, grid, align_corners=self.align_corners)


class BatchImageLabelRandomAffine(BatchImageRandomAffine):
    """Random affine transformation for image-label pairs."""

    def __call__(
        self, image: torch.Tensor, label: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply random affine transformation to the input.

        Args:
            image: (batch, ch, d1, ..., dn, ...), may have additional dimensions.
            label: (batch, d1, ..., dn, ...), may have additional dimensions.

        Returns:
            augmented data.
        """
        if label is None:
            raise ValueError("Label should be provided.")
        if image.ndim != label.ndim + 1:
            raise ValueError(
                f"Image should have one more dimension than label, "
                f"got image shape {image.shape} and label shape {label.shape}."
            )
        if image.ndim < self.n + 2:
            raise ValueError(f"Image should have at least {self.n+2} dimensions, got image shape {image.shape}.")

        batch_size = image.shape[0]
        max_rotation = torch.tile(self.max_rotation[None, ...], (batch_size, 1))
        max_zoom = torch.tile(self.max_zoom[None, ...], (batch_size, 1))
        max_shear = torch.tile(self.max_shear[None, ...], (batch_size, 1))
        max_shift = torch.tile(self.max_shift[None, ...], (batch_size, 1))

        # (batch, n+1, n+1)
        affine_matrix = batch_get_random_affine_matrix(
            half_image_size_mm=self.half_image_size_mm,
            max_rotation=max_rotation,
            max_zoom=max_zoom,
            max_shear=max_shear,
            max_shift=max_shift,
            p=self.p,
        )
        # (batch, n, n+1)
        affine_matrix = affine_matrix[:, :-1, :]

        # for 3D, affine_matrix is (4, 4) each row is x, y, z, 1
        # but in F.affine_grid, theta is (N, 3, 4) corresponding to a (x,y,z) * (x,y,z,1) matrix
        # but the size is (N, C, D, H, W), whereas we store in (N, C, H, W, D) or (N, C, H, W, D, T)

        # if keep label as integer, gets error "grid_sampler3d_cpu" not implemented for 'Long'
        label_dtype = label.dtype
        label = label.unsqueeze(1).to(image.dtype)  # (batch, 1, d1, ..., dn, ...) for grid_sample

        # (batch, d1, ..., dn, n)
        if image.ndim == self.n + 2:
            if self.n == 2:
                # (batch, ch, h, w)
                grid = F.affine_grid(
                    theta=affine_matrix,
                    size=list(image.shape[: self.n + 2]),
                    align_corners=self.align_corners,
                )
                image = self.grid_sample_image(image, grid)
                label = self.grid_sample_label(label, grid)
                label = label.squeeze(1).to(label_dtype)
                return image, label
            if self.n == 3:
                # (batch, ch, h, w, d) -> (batch, ch, d, h, w)
                image = image.permute(0, 1, 4, 2, 3).contiguous()
                label = label.permute(0, 1, 4, 2, 3).contiguous()
                grid = F.affine_grid(
                    theta=affine_matrix,
                    size=list(image.shape[: self.n + 2]),
                    align_corners=self.align_corners,
                )
                image = self.grid_sample_image(image, grid)
                label = self.grid_sample_label(label, grid)
                # (batch, ch, d, h, w) -> (batch, ch, h, w, d)
                image = image.permute(0, 1, 3, 4, 2).contiguous()
                label = label.permute(0, 1, 3, 4, 2).contiguous()
                label = label.squeeze(1).to(label_dtype)
                return image, label
            raise ValueError(f"Only support 2D/3D, got {self.n}.")

        # there are additional dimensions
        extra_dims = image.shape[self.n + 2 :]
        # (batch, ch, d1, ..., dn, -1)
        image = image.reshape((*image.shape[: self.n + 2], -1))
        label = label.reshape((*label.shape[: self.n + 2], -1))
        if self.n == 3:
            # (batch, ch, h, w, d, m) -> (batch, ch, d, h, w, m)
            image = image.permute(0, 1, 4, 2, 3, 5).contiguous()
            label = label.permute(0, 1, 4, 2, 3, 5).contiguous()

        grid = F.affine_grid(
            theta=affine_matrix,
            size=list(image.shape[: self.n + 2]),
            align_corners=self.align_corners,
        )
        image = torch.vmap(
            self.grid_sample_image,
            in_dims=(-1, None),
            out_dims=-1,
        )(image, grid)
        label = torch.vmap(
            self.grid_sample_label,
            in_dims=(-1, None),
            out_dims=-1,
        )(label, grid)

        if self.n == 3:
            # (batch, ch, d, h, w, m) -> (batch, ch, h, w, d, m)
            image = image.permute(0, 1, 3, 4, 2, 5).contiguous()
            label = label.permute(0, 1, 3, 4, 2, 5).contiguous()
        # (batch, ch, d1, ..., dn, ...)
        image = image.reshape((*image.shape[: self.n + 2], *extra_dims))
        label = label.reshape((*label.shape[: self.n + 2], *extra_dims))
        label = label.squeeze(1).to(label_dtype)
        return image, label

    def grid_sample_label(self, label: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Apply grid sample to the label."""
        if not label.is_signed():
            raise ValueError(f"Label should be signed integer, but got {label.dtype}.")
        return F.grid_sample(label, grid, mode="nearest", align_corners=self.align_corners)


def data_affine(image_np):
    spacing = (1.0, 1.0,10.0)
    image_size = image_np[...].shape
    image_np = image_np[None, None, ...]
    image_np = image_np.astype(np.float32) / 255.
    image = torch.from_numpy(image_np)

    align_corners = False
    max_rotation = (15, 15, 15)  # degrees for yz, xz, xy planes
    max_zoom = (0.2, 0.2, 0.2)  # as a fraction for x, y, z axes
    max_shear = (5, 5, 5,5, 5, 5)  # each value for yz, xz, xy planes
    max_shift = (0.15, 0.15, 0.15)  # as a fraction for x, y, z axes
    p = 1.0  # probability to apply transform


    image_transform = BatchImageRandomAffine(
        image_size=image_size,
        spacing=spacing,
        max_rotation=max_rotation,
        max_zoom=max_zoom,
        max_shear=max_shear,
        max_shift=max_shift,
        p=p,
        dtype=torch.float32,
        device=torch.device("cpu"),
        align_corners=align_corners,
    )

    transformed_image_np = image_transform(image).numpy()

    noise = np.random.normal(0, 0.1, transformed_image_np.shape)  # 生成高斯噪声
    noisy_image = transformed_image_np + noise

    return np.squeeze(noisy_image)