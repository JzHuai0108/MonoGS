## List of Known Dependencies
### MonoGS

In this document we list all the pieces of code included  by MonoGS and linked libraries which are not property of the authors of MonoGS.


##### Code in gaussian_splatting folder
Please follow the license of 3D Gaussian Splatting.

License URL: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md

##### Code in gui/gl_render folder
Please follow the license of Tiny Gaussian Splatting Viewer

Licence URL: https://github.com/limacv/GaussianSplattingViewer/blob/main/LICENSE

##### Code in submodules folder
Please follow the licenses of each repository.

##### simple-knn

By comparing the master branches of PGSR, LoopSplat, 2d-gaussian-splatting, we found that their simple-knn are literally the same as the bkerbl repo.

##### diff-gaussian-rasterization

LoopSplat's diff-gaussian-rasterization-w-pose is the same as the MonoGS submodule diff-gaussian-rasterization,
as both are from rmurai0610 and supports depth rendering and depth and pose back propagation in cuda.
LoopSplat uses this submodule for view registration to a Gaussian model, thus saving the need to transform Gaussians outside cuda.

LoopSplat also depends on a gaussian-rasterizer from VladimirYugay, it extends
[the original one](https://github.com/graphdeco-inria/diff-gaussian-rasterization) by supporting depth rendering and depth back propagation in cuda.

The diff-gaussian-rasterization-w-depth in SplaTAM does not to support depth or pose back propagation,
and computes median depth instead of alpha blended depth in rendering depth.
As a result, SplaTAM renders (depth, silhouette, depth^2) through the same mechanism as 3-channel colors, to support depth back propagation.
Also, SplaTAM and Gaussian-SLAM both transform the Gaussians before cuda rendering to support pose back propagation.
That is, their pose gradient is computed by PyTorch autograd.
SplaTAM ignores the spherical harmonics and uses one radius instead of three for the Gaussian covariance, i.e., isotropic.
MonoGS uses anisotropic Gaussians and supports using 3 degree spherical harmonics.

Both PGSR and 2d-gaussian-splatting customized diff-gaussian-rasterization to the 2D case, 
but in different ways, plane (squeezed 3d) and surfel (directly 2d), respectively.
PGSR's diff-plane-rasterization supports depth and normal rendering and back propagation in cuda.
2D-gaussian-splatting supports normal rendering and back propagation in cuda.
Both do not support pose back propagation in cuda.
Since it's a bit challenging to integrate the pose back propagation from MonoGS into cuda,
it is advised to use PyTorch to compute pose gradients and gain more flexibility relative to the different variants of Gaussian Rasterizers,
as done in SplaTAM and Gaussian-SLAM and LoopSplat.

##### Monocular or RGBD
From the MonoGS code, I see that the monocular case requires a random depth initialization, and has a larger position learning rate.
