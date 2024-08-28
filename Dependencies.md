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

LoopSplat's diff-gaussian-rasterization-w-pose is the same as the MonoGS submodule as both are from rmurai0610 and supports depth rendering and back propagation.

LoopSplat also depends on a gaussian-rasterizer from VladimirYugay. But I think it has the same functionality as [the original one](https://github.com/graphdeco-inria/diff-gaussian-rasterization), but enables debugging.

The diff-gaussian-rasterization-w-depth in SplaTAM does not to support depth back propagation,
and computes median depth instead of alpha blended depth in rendering depth.
As a result, SplaTAM renders (depth, silhouette, depth^2) through the same mechanism as 3-channel colors, to support back propagation.

Both PGSR and 2d-gaussian-splatting customized diff-gaussian-rasterization to the 2D case, but in different ways, plane (squeezed 3d) and surfel (directly 2d), respectively.



