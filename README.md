# Im2Hands (Implicit Two Hands)
## Im2Hands: Learning Attentive Implicit Representation of Interacting Two-Hand Shapes (CVPR 2023) ##

[Jihyun Lee](https://jyunlee.github.io/), [Minhyuk Sung](https://mhsung.github.io/), [Honggyu Choi](https://honggyuchoi.github.io/), [Tae-Kyun (T-K) Kim](https://sites.google.com/view/tkkim/home)

**[\[Project Page\]](https://jyunlee.github.io/projects/implicit-two-hands) [\[Paper\]](https://arxiv.org/abs/2302.14348) [\[Supplementary Video\]](https://youtu.be/3yNGSRz564A)**

**Code will be uploaded before CVPR 2023 (mid-June). Please stay tuned!**

<p align="center">
  <img src="teaser.gif" alt="animated" />
</p>

> We present Implicit Two Hands (Im2Hands), the first neural implicit representation of two interacting hands. Unlike existing methods on two-hand reconstruction that rely on a parametric hand model and/or low-resolution meshes, Im2Hands can produce fine-grained geometry of two hands with high hand-to-hand and hand-to-image coherency. To handle the shape complexity and interaction context between two hands, Im2Hands models the occupancy volume of two hands - conditioned on an RGB image and coarse 3D keypoints - by two novel attention-based modules responsible for initial occupancy estimation and context-aware occupancy refinement, respectively. Im2Hands first learns per-hand neural articulated occupancy in the canonical space designed for each hand using query-image attention. It then refines the initial two-hand occupancy in the posed space to enhance the coherency between the two hand shapes using query-anchor attention. In addition, we introduce an optional keypoint refinement module to enable robust two-hand shape estimation from predicted hand keypoints in a single-image reconstruction scenario. We experimentally demonstrate the effectiveness of Im2Hands on two-hand reconstruction in comparison to related methods, where ours achieves state-of-the-art results.

&nbsp;
