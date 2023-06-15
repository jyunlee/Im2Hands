# Im2Hands (Implicit Two Hands)
## Im2Hands: Learning Attentive Implicit Representation of Interacting Two-Hand Shapes (CVPR 2023) ##

[Jihyun Lee](https://jyunlee.github.io/), [Minhyuk Sung](https://mhsung.github.io/), [Honggyu Choi](https://honggyuchoi.github.io/), [Tae-Kyun (T-K) Kim](https://sites.google.com/view/tkkim/home)

**[\[Project Page\]](https://jyunlee.github.io/projects/implicit-two-hands) [\[Paper\]](https://arxiv.org/abs/2302.14348) [\[Supplementary Video\]](https://youtu.be/3yNGSRz564A)**

---

**CVPR 2023 Materials: [\[Presentation Video\]](https://youtu.be/hBSeN222Um4)** **<a href="https://jyunlee.github.io/projects/implicit-two-hands/data/cvpr2023_poster.pdf" class="image fit" type="application/pdf">\[Poster\]</a>**

<p align="center">
  <img src="teaser.gif" alt="animated" />
</p>

> We present Implicit Two Hands (Im2Hands), the first neural implicit representation of two interacting hands. Unlike existing methods on two-hand reconstruction that rely on a parametric hand model and/or low-resolution meshes, Im2Hands can produce fine-grained geometry of two hands with high hand-to-hand and hand-to-image coherency. To handle the shape complexity and interaction context between two hands, Im2Hands models the occupancy volume of two hands - conditioned on an RGB image and coarse 3D keypoints - by two novel attention-based modules responsible for initial occupancy estimation and context-aware occupancy refinement, respectively. Im2Hands first learns per-hand neural articulated occupancy in the canonical space designed for each hand using query-image attention. It then refines the initial two-hand occupancy in the posed space to enhance the coherency between the two hand shapes using query-anchor attention. In addition, we introduce an optional keypoint refinement module to enable robust two-hand shape estimation from predicted hand keypoints in a single-image reconstruction scenario. We experimentally demonstrate the effectiveness of Im2Hands on two-hand reconstruction in comparison to related methods, where ours achieves state-of-the-art results.

&nbsp;

## Environment Setup  

Clone this repository and install the dependencies specified in `requirements.txt`.

<pre><code> git clone https://github.com/jyunlee/Im2Hands.git
 mv Im2Hands
 pip install -r requirements.txt </pre></code>

Also, install [im2mesh library of Occupancy Networks](https://github.com/autonomousvision/occupancy_networks).

&nbsp;

## Data Preparation 

1. Download InterHand2.6M dataset from [its official website](https://mks0601.github.io/InterHand2.6M/). Set the value of `path` variable in your config file - which is either `configs/init_occ/init_occ.yaml`, `configs/ref_occ/ref_occ.yaml` or `configs/kpts_ref/kpts_ref.yaml` depending on the network - as the path of `annotation` directory of InterHand2.6M.
2. Follow the data pre-processing steps of [IntagHand](https://github.com/Dw1010/IntagHand) (`dataset/interhand.py`). Set the value of `img_path` variable in your config file as the path of the resulting data directory.
3. Place your hand keypoint files - either the ground truth keypoints of InterHand2.6M or predicted keypoints extracted using e.g. [IntagHand](https://github.com/Dw1010/IntagHand) - under `pred_joints` sub-directory under your pre-processed data directory.

&nbsp;

## Initial Occupancy Estimation Network

### Network Training

Place the pre-trained weights of [HALO](https://github.com/korrawe/halo) and [IntagHand](https://github.com/Dw1010/IntagHand) (`halo_baseline.pt` and `intaghand_baseline.pth`) under `out/init_occ` directory. These files can be also downloaded from [this Google Drive link](https://drive.google.com/drive/folders/1Kpoj1WW37hHvYgvhkwfmxNZVgPlyaApQ?usp=sharing).

Then, Run `init_occ_train.py` to train your own Initial Occupancy Estimation Network.

<pre><code> python init_occ_train.py </pre></code>

### Network Inference

Run `init_occ_generate.py` to generate the initial per-hand shapes.

<pre><code> python init_occ_generate.py </pre></code>

&nbsp;

## Two-Hand Occupancy Refinement Network

### Network Training

Place the weights of Initial Occupancy Estimation Network (`init_occ.pt`) under `out/ref_occ` directory. The pre-trained weights can be also downloaded from [this Google Drive link](https://drive.google.com/drive/folders/1Kpoj1WW37hHvYgvhkwfmxNZVgPlyaApQ?usp=sharing). 

Then, run `ref_occ_train.py` to train your own Refined Occupancy Estimation Network.

<pre><code> python ref_occ_train.py </pre></code>

For quantitative evaluation, please refer to `eval_meshes.py` script of [HALO](https://github.com/korrawe/halo).

### Network Inference

Run `ref_occ_generate.py` to generate the refined two-hand shapes.

<pre><code> python ref_occ_generate.py </pre></code>

&nbsp;

## [Optional] Keypoint Refinement Network

### Network Training

Place the pre-trained weights of [IntagHand](https://github.com/Dw1010/IntagHand) (`intaghand_baseline.pth`) under `out/kpts_ref` directory. This file can be also downloaded from [this Google Drive link](https://drive.google.com/drive/folders/1Kpoj1WW37hHvYgvhkwfmxNZVgPlyaApQ?usp=sharing).

Also, place (1) your initial hand keypoint files (e.g. predicted using [IntagHand](https://github.com/Dw1010/IntagHand)) under `pred_joints_before_ref` sub-directory and (2) the ground truth keypoints of InterHand2.6M under `gt_joints` sub-directory of your pre-processed data directory.

Then, Run `kpts_ref_train.py` to train your own Input Keypoint Refinement Network.

<pre><code> python kpts_ref_train.py </pre></code>

### Network Inference

Run `kpts_ref_generate.py` to save the refined two-hand keypoints.

<pre><code> python kpts_ref_generate.py </pre></code>

For quantitative evaluation, please refer to `apps/eval_interhand.py` script of [IntagHand](https://github.com/Dw1010/IntagHand).

&nbsp;

## Citation

If you find this work useful, please consider citing our paper.

```
@InProceedings{lee2023im2hands,
    author = {Lee, Jihyun and Sung, Minhyuk and Choi, Honggyu and Kim, Tae-Kyun},
    title = {Im2Hands: Learning Attentive Implicit Representation of Interacting Two-Hand Shapes},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2023}
}
```

&nbsp;

## Acknowledgements

 - Our code is based on [HALO](https://github.com/korrawe/halo), [IntagHand](https://github.com/Dw1010/IntagHand), and [AIR-Nets](https://github.com/SimonGiebenhain/AIR-Nets), and our model parameters are partially initialized from their pre-trained weights. We thank the authors of these three inspiring works.
 - We also thank the authors of [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) for the useful dataset.
