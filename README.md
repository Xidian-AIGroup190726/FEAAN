# FEAAN
Our code is released [here](https://github.com/LiuShutong04/FEAAN-main) !
# Network Architecture

FEAAN consists of a dual-branch encoder, feature differentiation units, a Feature Attribute Inference Module (FAIM), an Attribute-Specific Alignment Module (ASAM), and a downstream classifier. During pretraining, the encoders extract features at five levels. The feature differentiation units prepare these features for attribute inference in FAIM and attribute-specific alignment in ASAM. FAIM produces path weights that combine the alignment losses of ASAM. During fine-tuning, the pretrained encoders and the classifier are jointly updated using a small set of labeled samples for joint PAN and MS classification.

The following description uses multispectral (MS) patches of size $16\times16\times4$ and panchromatic (PAN) patches of size $64\times64\times1$. Feature map sizes are written as height × width × channels, with the batch dimension omitted. Superscripts $m$ and $p$ denote MS and PAN, respectively, and $l$ denotes the feature level.

1. Dual-branch Encoder

Both branches use ResNet-50 with a modified input layer, and their parameters are learned independently. The input layers accept four-channel MS patches and single-channel PAN patches, respectively. Each branch extracts features from the Stem and the ends of the four residual stages, giving $L=5$ feature levels. These features are passed to the corresponding feature differentiation units.

In the table below, $B(c,c_{\mathrm{out}},s)$ denotes a residual bottleneck block whose main path contains a $1\times1$ convolution with $c$ output channels, a $3\times3$ convolution with $c$ output channels and stride $s$, and a $1\times1$ convolution with $c_{\mathrm{out}}$ output channels. A stride of 2 halves the spatial dimensions. The multiplier gives the number of repeated blocks. BN denotes batch normalization.

Feature level

Network architecture

MS output

PAN output

Level 1: Stem

$3\times3$ Conv, 64 channels, stride 1 → BN → ReLU

$f_m^{(1)}$: $16\times16\times64$

$f_p^{(1)}$: $64\times64\times64$

Level 2: Residual stage 1

$B(64,256,1)\times3$

$f_m^{(2)}$: $16\times16\times256$

$f_p^{(2)}$: $64\times64\times256$

Level 3: Residual stage 2

$B(128,512,2)\rightarrow B(128,512,1)\times3$

$f_m^{(3)}$: $8\times8\times512$

$f_p^{(3)}$: $32\times32\times512$

Level 4: Residual stage 3

$B(256,1024,2)\rightarrow B(256,1024,1)\times5$

$f_m^{(4)}$: $4\times4\times1024$

$f_p^{(4)}$: $16\times16\times1024$

Level 5: Residual stage 4

$B(512,2048,2)\rightarrow B(512,2048,1)\times2$

$f_m^{(5)}$: $2\times2\times2048$

$f_p^{(5)}$: $8\times8\times2048$

2. Feature Differentiation Unit

At each level, the feature differentiation unit receives features from both modalities and prepares inputs for attribute inference and the three alignment paths. MS features are first upsampled to match the spatial dimensions of PAN features. Thus, $(\uparrow4,\uparrow1)$ means that the height and width of MS features are multiplied by 4, while PAN features retain their original size.

Each modality then passes through a $3\times3$ convolution, BN, and ReLU to obtain 128-channel base features $f_{\mathrm{base}}^{m,l}$ and $f_{\mathrm{base}}^{p,l}$. Their spatial sizes are

$$
(H_1,H_2,H_3,H_4,H_5)=(64,64,32,16,8).
$$

The base features from the two modalities are concatenated along the channel dimension and passed through a spatial branch to produce a shared guidance map $G^l$. This map determines the locations used for structural alignment. The base features also pass separately through $1\times1$ convolutions with an orthogonality constraint to obtain channel features $f_{\mathrm{chan}}^{m,l}$ and $f_{\mathrm{chan}}^{p,l}$ for semantic alignment.

Component

Input

Processing

Output

Base feature projection

$f_m^{(l)}$, $f_p^{(l)}$

MS: upsample by 4; PAN: unchanged.
Each modality: $3\times3$ Conv, 128 channels → BN → ReLU

$f_{\mathrm{base}}^{m,l}$, $f_{\mathrm{base}}^{p,l}$; each $H_l\times H_l\times128$

Spatial branch

Concatenated base features: $H_l\times H_l\times256$

$1\times1$ Conv, 1 channel → sigmoid

$G^l$: $H_l\times H_l\times1$

Channel branch

Base features from each modality: $H_l\times H_l\times128$

$1\times1$ Conv, 128 channels, with an orthogonality constraint;
$W_{\mathrm{orth}}\in\mathbb{R}^{128\times128}$

$f_{\mathrm{chan}}^{m,l}$, $f_{\mathrm{chan}}^{p,l}$; each $H_l\times H_l\times128$

The base features are used by FAIM and the texture and structure paths in ASAM. The shared guidance map is used by the structure path, while the channel features are used by the semantic path.

3. Feature Attribute Inference Module (FAIM)

FAIM determines the weights of the three alignment paths from the base features at the current level. It computes three indicators: local variation $\nu$, cross-modal structural similarity $\rho$, and channel diversity $\delta$. These indicators provide the basis for inferring texture, structure, and semantic attributes, respectively.

The indicators pass through three $1\rightarrow16\rightarrow16$ branches. Their outputs are concatenated into a 48-dimensional representation, which is mapped through $48\rightarrow32\rightarrow3$ to produce learned path scores $z_{\mathrm{learned}}$. These scores are combined with the dominance prior $z_{\mathrm{prior}}=[z_A,z_B,z_C]$ and the attribute evolution bias $b(p)$. Gumbel-Softmax then produces the path weights:

$$
w=[w_A,w_B,w_C]
=\operatorname{GumbelSoftmax}\left(
z_{\mathrm{learned}}+\lambda_r z_{\mathrm{prior}}+\lambda_b b(p)
\right).
$$

Here, $\lambda_r$ and $\lambda_b$ control the contributions of the dominance prior and the attribute evolution bias, respectively.

Step

Input

Processing

Output

Attribute measurement

$f_{\mathrm{base}}^{m,l}$, $f_{\mathrm{base}}^{p,l}$

Statistics: $\mathbb{E}_\Omega$, $\sigma_N(x)$, $\cos(\cdot,\cdot)$, $\mathrm{CV}$.
Scale-related terms: $r$, $g(r)$, $h(r)$

$[\nu,\rho,\delta]\in\mathbb{R}^3$

Learned score prediction

$[\nu,\rho,\delta]$

Three $1\rightarrow16\rightarrow16$ branches → concatenate to 48 dimensions → $48\rightarrow32\rightarrow3$

$z_{\mathrm{learned}}\in\mathbb{R}^3$

Path weighting

$z_{\mathrm{learned}}$, $z_{\mathrm{prior}}$, $b(p)$

Combine learned scores, the weighted dominance prior, and the weighted attribute evolution bias → Gumbel-Softmax

$[w_A,w_B,w_C]\in\mathbb{R}^3$

4. Attribute-Specific Alignment Module (ASAM)

ASAM contains three paths for texture, structure, and semantic alignment. These paths compute $\mathcal{L}_A$, $\mathcal{L}_B$, and $\mathcal{L}_C$, respectively. Their losses are combined using the path weights produced by FAIM during pretraining.

Path

Input

Processing

Representation size per modality

Output

A: Texture alignment

$f_{\mathrm{base}}^{m,l}$, $f_{\mathrm{base}}^{p,l}$

Divide into non-overlapping $4\times4$ regions → average within each region → regional contrastive learning

$(H_l^2/16)\times128$

Scalar loss $\mathcal{L}_A$

B: Structure alignment

$f_{\mathrm{base}}^{m,l}$, $f_{\mathrm{base}}^{p,l}$, $G^l$

Select the top 64 locations in $G^l$ → extract corresponding features → constrain feature similarity and relationships among locations

$64\times128$

Scalar loss $\mathcal{L}_B$

C: Semantic alignment

$f_{\mathrm{chan}}^{m,l}$, $f_{\mathrm{chan}}^{p,l}$

Pool to $1\times1$ and $2\times2$ → concatenate → divide into four subspaces → semantic alignment with relaxation

$640=4\times160$

Scalar loss $\mathcal{L}_C$

4.1. Texture Alignment

The texture path divides each base feature map into non-overlapping $4\times4$ regions and averages the 16 feature vectors within each region. This produces a 128-dimensional representation for each region. Contrastive learning is then applied to corresponding regions across the two modalities. In $(H_l^2/16)\times128$, the first dimension is the number of regions and the second is the feature dimension of each region.

4.2. Structure Alignment

The structure path selects the 64 locations with the highest responses in the shared guidance map $G^l$. Features at the same locations are extracted from both modalities, forming two matrices $Z_m,Z_p\in\mathbb{R}^{64\times128}$. This path constrains both feature similarity at corresponding locations and feature relationships among the selected locations.

4.3. Semantic Alignment

The semantic path pools the channel features of each modality to $1\times1$ and $2\times2$, then concatenates the pooled features into a $128(1^2+2^2)=640$-dimensional representation. This representation is divided into $H_s=4$ subspaces, each with 160 dimensions, for semantic alignment. When the similarity between features from the two modalities is high, a relaxation mechanism weakens the constraint that pulls them closer together.

5. Downstream Fine-tuning for Classification

After pretraining, the dual-branch encoder is used for supervised classification. Features from the five levels of each branch are globally average-pooled and concatenated:

$$
A_m=\left[\operatorname{GAP}(f_m^{(1)});\ldots;\operatorname{GAP}(f_m^{(5)})\right],
\qquad
A_p=\left[\operatorname{GAP}(f_p^{(1)});\ldots;\operatorname{GAP}(f_p^{(5)})\right].
$$

Each vector has $64+256+512+1024+2048=3904$ dimensions. Learnable gating coefficients $g_m$ and $g_p$ adjust the two vectors before concatenation:

$$
h_{\mathrm{fused}}=\left[g_m\odot A_m;;g_p\odot A_p\right]\in\mathbb{R}^{7808}.
$$

Here, $\odot$ denotes element-wise multiplication, and $[,;,]$ denotes concatenation. A $7808\rightarrow256\rightarrow C$ classifier produces the classification scores, where $C$ is the number of classes. The table below uses $C=11$ as an example. During fine-tuning, the encoders and classifier are jointly updated using a small set of labeled samples.

Step

Input

Processing

Output

Multi-level feature aggregation

Five feature levels from each branch

Global average pooling at each level → concatenate across levels

$A_m,A_p\in\mathbb{R}^{3904}$

Gated feature fusion

$A_m$, $A_p$

Apply $g_m$ and $g_p$ → concatenate the gated vectors

$h_{\mathrm{fused}}\in\mathbb{R}^{7808}$

Classification

$h_{\mathrm{fused}}$

$7808\rightarrow256\rightarrow C$

Classification scores in $\mathbb{R}^C$; $C=11$ in this example
