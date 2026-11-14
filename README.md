# FEAAN
Our code is released [here](https://github.com/LiuShutong04/FEAAN-main) !

## Network Architecture Details

The original manuscript describes the design of each module separately but does not adequately explain how features pass between modules or how the modules work together during training. We therefore provide a detailed explanation of the architectural components and their relationships.

The network consists of a dual-branch encoder, feature differentiation units, FAIM, ASAM, and a downstream classification component for fine-tuning. During pretraining, attribute inference and attribute-specific alignment support the learning of cross-modal representations. During fine-tuning, the pretrained encoder performs joint classification under the supervision of a small set of labeled samples.

<table>
<caption><strong>Table 1: Network Architecture Details of the Proposed Method</strong></caption>
<thead>
<tr>
<th>Module</th>
<th>Input</th>
<th>Network Architecture</th>
<th>Output</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="5"><strong>Dual-branch Encoder</strong></td>
<td>\(I_m\ (16 \times 16 \times 4)\)<br>\(I_p\ (64 \times 64 \times 1)\)</td>
<td>\(\text{Conv}[3 \times 3, 64] \to \text{BN} \to \text{ReLU}\)</td>
<td>\(f_m^{(1)}\ (16 \times 16 \times 64)\)<br>\(f_p^{(1)}\ (64 \times 64 \times 64)\)</td>
</tr>
<tr>
<td>\(f_m^{(1)}\ (16 \times 16 \times 64)\)<br>\(f_p^{(1)}\ (64 \times 64 \times 64)\)</td>
<td>\(\begin{bmatrix} 1 \times 1 & 64 \\\\ 3 \times 3 & 64 \\\\ 1 \times 1 & 256 \end{bmatrix} \times 3\)</td>
<td>\(f_m^{(2)}\ (16 \times 16 \times 256)\)<br>\(f_p^{(2)}\ (64 \times 64 \times 256)\)</td>
</tr>
<tr>
<td>\(f_m^{(2)}\ (16 \times 16 \times 256)\)<br>\(f_p^{(2)}\ (64 \times 64 \times 256)\)</td>
<td>\(\begin{bmatrix} 1 \times 1 & 128 \\\\ 3 \times 3 \downarrow 2 & 128 \\\\ 1 \times 1 & 512 \end{bmatrix} \to \begin{bmatrix} 1 \times 1 & 128 \\\\ 3 \times 3 & 128 \\\\ 1 \times 1 & 512 \end{bmatrix} \times 3\)</td>
<td>\(f_m^{(3)}\ (8 \times 8 \times 512)\)<br>\(f_p^{(3)}\ (32 \times 32 \times 512)\)</td>
</tr>
<tr>
<td>\(f_m^{(3)}\ (8 \times 8 \times 512)\)<br>\(f_p^{(3)}\ (32 \times 32 \times 512)\)</td>
<td>\(\begin{bmatrix} 1 \times 1 & 256 \\\\ 3 \times 3 \downarrow 2 & 256 \\\\ 1 \times 1 & 1024 \end{bmatrix} \to \begin{bmatrix} 1 \times 1 & 256 \\\\ 3 \times 3 & 256 \\\\ 1 \times 1 & 1024 \end{bmatrix} \times 5\)</td>
<td>\(f_m^{(4)}\ (4 \times 4 \times 1024)\)<br>\(f_p^{(4)}\ (16 \times 16 \times 1024)\)</td>
</tr>
<tr>
<td>\(f_m^{(4)}\ (4 \times 4 \times 1024)\)<br>\(f_p^{(4)}\ (16 \times 16 \times 1024)\)</td>
<td>\(\begin{bmatrix} 1 \times 1 & 512 \\\\ 3 \times 3 \downarrow 2 & 512 \\\\ 1 \times 1 & 2048 \end{bmatrix} \to \begin{bmatrix} 1 \times 1 & 512 \\\\ 3 \times 3 & 512 \\\\ 1 \times 1 & 2048 \end{bmatrix} \times 2\)</td>
<td>\(f_m^{(5)}\ (2 \times 2 \times 2048)\)<br>\(f_p^{(5)}\ (8 \times 8 \times 2048)\)</td>
</tr>
<tr>
<td rowspan="3"><strong>Feature Differentiation Unit</strong></td>
<td>\(\{f_m^{(l)}, f_p^{(l)}\}\)<br>\(l=1,2,3,4,5\)</td>
<td>\((\uparrow 4, \uparrow 1)\)<br>\(\text{Conv}[3 \times 3, 128] \to \text{BN} \to \text{ReLU}\)</td>
<td>\(\{f_{\text{base}}^{m,l}, f_{\text{base}}^{p,l}\}\ (H_l \times H_l \times 128)\)<br>\((H_1, H_2, H_3, H_4, H_5)\)<br>\(=(64, 64, 32, 16, 8)\)</td>
</tr>
<tr>
<td>\(f_{\text{base}}^{m,l}, f_{\text{base}}^{p,l}\)<br>\((H_l \times H_l \times 256)\)</td>
<td>\(\text{Conv}^{(\text{spa})}[1 \times 1, 1] \to \sigma\)</td>
<td>\(G^l\ (H_l \times H_l \times 1)\)</td>
</tr>
<tr>
<td>\(f_{\text{base}}^{m,l}, f_{\text{base}}^{p,l}\)<br>\((H_l \times H_l \times 128)\)</td>
<td>\(\text{Conv}^{(\text{orth})}[1 \times 1, 128]\)<br>\(W_{\text{orth}} \in \mathbb{R}^{128 \times 128}\)</td>
<td>\(\{f_{\text{chan}}^{m,l}, f_{\text{chan}}^{p,l}\}\)<br>\((H_l \times H_l \times 128)\)</td>
</tr>
<tr>
<td rowspan="3"><strong>FAIM</strong></td>
<td>\(f_{\text{base}}^{m,l}, f_{\text{base}}^{p,l}\)<br>\((H_l \times H_l \times 128)\)</td>
<td>\(\mathbb{E}_\Omega, \sigma_N(x), \cos(\cdot, \cdot), \text{CV}\)<br>\(r, g(r), h(r)\)</td>
<td>\([\nu, \rho, \delta] \in \mathbb{R}^3\)</td>
</tr>
<tr>
<td>\([\nu, \rho, \delta] \in \mathbb{R}^3\)</td>
<td>\((1 \to 16 \to 16) \times 3 \to 48\)<br>\(48 \to 32 \to 3\)</td>
<td>\(z_{\text{learned}} \in \mathbb{R}^3\)</td>
</tr>
<tr>
<td>\(z_{\text{learned}}\), \(z_{\text{prior}}\), \(b(p)\)<br>\(z_{\text{prior}} = [z_A, z_B, z_C]\)</td>
<td>\(h(z_{\text{learned}} + \lambda_r z_{\text{prior}} + \lambda_b b(p))\)</td>
<td>\(w = [w_A, w_B, w_C] \in \mathbb{R}^3\)</td>
</tr>
<tr>
<td rowspan="3"><strong>ASAM</strong></td>
<td>\(f_{\text{base}}^{m,l}, f_{\text{base}}^{p,l}\)<br>\((H_l \times H_l \times 128)\)</td>
<td>\(U_4 \to \frac{1}{16} \sum_{k=1}^{16}(\cdot)_k \to \{\tilde{z}_m; \tilde{z}_p\}\)<br>\((H_l^2/16) \times 128\)</td>
<td>\(\mathcal{L}_A\ (1)\)</td>
</tr>
<tr>
<td>\(f_{\text{base}}^{m,l}, f_{\text{base}}^{p,l}\), \(G^l\)</td>
<td>\(\text{TopK}(G^l, 64) \to \{Z_m, Z_p\}\)<br>\(64 \times 128\)</td>
<td>\(\mathcal{L}_B\ (1)\)</td>
</tr>
<tr>
<td>\(f_{\text{chan}}^{m,l}, f_{\text{chan}}^{p,l}\)<br>\((H_l \times H_l \times 128)\)</td>
<td>\(128(1^2 + 2^2) = 640 \to 4 \times 160\)<br>\(\{\bar{z}_h^m, \bar{z}_h^p\}_{h=1}^{H_s}, \quad H_s = 4\)</td>
<td>\(\mathcal{L}_C\ (1)\)</td>
</tr>
<tr>
<td rowspan="3"><strong>Downstream Fine-tuning<br>Classification Part</strong></td>
<td>\(\{f_m^{(l)}\}_{l=1}^5, \{f_p^{(l)}\}_{l=1}^5\)</td>
<td>\([\mathbb{E}_\Omega(f_*^{(1)}); \cdots; \mathbb{E}_\Omega(f_*^{(5)})]\)<br>\(64 + 256 + 512 + 1024 + 2048 = 3904\)</td>
<td>\(A_m\ (3904)\)<br>\(A_p\ (3904)\)</td>
</tr>
<tr>
<td>\(A_m\ (3904)\)<br>\(A_p\ (3904)\)</td>
<td>\([(g_m \odot A_m); (g_p \odot A_p)]\)</td>
<td>\(h_{\text{fused}}\ (7808)\)</td>
</tr>
<tr>
<td>\(h_{\text{fused}}\ (7808)\)</td>
<td>\(7808 \to 256 \to C\)</td>
<td>\(\mathbb{R}^C, \quad C=11\)</td>
</tr>
</tbody>
</table>

Table 1 lists the inputs, processing steps, and outputs of each module using multispectral patches of size \(16\times16\times4\) and panchromatic patches of size \(64\times64\times1\) as examples. Feature map sizes are given as height \(\times\) width \(\times\) number of channels.

### Dual-branch Encoder

Both branches use ResNet-50 with a modified input layer, and their parameters are learned independently. The network extracts features from the initial convolutional layer and the ends of the four residual stages, then passes these five levels of features to the corresponding feature differentiation units. Each group of three convolution rows in the table represents one residual bottleneck block. The number outside the parentheses gives its repeat count, and \(\downarrow2\) indicates a convolution with a stride of 2 that halves the spatial dimensions.

### Feature Differentiation Unit

This unit receives features from both modalities at the same level and prepares inputs for attribute inference and the different alignment paths. The multispectral features are first upsampled to match the spatial size of the panchromatic features. In the table, \((\uparrow4,\uparrow1)\) indicates that both the height and width of the multispectral features are multiplied by 4, while the panchromatic features retain their original size. The features of each modality then pass through a \(3\times3\) convolution, batch normalization, and an activation operation to obtain 128-channel base features \(f_{\mathrm{base}}^{m,l}\) and \(f_{\mathrm{base}}^{p,l}\). The base features from the two modalities are concatenated and passed through a spatial branch to generate a shared guidance map \(G^l\), which determines the locations on which structural alignment focuses. The base features also pass separately through \(1\times1\) convolutions with an orthogonality constraint to produce channel features \(f_{\mathrm{chan}}^{m,l}\) and \(f_{\mathrm{chan}}^{p,l}\) for semantic alignment.

### FAIM

This module determines the weights of the three alignment approaches from the base features at the current level. It first computes the local variation indicator \(\nu\), the cross-modal structural similarity indicator \(\rho\), and the channel diversity indicator \(\delta\). These indicators provide the basis for inferring texture, structure, and semantic attributes, respectively. The three indicators then pass through a prediction branch to produce learned scores for the three paths. These scores are combined with the dominance prior and attribute evolution bias, then passed through Gumbel-Softmax to obtain the path weights \(w_A,w_B,w_C\).

### ASAM

This module contains three alignment paths for texture, structure, and semantics, which compute \(\mathcal L_A\), \(\mathcal L_B\), and \(\mathcal L_C\), respectively. The texture path divides the base features into non-overlapping \(4\times4\) regions and averages the features within each region to obtain a 128-dimensional regional representation. It then applies contrastive learning to corresponding regions across the two modalities. In the table entry \((H_l^2/16)\times128\), the two dimensions represent the number of regions and the feature dimension of each region, respectively. The structure path selects the 64 locations with the highest responses in the shared guidance map \(G^l\). Features at the same locations are extracted from both modalities to form two \(64\times128\) feature matrices. This path constrains both the feature similarity at corresponding locations and the feature relationships among these locations. The semantic path pools the channel features of each modality to \(1\times1\) and \(2\times2\), then concatenates the results to obtain a \(128(1^2+2^2)=640\)-dimensional representation. This representation is divided into four 160-dimensional subspaces for semantic alignment. When the similarity between features from the two modalities is high, a relaxation mechanism weakens the constraint that pulls them closer together. The three path losses are combined using the weights produced by FAIM to form the pretraining loss function.

### Downstream Fine-tuning for Classification

After pretraining, the dual-branch encoder is used for supervised classification. Features from the five levels of each encoder are globally average-pooled and concatenated to obtain 3904-dimensional vectors \(A_m\) and \(A_p\). Learnable gating coefficients \(g_m\) and \(g_p\) adjust these vectors, which are then concatenated into a 7808-dimensional fused representation \(h_{\mathrm{fused}}\). A \(7808\rightarrow256\rightarrow C\) classifier outputs the classification scores, with \(C=11\) in the table. During fine-tuning, the encoders and classifier are jointly updated using a small number of labeled samples.
