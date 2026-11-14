# FEAAN
Our code is released [here](https://github.com/LiuShutong04/FEAAN-main) !

# Network Architecture

The network consists of a dual-branch encoder, feature differentiation units, FAIM, ASAM, and a downstream classification component for fine-tuning. During pretraining, attribute inference and attribute-specific alignment support the learning of cross-modal representations. During fine-tuning, the pretrained encoder performs joint classification under the supervision of a small set of labeled samples.

\begin{table*}[htbp]
	\centering
	\caption{Network Architecture Details of the Proposed Method}
	\label{tab:network_architecture}
	\renewcommand{\arraystretch}{1.5}
	\resizebox{\textwidth}{!}{
		\begin{tabular}{cccc}
			\toprule
			\textbf{Module} & \textbf{Input} & \textbf{Network Architecture} & \textbf{Output} \\
			\midrule
			\multirow{5}{*}{Dual-branch Encoder} 
			& \(I_m\ (16 \times 16 \times 4)\) \newline \(I_p\ (64 \times 64 \times 1)\) 
			& \(\text{Conv}[3 \times 3, 64] \to \text{BN} \to \text{ReLU}\) 
			& \(f_m^{(1)}\ (16 \times 16 \times 64)\) \newline \(f_p^{(1)}\ (64 \times 64 \times 64)\) \\
			\cmidrule{2-4}
			& \(f_m^{(1)}\ (16 \times 16 \times 64)\) \newline \(f_p^{(1)}\ (64 \times 64 \times 64)\) 
			& \(\left[ \begin{array}{c} 1 \times 1 \quad 64 \\ 3 \times 3 \quad 64 \\ 1 \times 1 \quad 256 \end{array} \right] \times 3\) 
			& \(f_m^{(2)}\ (16 \times 16 \times 256)\) \newline \(f_p^{(2)}\ (64 \times 64 \times 256)\) \\
			\cmidrule{2-4}
			& \(f_m^{(2)}\ (16 \times 16 \times 256)\) \newline \(f_p^{(2)}\ (64 \times 64 \times 256)\) 
			& \(\left[ \begin{array}{c} 1 \times 1 \quad 128 \\ 3 \times 3 \downarrow 2 \quad 128 \\ 1 \times 1 \quad 512 \end{array} \right] \to \left[ \begin{array}{c} 1 \times 1 \quad 128 \\ 3 \times 3 \quad 128 \\ 1 \times 1 \quad 512 \end{array} \right] \times 3\) 
			& \(f_m^{(3)}\ (8 \times 8 \times 512)\) \newline \(f_p^{(3)}\ (32 \times 32 \times 512)\) \\
			\cmidrule{2-4}
			& \(f_m^{(3)}\ (8 \times 8 \times 512)\) \newline \(f_p^{(3)}\ (32 \times 32 \times 512)\) 
			& \(\left[ \begin{array}{c} 1 \times 1 \quad 256 \\ 3 \times 3 \downarrow 2 \quad 256 \\ 1 \times 1 \quad 1024 \end{array} \right] \to \left[ \begin{array}{c} 1 \times 1 \quad 256 \\ 3 \times 3 \quad 256 \\ 1 \times 1 \quad 1024 \end{array} \right] \times 5\) 
			& \(f_m^{(4)}\ (4 \times 4 \times 1024)\) \newline \(f_p^{(4)}\ (16 \times 16 \times 1024)\) \\
			\cmidrule{2-4}
			& \(f_m^{(4)}\ (4 \times 4 \times 1024)\) \newline \(f_p^{(4)}\ (16 \times 16 \times 1024)\) 
			& \(\left[ \begin{array}{c} 1 \times 1 \quad 512 \\ 3 \times 3 \downarrow 2 \quad 512 \\ 1 \times 1 \quad 2048 \end{array} \right] \to \left[ \begin{array}{c} 1 \times 1 \quad 512 \\ 3 \times 3 \quad 512 \\ 1 \times 1 \quad 2048 \end{array} \right] \times 2\) 
			& \(f_m^{(5)}\ (2 \times 2 \times 2048)\) \newline \(f_p^{(5)}\ (8 \times 8 \times 2048)\) \\
			\midrule
			\multirow{3}{*}{Feature Differentiation Unit} 
			& \(\{f_m^{(l)}, f_p^{(l)}\}\) \newline \(l=1,2,3,4,5\) 
			& \((\uparrow 4, \uparrow 1)\) \newline \(\text{Conv}[3 \times 3, 128] \to \text{BN} \to \text{ReLU}\) 
			& \begin{tabular}{@{}c@{}}\(\{f_{\text{base}}^{m,l}, f_{\text{base}}^{p,l}\}\ (H_l \times H_l \times 128)\) \\ \((H_1, H_2, H_3, H_4, H_5)\) \\ \(=(64, 64, 32, 16, 8)\)\end{tabular} \\
			\cmidrule{2-4}
			& \begin{tabular}{@{}c@{}}\(f_{\text{base}}^{m,l}, f_{\text{base}}^{p,l}\) \\ \((H_l \times H_l \times 256)\)\end{tabular} 
			& \(\text{Conv}^{(\text{spa})}[1 \times 1, 1] \to \sigma\) 
			& \(G^l\ (H_l \times H_l \times 1)\) \\
			\cmidrule{2-4}
			& \begin{tabular}{@{}c@{}}\(f_{\text{base}}^{m,l}, f_{\text{base}}^{p,l}\) \\ \((H_l \times H_l \times 128)\)\end{tabular} 
			& \begin{tabular}{@{}c@{}}\(\text{Conv}^{(\text{orth})}[1 \times 1, 128]\) \\ \(W_{\text{orth}} \in \mathbb{R}^{128 \times 128}\)\end{tabular} 
			& \begin{tabular}{@{}c@{}}\(\{f_{\text{chan}}^{m,l}, f_{\text{chan}}^{p,l}\}\) \\ \((H_l \times H_l \times 128)\)\end{tabular} \\
			\midrule
			\multirow{3}{*}{FAIM} 
			& \begin{tabular}{@{}c@{}}\(f_{\text{base}}^{m,l}, f_{\text{base}}^{p,l}\) \\ \((H_l \times H_l \times 128)\)\end{tabular} 
			& \begin{tabular}{@{}c@{}}\(\mathbb{E}_\Omega, \sigma_N(x), \cos(\cdot, \cdot), \text{CV}\) \\ \(r, g(r), h(r)\)\end{tabular} 
			& \([\nu, \rho, \delta] \in \mathbb{R}^3\) \\
			\cmidrule{2-4}
			& \([\nu, \rho, \delta] \in \mathbb{R}^3\) 
			& \begin{tabular}{@{}c@{}}\((1 \to 16 \to 16) \times 3 \to 48\) \\ \( 48 \to 32 \to 3 \)\end{tabular} 
			& \begin{tabular}{@{}c@{}}\(z_{\text{learned}} \in \mathbb{R}^3\) \end{tabular} \\
			\cmidrule{2-4}
			& \begin{tabular}{@{}c@{}}\(z_{\text{learned}}\), \(z_{\text{prior}}\), \(b(p)\) \\ \(z_{\text{prior}} = [z_A, z_B, z_C]\)\end{tabular} 
			& \(h(z_{\text{learned}} + \lambda_r z_{\text{prior}} + \lambda_b b(p))\) 
			& \(w = [w_A, w_B, w_C] \in \mathbb{R}^3\) \\
			\midrule
			\multirow{3}{*}{ASAM} 
			& \begin{tabular}{@{}c@{}}\(f_{\text{base}}^{m,l}, f_{\text{base}}^{p,l}\) \\ \((H_l \times H_l \times 128)\)\end{tabular} 
			& \begin{tabular}{@{}c@{}}\(U_4 \to \frac{1}{16} \sum_{k=1}^{16}(\cdot)_k \to \{\tilde{z}_m; \tilde{z}_p\}\) \\ \((H_l^2/16) \times 128\)\end{tabular} 
			& \(\mathcal{L}_A\ (1)\) \\
			\cmidrule{2-4}
			& \(f_{\text{base}}^{m,l}, f_{\text{base}}^{p,l}\), \(G^l\) 
			& \begin{tabular}{@{}c@{}}\(\text{TopK}(G^l, 64) \to \{Z_m, Z_p\}\) \\ \(64 \times 128\)\end{tabular} 
			& \(\mathcal{L}_B\ (1)\) \\
			\cmidrule{2-4}
			& \begin{tabular}{@{}c@{}}\(f_{\text{chan}}^{m,l}, f_{\text{chan}}^{p,l}\) \\ \((H_l \times H_l \times 128)\)\end{tabular} 
			& \begin{tabular}{@{}c@{}}\(128(1^2 + 2^2) = 640 \to 4 \times 160\) \\ \(\{\bar{z}_h^m, \bar{z}_h^p\}_{h=1}^{H_s}, \quad H_s = 4\)\end{tabular} 
			& \(\mathcal{L}_C\ (1)\) \\
			\midrule
			\multirow{3}{*}{\begin{tabular}{@{}c@{}}Downstream Fine-tuning \\ Classification Part\end{tabular}} 
			& \(\{f_m^{(l)}\}_{l=1}^5, \{f_p^{(l)}\}_{l=1}^5\) 
			& \begin{tabular}{@{}c@{}}\([\mathbb{E}_\Omega(f_*^{(1)}); \cdots; \mathbb{E}_\Omega(f_*^{(5)})]\) \\ \(64 + 256 + 512 + 1024 + 2048 = 3904\)\end{tabular} 
			& \begin{tabular}{@{}c@{}}\(A_m\ (3904)\) \\ \(A_p\ (3904)\)\end{tabular} \\
			\cmidrule{2-4}
			& \begin{tabular}{@{}c@{}}\(A_m\ (3904)\) \\ \(A_p\ (3904)\)\end{tabular} 
			& \([(g_m \odot A_m); (g_p \odot A_p)]\) 
			& \(h_{\text{fused}}\ (7808)\) \\
			\cmidrule{2-4}
			& \(h_{\text{fused}}\ (7808)\) 
			& \(7808 \to 256 \to C\) 
			& \(\mathbb{R}^C, \quad C=11\) \\
			\bottomrule
		\end{tabular}
	}
\end{table*}

Table~\ref{tab:network_architecture} lists the inputs, processing steps, and outputs of each module using multispectral patches of size $16\times16\times4$ and panchromatic patches of size $64\times64\times1$ as examples. Feature map sizes are given as height $\times$ width $\times$ number of channels.

Dual-branch encoder: Both branches use ResNet-50 with a modified input layer, and their parameters are learned independently. The network extracts features from the initial convolutional layer and the ends of the four residual stages, then passes these five levels of features to the corresponding feature differentiation units. Each group of three convolution rows in the table represents one residual bottleneck block. The number outside the parentheses gives its repeat count, and $\downarrow2$ indicates a convolution with a stride of 2 that halves the spatial dimensions.

Feature differentiation unit: This unit receives features from both modalities at the same level and prepares inputs for attribute inference and the different alignment paths. The multispectral features are first upsampled to match the spatial size of the panchromatic features. In the table, $(\uparrow4,\uparrow1)$ indicates that both the height and width of the multispectral features are multiplied by 4, while the panchromatic features retain their original size. The features of each modality then pass through a $3\times3$ convolution, batch normalization, and an activation operation to obtain 128-channel base features $f_{\mathrm{base}}^{m,l}$ and $f_{\mathrm{base}}^{p,l}$. The base features from the two modalities are concatenated and passed through a spatial branch to generate a shared guidance map $G^l$, which determines the locations on which structural alignment focuses. The base features also pass separately through $1\times1$ convolutions with an orthogonality constraint to produce channel features $f_{\mathrm{chan}}^{m,l}$ and $f_{\mathrm{chan}}^{p,l}$ for semantic alignment.

FAIM: This module determines the weights of the three alignment approaches from the base features at the current level. It first computes the local variation indicator $\nu$, the cross-modal structural similarity indicator $\rho$, and the channel diversity indicator $\delta$. These indicators provide the basis for inferring texture, structure, and semantic attributes, respectively. The three indicators then pass through a prediction branch to produce learned scores for the three paths. These scores are combined with the dominance prior and attribute evolution bias, then passed through Gumbel-Softmax to obtain the path weights $w_A,w_B,w_C$.

ASAM: This module contains three alignment paths for texture, structure, and semantics, which compute $\mathcal L_A$, $\mathcal L_B$, and $\mathcal L_C$, respectively. The texture path divides the base features into non-overlapping $4\times4$ regions and averages the features within each region to obtain a 128-dimensional regional representation. It then applies contrastive learning to corresponding regions across the two modalities. In the table entry $(H_l^2/16)\times128$, the two dimensions represent the number of regions and the feature dimension of each region, respectively. The structure path selects the 64 locations with the highest responses in the shared guidance map $G^l$. Features at the same locations are extracted from both modalities to form two $64\times128$ feature matrices. This path constrains both the feature similarity at corresponding locations and the feature relationships among these locations. The semantic path pools the channel features of each modality to $1\times1$ and $2\times2$, then concatenates the results to obtain a $128(1^2+2^2)=640$-dimensional representation. This representation is divided into four 160-dimensional subspaces for semantic alignment. When the similarity between features from the two modalities is high, a relaxation mechanism weakens the constraint that pulls them closer together. The three path losses are combined using the weights produced by FAIM to form the pretraining loss function.

Downstream fine-tuning for classification: After pretraining, the dual-branch encoder is used for supervised classification. Features from the five levels of each encoder are globally average-pooled and concatenated to obtain 3904-dimensional vectors $A_m$ and $A_p$. Learnable gating coefficients $g_m$ and $g_p$ adjust these vectors, which are then concatenated into a 7808-dimensional fused representation $h_{\mathrm{fused}}$. A $7808\rightarrow256\rightarrow C$ classifier outputs the classification scores, with $C=11$ in the table. During fine-tuning, the encoders and classifier are jointly updated using a small number of labeled samples.
