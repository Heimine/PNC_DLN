---
title: "Understanding Deep Representation Learning via Layerwise Feature Compression and Discrimination"
permalink: /
layout: single
classes: wide
---

<p class="home-heading"><a href="#" aria-label="Back to homepage"><span aria-hidden="true">&larr;</span> Back to homepage</a></p>

<p class="button-row">
<a class="btn btn--success" href="https://github.com/Heimine/PNC_DLN"><i class="fab fa-github" aria-hidden="true"></i> Code</a>
<a class="btn btn--arxiv" href="https://arxiv.org/abs/2311.02960"><i class="fas fa-file-alt" aria-hidden="true"></i> arXiv</a>
<a class="btn btn--openreview" href="#"><i class="fas fa-book-open" aria-hidden="true"></i> JMLR</a>
</p>

<p class="author-row">
<a class="author-link" href="https://peng8wang.github.io/"><strong>Peng Wang</strong></a><sup>1,*</sup>, <a class="author-link" href="https://heimine.github.io/"><strong>Xiao Li</strong></a><sup>1,*</sup>, <a class="author-link" href="https://canyaras.com/">Can Yaras</a><sup>1</sup>, <a class="author-link" href="https://zhihuizhu.github.io/">Zhihui Zhu</a><sup>2</sup>, <a class="author-link" href="https://web.eecs.umich.edu/~girasole/">Laura Balzano</a><sup>1</sup>, <a class="author-link" href="https://weihu.me/">Wei Hu</a><sup>1</sup>, and <a class="author-link" href="https://qingqu.engin.umich.edu/"><strong>Qing Qu</strong></a><sup>1</sup>
</p>
<p class="affiliation-row">
<sup>1</sup>University of Michigan &nbsp;&middot;&nbsp; <sup>2</sup>Ohio State University
&nbsp;&middot;&nbsp; <sup>*</sup>Equal contribution
</p>

<p class="tldr-box"><strong>TL;DR.</strong> Each layer of a trained classification deep network compresses within-class features and discriminates between-class features at precise, predictable rates.</p>

---

<p class="lead-italic"><em>What happens inside a deep network, layer by layer?</em></p>

<p>Deep networks are believed to perform hierarchical feature learning: each layer builds on the last to extract increasingly meaningful structure from raw data. But despite a decade of empirical study, it remains unclear exactly how this happens — what quantitative pattern, if any, governs the transformation of features from shallow to deep layers.</p>

<p>Empirical studies have observed two consistent trends. First, intermediate layers <button class="inline-note-trigger" type="button" aria-expanded="false" aria-controls="note-expand" data-note-target="note-expand">expand and then compress</button> the intrinsic dimension of features as depth increases. Second, deep networks tend to produce features that are <em>within-class compressed</em> and <em>between-class discriminative</em> — a pattern closely related to the <button class="inline-note-trigger" type="button" aria-expanded="false" aria-controls="note-nc" data-note-target="note-nc">neural collapse</button> phenomenon observed at the final layer.</p>

<div id="note-expand" class="inline-note-body" hidden>
  <p>Initial layers expand the intrinsic dimension of features to make them linearly separable, while subsequent layers progressively compress them. See Alain and Bengio (2017), Ansuini et al. (2019), and Masarczyk et al. (2023).</p>
</div>
<div id="note-nc" class="inline-note-body" hidden>
  <p>Neural collapse refers to a phenomenon in which last-layer features from the same class become nearly identical, while features from different classes become maximally separated. See Papyan, Han, and Donoho (2020).</p>
</div>

<img class="feature-figure" src="{{ '/assets/figures/fig1_rank_acc.png' | relative_url }}" alt="Numerical rank and training accuracy plotted against layer index for an MLP and a hybrid network, showing rank rising then falling while accuracy saturates." width="85%" style="display:block;margin:auto;" />
<p class="figure-caption"><strong>Feature rank rises then falls, while accuracy saturates early.</strong> Initial layers make features linearly separable; later layers compress them further.</p>

<p>However, no theoretical framework has explained <em>why</em> this happens, or at what rate. We take a first step toward closing that gap.</p>

---

<p class="lead-italic"><em>Why deep linear networks?</em></p>

<p>Deep linear networks (DLNs) — networks with no nonlinear activation at all — might seem like an odd place to look for insight into nonlinear feature learning. But we find that linear layers placed deep in a network behave remarkably like their nonlinear counterparts: once early layers have made features linearly separable, the deeper layers' job is simply to compress and discriminate, a role that linear layers perform just as well.</p>

<img class="feature-figure" src="{{ '/assets/figures/fig2_umap.png' | relative_url }}" alt="UMAP visualization of features at layers 1, 2, 4, and 6 for a nonlinear MLP and a hybrid network, showing increasingly separated and compact class clusters." width="85%" style="display:block;margin:auto;" />
<p class="figure-caption"><strong>Linear layers replicate the deep-layer behavior of nonlinear networks.</strong> A hybrid network (nonlinear layers followed by linear layers) shows the same progressive class separation as a fully nonlinear MLP.</p>

<p>This lets us study a simple, tractable model — and the simplicity is not a weakness here, since DLNs are <em>over-parameterized</em>: stacking linear layers does not reduce to a single linear operator in any meaningful training sense, and depth in DLNs still meaningfully improves generalization, exactly as in their nonlinear cousins.</p>

---

<p class="lead-italic"><em>Two metrics: compression and discrimination</em></p>

<p>To make "feature evolution" precise, we define two metrics at each layer <span class="math-inline">\(l\)</span>. Let <span class="math-inline">\(\Sigma_W^l\)</span> and <span class="math-inline">\(\Sigma_B^l\)</span> denote the within-class and between-class covariance of features at that layer. We define:</p>

$$
C_l = \frac{\mathrm{Tr}(\Sigma_W^l)}{\mathrm{Tr}(\Sigma_B^l)}, \qquad D_l = 1 - \max_{k \neq k'} \frac{\langle \mu_k^l, \mu_{k'}^l \rangle}{\|\mu_k^l\| \|\mu_{k'}^l\|}.
$$

<p><span class="math-inline">\(C_l\)</span> measures within-class <em>compression</em> (smaller is more compressed), and <span class="math-inline">\(D_l\)</span> measures between-class <em>discrimination</em> (larger is more separated, computed from the smallest angle between any two class means).</p>

<p>Our main theoretical result, proved under <button class="inline-note-trigger" type="button" aria-expanded="false" aria-controls="note-assump" data-note-target="note-assump">mild assumptions</button> on the data and trained weights, is strikingly simple:</p>

<div id="note-assump" class="inline-note-body" hidden>
  <p>We assume the training data is nearly orthonormal (a common simplifying assumption for studying implicit bias of gradient descent), and that the trained weights are minimum-norm, approximately balanced, and approximately low-rank — properties that gradient descent is known to favor.</p>
</div>

<p class="tldr-box"><em>The compression metric <strong>decays at a geometric rate</strong>, while the discrimination metric <strong>increases at a linear rate</strong>, with respect to the number of layers.</em></p>

<p>More precisely, under these assumptions, the ratio of within-class compression between consecutive layers satisfies</p>

$$
\frac{C_{l+1}}{C_l} = O\!\left(\frac{\varepsilon^2}{n^{1/L}}\right),
$$

<p>while the between-class discrimination metric satisfies</p>

$$
D_l \geq 1 - O\!\left(\frac{\theta + \delta}{L}\right) \cdot (\text{linear in } l).
$$

<p>Here <span class="math-inline">\(n\)</span> is the number of samples per class, <span class="math-inline">\(L\)</span> is the network depth, and <span class="math-inline">\(\varepsilon,\theta,\delta\)</span> are the small constants from the assumptions above, controlling the data's near-orthogonality and the weights' balancedness and low-rankness. In words: with more samples or smaller residual constants, compression happens faster per layer; discrimination grows steadily and linearly regardless.</p>

<img class="feature-figure" src="{{ '/assets/figures/synthetic_data.png' | relative_url }}" alt="Compression and discrimination metrics plotted against layer index for a linear network and a nonlinear network, both showing geometric decay and linear increase respectively." width="85%" style="display:block;margin:auto;" />
<p class="figure-caption"><strong>Theory matches practice in both linear and nonlinear networks.</strong> Compression decays geometrically (log-scale plot, left); discrimination increases linearly (right) — for both a DLN and a trained MLP.</p>

---

<p class="lead-italic"><em>This pattern holds well beyond the assumptions</em></p>

<p>We validate the theorem with synthetic data matching our assumptions exactly, then show the same trend holds approximately under default PyTorch initialization, on real datasets (CIFAR-10, FashionMNIST), and even in nonlinear MLPs of varying depth — and beyond images, on a text classification benchmark.</p>

<img class="feature-figure" src="{{ '/assets/figures/real_data.png' | relative_url }}" alt="Within-class compression metric decaying approximately geometrically across layers for networks of varying depths (L=5,7,9) trained on FashionMNIST and CIFAR-10 with default initialization." width="75%" style="display:block;margin:auto;" />
<p class="figure-caption"><strong>Progressive compression persists with real data and default initialization, across network depths.</strong> Our theoretical assumptions are a simplification — the pattern survives without them.</p>

---

<p class="lead-italic"><em>Why this matters</em></p>

<p>Beyond characterizing a previously undescribed pattern, our result has concrete implications:</p>

<ul>
  <li><strong>Neural collapse, without the unconstrained features model.</strong> Most prior neural collapse analyses treat last-layer features as free optimization variables, ignoring the network's hierarchical structure. Our theorem shows that sufficiently deep linear networks naturally exhibit neural collapse as a <em>consequence</em> of layerwise dynamics — without that assumption.</li>
  <li><strong>Why projection heads help transfer learning.</strong> Contrastive learning methods discard a few extra MLP layers (the "projection head") after pretraining, keeping only the features before them. Our theory explains why: features closer to the final layer are more collapsed and thus less transferable, so backing off by a few layers preserves more usable structure.</li>
  <li><strong>Guidance on architecture depth.</strong> Since compression and discrimination both strengthen with depth, deeper networks better separate data — but only up to a point, since over-compression can hurt out-of-distribution generalization.</li>
</ul>

<img class="feature-figure" src="{{ '/assets/figures/projection_head.png' | relative_url }}" alt="Diagram of projection head usage during pretraining and transfer, alongside a plot showing transfer accuracy improving and feature compression decreasing as more projection head layers are added." width="85%" style="display:block;margin:auto;" />
<p class="figure-caption"><strong>More projection head layers mean less feature collapse and better transfer accuracy.</strong> This empirically confirms the mechanism predicted by our theory.</p>

---

<p class="lead-italic"><em>In short</em></p>

<p>Even though deep linear networks lack the expressive power of their nonlinear counterparts, they are powerful enough to reveal — and prove — a precise, quantitative law governing how deep networks transform data into discriminative representations, layer by layer.</p>