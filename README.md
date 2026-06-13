# Awesome One-Step Generation

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

<p align="center">
  <em>"Nature is pleased with simplicity."</em><br>
  — Isaac Newton
</p>

> A curated list of papers, code, and resources for **one-step diffusion models** — methods that turn noise into high-quality samples in a *single* neural network forward pass.
>
> Unlike most lists that simply split work into "native vs. distillation", this list is organized by the **core mathematical principle that makes one step generation possible**. Within each principle we then separate 🟢 **native** (from-scratch) and 🔵 **distillation** (teacher-dependent) variants.

---

## 🧭 Classification Rationale

Fast-sampling research has historically been catalogued either by application (text-to-image, super-resolution, video) or by a coarse *native vs. distillation* dichotomy. Both axes obscure the deeper question: **what mathematical mechanism actually collapses an iterative sampler into one step?** This list answers that question by grouping methods according to their *primary theoretical contribution*.

This taxonomy is grounded in recent unifying literature. **Uni-Instruct** (NeurIPS 2025) shows that a large family of one-step diffusion methods — DMD, SiD, Diff-Instruct, score-based and variational approaches — are all special cases of **diffusion divergence minimization**, differing only in the choice of divergence. This motivates our dedicated *Distribution Divergence Minimization* category with sub-axes for Forward KL, Reverse KL, Fisher divergence, and variational bounds. **One Step Diffusion via Shortcut Models** (ICLR 2025 Oral) explicitly frames direct noise-to-data mapping as a distinct paradigm separate from trajectory-based reasoning, justifying the *Shortcut / Direct Mapping* category. And **iCT** (ICLR 2024) demonstrates that consistency *training* (CT) and consistency *distillation* (CD) are two regimes of one principle — self-consistency along the PF-ODE — motivating our native/distillation split *inside* each principle rather than across the whole list.

We classify each paper by its **primary contribution**. Many works combine several principles (e.g. adversarial losses layered on score distillation); cross-references are added where a method materially spans categories. Within every category, entries are ordered chronologically (oldest → newest) so the conceptual lineage is easy to follow.

| Legend | Meaning |
|:------:|---------|
| 🟢 **Native** | Trained from scratch / teacher-free; one-step is a native training objective |
| 🔵 **Distill** | Distills a pretrained multi-step diffusion teacher into one/few steps |
| 🟡 **Hybrid** | Supports both native training and distillation |

---

## Table of Contents

- [Consistency Models](#consistency-models) — *Consistency Models, iCT, LCM, CTM*
- [Flow Straightening](#flow-straightening) — *Rectified Flow, Flow Matching, MeanFlow, InstaFlow*
- [Distribution Divergence Minimization](#distribution-divergence-minimization) — *DMD, SiD, Diff-Instruct, SwiftBrush*
  - [Forward KL / Regression](#forward-kl--regression)
  - [Reverse KL / Distribution Matching](#reverse-kl--distribution-matching)
  - [Fisher Divergence / Score Matching](#fisher-divergence--score-matching)
  - [Variational Bounds](#variational-bounds)
- [Adversarial Training](#adversarial-training) — *ADD, LADD, YOSO, NitroFusion*
- [Shortcut / Direct Mapping](#shortcut--direct-mapping) — *Shortcut Models, D2O, NCT*
- [Applications](#applications)
  - [Image Super-Resolution](#image-super-resolution)
  - [Video Generation](#video-generation)
  - [Controllable Generation](#controllable-generation)
  - [Other Applications](#other-applications)
- [Unified Frameworks & Theory](#unified-frameworks--theory)
- [Full Paper List](#full-paper-list-newest--oldest)
- [Related Resources](#related-resources)
- [Contributing](#contributing)

---

## Consistency Models

> **Core principle — self-consistency along the PF-ODE.** Any two points on the same probability-flow ODE trajectory must map to the *same* endpoint. Enforcing this consistency constraint lets a network jump directly from noise to data. Following iCT, the principle admits both consistency *training* (CT, 🟢 native) and consistency *distillation* (CD, 🔵 teacher-based).

### Native (🟢)

- **Consistency Models** [ICML 2023] 🟡  
  *Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever*  
  [[Paper](https://arxiv.org/abs/2303.01469)]  
  Self-consistency for direct noise-to-data mapping; supports both distillation (CD) and native training (CT).

- **Improved Techniques for Training Consistency Models (iCT)** [ICLR 2024] 🟢  
  *OpenAI*  
  [[Paper](https://arxiv.org/abs/2310.14189)]  
  Makes consistency training from scratch competitive with distillation.

- **Improving Consistency Models with Generator-Augmented Flows** [ICML 2025] 🟢  
  [[Paper](https://arxiv.org/abs/2406.09570)] [[Code](https://github.com/thibautissenhuth/consistency)]  
  Generator-augmented flows for improved consistency models.

- **Consistency Models Made Easy** [ICLR 2025] 🟢  
  [[Paper](https://arxiv.org/abs/2406.14548)]  
  Simplified and efficient consistency-model training recipe.

- **Understanding and Improving Consistency Models** [ICLR 2025] 🟢  
  [[Paper](https://arxiv.org/abs/2410.18958)] [[Code](https://github.com/G-U-N/Stable-Consistency-Tuning)]  
  An MDP framework for understanding and improving consistency models.

- **Inverse Flow and Consistency Models** [ICML 2025] 🟢  
  [[Paper](https://arxiv.org/abs/2502.11333)]  
  Inverse-flow approach for consistency model training.

- **Multi-step Consistency Models** [2025] 🟢  
  [[Paper](https://arxiv.org/abs/2505.01049)]  
  Theoretical analysis and multi-step inference for consistency models.

- **How to Build a Consistency Model: Learning Flow Maps via Self-Distillation** [NeurIPS 2025] 🟡  
  *Nicholas Boffi, et al.*  
  [[Paper](https://arxiv.org/abs/2505.18825)] [[Code](https://github.com/nmboffi/flow-maps)]  
  A systematic approach to learning flow maps via self-distillation.

### Distillation (🔵)

- **Latent Consistency Models (LCM)** [ICLR 2024] 🔵  
  *Simian Luo, et al.*  
  [[Paper](https://arxiv.org/abs/2310.04378)]  
  Consistency distillation in latent space from pretrained Stable Diffusion.

- **Consistency Trajectory Models (CTM)** [ICLR 2024] 🔵  
  [[Paper](https://arxiv.org/abs/2310.02279)] [[Code](https://github.com/sony/ctm)]  
  Learning the probability-flow ODE trajectory for consistency.

- **TLCM: Training-efficient Latent Consistency Model** [2024] 🔵  
  *OPPO Mente Lab*  
  [[Paper](https://arxiv.org/abs/2406.05768)] [[Code](https://github.com/OPPO-Mente-Lab/TLCM)]  
  Two-stage data-free consistency distillation.

- **SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation** [ICCV 2025] 🔵  
  *NVIDIA*  
  [[Paper](https://arxiv.org/abs/2503.09641)]  
  Continuous-time consistency distillation for ultra-fast text-to-image.

- **Simple Distillation for One-Step Diffusion Models (CED)** [NeurIPS 2025] 🔵  
  [[Paper](https://arxiv.org/abs/2502.00883)]  
  Contrastive energy distillation for one-step diffusion.

---

## Flow Straightening

> **Core principle — straighten the transport path.** If the trajectory connecting noise and data is (close to) a straight line, a single Euler step approximates the full ODE integration. Rectified-flow and flow-matching methods learn or refine such straight paths so that one-step Euler ≈ the complete ODE solution.

### Native (🟢)

> **Rectified Flow** and **Flow Matching** are catalogued under [Related Resources → Foundational Works](#foundational-works) as general frameworks. The methods below specifically target or achieve one-step generation.

- **Optimal Flow Matching: Learning Straight Trajectories in Just One Step** [NeurIPS 2024] 🟢  
  *Alexander Korotin, et al.*  
  [[Paper](https://arxiv.org/abs/2403.13117)] [[Code](https://github.com/Jhomanik/Optimal-Flow-Matching)]  
  Restricting flow matching to obtain one-step straight trajectories.

- **Improving the Training of Rectified Flows** [NeurIPS 2024] 🟢  
  [[Paper](https://arxiv.org/abs/2405.20320)] [[Code](https://github.com/sangyun884/rfpp)]  
  Improved rectified-flow training enabling stronger one-step performance.

- **End-to-End Single-Step Flow Matching via Direct Models (FlowFit)** [ICLR 2025] 🟢  
  [[Paper](https://arxiv.org/abs/2502.01441)]  
  Direct model learning for single-step flow matching.

- **Mean Flows for One-step Generative Modeling (MeanFlow)** [NeurIPS 2025] 🟢  
  *Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, Kaiming He*  
  [[Paper](https://arxiv.org/abs/2505.13447)] [[Code](https://github.com/haidog-yaqub/MeanFlow)]  
  Native one-step generation via mean-velocity-field properties; trains from scratch.

- **ODE-free Neural Flow Matching for One-Step Generative Modeling** [2025] 🟢  
  [[Paper](https://arxiv.org/abs/2604.06413)]  
  ODE-free neural flow matching that achieves one-step generation.

- **Blockwise Flow Matching (BFM)** [NeurIPS 2025] 🟢  
  [[Paper](https://arxiv.org/abs/2506.14603)]  
  Block-level flow matching for efficient high-quality generation.

- **SoFlow: Solution Flow Models for One-Step Generative Modeling** [ICLR 2026] 🟢  
  *Princeton ZLab*  
  [[Paper](https://arxiv.org/abs/2512.15657)] [[Code](https://github.com/zlab-princeton/SoFlow)]  
  Learning the solution map of flow ODEs for direct one-step generation.

### Distillation (🔵)

> **Progressive Distillation** is catalogued under [Related Resources → Foundational Works](#foundational-works) as it is a general step-compression method rather than a one-step-specific technique. It is cross-referenced here as it inspired much of the flow-distillation lineage.

- **InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation** [ICLR 2024] 🔵  
  *Xingchao Liu, et al.*  
  [[Paper](https://arxiv.org/abs/2309.06380)] [[Code](https://github.com/gnobitab/InstaFlow)]  
  Rectified-Flow reflow technique for one-step text-to-image.

- **Training Smaller One-Step Diffusion Models with Rectified Flow (SlimFlow)** [ECCV 2024] 🔵  
  *Yuanzhi Zhu, et al.*  
  [[Paper](https://arxiv.org/abs/2407.12718)] [[Code](https://github.com/yuanzhi-zhu/SlimFlow)]  
  Parameter-efficient one-step models via rectified-flow compression.

- **Self-Corrected Flow Distillation (SCFlow)** [AAAI 2025] 🔵  
  *Hao Phung, et al.*  
  [[Paper](https://arxiv.org/abs/2412.16906)] [[Code](https://github.com/hao-pt/SCFlow)]  
  Integrates consistency models and adversarial training within the flow-matching framework for one-step/few-step generation.  
  *(Cross-ref: also uses [Consistency](#consistency-models) and [Adversarial](#adversarial-training) principles.)*

- **Learning Few-Step Diffusion Models by Trajectory Distribution Matching (TDM)** [ICCV 2025] 🔵  
  *Yihong Luo, et al.*  
  [[Paper](https://arxiv.org/abs/2503.06674)] [[Code](https://github.com/Luo-Yihong/TDM)]  
  Unified trajectory and distribution matching for few-step distillation.

- **Align Your Flow: Scaling Continuous-Time Flow Map Distillation** [NeurIPS 2025] 🔵  
  *NVIDIA Toronto AI*  
  [[Paper](https://arxiv.org/abs/2506.14603)]  
  Continuous-time objectives for training flow maps at scale.

---

## Distribution Divergence Minimization

> **Core principle — minimize a statistical divergence between distributions.** These methods train the one-step generator so that its output distribution `p_θ` matches the target distribution `p_target` by minimizing some divergence `D(p_θ ∥ p_target)`. As shown by **Uni-Instruct** (NeurIPS 2025), DMD, SiD, Diff-Instruct and many others are special cases of a single *diffusion divergence minimization* framework; they differ chiefly in **which divergence** they minimize. We organize them along that axis.

### Forward KL / Regression

> Minimizing the forward KL `D(p_target ∥ p_θ)` is mode-covering and reduces to maximum-likelihood / regression-style objectives.

- **Progressive Distillation** is catalogued under [Related Resources → Foundational Works](#foundational-works), since its core contribution is *step compression* rather than a divergence-minimization result. It is cross-referenced here because its regression-on-teacher objective is forward-KL in spirit.

- **BOOT: Data-free Distillation of Denoising Diffusion Models with Bootstrapping** [ICML 2023] 🔵  
  *Apple ML Research*  
  [[Paper](https://arxiv.org/abs/2306.05544)]  
  Data-free bootstrapping distillation via sequential prediction on the signal ODE.

- **EM Distillation for One-step Diffusion Models** [NeurIPS 2024] 🔵  
  *Ruiqi Gao, et al.*  
  [[Paper](https://arxiv.org/abs/2405.16852)]  
  Maximum-likelihood (forward-KL) distillation via an Expectation-Maximization formulation.

### Reverse KL / Distribution Matching

> Minimizing the reverse KL `D(p_θ ∥ p_target)` is mode-seeking; in practice it is estimated through the score functions of the real and the generator ("fake") distributions.

- **One-step Diffusion with Distribution Matching Distillation (DMD)** [CVPR 2024] 🔵  
  *Tianwei Yin, et al.*  
  [[Paper](https://arxiv.org/abs/2311.18828)]  
  Learning score functions of real and fake distributions for reverse-KL distribution matching.

- **Improved Distribution Matching Distillation (DMD2)** [NeurIPS 2024 Oral] 🔵  
  *Tianwei Yin, et al.*  
  [[Paper](https://arxiv.org/abs/2405.14867)] [[Code](https://github.com/tianweiy/dmd2)]  
  Improved DMD that removes the costly regression loss.

- **One-step Diffusion Models with f-Divergence Distribution Matching (f-distill)** [2025] 🔵  
  *NVIDIA Gen AI Lab*  
  [[Paper](https://arxiv.org/abs/2502.15681)]  
  A generalized f-divergence framework subsuming reverse-KL distribution matching.

### Fisher Divergence / Score Matching

> Matching the *scores* (gradients of log-density) of the generator and target corresponds to minimizing a Fisher-type divergence — the basis of score-identity and score-implicit methods.

- **Score identity Distillation (SiD)** [ICML 2024] 🔵  
  *Mingyuan Zhou, et al.*  
  [[Paper](https://arxiv.org/abs/2404.04057)] [[Code](https://github.com/mingyuanzhou/SiD)]  
  Data-free score-identity matching for exponentially fast distillation.

- **One-Step Diffusion Distillation through Score Implicit Matching (SIM)** [NeurIPS 2024] 🔵  
  [[Paper](https://arxiv.org/abs/2410.16794)] [[Code](https://github.com/maple-research-lab/SIM)]  
  Score implicit matching that preserves the teacher's denoising capability.

- **DiffRatio: Training One-Step Diffusion Models Without Teacher Supervision** [ICLR 2025] 🟢  
  *Wenlin Chen, et al.*  
  [[Paper](https://arxiv.org/abs/2502.08005)] [[Code](https://github.com/Wenlin-Chen/DiffRatio)]  
  Direct score-ratio estimation for teacher-free one-step training.

- **Guided Score identity Distillation (SiD-LSG)** [ICLR 2025] 🔵  
  *Mingyuan Zhou, et al.*  
  [[Paper](https://arxiv.org/abs/2406.01561)] [[Code](https://github.com/mingyuanzhou/SiD-LSG)]  
  Long-short classifier-free guidance enhanced SiD.

- **Adversarial Score identity Distillation (SiDA)** [ICLR 2025] 🔵  
  *Mingyuan Zhou, et al.*  
  [[Paper](https://arxiv.org/abs/2410.14919)] [[Code](https://github.com/mingyuanzhou/SiD/tree/sida)]  
  Adversarial enhancement of SiD, surpassing the teacher model.

- **Score-of-Mixture Training** [ICML 2025 Spotlight] 🟢  
  *Jongha Ryu, et al.*  
  [[Paper](https://arxiv.org/abs/2502.09609)]  
  Minimizing mixture-distribution divergence for pretrain-free one-step generation.

### Variational Bounds

> When the divergence cannot be optimized directly, a tractable *variational bound* (e.g. variational score distillation) is minimized instead, using a learned score/critic to guide the generator.

- **Diff-Instruct: A Universal Approach for Transferring Knowledge from Pre-trained Diffusion Models** [NeurIPS 2023] 🔵  
  *Weijian Luo, et al.*  
  [[Paper](https://arxiv.org/abs/2305.18455)]  
  Universal framework using variational bounds to guide any generator; conceptual precursor to many one-step methods.

- **SwiftBrush: One-Step Text-to-Image Diffusion Model with Variational Score Distillation** [CVPR 2024] 🔵  
  *Khuong Nguyen, et al.*  
  [[Paper](https://arxiv.org/abs/2312.05239)] [[Code](https://github.com/VinAIResearch/SwiftBrush)]  
  Image-free variational score distillation for one-step text-to-image.

- **SwiftBrush v2: Make Your One-step Model Better Than Its Teacher** [ECCV 2024] 🔵  
  [[Paper](https://arxiv.org/abs/2408.14176)] [[Code](https://github.com/vinairesearch/swiftbrushv2)]  
  Improved SwiftBrush that surpasses teacher-model performance.

- **Diff-Instruct++: Training One-step Text-to-image Generator Model to Human-Preferred** [2024] 🔵  
  [[Paper](https://arxiv.org/abs/2410.18881)] [[Code](https://github.com/pkulwj1994/diff_instruct_pp)]  
  Human-preference alignment for one-step text-to-image generators.

---

## Adversarial Training

> **Core principle — push samples onto the data manifold with a discriminator.** Borrowing the GAN principle, an adversarial discriminator drives generated samples toward the real-data distribution, yielding sharp, high-fidelity outputs in a single step. Often combined with score- or consistency-based objectives.

- **You Only Sample Once (YOSO)** [ICLR 2024] 🟢  
  *Yihong Luo, Xiaolong Chen, Xinghua Qu, et al.*  
  [[Paper](https://arxiv.org/abs/2403.12931)] [[Code](https://github.com/Luo-Yihong/YOSO)]  
  Self-cooperative diffusion-GAN for one-step text-to-image, trained without a frozen teacher.

- **Adversarial Diffusion Distillation (ADD)** [ECCV 2024] 🔵  
  *Stability AI*  
  [[Paper](https://arxiv.org/abs/2311.17042)]  
  Adversarial training for distilling diffusion into 1–4 step generators (SDXL-Turbo).

- **Latent Adversarial Diffusion Distillation (LADD)** [ECCV 2024] 🔵  
  *Axel Sauer, Frederic Boesel, et al.*  
  [[Paper](https://arxiv.org/abs/2403.12015)]  
  Latent-space adversarial distillation overcoming ADD limitations (powers FLUX.1 [schnell]).

- **NitroFusion: High-Fidelity Single-Step Diffusion through Dynamic Adversarial Training** [CVPR 2025] 🔵  
  *Dar-Yen Chen, et al.*  
  [[Paper](https://arxiv.org/abs/2412.02030)]  
  Dynamic adversarial framework for high-fidelity one-step diffusion.

- **Distillation-Free One-Step Diffusion for Real-World Image Super-Resolution (DFOSD)** [ICLR 2025] 🟢  
  *Jianze Li, et al.*  
  [[Paper](https://arxiv.org/abs/2410.04224)] [[Code](https://github.com/JianzeLi-114/D3SR)]  
  Noise-aware discriminator enabling teacher-free one-step diffusion SR.

---

## Shortcut / Direct Mapping

> **Core principle — learn the noise-to-data map directly.** Rather than reasoning about trajectories or iterative denoising, these methods propose a new paradigm: a single conditioned network that maps noise straight to data, bypassing the ODE/SDE machinery entirely.

- **One Step Diffusion via Shortcut Models** [ICLR 2025 Oral] 🟢  
  *Kevin Frans, Danijar Hafner*  
  [[Paper](https://arxiv.org/abs/2410.12557)] [[Code](https://github.com/kvfrans/shortcut-models)]  
  A single network and single training phase for direct generation without iterative denoising.

- **High-Order Matching for One-Step Shortcut Diffusion Models** [ICLR 2025] 🟢  
  [[Paper](https://arxiv.org/abs/2502.00688)]  
  High-order matching to improve shortcut models.

- **Revisiting Diffusion Models: From Generative Pre-training to One-Step Generation (D2O)** [ICML 2025] 🟢  
  *Zheng Yang, et al.*  
  [[Paper](https://arxiv.org/abs/2506.09376)] [[Code](https://github.com/Zyriix/D2O)]  
  Treating diffusion as generative pre-training and fine-tuning for one-step generation.

- **Noise Consistency Training (NCT)** [NeurIPS 2025] 🟢  
  *Yihong Luo, Shuchen Xue, et al.*  
  [[Paper](https://arxiv.org/abs/2506.19741)] [[Code](https://github.com/Luo-Yihong/NCT)]  
  Native integration of new control signals directly into one-step generators.

---

## Applications

### Image Super-Resolution

- **One-Step Effective Diffusion Network for Real-World Image Super-Resolution (OSEDiff)** [NeurIPS 2024] 🔵  
  [[Paper](https://arxiv.org/abs/2406.08177)] [[Code](https://github.com/cswry/OSEDiff)]

- **One Step Diffusion-based Super-Resolution with Time-Aware Distillation (TAD-SR)** [NeurIPS 2024] 🔵  
  [[Paper](https://arxiv.org/abs/2408.07476)] [[Code](https://github.com/LearningHx/TAD-SR)]

- **TSD-SR: One-Step Diffusion with Target Score Distillation** [CVPR 2025] 🔵  
  [[Paper](https://arxiv.org/abs/2411.18263)] [[Code](https://github.com/Microtreei/TSD-SR)]

- **PassionSR: Post-Training Quantization for One-Step Diffusion SR** [CVPR 2025] 🔵  
  [[Paper](https://arxiv.org/abs/2411.17106)] [[Code](https://github.com/libozhu03/PassionSR)]

- **Consistency Trajectory Matching for One-Step Generative Super-Resolution** [ICCV 2025] 🔵  
  [[Paper](https://arxiv.org/abs/2503.20349)] [[Code](https://github.com/LabShuHangGU/CTMSR)]

- **FlowSR: Fast Image Super-Resolution via Consistency Rectified Flow** [ICCV 2025] 🟢  
  [[Paper](https://arxiv.org/abs/2605.12377)]

- **One-Step Diffusion Transformer for Controllable Real-World Image SR (ODTSR)** [CVPR 2026] 🔵  
  [[Paper](https://arxiv.org/abs/2511.17138)] [[Code](https://github.com/RedMediaTech/ODTSR)]

### Video Generation

- **SF-V: Single Forward Video Generation Model** [NeurIPS 2024] 🔵  
  *Snap Research*  
  [[Paper](https://arxiv.org/abs/2406.04324)] [[Code](https://github.com/snap-research/SF-V)]

- **OSV: One Step is Enough for High-Quality Image to Video Generation** [CVPR 2025] 🔵  
  [[Paper](https://arxiv.org/abs/2409.11367)]

- **Diffusion Adversarial Post-Training for One-Step Video Generation** [ICML 2025] 🔵  
  [[Paper](https://arxiv.org/abs/2501.08316)]

- **DOVE: Efficient One-Step Diffusion for Real-World Video Super-Resolution** [NeurIPS 2025] 🔵  
  [[Paper](https://arxiv.org/abs/2505.16239)] [[Code](https://github.com/zhengchen1999/DOVE)]

- **One-Step Diffusion for Detail-Rich Video Super-Resolution (DLoRAL)** [NeurIPS 2025] 🔵  
  [[Paper](https://arxiv.org/abs/2506.15591)]

- **Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation** [NeurIPS 2025] 🔵  
  [[Paper](https://arxiv.org/abs/2506.09350)]

### Controllable Generation

- **Adding Additional Control to One-Step Diffusion with Joint Distribution Matching** [ICCV 2025] 🔵  
  [[Paper](https://arxiv.org/abs/2503.06652)]

- **Supercharged One-step Text-to-Image Diffusion Models with Negative Prompts (SNOOPI)** [ICCV 2025] 🔵  
  [[Paper](https://arxiv.org/abs/2412.02687)]

> See also: **Noise Consistency Training (NCT)** under [Shortcut / Direct Mapping](#shortcut--direct-mapping) — a 🟢 native approach to adding controls.

### Other Applications

- **MoFlow: One-Step Flow Matching for Human Trajectory Forecasting** [CVPR 2025] 🟢  
  [[Paper](https://arxiv.org/abs/2503.09950)]

- **OSDFace: One-Step Diffusion Model for Face Restoration** [CVPR 2025] 🔵  
  [[Paper](https://arxiv.org/abs/2411.17163)] [[Code](https://github.com/jkwang28/OSDFace)]

- **Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models** [ICLR 2025] 🟢  
  [[Paper](https://arxiv.org/abs/2409.08861)]

- **Reward Guided Latent Consistency Distillation (RG-LCD)** [ICLR 2025] 🔵  
  [[Paper](https://arxiv.org/abs/2403.11027)]

- **Diffusion2GAN: Distilling Diffusion Models into Conditional GANs** [ECCV 2024] 🔵  
  [[Paper](https://arxiv.org/abs/2405.05967)]

- **Flash Diffusion: Accelerating Any Conditional Diffusion Model for Few Steps Image Generation** [AAAI 2024] 🔵  
  [[Paper](https://arxiv.org/abs/2406.02347)] [[Code](https://github.com/gojasper/flash-diffusion)]

- **DKDM: Data-Free Knowledge Distillation for Diffusion Models with Any Architecture** [CVPR 2025] 🔵  
  [[Paper](https://arxiv.org/abs/2409.03550)] [[Code](https://github.com/qianlong0502/DKDM)]

- **One-Step Specular Highlight Removal with Adapted Diffusion Models** [ICCV 2025] 🔵  
  [[Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Atmis_One-Step_Specular_Highlight_Removal_with_Adapted_Diffusion_Models_ICCV_2025_paper.pdf)]

- **OSCAR: One-Step Diffusion Codec Across Multiple Bit-rates** [NeurIPS 2025] 🔵  
  [[Paper](https://arxiv.org/abs/2505.16091)]

- **Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching** [NeurIPS 2025] 🟢  
  [[Paper](https://arxiv.org/abs/2510.18328)] [[Code](https://github.com/ZhongLIFR/TCCM-NIPS)]

---

## Unified Theory

- **Uni-Instruct: One-step Diffusion Model through Unified Diffusion Divergence Instruction** [NeurIPS 2025]  
  *Yifei Wang, et al. (PKU & Xiaohongshu)*  
  [[Paper](https://arxiv.org/abs/2505.20755)]  
  A unified theoretical framework proving 10+ existing distillation methods are special cases of diffusion divergence minimization.

- **High-Order Flow Matching: Unified Framework** [NeurIPS 2025]  
  A unified high-order flow-matching framework.


---

## Full Paper List


| Date | Paper | Venue | Type | Category | Links |
|------|-------|-------|:----:|:--------:|------|
| 2026-05 | FlowSR: Fast Image SR via Consistency Rectified Flow | ICCV 2025 | 🟢 | Applications | [[Paper](https://arxiv.org/abs/2605.12377)] |
| 2026-04 | ODE-free Neural Flow Matching for One-Step Generative Modeling | 2025 | 🟢 | Flow Straightening | [[Paper](https://arxiv.org/abs/2604.06413)] |
| 2026-03 | SoFlow: Solution Flow Models for One-Step Generative Modeling | ICLR 2026 | 🟢 | Flow Straightening | [[Paper](https://arxiv.org/abs/2512.15657)] [[Code](https://github.com/zlab-princeton/SoFlow)] |
| 2026-03 | Self-Corrected Flow Distillation (SCFlow) | AAAI 2025 | 🔵 | Flow Straightening | [[Paper](https://arxiv.org/abs/2412.16906)] [[Code](https://github.com/hao-pt/SCFlow)] |
| 2026-01 | DiffRatio: Training One-Step Diffusion Models Without Teacher Supervision | ICLR 2025 | 🟢 | Distribution Divergence Minimization | [[Paper](https://arxiv.org/abs/2502.08005)] [[Code](https://github.com/Wenlin-Chen/DiffRatio)] |
| 2025-12 | High-Order Flow Matching: Unified Framework | NeurIPS 2025 | — | Unified Theory | [[Paper](https://openreview.net/forum?id=ib0aV2hphN)] |
| 2025-11 | ODTSR: One-Step Diffusion Transformer for Controllable Real-World Image SR | CVPR 2026 | 🔵 | Applications | [[Paper](https://arxiv.org/abs/2511.17138)] [[Code](https://github.com/RedMediaTech/ODTSR)] |
| 2025-11 | Consistency Trajectory Matching for One-Step Generative SR | ICCV 2025 | 🔵 | Applications | [[Paper](https://arxiv.org/abs/2503.20349)] [[Code](https://github.com/LabShuHangGU/CTMSR)] |
| 2025-11 | DOVE: Efficient One-Step Diffusion for Real-World Video SR | NeurIPS 2025 | 🔵 | Applications | [[Paper](https://arxiv.org/abs/2505.16239)] [[Code](https://github.com/zhengchen1999/DOVE)] |
| 2025-10 | Scalable, Explainable & Robust Anomaly Detection with One-Step Flow Matching | NeurIPS 2025 | 🟢 | Applications | [[Paper](https://arxiv.org/abs/2510.18328)] [[Code](https://github.com/ZhongLIFR/TCCM-NIPS)] |
| 2025-10 | One-Step Diffusion for Detail-Rich Video SR (DLoRAL) | NeurIPS 2025 | 🔵 | Applications | [[Paper](https://arxiv.org/abs/2506.15591)] |
| 2025-10 | Autoregressive Adversarial Post-Training for Real-Time Interactive Video Gen | NeurIPS 2025 | 🔵 | Applications | [[Paper](https://arxiv.org/abs/2506.09350)] |
| 2025-10 | Uni-Instruct: One-step Diffusion via Unified Diffusion Divergence Instruction | NeurIPS 2025 | — | Unified Theory | [[Paper](https://arxiv.org/abs/2505.20755)] |
| 2025-10 | How to Build a Consistency Model: Learning Flow Maps via Self-Distillation | NeurIPS 2025 | 🟡 | Consistency Models | [[Paper](https://arxiv.org/abs/2505.18825)] [[Code](https://github.com/nmboffi/flow-maps)] |
| 2025-10 | Diffusion Adversarial Post-Training for One-Step Video Generation | ICML 2025 | 🔵 | Applications | [[Paper](https://arxiv.org/abs/2501.08316)] |
| 2025-10 | One-Step Specular Highlight Removal with Adapted Diffusion Models | ICCV 2025 | 🔵 | Applications | [[Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Atmis_One-Step_Specular_Highlight_Removal_with_Adapted_Diffusion_Models_ICCV_2025_paper.pdf)] |
| 2025-09 | SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation | ICCV 2025 | 🔵 | Consistency Models | [[Paper](https://arxiv.org/abs/2503.09641)] |
| 2025-09 | SNOOPI: Supercharged One-step T2I Diffusion with Negative Prompts | ICCV 2025 | 🔵 | Applications | [[Paper](https://arxiv.org/abs/2412.02687)] |
| 2025-07 | Score-of-Mixture Training | ICML 2025 Spotlight | 🟢 | Distribution Divergence Minimization | [[Paper](https://arxiv.org/abs/2502.09609)] |
| 2025-07 | Improving Consistency Models with Generator-Augmented Flows | ICML 2025 | 🟢 | Consistency Models | [[Paper](https://arxiv.org/abs/2406.09570)] [[Code](https://github.com/thibautissenhuth/consistency)] |
| 2025-06 | Noise Consistency Training (NCT) | NeurIPS 2025 | 🟢 | Shortcut / Direct Mapping | [[Paper](https://arxiv.org/abs/2506.19741)] [[Code](https://github.com/Luo-Yihong/NCT)] |
| 2025-06 | Align Your Flow: Scaling Continuous-Time Flow Map Distillation | NeurIPS 2025 | 🔵 | Flow Straightening | [[Paper](https://arxiv.org/abs/2506.14603)] |
| 2025-06 | Blockwise Flow Matching (BFM) | NeurIPS 2025 | 🟢 | Flow Straightening | [[Paper](https://arxiv.org/abs/2506.14603)] |
| 2025-06 | Revisiting Diffusion Models: From Pre-training to One-Step Generation (D2O) | ICML 2025 | 🟢 | Shortcut / Direct Mapping | [[Paper](https://arxiv.org/abs/2506.09376)] [[Code](https://github.com/Zyriix/D2O)] |
| 2025-06 | Diff-Instruct++: Training One-step T2I Generator to Human-Preferred | 2024 | 🔵 | Distribution Divergence Minimization | [[Paper](https://arxiv.org/abs/2410.18881)] [[Code](https://github.com/pkulwj1994/diff_instruct_pp)] |
| 2025-06 | One Step Diffusion via Shortcut Models | ICLR 2025 Oral | 🟢 | Shortcut / Direct Mapping | [[Paper](https://arxiv.org/abs/2410.12557)] [[Code](https://github.com/kvfrans/shortcut-models)] |
| 2025-05 | TSD-SR: One-Step Diffusion with Target Score Distillation | CVPR 2025 | 🔵 | Applications | [[Paper](https://arxiv.org/abs/2411.18263)] [[Code](https://github.com/Microtreei/TSD-SR)] |
| 2025-05 | OSCAR: One-Step Diffusion Codec Across Multiple Bit-rates | NeurIPS 2025 | 🔵 | Applications | [[Paper](https://arxiv.org/abs/2505.16091)] |
| 2025-05 | Mean Flows for One-step Generative Modeling (MeanFlow) | NeurIPS 2025 | 🟢 | Flow Straightening | [[Paper](https://arxiv.org/abs/2505.13447)] [[Code](https://github.com/haidog-yaqub/MeanFlow)] |
| 2025-05 | Multi-step Consistency Models | 2025 | 🟢 | Consistency Models | [[Paper](https://arxiv.org/abs/2505.01049)] |
| 2025-04 | OSV: One Step is Enough for High-Quality Image to Video Generation | CVPR 2025 | 🔵 | Applications | [[Paper](https://arxiv.org/abs/2409.11367)] |
| 2025-03 | Distillation-Free One-Step Diffusion for Real-World Image SR (DFOSD) | ICLR 2025 | 🟢 | Adversarial Training | [[Paper](https://arxiv.org/abs/2410.04224)] [[Code](https://github.com/JianzeLi-114/D3SR)] |
| 2025-03 | MoFlow: One-Step Flow Matching for Human Trajectory Forecasting | CVPR 2025 | 🟢 | Applications | [[Paper](https://arxiv.org/abs/2503.09950)] |
| 2025-03 | Learning Few-Step Diffusion Models by Trajectory Distribution Matching (TDM) | ICCV 2025 | 🔵 | Flow Straightening | [[Paper](https://arxiv.org/abs/2503.06674)] [[Code](https://github.com/Luo-Yihong/TDM)] |
| 2025-03 | Adding Additional Control to One-Step Diffusion with Joint Distribution Matching | ICCV 2025 | 🔵 | Applications | [[Paper](https://arxiv.org/abs/2503.06652)] |
| 2025-02 | Guided Score identity Distillation (SiD-LSG) | ICLR 2025 | 🔵 | Distribution Divergence Minimization | [[Paper](https://arxiv.org/abs/2406.01561)] [[Code](https://github.com/mingyuanzhou/SiD-LSG)] |
| 2025-02 | You Only Sample Once (YOSO) | ICLR 2024 | 🟢 | Adversarial Training | [[Paper](https://arxiv.org/abs/2403.12931)] [[Code](https://github.com/Luo-Yihong/YOSO)] |
| 2025-02 | One-step Diffusion Models with f-Divergence Distribution Matching (f-distill) | 2025 | 🔵 | Distribution Divergence Minimization | [[Paper](https://arxiv.org/abs/2502.15681)] |
| 2025-02 | Inverse Flow and Consistency Models | ICML 2025 | 🟢 | Consistency Models | [[Paper](https://arxiv.org/abs/2502.11333)] |
| 2025-02 | High-Order Matching for One-Step Shortcut Diffusion Models | ICLR 2025 | 🟢 | Shortcut / Direct Mapping | [[Paper](https://arxiv.org/abs/2502.00688)] |
| 2025-02 | Simple Distillation for One-Step Diffusion Models (CED) | NeurIPS 2025 | 🔵 | Consistency Models | [[Paper](https://arxiv.org/abs/2502.00883)] |
| 2025-02 | End-to-End Single-Step Flow Matching via Direct Models (FlowFit) | ICLR 2025 | 🟢 | Flow Straightening | [[Paper](https://arxiv.org/abs/2502.01441)] |
| 2024-12 | Understanding and Improving Consistency Models | ICLR 2025 | 🟢 | Consistency Models | [[Paper](https://arxiv.org/abs/2410.18958)] [[Code](https://github.com/G-U-N/Stable-Consistency-Tuning)] |
| 2024-12 | Adversarial Score identity Distillation (SiDA) | ICLR 2025 | 🔵 | Distribution Divergence Minimization | [[Paper](https://arxiv.org/abs/2410.14919)] [[Code](https://github.com/mingyuanzhou/SiD/tree/sida)] |
| 2024-12 | EM Distillation for One-step Diffusion Models | NeurIPS 2024 | 🔵 | Distribution Divergence Minimization | [[Paper](https://arxiv.org/abs/2405.16852)] |
| 2024-12 | Flash Diffusion: Accelerating Any Conditional Diffusion Model for Few Steps | AAAI 2024 | 🔵 | Applications | [[Paper](https://arxiv.org/abs/2406.02347)] [[Code](https://github.com/gojasper/flash-diffusion)] |
| 2024-12 | NitroFusion: High-Fidelity Single-Step Diffusion via Dynamic Adversarial Training | CVPR 2025 | 🔵 | Adversarial Training | [[Paper](https://arxiv.org/abs/2412.02030)] |
| 2024-11 | TLCM: Training-efficient Latent Consistency Model | 2024 | 🔵 | Consistency Models | [[Paper](https://arxiv.org/abs/2406.05768)] [[Code](https://github.com/OPPO-Mente-Lab/TLCM)] |
| 2024-11 | Optimal Flow Matching: Learning Straight Trajectories in Just One Step | NeurIPS 2024 | 🟢 | Flow Straightening | [[Paper](https://arxiv.org/abs/2403.13117)] [[Code](https://github.com/Jhomanik/Optimal-Flow-Matching)] |
| 2024-11 | SwiftBrush: One-Step T2I Diffusion with Variational Score Distillation | CVPR 2024 | 🔵 | Distribution Divergence Minimization | [[Paper](https://arxiv.org/abs/2312.05239)] [[Code](https://github.com/VinAIResearch/SwiftBrush)] |
| 2024-11 | OSDFace: One-Step Diffusion Model for Face Restoration | CVPR 2025 | 🔵 | Applications | [[Paper](https://arxiv.org/abs/2411.17163)] [[Code](https://github.com/jkwang28/OSDFace)] |
| 2024-11 | PassionSR: Post-Training Quantization for One-Step Diffusion SR | CVPR 2025 | 🔵 | Applications | [[Paper](https://arxiv.org/abs/2411.17106)] [[Code](https://github.com/libozhu03/PassionSR)] |
| 2024-10 | Consistency Models Made Easy | ICLR 2025 | 🟢 | Consistency Models | [[Paper](https://arxiv.org/abs/2406.14548)] |
| 2024-10 | One-Step Effective Diffusion Network for Real-World Image SR (OSEDiff) | NeurIPS 2024 | 🔵 | Applications | [[Paper](https://arxiv.org/abs/2406.08177)] [[Code](https://github.com/cswry/OSEDiff)] |
| 2024-10 | SF-V: Single Forward Video Generation Model | NeurIPS 2024 | 🔵 | Applications | [[Paper](https://arxiv.org/abs/2406.04324)] [[Code](https://github.com/snap-research/SF-V)] |
| 2024-10 | Improving the Training of Rectified Flows | NeurIPS 2024 | 🟢 | Flow Straightening | [[Paper](https://arxiv.org/abs/2405.20320)] [[Code](https://github.com/sangyun884/rfpp)] |
| 2024-10 | Reward Guided Latent Consistency Distillation (RG-LCD) | ICLR 2025 | 🔵 | Applications | [[Paper](https://arxiv.org/abs/2403.11027)] |
| 2024-10 | One-step Diffusion with Distribution Matching Distillation (DMD) | CVPR 2024 | 🔵 | Distribution Divergence Minimization | [[Paper](https://arxiv.org/abs/2311.18828)] |
| 2024-10 | One-Step Diffusion Distillation through Score Implicit Matching (SIM) | NeurIPS 2024 | 🔵 | Distribution Divergence Minimization | [[Paper](https://arxiv.org/abs/2410.16794)] [[Code](https://github.com/maple-research-lab/SIM)] |
| 2024-09 | Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models | ICLR 2025 | 🟢 | Applications | [[Paper](https://arxiv.org/abs/2409.08861)] |
| 2024-09 | DKDM: Data-Free Knowledge Distillation for Diffusion Models | CVPR 2025 | 🔵 | Applications | [[Paper](https://arxiv.org/abs/2409.03550)] [[Code](https://github.com/qianlong0502/DKDM)] |
| 2024-08 | One Step Diffusion-based SR with Time-Aware Distillation (TAD-SR) | NeurIPS 2024 | 🔵 | Applications | [[Paper](https://arxiv.org/abs/2408.07476)] [[Code](https://github.com/LearningHx/TAD-SR)] |
| 2024-08 | SwiftBrush v2: Make Your One-step Model Better Than Its Teacher | ECCV 2024 | 🔵 | Distribution Divergence Minimization | [[Paper](https://arxiv.org/abs/2408.14176)] [[Code](https://github.com/vinairesearch/swiftbrushv2)] |
| 2024-07 | Training Smaller One-Step Diffusion Models with Rectified Flow (SlimFlow) | ECCV 2024 | 🔵 | Flow Straightening | [[Paper](https://arxiv.org/abs/2407.12718)] [[Code](https://github.com/yuanzhi-zhu/SlimFlow)] |
| 2024-05 | Improved Distribution Matching Distillation (DMD2) | NeurIPS 2024 Oral | 🔵 | Distribution Divergence Minimization | [[Paper](https://arxiv.org/abs/2405.14867)] [[Code](https://github.com/tianweiy/dmd2)] |
| 2024-05 | Diffusion2GAN: Distilling Diffusion Models into Conditional GANs | ECCV 2024 | 🔵 | Applications | [[Paper](https://arxiv.org/abs/2405.05967)] |
| 2024-04 | Score identity Distillation (SiD) | ICML 2024 | 🔵 | Distribution Divergence Minimization | [[Paper](https://arxiv.org/abs/2404.04057)] [[Code](https://github.com/mingyuanzhou/SiD)] |
| 2024-03 | Consistency Trajectory Models (CTM) | ICLR 2024 | 🔵 | Consistency Models | [[Paper](https://arxiv.org/abs/2310.02279)] [[Code](https://github.com/sony/ctm)] |
| 2024-03 | InstaFlow: One Step is Enough for High-Quality T2I Generation | ICLR 2024 | 🔵 | Flow Straightening | [[Paper](https://arxiv.org/abs/2309.06380)] [[Code](https://github.com/gnobitab/InstaFlow)] |
| 2024-03 | Latent Adversarial Diffusion Distillation (LADD) | ECCV 2024 | 🔵 | Adversarial Training | [[Paper](https://arxiv.org/abs/2403.12015)] |
| 2023-11 | Adversarial Diffusion Distillation (ADD) | ECCV 2024 | 🔵 | Adversarial Training | [[Paper](https://arxiv.org/abs/2311.17042)] |
| 2023-10 | Improved Techniques for Training Consistency Models (iCT) | ICLR 2024 | 🟢 | Consistency Models | [[Paper](https://arxiv.org/abs/2310.14189)] |
| 2023-10 | Latent Consistency Models (LCM) | ICLR 2024 | 🔵 | Consistency Models | [[Paper](https://arxiv.org/abs/2310.04378)] |
| 2023-06 | BOOT: Data-free Distillation of Denoising Diffusion Models with Bootstrapping | ICML 2023 | 🔵 | Distribution Divergence Minimization | [[Paper](https://arxiv.org/abs/2306.05544)] |
| 2023-05 | Diff-Instruct: Universal Knowledge Transfer from Pre-trained Diffusion Models | NeurIPS 2023 | 🔵 | Distribution Divergence Minimization | [[Paper](https://arxiv.org/abs/2305.18455)] |
| 2023-03 | Consistency Models | ICML 2023 | 🟡 | Consistency Models | [[Paper](https://arxiv.org/abs/2303.01469)] |
| 2022-10 | Flow Matching for Generative Modeling | ICLR 2023 | 🟢 | Flow Straightening | [[Paper](https://arxiv.org/abs/2210.02747)] |
| 2022-09 | Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow | ICLR 2023 Spotlight | 🟢 | Flow Straightening | [[Paper](https://arxiv.org/abs/2209.03003)] |
| 2022-02 | Progressive Distillation for Fast Sampling of Diffusion Models | ICLR 2023 | 🔵 | Flow Straightening | [[Paper](https://arxiv.org/abs/2202.00512)] |

---

## Related Resources

### Foundational Works

> These papers are **not** one-step generation methods themselves, but provide the theoretical foundations upon which many one-step methods are built.

- **Progressive Distillation for Fast Sampling of Diffusion Models** [ICLR 2023] 🔵  
  *Tim Salimans, Jonathan Ho*  
  [[Paper](https://arxiv.org/abs/2202.00512)]  
  Iterative distillation that progressively halves the number of sampling steps; foundational for step-compression research.

- **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow** [ICLR 2023 Spotlight] 🟢  
  *Xingchao Liu, Chengyue Gong, Qiang Liu*  
  [[Paper](https://arxiv.org/abs/2209.03003)]  
  Learning ODEs that follow straight paths; the reflow procedure inspired many one-step flow methods.

- **Flow Matching for Generative Modeling** [ICLR 2023] 🟢  
  *Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le*  
  [[Paper](https://arxiv.org/abs/2210.02747)]  
  Simulation-free training of continuous normalizing flows; the foundational framework behind MeanFlow, SoFlow, etc.

---

## Contributing

Contributions are very welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for paper-entry formatting, category rules, and the native-vs-distillation labeling convention before submitting a pull request.

When adding a paper, classify it by its **primary mathematical contribution** (the principle that makes one-step generation work), then mark it 🟢 native, 🔵 distillation, or 🟡 hybrid, and place it chronologically within the category.

---

## Citation

If you find this list useful in your research, please consider citing it:

```bibtex
@misc{awesome-one-step-generation,
  title        = {Awesome One-Step Generation},
  author       = {Awesome One-Step Generation Contributors},
  howpublished = {\url{https://github.com/Luciferbobo/Awesome-One-Step-Generation}},
  year         = {2026}
}
```

