<h1 align="center">RoboBench</h1>

<p align="center">
  <strong>A Comprehensive Evaluation Benchmark for Multimodal Large Language Models as Embodied Brain</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.17801">Paper</a> ·
  <a href="https://robo-bench.github.io/">Project website</a> ·
  <a href="https://huggingface.co/datasets/LeoFan01/RoboBench">Dataset</a> ·
  <a href="https://huggingface.co/datasets/lyl010221-pku/RoboBench-Results">Official results</a> ·
  <a href="https://eccv.ecva.net/virtual/2026/poster/4213">ECCV 2026</a> ·
  <a href="https://robo-bench.github.io/static/pdfs/robobench_eccv2026_poster.pdf">Poster</a> ·
  <a href="https://www.youtube.com/watch?v=_W3FSK75IRQ">Introduction video</a>
</p>

## Overview

RoboBench evaluates MLLMs on robotic manipulation tasks by decomposing embodied intelligence into diagnostic abilities rather than reporting only end-to-end task success. It covers the full execution pipeline from instruction comprehension and perception to generalized planning, affordance reasoning, and failure analysis. The ECCV 2026 release contains **5 dimensions, 14 capability groups, 25 tasks (released as 32 subtask manifests), 6,092 QA pairs, and results for 18 state-of-the-art MLLMs**.

<p align="center">
  <img src="assets/teaser.jpg" alt="RoboBench overview" width="900">
</p>

## Quick start

```bash
git clone https://github.com/yulin-luo/RoboBench.git
cd RoboBench
pip install -e .
```

- [Run RoboBench](docs/guides/QUICKSTART.md) to download and validate the dataset, then run inference and scoring.
- [Evaluate RoboBench-MCQ and RoboBench-Planning](docs/guides/HY_EMBODIED_EVAL.md) with the HY-Embodied helper.
- [Browse the documentation](docs/README.md) for benchmark metrics, configuration, prompts, and development.

## External evaluations

ACE Robotics (ACE-Brain-0.7), Li Auto (ME-VLM), Tencent (HY-Embodied), and ZTE (EmbodiedMind) report RoboBench results under their own evaluation settings. See [external results and evaluation protocols](docs/EXTERNAL_EVALUATIONS.md) for scores and source reports.

## Repository layout

| Directory | Contents |
| --- | --- |
| [docs/](docs/) | Run guides, evaluation protocols, and reference |
| [config/](config/) | Benchmark configuration template |
| [scripts/](scripts/) | Evaluation helper |
| [src/robobench/](src/robobench/) | Dataset, inference, evaluation, and scoring code |
| [tests/](tests/) | Automated tests |

## News

- **2026.11** — Congratulations to the [ACE-Brain team](https://arxiv.org/abs/2611.26519) on ACE-Brain-0.7! Their evaluation uses 3,979 RoboBench questions on task decomposition, next-subtask prediction, and step completion, with natural-language answers for the first two tasks.
- **2026.09** — Li Auto reports RoboBench results for both [ME-VLM variants](https://arxiv.org/html/2609.24526v1#S6), also featured in the [ME-Brain-1.0](https://arxiv.org/html/2609.24271v1#S7) cognitive-core evaluation.
- 🖼️ **2026.09** - The official **ECCV 2026 poster** is released ([PDF](https://robo-bench.github.io/static/pdfs/robobench_eccv2026_poster.pdf)). Meet us at ECCV 2026 in Malmö, Sept 8–12 — see the [ECCV virtual poster page](https://eccv.ecva.net/virtual/2026/poster/4213).
- 📊 **2026.07** - Congratulations to [HY-Embodied](https://github.com/Tencent-Hunyuan/HY-Embodied)! **HY-Embodied-0.5 MoT-2B** and **Hy-Embodied-VLM-1.0 A3B** include **RoboBench-MCQ** and **RoboBench-Planning** as part of their official evaluation suite, highlighting RoboBench as a recognized benchmark for embodied foundation models.
- 🎉 **2026.07** - RoboBench is accepted to **ECCV 2026**. The official release reports results for **18 state-of-the-art MLLMs** with the MLLM-as-world-simulator planning framework.
- 📰 **2026.07** - RoboBench is featured by [具身智能之心](https://mp.weixin.qq.com/s/SdGEqu_1mz14DhUumTZ3FQ), introducing our benchmark for evaluating MLLMs as embodied brains.

## Citation

If you use RoboBench in your research, please cite:

```bibtex
@misc{luo2026robobenchcomprehensiveevaluationbenchmark,
      title={Robobench: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models as Embodied Brain},
      author={Yulin Luo and Chun-Kai Fan and Menghang Dong and Jiayu Shi and Xiangju Mi and Mengdi Zhao and Bo-Wen Zhang and Cheng Chi and Jiaming Liu and Gaole Dai and Rongyu Zhang and Ruichuan An and Kun Wu and Zhengping Che and Shaoxuan Xie and Guocai Yao and Zhongxia Zhao and Pengwei Wang and Guang Liu and Zhongyuan Wang and Tiejun Huang and Shanghang Zhang},
      year={2026},
      eprint={2510.17801},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.17801},
}
```

## License

This repository is released under the [MIT License](LICENSE).
