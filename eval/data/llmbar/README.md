# LLMBar (vendored)

The LLMBar benchmark of pairwise instruction-following judgements, used by
`eval/llmbar.py` (`make eval-llmbar`) to measure the judge's accuracy and position
bias (PLAN.md dataset D2).

- Source: https://github.com/princeton-nlp/LLMBar, directory `Dataset/LLMBar/`
- Commit: `900616bff90b6c6c8e1681f7d079250637c55992` (`main` on 2026-10-14)
- Licence: MIT, copied from the upstream repository as `LICENSE-LLMBar`
- Files are byte-for-byte copies; only the `Dataset/LLMBar/` prefix is dropped.
  `python -I eval/llmbar_fetch.py` re-downloads them at the commit and verifies the
  hashes below; `--verify` checks this copy offline.

Each `dataset.json` is a JSON list of `{"input", "output_1", "output_2", "label"}`,
where `label` (1 or 2) names the output that follows the instruction.

| File | Instances | Label 1 / 2 | Bytes | sha256 |
|---|---|---|---|---|
| `Natural/dataset.json` | 100 | 42 / 58 | 91269 | `f7127b7aa78104cb8e03a2b1b3ac0077e95b125714dd441a8ee63666145a3375` |
| `Adversarial/Neighbor/dataset.json` | 134 | 75 / 59 | 131988 | `6712c7eda6342602e0bc7c06c282fa95e2110d4e741a284026a62d937f9780a0` |
| `Adversarial/GPTInst/dataset.json` | 92 | 45 / 47 | 211768 | `1002425f73c2cf54b6e6de99bce3f6c60fa2a17d13194772e37d7481c1351d42` |
| `Adversarial/GPTOut/dataset.json` | 47 | 22 / 25 | 32640 | `1ff56e520d7607750a9866a3be0f5b850392ca4477a1f69d80f32bd3e310f561` |
| `Adversarial/Manual/dataset.json` | 46 | 22 / 24 | 79050 | `0d9598769a39cdbbd265778c37e49939d41d5f76dcfb17bd75dcfa07516d3837` |
| `LICENSE-LLMBar` | | | 1094 | `8872dbf8660b00890b5e07ce1ea1f7a44fa7ae4e1857da56caba172b51dab3cb` |

Total: 419 instances (Natural 100, Adversarial 319), as counted in these files.

## Citation

```bibtex
@inproceedings{zeng2024llmbar,
   title={Evaluating Large Language Models at Evaluating Instruction Following},
   author={Zeng, Zhiyuan and Yu, Jiatong and Gao, Tianyu and Meng, Yu and Goyal, Tanya and Chen, Danqi},
   booktitle = {International Conference on Learning Representations (ICLR)},
   year={2024}
}
```
