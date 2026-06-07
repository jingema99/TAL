# Temporal-Adjusted Loss (TAL)

This repository contains the official implementation of the CVPR 2026 paper [Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning](https://arxiv.org/abs/2603.02280).

TAL studies temporal imbalance in class-incremental learning and dynamically reweights negative supervision in cross-entropy loss. The code framework is based on [PyCIL](https://github.com/LAMDA-CL/PyCIL), with TAL added as the core loss implementation and integrated into selected class-incremental learning baselines. This repository currently focuses on the reproduction paths used in our released experiments.

Slides: [TAL slides](scripts/TAL.pptx)

## Status

**The code is still being completed and cleaned up. Documentation and packaging will be improved over time.**

## Usage

Install the required Python dependencies, including PyTorch, torchvision, numpy, scipy, tqdm, scikit-learn, and POT.

From the repository root, run an experiment with:

```bash
python main.py --config=./exps/[CONFIG].json
```

The reproduction scripts in `scripts/` automatically switch to the repository root. From the repository root, use:

```bash
# iCaRL on CIFAR-100
bash scripts/icarl.sh

# iCaRL + TAL on CIFAR-100
bash scripts/icarl_tal.sh

# DER on ImageNet-100
bash scripts/der.sh

# DER + TAL on ImageNet-100
bash scripts/der_tal.sh
```

For ImageNet-100, update the dataset paths or file lists in `utils/data.py` and `data/imagenet_subset/` for your local environment.

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{ma2026temporal,
  title={Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning},
  author={Ma, Jinge and Zhu, Fengqing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={32299--32308},
  year={2026}
}
```

## Acknowledgments

This implementation is based on [PyCIL](https://github.com/LAMDA-CL/PyCIL). We thank the PyCIL authors for their open-source codebase.
