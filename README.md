# ImageNet Challenge Paper Reproductions

[![Github Workflow Status](https://img.shields.io/github/workflow/status/amritpurshotam/computer-vision-papers/CI)](https://github.com/amritpurshotam/computer-vision-papers/actions/workflows/ci.yaml)
[![wandb](https://img.shields.io/static/v1?message=Weights+%26+Biases&color=222222&logo=Weights+%26+Biases&logoColor=ffcc33&label=)](https://wandb.ai/amrit/computer-vision-papers)

Reproduction of the architectures and results of some of the most significant papers that came out of the ImageNet challenge. To limit the scope of this project, I will be focusing on the top-5 % accuracy of a single model (i.e. no ensembles) on the validation set specifically.

## Getting Started

This project uses Python 3.7.5. Setup your virtual environmnet of choice then install with the below

```console
pip install -e .
```

## Papers

| Paper | Year | Val Top-5 % | Reproduction |
| :--- | ---: | ---: | ---: |
| [AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) | 2012 | 83.00 | |
| [ZFNet](https://arxiv.org/abs/1311.2901) | 2013 | 84.00 |  |
| [GoogLeNet](https://arxiv.org/abs/1409.4842) | 2014 | 89.93 |  |
| [VGGNet](https://arxiv.org/abs/1409.1556) | 2014 | 92.00 |  |
| [ResNet](https://arxiv.org/abs/1512.03385) | 2015 | 95.51 |  |
| [Xception](https://arxiv.org/abs/1610.02357) | 2016 | 94.50 |  |
| [SENet](https://arxiv.org/abs/1709.01507) | 2017 | 97.75 |  |

## License

[MIT](LICENSE)