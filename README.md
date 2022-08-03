<div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div>

<div align="center"><a title="" href="https://github.com/ZJCV/ZCls2"><img align="center" src="./imgs/ZCls2.png" alt=""></a></div>

<p align="center">
  «ZCls2» is a more faster classification model training framework
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square" alt=""></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg" alt=""></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg" alt=""></a>
  <a href="https://libraries.io/pypi/zcls2"><img src="https://img.shields.io/librariesio/github/ZJCV/ZCls2" alt=""></a>
<br>
  <a href="https://pypi.org/project/zcls2/"><img src="https://img.shields.io/badge/PYPI-zcls2-brightgreen" alt=""></a>
  <a href="https://pypi.org/project/zcls2/"><img src="https://img.shields.io/pypi/pyversions/zcls2" alt=""></a>
  <a href="https://pypi.org/project/zcls2/"><img src="https://img.shields.io/pypi/v/zcls2" alt=""></a>
  <a href="https://pypi.org/project/zcls2/"><img src="https://img.shields.io/pypi/l/zcls2" alt=""></a>
  <a href="https://pypi.org/project/zcls2/"><img src="https://img.shields.io/pypi/dd/zcls2?style=plastic" alt=""></a>
<br>
  <a href='https://zcls2.readthedocs.io/en/latest/?badge=latest'>
      <img src='https://readthedocs.org/projects/zcls2/badge/?version=latest' alt='Documentation Status' />
  </a>
  <a href="https://github.com/ZJCV/ZCls2"><img src="https://img.shields.io/github/v/tag/ZJCV/ZCls2" alt=""></a>
  <a href="https://github.com/ZJCV/ZCls2"><img src="https://img.shields.io/github/repo-size/ZJCV/ZCls2" alt=""></a>
  <a href="https://github.com/ZJCV/ZCls2"><img src="https://img.shields.io/github/forks/ZJCV/ZCls2?style=social" alt=""></a>
  <a href="https://github.com/ZJCV/ZCls2"><img src="https://img.shields.io/github/stars/ZJCV/ZCls2?style=social" alt=""></a>
  <a href="https://github.com/ZJCV/ZCls2"><img src="https://img.shields.io/github/downloads/ZJCV/ZCls2/total" alt=""></a>
  <a href="https://github.comZJCV/ZCls2"><img src="https://img.shields.io/github/commit-activity/y/ZJCV/ZCls2" alt=""></a>
</p>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Background

After nearly one and a half years of development, [ZCls](https://github.com/ZJCV/ZCls) has integrated many training features, includes configuration module, register module, training module, and many model implementations (`resnet/mobilenet/senet-sknet-resnest/acbnet-repvgg-dbbnet/ghostnet/gcnet...`) and so on. In the development process, it is found that compared with the current excellent classification training framework, such as [apex](https://github.com/NVIDIA/apex/tree/master/examples/imagenet), the training speed of [ZCls](https://github.com/ZJCV/ZCls) is not outstanding. 

In order to better improve the training speed, we decided to develop a new training framework [ZCls2](https://github.com/ZJCV/ZCls2), which is implemented based on [apex](https://github.com/NVIDIA/apex/tree/master/examples/imagenet) and provides more friendly and powerful functions. In the preliminary implementation, it can be found that [ZCls2](https://github.com/ZJCV/ZCls2) improves the training speed by at least 50% compared with [ZCls](https://github.com/ZJCV/ZCls).  More functions are being added.

## Installation

See [Install](https://zcls2.readthedocs.io/en/latest/install/)

## Usage

See [Get started](https://zcls2.readthedocs.io/en/latest/get-started/)

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [NVIDIA/apex](https://github.com/NVIDIA/apex/tree/master/examples/imagenet)
* [ZJCV/ZCls](https://github.com/ZJCV/ZCls)

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/ZJCV/ZCls2/issues) or submit PRs.

Small note:

* Git submission specifications should be complied
  with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* If versioned, please conform to the [Semantic Versioning 2.0.0](https://semver.org) specification
* If editing the README, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme)
  specification.

## License

[Apache License 2.0](LICENSE) © 2022 zjykzj