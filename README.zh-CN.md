<div align="right">
  语言:
    🇨🇳
  <a title="英语" href="./README.md">🇺🇸</a>
</div>

 <div align="center"><a title="" href="https://github.com/ZJCV/ZCls2"><img align="center" src="./imgs/ZCls2.png"></a></div>

<p align="center">
  «ZCls2»是一款更快速的分类模型训练框架
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
  <a href="https://libraries.io/pypi/zcls2"><img src="https://img.shields.io/librariesio/github/ZJCV/ZCls2"></a>
<br>
  <a href="https://pypi.org/project/zcls2/"><img src="https://img.shields.io/badge/PYPI-zcls2-brightgreen"></a>
  <a href="https://pypi.org/project/zcls2/"><img src="https://img.shields.io/pypi/pyversions/zcls2"></a>
  <a href="https://pypi.org/project/zcls2/"><img src="https://img.shields.io/pypi/v/zcls2"></a>
  <a href="https://pypi.org/project/zcls2/"><img src="https://img.shields.io/pypi/l/zcls2"></a>
  <a href="https://pypi.org/project/zcls2/"><img src="https://img.shields.io/pypi/dd/zcls2?style=plastic"></a>
<br>
  <a href='https://zcls2.readthedocs.io/en/latest/?badge=latest'>
      <img src='https://readthedocs.org/projects/zcls2/badge/?version=latest' alt='Documentation Status' />
  </a>
  <a href="https://github.com/ZJCV/ZCls2"><img src="https://img.shields.io/github/v/tag/ZJCV/ZCls2"></a>
  <a href="https://github.com/ZJCV/ZCls2"><img src="https://img.shields.io/github/repo-size/ZJCV/ZCls2"></a>
  <a href="https://github.com/ZJCV/ZCls2"><img src="https://img.shields.io/github/forks/ZJCV/ZCls2?style=social"></a>
  <a href="https://github.com/ZJCV/ZCls2"><img src="https://img.shields.io/github/stars/ZJCV/ZCls2?style=social"></a>
  <a href="https://github.com/ZJCV/ZCls2"><img src="https://img.shields.io/github/downloads/ZJCV/ZCls2/total"></a>
  <a href="https://github.comZJCV/ZCls2"><img src="https://img.shields.io/github/commit-activity/y/ZJCV/ZCls2"></a>
</p>

## 内容列表

- [内容列表](#内容列表)
- [背景](#背景)
- [安装](#安装)
- [用法](#用法)
- [主要维护人员](#主要维护人员)
- [致谢](#致谢)
- [参与贡献方式](#参与贡献方式)
- [许可证](#许可证)

## 背景

经过近一年半的开发，[ZCls](https://github.com/ZJCV/ZCls)已经集成了许多优秀的训练特性，包括配置模块，注册模块，训练模块，以及许多模型实现(`resnet/mobilenet/senet-sknet-resnest/acbnet-repvgg-dbbnet/ghostnet/gcnet...`)等等。在开发过程中，逐渐发现和当前最好的分类训练框架（比如[apex](https://github.com/NVIDIA/apex/tree/master/examples/imagenet)）相比，[ZCls](https://github.com/ZJCV/ZCls)的训练速度并不突出。

为了更好的提高训练速度，我决定重新开发一款更快速的分类模型训练框架[ZCls2](https://github.com/ZJCV/ZCls2)，基于[apex](https://github.com/NVIDIA/apex/tree/master/examples/imagenet)的同时能够提供更多更友好和更强大的特性。在初步的实现中，可以发现相比于[ZCls](https://github.com/ZJCV/ZCls)，新开发的[ZCls2](https://github.com/ZJCV/ZCls2)能够提高至少`50%`的训练速度。更多功能正在开发中。

## 安装

查看[Install](https://zcls2.readthedocs.io/en/latest/install/)

## 用法

查看[Get started](https://zcls2.readthedocs.io/en/latest/get-started/)

## 主要维护人员

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## 致谢

* [NVIDIA/apex](https://github.com/NVIDIA/apex/tree/master/examples/imagenet)
* [ZJCV/ZCls](https://github.com/ZJCV/ZCls)

## 参与贡献方式

欢迎任何人的参与！打开[issue](https://github.com/ZJCV/ZCls2/issues)或提交合并请求。

注意:

* `GIT`提交，请遵守[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)规范
* 语义版本化，请遵守[Semantic Versioning 2.0.0](https://semver.org)规范
* `README`编写，请遵守[standard-readme](https://github.com/RichardLitt/standard-readme)规范

## 许可证

[Apache License 2.0](LICENSE) © 2022 zjykzj