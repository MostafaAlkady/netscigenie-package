# netscigenie-package

[![PyPI - Version](https://img.shields.io/pypi/v/netscigenie.svg)](https://pypi.org/project/netscigenie)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/netscigenie.svg)](https://pypi.org/project/netscigenie)

-----

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [License](#license)

## Introduction
Netscigenie is a python package that provides the functions that are essential for Network Science research. The module `genie` contains the following functions:
- Degree Distribution: Given a degree sequence, the function returns the y values (probability) and the x values (support) of the degree distribution.
- Average Clustering by Degree: A function that calculate the average clustering coefficient of nodes of degree k (C(k)) for a NetworkX graph G.
- Average Neighbors Degree by Degree: A function that calculates the average degree of neighbors of nodes of degree k for a NetworkX graph G.
- Log-binning for functions: A function that does log-binning on X and Y data.
- Log-binning for distributions: A function that does log-binning on a given array X.
- Pareto Inverse CDF sampling: A function that generates a sample from Pareto Distribution using Inverse CDF method.
- HSCM Network: A function that generates a Hypersoft configuration model graph.
- HER Network: A function that generates a hypercanonical Erdos-Renyi graph.
- Add Numbers: A test function that adds two numbers :)

## Installation

```console
pip install netscigenie
```

## License

`netscigenie` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
