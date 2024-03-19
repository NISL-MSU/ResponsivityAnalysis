[![arXiv](https://img.shields.io/badge/arXiv-2304.04063-b31b1b.svg)](https://arxiv.org/abs/2304.04063)

# Counterfactual Explanations of Neural Network-Generated Response Curves

## Description

Response curves exhibit the magnitude of the response of a sensitive system to a varying stimulus. However,
response of such systems may be sensitive to multiple stimuli
(i.e., input features) that are not necessarily independent. As
a consequence, the shape of response curves generated for a
selected input feature (referred to as “active feature”) might
depend on the values of the other input features (referred to
as “passive features”). In this work we consider the case of
systems whose response is approximated using regression neural
networks. We propose to use counterfactual explanations (CFEs)
for the identification of the features with the highest relevance
on the shape of response curves generated by neural network
black boxes. CFEs are generated by a genetic algorithm-based
approach that solves a multi-objective optimization problem. In
particular, given a response curve generated for an active feature,
a CFE finds the minimum combination of passive features that
need to be modified to alter the shape of the response curve.
We tested our method on a synthetic dataset with 1-D inputs
and two crop yield prediction datasets with 2-D inputs. The
relevance ranking of features and feature combinations obtained
on the synthetic dataset coincided with the analysis of the
equation that was used to generate the problem. Results obtained
on the yield prediction datasets revealed that the impact on
fertilizer responsivity of passive features depends on the terrain
characteristics of each field.


<img src=https://raw.githubusercontent.com/GiorgioMorales/ResponsivityAnalysis/master/images/resp.jpg alt="alt text" width=300 height=400>

Please read our [IJCNN paper](https://arxiv.org/abs/2304.04063) for more information.


## Usage

This repository contains the following scripts:

* `FeatureResponsivity.py`: Contains the FeatureResponsivity class. The main method of the class is `impact` whose parameters are:
  
        *`s`: Index of the feature whose responsivity w.r.t response variable will be assessed.
        *`epsi`: List of tolerance errors that will be tested.
        *`replace`: If True, overwrite existing results.
        *NOTE: This implementation assumes that the models are already trained using Trainer.py and 10x1 CV
        
  The class can be instantiated and executed as follows:

  ```
  name = 'Synth'
  fresp = FeatureResponsivity(dataset=name)
  fresp.impact(s=0, epsi=[0.4, 0.6, 0.8], replace=True)
  ```
  
* `Trainer.py`: Class used for training the NNs using 10x1 cross-validation.

* `MOO.py`: Implements the multi-objective optimization (MOO) framework and defines the objective functions.

* `DataLoader.py`: Load and pre-process the datasets.
        
* `utils.py`: Additional methods used to transform the data and calculate the metrics.   



# Citation
Use this Bibtex to cite this repository

```
@INPROCEEDINGS{Morales:Counterfactual,
AUTHOR="Giorgio Morales and John W. Sheppard",
TITLE="Counterfactual Explanations of Neural {Network-Generated} Response Curves",
BOOKTITLE="2023 International Joint Conference on Neural Networks (IJCNN) (IJCNN 2023)",
ADDRESS="Queensland, Australia",
DAYS="17",
MONTH=jun,
YEAR=2023,
}
```
