# Interpretable Lazy Classification for Numerical Data using Interval Pattern Structures

This repository contains the code developed for the paper titled "Interpretable Lazy Classification for Numerical Data using Interval Pattern Structures".

## Abstract
This paper evaluates interpretable lazy classification methods based on Interval Pattern Structures (IPS), focusing on the Formal Concept Analysis Lazy Classifier (FCALC) and an IPS-based k-Nearest Neighbor (IPS-KNN) classifier. For FCALC, we tested multiple decision functions and treated the function as a hyperparameter, increasing the model's flexibility. For IPS-KNN, we extended the method to support multi-class classification and improved interpretability by introducing feature importance scoring. We further simplified IPS-KNN through localized feature selection tailored to each query object. These improvements were evaluated across diverse datasets. The results highlight the robustness and interpretability of IPS-based models—especially IPS-KNN—while maintaining compact representations and delivering competitive classification performance.


## Usage Instructions
The instructions for FCALC and its randomized version can be seen in  [FCALC/README.md](FCALC/README.md). 

The instruction for all other classifiers are here.

### Prerequisites
Ensure you have the necessary Python packages installed. You can install them using pip with the provided `requirements.txt`:

`pip install -r requirements.txt`


### Running Experiments
To reproduce the experimental results discussed in the paper, use `run_experiments.py`. This script allows you to run different experiments:

1. f1: Compute the F1 score of each tested model on each dataset.
2. param_search: Perform hyperparameter search for all tested models.
3. size: Evaluate the size of classifiers for testing local interpretability.
4. time: Measure the training-prediction time and explanation component times.



All results will be stored in the `output` folder. Example outputs can be found in the `example_output` folder.



