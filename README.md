# OpenFoodFact_Cluster

Machine learning project on the OpenFoodFact project whose goal is to do clustering on products

## Links (to delete when finished)
URl for project :

Documentations : https://docs.google.com/document/d/1MhWQtcY0oPlCQaRdboHdS-F7hMPLMvs6gueb30spwxo/edit?usp=sharing

Link containing all the fields of the dataset:
https://static.openfoodfacts.org/data/data-fields.txt

## Content directories

- data: small additional datasets (<50Mo)
- models : trained models
- notebooks: notebooks dedicated to analyses
- results : results (pictures, model performance, ... )

## Steps to start working on the project

- Create a new virtual environment and activate it :

with pyenv virtualenv:
```bash
	pyenv virtualenv clustering_OFF && pyenv activate clustering_OFF
```
or conda:
```bash
	conda clustering_OFF && conda activate clustering_OFF
```

- Install required librairies :
```bash
  	pip install -r requirements
```
