# Amadeus
Public Copy

CS 172B Winter 2018 - Team 16

## Goal


## Setting up Dev Environment

To use the group environment run the following in the top level of the repo

	conda env create -f environment.yml


Everytime you want to use the amadeus envrionment run:

	source activate amadeus

If you want to add a library to the environment:

If it is on conda you can run the following:
	
	conda install --name amadeus <package name>

Otherwise refer to: https://conda.io/docs/user-guide/tasks/manage-pkgs.html#installing-non-conda-packages

once you have installed the package you need to build a new yml file

	conda env export > environment.yml

Then push it back to the repo so we all can install it

## Adding the data

After downloading the data, copy them in top level folder the repository. Currently we are using these two `./fma_metadata/` and `./fma_small/`.

