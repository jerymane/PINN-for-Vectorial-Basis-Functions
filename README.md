# PINN-based Generalized Polygonal Vectorial Basis Function Model

This repo contains a code developed to generate the results published at the COMPUMAG 2023 conference (proceeding comming soon).

## About.
Physics Informed Neural Networks (PINNs) are a new approach to solving partial differential equation (PDE) systems. Since [Raissi et al.](https://maziarraissi.github.io/PINNs/) introduced this concept in 2019, it gained a large popularity in the Scientific Machine Learning and in Computational Sciences in general. 

Vectorial basis functions (VBFs) are used to model and calculate vector fields of physical phenomena. Although the commonly used polygonal vectorial basis functions are restricted to rectangles and triangles, having access to a wider range of polygonal basis functions would provide greater flexibility in modeling intricate domains without compromising the tradeoff between accuracy and computational resource demands.

This project aims to train a simple neural network model to solve a PDE system which yields a VBF for any given polygon. The PDE system and the boundary conditions are described by [L. Knockaert et al](https://ieeexplore.ieee.org/document/967119).

A set of x, y coordinates describing the polygon domain are given at the input and a VBF value is at each coordinate is generated at the output.

## Repo structure.
This repo contains:
- a script to generate semi-random polygonal domains used for training training (generate_polygon.py) 
- a set of generated semi-random quadrilateral domains (training_sets_quads)
- the main training and output analysis script (PINN_solver.py). 


This repo also serves as a public data access for the MyWave project. 
