# Particle filter of a network of coupled oscillators
Particle filter implementation for a network of coupled Kuramoto oscillators (in MATLAB and Julia programming
language) and a network of coupled Rossler systems (in MATLAB programming
language), as described in 

- Arthur N. Montanari, Luis A. Aguirre (**2019**). Particle filtering of dynamical networks: Highlighting observability issues. *Chaos*, 29, 033118. https://doi.org/10.1063/1.5085321

Please, refer to this paper for further details. The codes syntax were adjusted to be better aligned with the paper notation.
This is not the most computationally efficient implementation of the particle
filter, however it is, in our opinion, the most readable one.
The contents of this repository were firstly reported at https://doi.org/10.13140/RG.2.2.13879.37288.

# Usage

- `KuramotoPF.m` : Particle filter implementation for a network of Kuramoto oscillators (MATLAB).
- `Rossler PF.m` : Particle filter implementation for a network of Rossler systems (MATLAB).
- `KuramotoPF.ipynb` : Particle filter implementation for a network of Kuramoto oscillators	(Julia language, in a Jupyter Notebook).
