# Code Base of the Semester Thesis: Numerical implementation of a discrete-time controller based on variational integrators

This repository consists of various numerical methods for solving systems of nonlinear equations implemented in Matlab. The implemented functions were used in the thesis - Numerical implementation of a discrete-time controller based on variational integrators – for solving implicit variational integrators of a 3R robot for control purposes.

The different numerical methods can be grouped in accordance with the repository’s
folder structure. The groups are:

- `dogleg`: Trust-Region methods with the Dogleg method for solving the Trust-Region-Subproblem
- `levenberg-marquardt`: Different adaptations/extensions of the Levenberg-Marquardt method
- `newton`: Methods based on the Newton-Raphson method
- `quasi-newton`: Different Quasi-Newton methods
- `trust-region`: Trust-Region methods that use the exact solution of the Trust-Region-Subproblem

The sources of the different algorithms are given in the respective function files and bibliography file `bibliography.pdf`.
