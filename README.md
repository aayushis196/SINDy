# SINDy
Sparse Identification of Nonlinear Dynamical systems (SINDy)

Discovering the governing equations from scientific
data becomes easier using data-driven approaches. Sparse regression
enables the tractable identification of both the structure
and parameters of a nonlinear dynamical system from data. The
resulting models have the fewest terms necessary to describe the
dynamics, balancing model complexity with the descriptive ability
and thus promoting interpretability and generalizability. In this
work, we design a custom autoencoder to discover a coordinate
transformation into a reduced space where the dynamics may be
sparsely represented.We combine the strength of the autoencoder
for coordinate representation in a reduced state of the system
and sparse identification of nonlinear dynamics (SINDy) for
parsimonious models. We implemented this method on the planar
pushing task and compared it against a globally linear, Embed
to Control(E2C) latent space model.
