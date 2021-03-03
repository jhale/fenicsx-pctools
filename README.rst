============================
FEniCS Preconditioning Tools
============================

.. only:: html

   .. image:: https://img.shields.io/badge/code%20style-black-000000.svg
       :target: https://github.com/psf/black

Description
===========

This repository contains various tools for preconditioning of systems of linear algebraic
equations in `DOLFINX <https://github.com/FEniCS/dolfinx>`_, which is the computational
environment of `FEniCS <https://fenicsproject.org/>`_. These tools, delivered as a Python
library *fenics_pctools*, aim to facilitate the specification of problems that can be tackled
by means of composable linear solvers offered by `PETSc <https://www.mcs.anl.gov/petsc/>`_.
The idea is inspired by the work of Kirby and Mitchell [1]_ that was originally implemented in
the `Firedrake Project <https://firedrakeproject.org/>`_.

Implementation of chosen advanced preconditioners is part of the library, e.g. variants of the
pressure-convection-diffusion (PCD) preconditioner for the system of incompressible Navier-Stokes
equations originally implemented in the work of Blechta and Řehoř [2]_.

The library is developed by `Rafinex <https://www.rafinex.com/>`_ within
the `FNR <https://www.fnr.lu/>`_ Industrial Fellowship project
`RIFLE <https://www.fnr.lu/projects/robust-incompressible-flow-solver-enhancement/>`_
(Host Institution: `Rafinex S.à r.l. <https://www.rafinex.com/>`_,
Academic Partner: `University of Luxembourg <https://wwwen.uni.lu/>`_).

References
----------

.. [1] \ R. C. Kirby and L. Mitchell, "Solver Composition Across the PDE/Linear Algebra Barrier,"
         SIAM J. Sci. Comput., vol. 40, no. 1, pp. C76–C98, 2017, doi: 10.1137/17M1133208.

.. [2] \ J. Blechta and M. Řehoř, Fenapack 2018.1.0. Zenodo, 2018, doi: 10.5281/zenodo.1308015.

Requirements
============

The easiest way to start using *fenics_pctools* is to run it within the official
`docker image <https://hub.docker.com/r/dolfinx/dolfinx>`_ with *dolfinx* installed.
Some examples have been implemented using high-level wrappers from another FEniCS-based
library *dolfiny* (available `here <https://github.com/michalhabera/dolfiny>`_).

Quickstart
==========

Assuming that the current working directory is the root of this repository.

1. Install *fenics_pctools*:

   .. code-block:: console

      pip3 install .

2. Run unit tests to verify the installation:

   .. code-block:: console

      pytest fenics_pctools/

3. Run an example:

   .. code-block:: console

      cd examples/rayleigh-benard-convection/
      pytest -sv test_rayleigh_benard.py [--resultsdir=./output] [--overwrite] [--warmup] [--help]

   Hint: Run the last command with ``--help`` and search for the section *custom options*
   to get the detailed information about the optional arguments in square brackets.

Authors
=======

- Martin Řehoř <martin.rehor@rafinex.com>

License
=======

.. |(C)| unicode:: U+000A9

Copyright |(C)| 2021 Rafinex S. á r. l. <info@rafinex.com>

The rights under which the University of Luxembourg and Rafinex can use this work are detailed in
the *Collaboration Agreement in the frame of FNR Industrial Fellowships* concluded between both
parties.

The code is a property of the copyright holder. No distribution and/or modification allowed
without written permission.
