==================================================
FEniCSx-pctools: FEniCSx Preconditioning Tools
==================================================

.. only:: html

    .. image:: https://img.shields.io/badge/docs-ready-success
       :target: https://rafinex-external-rifle.gitlab.io/fenicsx-pctools/

    .. image:: https://gitlab.com/rafinex-external-rifle/fenicsx-pctools/badges/main/pipeline.svg
       :target: https://gitlab.com/rafinex-external-rifle/fenicsx-pctools/-/pipelines

Description
===========

This repository contains various tools for preconditioning of systems of linear
algebraic equations in `DOLFINx <https://github.com/FEniCS/dolfinx>`_, the
computational environment of the `FEniCS Project
<https://fenicsproject.org/>`_. These tools, delivered as a Python package
FEniCSx-pctools, aim to facilitate the specification of problems that can
be tackled by means of composable linear solvers offered by `PETSc
<https://www.mcs.anl.gov/petsc/>`_. The idea is inspired by the work of
:cite:t:`kirby_solver_2018` that was originally implemented in the `Firedrake
Project <https://firedrakeproject.org/>`_.

Implementation of chosen advanced preconditioners is part of the package, e.g.
variants of the pressure-convection-diffusion (PCD) preconditioner for the
system of incompressible Navier-Stokes equations originally implemented in the
work of :cite:t:`blechta_fenapack_2018`.

Synopsis
========

The main feature of FEniCSx-pctools is its ability to take block matrices
constructed using DOLFINx's built-in assembly functions

    .. code-block:: python

       a = [[inner(q, q_t) * dx, inner(p, div(q_t)) * dx], [inner(div(q), p_t) * dx, None]]
       a_dolfinx = fem.form(a)


       A = fem.petsc.create_matrix_block(a_dolfinx)
       fem.petsc.assemble_matrix_block(A, a_dolfinx, bcs)
       A.assemble()

and then automatically create a wrapper that is compatible with PETSc's
field split preconditioning features

     .. code-block:: python

       A_splittable = create_splittable_matrix_block(A, a)

Advanced constructions are supported, including nested field splits (splits within splits).
For more details, see :doc:`Documented demos <demos>` and :doc:`API documentation <api>`.

Citing
======

FEniCSx-pctools is described in the preprint at https://arxiv.org/abs/2402.02523

     .. code-block::

       @misc{rehor2024fenicsx,
             title={FEniCSx Preconditioning Tools (FEniCSx-pctools)},
             author={Martin Řehoř and Jack S. Hale},
             year={2024},
             eprint={2402.02523},
             archivePrefix={arXiv},
             primaryClass={cs.MS}
       }

The raw logs and data from the numerical experiments presented in the pre-print are contained
in the directory ``examples/rayleigh-benard-convection/aion_results_WeakRB100k-r1_2023-10-11``.

Documentation
=============

Full documentation is available at
https://rafinex-external-rifle.gitlab.io/fenicsx-pctools/.

Dependencies
============

Please see ``pyproject.toml`` file for a precise list of dependencies and
optional dependency sets.

Quickstart
==========

Before installing `fenicsx-pctools` you must ensure that DOLFINx built against
PETSc is already installed. If not, follow the upstream instructions
`here <https://github.com/fenics/dolfinx?tab=readme-ov-file#installation>`_.

To get started quickly you can use the upsteam Docker images:

   .. code-block:: console

      docker run -ti -v $(pwd):/shared -w /shared dolfinx/dolfinx:v0.9.0r1

And then in any system with DOLFINx installed:

1. Install FEniCSx-pctools with the optional dependencies to run the demos:

   .. code-block:: console

      python3 -m pip install .[demos]

2. Run unit tests to verify the installation:

   .. code-block:: console

      python3 -m pytest tests/

3. Run an example:

   .. code-block:: console

      cd demo/navier-stokes-pcd
      python3 demo_navier-stokes-pcd.py


Authors
=======

- Martin Řehoř <martin.rehor@rafinex.com>
- Jack S. Hale <jack.hale@uni.lu>

This package was developed by `Rafinex <https://www.rafinex.com/>`_ within the
`FNR <https://www.fnr.lu/>`_ Industrial Fellowship project `RIFLE
<https://www.fnr.lu/projects/robust-incompressible-flow-solver-enhancement/>`_
(Host Institution: `Rafinex S.à r.l. <https://www.rafinex.com/>`_ <info@rafinex.com>,
Academic Partner: `University of Luxembourg <https://wwwen.uni.lu/>`_).

License
=======

.. |(C)| unicode:: U+000A9

Copyright |(C)| 2021-2025 Rafinex S.à r.l. and Jack S. Hale

FEniCSx-pctools is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FEniCSx-pctools is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with FEniCSx-pctools. If not, see
<http://www.gnu.org/licenses/>.

In addition to the LGPL license detailed above, the additional rights under
which the University of Luxembourg and Rafinex can use this work are detailed
in the *Collaboration Agreement in the frame of FNR Industrial Fellowships*
concluded between both parties.
