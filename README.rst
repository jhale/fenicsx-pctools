==================================================
``fenicsx-pctools``: FEniCSx Preconditioning Tools
==================================================

Description
===========

This repository contains various tools for preconditioning of systems of linear
algebraic equations in `DOLFINx <https://github.com/FEniCS/dolfinx>`_, the
computational environment of the `FEniCS Project
<https://fenicsproject.org/>`_. These tools, delivered as a Python library
``fenics_pctools``, aim to facilitate the specification of problems that can be
tackled by means of composable linear solvers offered by `PETSc
<https://www.mcs.anl.gov/petsc/>`_. The idea is inspired by the work of Kirby
and Mitchell [1]_ that was originally implemented in the `Firedrake Project
<https://firedrakeproject.org/>`_.

Implementation of chosen advanced preconditioners is part of the library, e.g.
variants of the pressure-convection-diffusion (PCD) preconditioner for the
system of incompressible Navier-Stokes equations originally implemented in the
work of Blechta and Řehoř [2]_.

Dependencies
============

``fenicsx-pctools`` depends on the Python interface to DOLFINx.

Quickstart
==========

Assuming that the current working directory is the root of this repository.

1. Install ``fenics_pctools``:

   .. code-block:: console

      python3 -m pip install .

2. Run unit tests to verify the installation:

   .. code-block:: console

      python3 -m pytest .

3. Run an example:

   .. code-block:: console

      TODO


Authors
=======

- Martin Řehoř <martin.rehor@rafinex.com>
- Jack S. Hale <jack.hale@uni.lu>

This package was developed by `Rafinex <https://www.rafinex.com/>`_ within the
`FNR <https://www.fnr.lu/>`_ Industrial Fellowship project `RIFLE
<https://www.fnr.lu/projects/robust-incompressible-flow-solver-enhancement/>`_
(Host Institution: `Rafinex S.à r.l. <https://www.rafinex.com/>`_, Academic
Partner: `University of Luxembourg <https://wwwen.uni.lu/>`_).

References
==========

.. [1] \ R. C. Kirby and L. Mitchell, "Solver Composition Across the PDE/Linear Algebra Barrier,"
         SIAM J. Sci. Comput., vol. 40, no. 1, pp. C76–C98, 2017, doi: 10.1137/17M1133208.

.. [2] \ J. Blechta and M. Řehoř, Fenapack 2018.1.0. Zenodo, 2018, doi: 10.5281/zenodo.1308015.

License
=======

.. |(C)| unicode:: U+000A9

Copyright |(C)| 2021-2022 Rafinex S. á r. l. <info@rafinex.com>

``fenicsx-pctools`` is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

``fenicsx-pctools`` is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with fenicsx-pctools. If not, see
<http://www.gnu.org/licenses/>.

In addition to the LGPL license detailed above, the additional rights under
which the University of Luxembourg and Rafinex can use this work are detailed
in the *Collaboration Agreement in the frame of FNR Industrial Fellowships*
concluded between both parties.
