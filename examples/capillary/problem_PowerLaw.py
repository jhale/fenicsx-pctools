from functools import cached_property

from problem_NavierStokes import Problem as NavierStokesProblem

import ufl


class CarreauYasudaViscosity(object):
    def __init__(self, mu0, mu8, alpha, n):
        self.mu0 = mu0
        self.mu8 = mu8
        self.alpha = alpha
        self.n = n

    def __call__(self, dgamma_squared):
        n = self.n
        mu0, mu8, alpha = float(self.mu0), float(self.mu8), float(self.alpha)

        return mu8 + (mu0 - mu8) * (1.0 + (alpha ** 2.0) * dgamma_squared) ** (0.5 * (n - 1.0))


class Problem(NavierStokesProblem):

    model_parameters = ("rho", "mu_0", "mu_8", "alpha", "n")

    def parse_options(self, **kwargs):
        super().parse_options(**kwargs)

        # Non-constant coefficients
        self._coeffs["mu"] = self.mu

        # Mapping to Navier-Stokes problem options
        self._ns_opts["mu"] = self.mu_0

    @cached_property
    def viscosity_law(self):
        # TODO: Any other useful formulas?
        mu = CarreauYasudaViscosity(
            self.mu_0, self.mu_8, self.alpha, self.n
        )
        return mu

    @cached_property
    def mu(self):
        # TODO: Experiment with explicit formulations (e.g. using the last known velocity iterate)!
        v = self.solution_vars[0]
        mu = self.viscosity_law
        return mu(2.0 * ufl.inner(self.D(v), self.D(v)))

    # NOTE:
    #   Basic Newton solver does not converge if the viscosity depends directly on
    #   the unknown velocity field. A more sophisticated line search is typically required
    #   when using this "fully-implicit" formulation (e.g. backtracing line search), see
    #
    #   https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/SNES/SNESLineSearchSetType.html#SNESLineSearchSetType

    def Re(self, v_char, x_char):
        """For given characteristic quantities, compute the Reynolds number."""
        mu = self.viscosity_law
        return self.rho * v_char * x_char / mu((v_char / x_char) ** 2)

    @cached_property
    def F_form(self):
        v, p = self.solution_vars
        v_te, p_te = self.test_functions

        dx = ufl.dx
        dx_fixqd = dx.reconstruct(metadata={"quadrature_degree": 3})

        # Volume contributions
        F_v = ufl.inner(self.rT(v, p), self.D_base(v_te)) * dx_fixqd
        F_v += ufl.inner(self.T(v, p), self.D_xtra(v_te)) * dx_fixqd
        F_v += self.coeff("rho") * ufl.inner(ufl.dot(self.rgrad(v), v), v_te) * dx

        F_p = -self.rdiv(v) * p_te * dx

        # Boundary contributions
        if self.application_opts["bc_outlet"] == "NoEnd":
            ds_outlet = self.domain.ds("outlet").reconstruct(metadata=dx_fixqd.metadata())
            e, r_index, z_index = self.e, self.r_index, self.z_index
            n = ufl.FacetNormal(self.domain.mesh)
            n = n[r_index] * e[r_index] + n[z_index] * e[z_index]
            F_v += -ufl.inner(2.0 * self.coeff("mu") * ufl.dot(self.rD(v), n), v_te) * ds_outlet

        return [F_v, F_p]
