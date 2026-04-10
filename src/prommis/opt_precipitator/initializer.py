#####################################################################################################
# "PrOMMiS" was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# ("PrOMMiS") initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
IDAES-compatible initializer for the OptPrecipitator unit model.

Authors: Chris Laliwala

Design pattern
--------------
Follows the FixedBedTSA0DInitializer pattern from
``idaes/models_extra/temperature_swing_adsorption/initializer.py``:
domain-specific numeric kwargs are added to ``initialize()`` and stored as
instance attributes for use inside ``initialization_routine()``.

Chemistry-specific knowledge (equilibrium estimates, dissolution extents) is
provided by the *caller* through ``outlet_conc`` and ``rxn_extent_seeds``.
The initializer applies those seeds, scales the model, and solves with IPOPT.
No seed_fn callback pattern is used — that pattern has no precedent in IDAES or
PrOMMiS.
"""
import math

import idaes.core.util.scaling as iscale
import pyomo.environ as pyo
from pyomo.environ import Block

from idaes.core.initialization.initializer_base import InitializerBase


class OptPrecipitatorInitializer(InitializerBase):
    """
    Initializer for OptPrecipitator unit models.

    Follows the FixedBedTSA0DInitializer pattern: domain-specific numeric
    kwargs are accepted on ``initialize()`` and stored as instance attributes
    for use in ``initialization_routine()``.

    The caller is responsible for computing chemistry-specific seeds
    (e.g., equilibrium estimates for complex species) and passing them via
    ``outlet_conc`` and ``rxn_extent_seeds``.  The initializer applies the
    seeds, computes IPOPT scaling, and solves.

    Parameters accepted by ``initialize()`` beyond the IDAES base class:
    outlet_conc : dict, optional
        ``{species: concentration (mol/L)}`` seeds for ``log_conc_out`` and
        ``flow_mol_comp_out``.  Species absent from this dict fall back to the
        corresponding inlet concentration.
    rxn_extent_seeds : dict, optional
        ``{rxn_idx: extent (mol/L)}`` seeds for ``rxn_extent`` variables.
    flow_vol : float, optional
        Volumetric flow rate (L/s).  Read from the model's aqueous inlet if
        not provided.
    m0_dissolution_solid : float, optional
        Moles of the primary dissolving solid (e.g. CuSO4(s) for R1).  Used
        to set appropriate scaling floors for dissolution reactions.
    rxn_extent_bounds : dict, optional
        ``{rxn_idx: (lb, ub)}`` bounds to tighten before solving.
    """

    CONFIG = InitializerBase.CONFIG()

    def initialize(
        self,
        model: Block,
        initial_guesses: dict = None,
        json_file: str = None,
        output_level=None,
        exclude_unused_vars: bool = False,
        outlet_conc: dict = None,
        rxn_extent_seeds: dict = None,
        flow_vol: float = None,
        m0_dissolution_solid: float = None,
        rxn_extent_bounds: dict = None,
    ):
        self._outlet_conc = outlet_conc or {}
        self._rxn_extent_seeds = rxn_extent_seeds or {}
        self._flow_vol = flow_vol
        self._m0_dissolution_solid = m0_dissolution_solid
        self._rxn_extent_bounds = rxn_extent_bounds or {}

        return super().initialize(
            model=model,
            initial_guesses=initial_guesses,
            json_file=json_file,
            output_level=output_level,
            exclude_unused_vars=exclude_unused_vars,
        )

    def precheck(self, model: Block):
        """Skip the standard DoF equality check.

        OptPrecipitator includes precipitation saturation inequalities
        (``precip_sat_ineq``) that are not counted by the standard
        ``degrees_of_freedom`` function (which only counts equality
        constraints).  When a precipitation reaction is present the apparent
        DoF = 1, but the system is square at the optimum because the
        inequality is active.  IPOPT handles this correctly via slack
        variables; the precheck would raise a false error.
        """

    def initialization_routine(self, model: Block):
        """Seed log_conc_out, apply IPOPT scaling, and solve."""
        t0 = model.flowsheet().time.first()
        aq_list = model.config.property_package_aqueous.component_list
        props_in = model.cv_aqueous.properties_in[t0]
        props_out = model.cv_aqueous.properties_out[t0]

        fv = (
            self._flow_vol
            if self._flow_vol is not None
            else pyo.value(props_in.flow_vol)
        )

        # ── 1. Seed log_conc_out and flow_mol_comp_out ─────────────────────
        # Priority: caller-supplied outlet_conc > inlet concentration.
        for sp in aq_list:
            if sp in self._outlet_conc:
                c_seed = max(self._outlet_conc[sp], 1e-20)
            else:
                c_in = pyo.value(props_in.flow_mol_comp[sp]) / max(fv, 1e-20)
                c_seed = max(c_in, 1e-20)
            model.log_conc_out[sp].set_value(math.log(c_seed))
            props_out.flow_mol_comp[sp].set_value(c_seed * fv)
        props_out.flow_vol.set_value(fv)

        # ── 2. Seed rxn_extent ──────────────────────────────────────────────
        for r, ext in self._rxn_extent_seeds.items():
            model.rxn_extent[r].set_value(ext)

        # ── 3. Tighten rxn_extent bounds ────────────────────────────────────
        for r, (lb, ub) in self._rxn_extent_bounds.items():
            model.rxn_extent[r].setlb(lb)
            model.rxn_extent[r].setub(ub)

        # ── 4. Scaling ──────────────────────────────────────────────────────
        iscale.calculate_scaling_factors(model)
        self._apply_scaling(model, t0, fv, aq_list)

        # ── 5. Solve ────────────────────────────────────────────────────────
        solver = pyo.SolverFactory("ipopt")
        solver.options["nlp_scaling_method"] = "user-scaling"
        solver.options["tol"] = 1e-10
        solver.options["max_iter"] = 10_000
        return solver.solve(model, tee=True)

    def _apply_scaling(self, model, t0, fv, aq_list):
        """Apply flux-based IPOPT scaling to all OptPrecipitator variables."""
        props_in = model.cv_aqueous.properties_in[t0]
        props_out = model.cv_aqueous.properties_out[t0]

        m0 = self._m0_dissolution_solid
        floor_flow = max(m0 * 1e-2, 1e-3 * fv) if m0 is not None else 1e-3 * fv

        # Flow_vol
        sf_fv = 1.0 / max(fv, 1e-20)
        iscale.set_scaling_factor(props_in.flow_vol, sf_fv)
        iscale.set_scaling_factor(props_out.flow_vol, sf_fv)

        # Aqueous molar flows
        for sp in aq_list:
            n_in = pyo.value(props_in.flow_mol_comp[sp])
            sf_in = 1.0 / max(n_in, floor_flow)
            iscale.set_scaling_factor(props_in.flow_mol_comp[sp], sf_in)

            if sp in self._outlet_conc:
                n_out_nom = max(self._outlet_conc[sp] * fv, floor_flow)
            else:
                n_out_nom = max(n_in, floor_flow)
            iscale.set_scaling_factor(props_out.flow_mol_comp[sp], 1.0 / n_out_nom)

        # rxn_extent — use seeds > bounds > dissolution-based fallback
        if hasattr(model, "rxn_extent"):
            for r in model.rxn_extent:
                if r in self._rxn_extent_seeds:
                    nom = max(abs(self._rxn_extent_seeds[r]), 1e-10)
                elif r in self._rxn_extent_bounds:
                    nom = max(abs(self._rxn_extent_bounds[r][1]), 1e-10)
                elif m0 is not None:
                    nom = max(m0 / max(fv, 1e-20), 1e-10)
                else:
                    nom = 1e-3
                iscale.set_scaling_factor(model.rxn_extent[r], 1.0 / nom)

        # Precipitate solid (if present) — scale from inlet molar flows
        if hasattr(model, "cv_precipitate"):
            sp_list = model.config.property_package_precipitate.component_list
            sp_props_in = model.cv_precipitate.properties_in[t0]
            sp_props_out = model.cv_precipitate.properties_out[t0]
            for sp in sp_list:
                n_sp = pyo.value(sp_props_in.moles_precipitate_comp[sp])
                sf_sp = 1.0 / max(n_sp, floor_flow)
                iscale.set_scaling_factor(sp_props_in.moles_precipitate_comp[sp], sf_sp)
                iscale.set_scaling_factor(sp_props_out.moles_precipitate_comp[sp], sf_sp)
