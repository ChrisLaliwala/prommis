#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
r"""
Optimization-based Precipitator
===============================

Author: Chris Laliwala

Equilibrium precipitator: aqueous speciation, precipitation/dissolution and (optionally)
gas-liquid equilibria at a process temperature, written in log space. The saturation rule
of the precipitation reactions is imposed either through the objective function of the
original formulation (``formulation="objective"``) or as a complementarity constraint
(``formulation="complementarity"``).
"""

import math

import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.environ import units as pyunits
from pyomo.mpec import Complementarity, complements

import idaes.logger as idaeslog
from idaes.core import (
    ControlVolume0DBlock,
    UnitModelBlockData,
    declare_process_block_class,
    useDefault,
)
from idaes.core.initialization import ModularInitializerBase
from idaes.core.scaling import ConstraintScalingScheme, CustomScalerBase
from idaes.core.solvers import get_solver
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.exceptions import InitializationError
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.tables import create_stream_table_dataframe

_log = idaeslog.getLogger(__name__)

R_GAS = 8.314  # J/(mol K), Van't Hoff correction
R_GAS_L_BAR = 0.08314  # L bar/(mol K), ideal gas law
T_REF = 298.15  # K, reference temperature of the equilibrium constants


def _constituent_inventory(model, sb_in, prop, j):
    """
    Largest inlet flow (mol/s) among the aqueous species taking part in the reactions
    that form component ``j`` of property package ``prop``; a solid or gas formed in
    the unit is bounded by the aqueous inventory of its constituents.
    """
    prop_aq = model.config.property_package_aqueous
    inventory = 0.0
    for r, coeffs in prop.stoich_dict.items():
        if coeffs.get(j, 0) == 0:
            continue
        for k, nu in prop_aq.stoich_dict.get(r, {}).items():
            if nu != 0 and k in sb_in.flow_mol_comp:
                inventory = max(
                    inventory,
                    abs(pyo.value(sb_in.flow_mol_comp[k], exception=False) or 0.0),
                )
    return inventory


class PrecipitatorScaler(CustomScalerBase):
    """
    Scaler for the optimization-based precipitator.

    Flows are scaled by their magnitude: the inlet value, the current value, and for
    a solid or gas formed in the unit the aqueous inventory of its constituents. The
    log-space and temperature variables use the default factors below.
    """

    DEFAULT_SCALING_FACTORS = {
        "temperature": 1e-2,
        "inv_temp": 1e2,
        "log_conc_out": 1e-1,
        "log_q_sp": 1e-1,
        "rxn_extent": 1e0,
        "log_partial_pressure": 1e-1,
        "partial_pressure": 1e0,
        "log_moles_gas_out": 1e-1,
    }

    def _scale_by_value(
        self, var, overwrite, reference=None, magnitude=0.0, floor=1e-10
    ):
        """
        Scale a variable by the inverse of its magnitude. The magnitude is the larger
        of the variable's current value and that of ``reference`` (an inlet
        counterpart), so outlet flows inherit the inlet scaling until they have been
        initialized; re-running the scaler with ``overwrite=True`` after initialization
        refreshes the factors from the initialized values.
        """
        if not overwrite and self.get_scaling_factor(var) is not None:
            return
        val = abs(pyo.value(var, exception=False) or 0.0)
        if reference is not None:
            val = max(val, abs(pyo.value(reference, exception=False) or 0.0))
        val = max(val, magnitude)
        self.set_variable_scaling_factor(
            var, 1.0 / max(val, floor), overwrite=overwrite
        )

    def variable_scaling_routine(
        self, model, overwrite: bool = False, submodel_scalers: ComponentMap = None
    ):
        """
        Variable scaling routine.

        Args:
            model: instance of Precipitator to be scaled
            overwrite: whether to overwrite existing scaling factors
            submodel_scalers: ComponentMap of Scalers to use for sub-models

        Returns:
            None
        """
        for t in model.flowsheet().time:
            sb_in = model.cv_aqueous.properties_in[t]
            sb_out = model.cv_aqueous.properties_out[t]
            self._scale_by_value(sb_in.flow_vol, overwrite)
            self._scale_by_value(sb_out.flow_vol, overwrite, reference=sb_in.flow_vol)
            for j in sb_in.flow_mol_comp:
                self._scale_by_value(sb_in.flow_mol_comp[j], overwrite)
                self._scale_by_value(
                    sb_out.flow_mol_comp[j], overwrite, reference=sb_in.flow_mol_comp[j]
                )
            if hasattr(model, "cv_precipitate"):
                sp_in = model.cv_precipitate.properties_in[t]
                sp_out = model.cv_precipitate.properties_out[t]
                for j in sp_in.moles_precipitate_comp:
                    self._scale_by_value(sp_in.moles_precipitate_comp[j], overwrite)
                    self._scale_by_value(
                        sp_out.moles_precipitate_comp[j],
                        overwrite,
                        reference=sp_in.moles_precipitate_comp[j],
                        magnitude=_constituent_inventory(
                            model, sb_in, model.config.property_package_precipitate, j
                        ),
                    )
            if hasattr(model, "cv_gas"):
                g_in = model.cv_gas.properties_in[t]
                g_out = model.cv_gas.properties_out[t]
                for j in g_in.moles_gas_comp:
                    self._scale_by_value(g_in.moles_gas_comp[j], overwrite)
                    self._scale_by_value(
                        g_out.moles_gas_comp[j],
                        overwrite,
                        reference=g_in.moles_gas_comp[j],
                        magnitude=_constituent_inventory(
                            model, sb_in, model.config.property_package_gas, j
                        ),
                    )

        self.scale_variable_by_default(model.temperature, overwrite=overwrite)
        self.scale_variable_by_default(model.inv_temp, overwrite=overwrite)
        for name in (
            "log_conc_out",
            "log_q_sp",
            "rxn_extent",
            "log_partial_pressure",
            "partial_pressure",
            "log_moles_gas_out",
        ):
            if hasattr(model, name):
                for v in getattr(model, name).values():
                    self.scale_variable_by_default(v, overwrite=overwrite)

    def constraint_scaling_routine(
        self, model, overwrite: bool = False, submodel_scalers: ComponentMap = None
    ):
        """
        Constraint scaling routine.

        Args:
            model: instance of Precipitator to be scaled
            overwrite: whether to overwrite existing scaling factors
            submodel_scalers: ComponentMap of Scalers to use for sub-models

        Returns:
            None
        """
        for name in (
            "inv_temp_definition",
            "log_conc_linking_eqns",
            "aqueous_equilibrium_eqns",
            "log_q_precipitate_equilibrium_rxn_eqns",
            "precip_sat_ineq",
            "aqueous_mole_balance_eqns",
            "vol_balance",
            "precipitate_mole_balance_eqns",
            "log_pressure_linking_eqns",
            "log_moles_gas_linking_eqns",
            "gas_equilibrium_eqns",
            "ideal_gas_eqns",
            "gas_mole_balance_eqns",
        ):
            if hasattr(model, name):
                for condata in getattr(model, name).values():
                    self.scale_constraint_by_nominal_value(
                        condata,
                        scheme=ConstraintScalingScheme.inverseMaximum,
                        overwrite=overwrite,
                    )


class PrecipitatorInitializer(ModularInitializerBase):
    """
    Initializer for the optimization-based precipitator.

    Outlet flows and log-space variables are seeded from the inlet (or from the
    ``outlet_conc`` guesses, mol/L, when given) and the unit is solved. With
    ``formulation="complementarity"`` and an untransformed complementarity component,
    the solve is an aqueous-equilibrium seed with the precipitation extents held at
    zero; the complementarity itself must be transformed (``pyomo.mpec``) by the user
    before the full model is solved.
    """

    CONFIG = ModularInitializerBase.CONFIG()
    CONFIG.declare(
        "outlet_conc",
        ConfigValue(
            default=None,
            description="Optional {aqueous component: concentration (mol/L)} guesses "
            "for the outlet; components not listed start from the inlet",
        ),
    )
    CONFIG.declare(
        "rxn_extent_seeds",
        ConfigValue(
            default=None,
            description="Optional {reaction: extent (mol/L)} guesses",
        ),
    )

    def precheck(self, model):
        """
        Degrees-of-freedom check: one degree of freedom per precipitation reaction is
        expected (the amount of each solid is decided by the saturation rule).
        """
        expected = 0
        if model.config.property_package_precipitate is not None:
            expected = len(model.config.property_package_precipitate.rxn_set)
        dof = degrees_of_freedom(model)
        if dof != expected:
            raise InitializationError(
                f"Degrees of freedom of {model.name} were not equal to the number of "
                f"precipitation reactions ({expected}): {dof}."
            )

    def _get_solver(self):
        # trace species: an absolute tolerance looser than 1e-8 leaves their flows,
        # and hence their log-space variables, unresolved
        if self._solver is None:
            options = dict(self.config.solver_options)
            options.setdefault("tol", 1e-8)
            self._solver = get_solver(
                self.config.solver,
                solver_options=options,
                writer_config=self.config.writer_config,
            )
        return self._solver

    @staticmethod
    def _seed_gas_phase(model, t, props_in):
        """
        Seed the gas outlet from Henry's law at the inlet composition, capped by the
        aqueous inventory of the constituents of each gas species.
        """
        prop_aq = model.config.property_package_aqueous
        prop_gas = model.config.property_package_gas
        g_in = model.cv_gas.properties_in[t]
        g_out = model.cv_gas.properties_out[t]
        ln_rho_over_mw = math.log(prop_gas.rho_solvent / prop_gas.MW_solvent)
        rt = R_GAS_L_BAR * pyo.value(model.temperature)  # L bar / mol
        vol = pyo.value(model.gas_volume_basis[t])  # L
        t_batch = pyo.value(model.t_batch)  # s
        for j in g_in.moles_gas_comp:
            n = max(pyo.value(g_in.moles_gas_comp[j]), 1e-20)
            for r, coeffs in prop_gas.stoich_dict.items():
                nu_j = coeffs.get(j, 0)
                if nu_j == 0:
                    continue
                log_p = (
                    pyo.value(model.log_k[r])
                    - sum(
                        nu * pyo.value(model.log_conc_out[t, k])
                        for k, nu in prop_aq.stoich_dict.get(r, {}).items()
                        if nu != 0
                    )
                ) / nu_j - ln_rho_over_mw
                n_henry = math.exp(log_p) * vol / rt / t_batch
                inventory = _constituent_inventory(model, props_in, prop_gas, j)
                n = max(n, min(n_henry, inventory))
            g_out.moles_gas_comp[j].set_value(n)
            model.log_moles_gas_out[t, j].set_value(math.log(n))
            p = max(pyo.value(model.partial_pressure_from_moles[t, j]), 1e-15)
            model.partial_pressure[t, j].set_value(p)
            model.log_partial_pressure[t, j].set_value(math.log(p))

    def initialize_main_model(self, model):
        """
        Seed the outlet and log-space variables and solve the unit.

        Args:
            model: instance of Precipitator to be initialized

        Returns:
            solver results
        """
        outlet_conc = self.config.outlet_conc or {}
        extent_seeds = self.config.rxn_extent_seeds or {}
        prop_aq = model.config.property_package_aqueous

        model.inv_temp.set_value(1.0 / pyo.value(model.temperature))
        for t in model.flowsheet().time:
            props_in = model.cv_aqueous.properties_in[t]
            props_out = model.cv_aqueous.properties_out[t]
            fv = pyo.value(props_in.flow_vol)
            props_out.flow_vol.set_value(fv)
            for j in prop_aq.component_list:
                if j in outlet_conc:
                    c = max(outlet_conc[j], 1e-20)
                else:
                    c = max(pyo.value(props_in.flow_mol_comp[j]) / fv, 1e-20)
                props_out.flow_mol_comp[j].set_value(c * fv)
                model.log_conc_out[t, j].set_value(math.log(c))
            if hasattr(model, "cv_precipitate"):
                sp_in = model.cv_precipitate.properties_in[t]
                sp_out = model.cv_precipitate.properties_out[t]
                for j in sp_in.moles_precipitate_comp:
                    sp_out.moles_precipitate_comp[j].set_value(
                        pyo.value(sp_in.moles_precipitate_comp[j])
                    )
                for r in model.config.property_package_precipitate.rxn_set:
                    model.log_q_sp[t, r].set_value(
                        pyo.value(
                            sum(
                                prop_aq.stoich_dict[r][j] * model.log_conc_out[t, j]
                                for j in prop_aq.stoich_dict.get(r, {})
                            )
                        )
                    )
            if hasattr(model, "cv_gas"):
                self._seed_gas_phase(model, t, props_in)
            for r in model.merged_rxns:
                model.rxn_extent[t, r].set_value(extent_seeds.get(r, 0.0))

        solver = self._get_solver()

        untransformed = hasattr(model, "precipitation_complementarity") and any(
            c.active for c in model.precipitation_complementarity.values()
        )
        if untransformed:
            # aqueous-equilibrium seed: no precipitation, complementarity switched off
            model.precipitation_complementarity.deactivate()
            held = []
            for t in model.flowsheet().time:
                for r in model.config.property_package_precipitate.rxn_set:
                    v = model.rxn_extent[t, r]
                    if not v.fixed:
                        v.fix(0.0)
                        held.append(v)
            try:
                results = solver.solve(
                    model, tee=self.config.output_level < idaeslog.INFO
                )
            finally:
                for v in held:
                    v.unfix()
                model.precipitation_complementarity.activate()
            _log.info(
                "%s: complementarity formulation seeded with the precipitation "
                "extents at zero; transform the complementarity before solving.",
                model.name,
            )
            return results

        return solver.solve(model, tee=self.config.output_level < idaeslog.INFO)


@declare_process_block_class("Precipitator")
class PrecipitatorData(UnitModelBlockData):
    """
    Optimization-based Precipitator Unit Model Class
    """

    default_initializer = PrecipitatorInitializer
    default_scaler = PrecipitatorScaler

    CONFIG = UnitModelBlockData.CONFIG()

    CONFIG.declare(
        "property_package_aqueous",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use for aqueous control volume",
            doc="""Property parameter object used to define property calculations,
**default** - useDefault.
**Valid values:** {
**useDefault** - use default package from parent model or flowsheet,
**PropertyParameterObject** - a PropertyParameterBlock object.}""",
        ),
    )
    CONFIG.declare(
        "property_package_args_aqueous",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing aqueous property packages",
            doc="""A ConfigBlock with arguments to be passed to a property block(s)
and used when constructing these,
**default** - None.
**Valid values:** {
see property package for documentation.}""",
        ),
    )
    CONFIG.declare(
        "property_package_precipitate",
        ConfigValue(
            default=None,
            domain=is_physical_parameter_block,
            description="Property package to use for precipitate control volume",
            doc="""Property parameter object used to define property calculations,
**default** - None (no precipitates).
**Valid values:** {
**None** - no precipitate phase,
**PropertyParameterObject** - a PropertyParameterBlock object.}""",
        ),
    )
    CONFIG.declare(
        "property_package_args_precipitate",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing precipitate property packages",
            doc="""A ConfigBlock with arguments to be passed to a property block(s)
and used when constructing these,
**default** - None.
**Valid values:** {
see property package for documentation.}""",
        ),
    )
    CONFIG.declare(
        "property_package_gas",
        ConfigValue(
            default=None,
            domain=is_physical_parameter_block,
            description="Property package to use for gas control volume",
            doc="""Property parameter object used to define property calculations,
**default** - None (no gas phase).
**Valid values:** {
**None** - no gas phase,
**PropertyParameterObject** - a PropertyParameterBlock object.}""",
        ),
    )
    CONFIG.declare(
        "property_package_args_gas",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing gas property packages",
            doc="""A ConfigBlock with arguments to be passed to a property block(s)
and used when constructing these,
**default** - None.
**Valid values:** {
see property package for documentation.}""",
        ),
    )
    CONFIG.declare(
        "formulation",
        ConfigValue(
            default="objective",
            domain=In(["objective", "complementarity"]),
            description="Formulation of the saturation rule of the precipitation reactions",
            doc="""Formulation of the saturation rule of the precipitation reactions,
**default** - "objective".
**Valid values:** {
**"objective"** - the unit carries the objective min sum (ln K - ln Q)^2 with ln Q <= ln K,
**"complementarity"** - n_solid >= 0 complements (ln K - ln Q) >= 0, built as a
pyomo.mpec Complementarity component that the user transforms before solving.}""",
        ),
    )
    CONFIG.declare(
        "temperature",
        ConfigValue(
            default=T_REF,
            domain=float,
            description="Initial value of the process temperature (K)",
            doc="""Initial value of the process temperature variable (K), bounded to
273.15-373.15 K; fix it for a single-temperature solve or leave it free as a decision.
The equilibrium constants are corrected from 298.15 K with the Van't Hoff equation.""",
        ),
    )
    CONFIG.declare(
        "gas_volume",
        ConfigValue(
            default=None,
            description="Gas-phase volume (L) for the ideal gas law",
            doc="""Gas-phase volume (L) used in the ideal gas law of the gas species,
**default** - None: the volume of solution treated per batch, i.e. the aqueous inlet
volumetric flow rate over one second.
**Valid values:** {
**None** - use the solution volume,
**float** - a fixed gas volume (L), stored as a mutable Parameter.}""",
        ),
    )

    def build(self):
        """Building model

        Args:
            None
        Return:
            None
        """
        super().build()

        prop_aq = self.config.property_package_aqueous
        prop_sp = self.config.property_package_precipitate
        prop_gas = self.config.property_package_gas
        time = self.flowsheet().time

        # ----------------------------------------------------------------- #
        # Control volumes and ports
        # ----------------------------------------------------------------- #
        self.cv_aqueous = ControlVolume0DBlock(
            dynamic=False,
            has_holdup=False,
            property_package=prop_aq,
            property_package_args=self.config.property_package_args_aqueous,
        )
        self.cv_aqueous.add_state_blocks(has_phase_equilibrium=False)
        self.add_inlet_port(block=self.cv_aqueous, name="aqueous_inlet")
        self.add_outlet_port(block=self.cv_aqueous, name="aqueous_outlet")

        if prop_sp is not None:
            self.cv_precipitate = ControlVolume0DBlock(
                dynamic=False,
                has_holdup=False,
                property_package=prop_sp,
                property_package_args=self.config.property_package_args_precipitate,
            )
            self.cv_precipitate.add_state_blocks(has_phase_equilibrium=False)
            self.add_inlet_port(block=self.cv_precipitate, name="precipitate_inlet")
            self.add_outlet_port(block=self.cv_precipitate, name="precipitate_outlet")

        if prop_gas is not None:
            self.cv_gas = ControlVolume0DBlock(
                dynamic=False,
                has_holdup=False,
                property_package=prop_gas,
                property_package_args=self.config.property_package_args_gas,
            )
            self.cv_gas.add_state_blocks(has_phase_equilibrium=False)
            self.add_inlet_port(block=self.cv_gas, name="gas_inlet")
            self.add_outlet_port(block=self.cv_gas, name="gas_outlet")

        # ----------------------------------------------------------------- #
        # Reaction data: ln(K) at 298.15 K and reaction enthalpies
        # ----------------------------------------------------------------- #
        ln_k_ref = dict(prop_aq.ln_k_dict)
        dHr = dict(prop_aq.dHr_dict)
        if prop_sp is not None:
            ln_k_ref.update(prop_sp.ln_k_dict)
            dHr.update(prop_sp.dHr_dict)
        if prop_gas is not None:
            ln_k_ref.update(prop_gas.ln_k_dict)
            dHr.update(prop_gas.dHr_dict)
        self.merged_rxns = pyo.Set(initialize=list(ln_k_ref.keys()))

        self.log_k_ref = pyo.Param(
            self.merged_rxns,
            initialize=ln_k_ref,
            within=pyo.Reals,
            doc="ln(K) of each reaction at the reference temperature",
        )
        self.dHr = pyo.Param(
            self.merged_rxns,
            initialize=lambda b, r: dHr.get(r, 0.0),
            within=pyo.Reals,
            units=pyunits.J / pyunits.mol,
            doc="Standard enthalpy of reaction; zero (isothermal) when not supplied",
        )

        # ----------------------------------------------------------------- #
        # Temperature and the Van't Hoff correction
        # ----------------------------------------------------------------- #
        self.T_ref = pyo.Param(
            initialize=T_REF,
            units=pyunits.K,
            doc="Reference temperature of the equilibrium constants",
        )
        self.temperature = pyo.Var(
            initialize=self.config.temperature,
            bounds=(273.15, 373.15),
            units=pyunits.K,
            doc="Process temperature",
        )
        self.inv_temp = pyo.Var(
            initialize=1.0 / float(self.config.temperature),
            bounds=(1.0 / 373.15, 1.0 / 273.15),
            units=pyunits.K**-1,
            doc="Reciprocal of the process temperature",
        )

        @self.Constraint(doc="Reciprocal temperature definition")
        def inv_temp_definition(blk):
            return blk.inv_temp * blk.temperature == 1.0

        @self.Expression(
            self.merged_rxns, doc="ln(K) at the process temperature (Van't Hoff)"
        )
        def log_k(blk, r):
            return blk.log_k_ref[r] + (
                blk.dHr[r] / (R_GAS * pyunits.J / pyunits.mol / pyunits.K)
            ) * (1 / blk.T_ref - blk.inv_temp)

        # ----------------------------------------------------------------- #
        # Reference quantities that make the log arguments dimensionless
        # ----------------------------------------------------------------- #
        self.c_ref = pyo.Param(
            initialize=1.0, units=pyunits.mol / pyunits.L, doc="Reference concentration"
        )
        self.t_batch = pyo.Param(
            initialize=1.0,
            units=pyunits.s,
            doc="Time basis converting flow rates to amounts per batch",
        )

        # ----------------------------------------------------------------- #
        # Variables
        # ----------------------------------------------------------------- #
        self.rxn_extent = pyo.Var(
            time,
            self.merged_rxns,
            initialize=0.0,
            bounds=(-1e3, 1e3),
            units=pyunits.mol / pyunits.L,
            doc="Extent of reaction per unit volume of solution",
        )
        self.log_conc_out = pyo.Var(
            time,
            prop_aq.component_list,
            initialize=-5.0,
            bounds=(-60.0, 10.0),
            units=pyunits.dimensionless,
            doc="ln(C_out / c_ref) of each aqueous component",
        )
        if prop_sp is not None:
            self.log_q_sp = pyo.Var(
                time,
                prop_sp.rxn_set,
                initialize=-5.0,
                bounds=(-200.0, 50.0),
                units=pyunits.dimensionless,
                doc="ln(Q) of each precipitation reaction",
            )

        # ----------------------------------------------------------------- #
        # Log linking and equilibrium constraints
        # ----------------------------------------------------------------- #
        @self.Constraint(
            time,
            prop_aq.component_list,
            doc="Outlet molar flow from its log concentration",
        )
        def log_conc_linking_eqns(blk, t, j):
            return blk.cv_aqueous.properties_out[t].flow_mol_comp[j] == (
                pyo.exp(blk.log_conc_out[t, j])
                * blk.c_ref
                * blk.cv_aqueous.properties_out[t].flow_vol
            )

        @self.Constraint(time, prop_aq.rxn_set, doc="Aqueous equilibrium in log form")
        def aqueous_equilibrium_eqns(blk, t, r):
            return blk.log_k[r] == sum(
                prop_aq.stoich_dict[r][j] * blk.log_conc_out[t, j]
                for j in prop_aq.stoich_dict[r]
            )

        if prop_sp is not None:

            @self.Constraint(
                time, prop_sp.rxn_set, doc="ln(Q) of the precipitation reactions"
            )
            def log_q_precipitate_equilibrium_rxn_eqns(blk, t, r):
                return blk.log_q_sp[t, r] == sum(
                    prop_aq.stoich_dict[r][j] * blk.log_conc_out[t, j]
                    for j in prop_aq.stoich_dict.get(r, {})
                )

            if self.config.formulation == "objective":

                @self.Constraint(time, prop_sp.rxn_set, doc="ln(Q) at most ln(Ksp)")
                def precip_sat_ineq(blk, t, r):
                    return blk.log_q_sp[t, r] <= blk.log_k[r]

                @self.Objective(
                    doc="Minimize the distance to saturation of the precipitation reactions"
                )
                def min_logs(blk):
                    return sum(
                        (blk.log_k[r] - blk.log_q_sp[t, r]) ** 2
                        for t in time
                        for r in prop_sp.rxn_set
                    )

            else:

                def _precipitation_complementarity_rule(blk, t, r):
                    solid = blk._solid_of_reaction(r)
                    return complements(
                        blk.cv_precipitate.properties_out[t].moles_precipitate_comp[
                            solid
                        ]
                        >= 0.0 * pyunits.mol / pyunits.s,
                        blk.log_k[r] - blk.log_q_sp[t, r] >= 0.0,
                    )

                self.precipitation_complementarity = Complementarity(
                    time,
                    prop_sp.rxn_set,
                    rule=_precipitation_complementarity_rule,
                    doc="Saturation rule: the solid is absent or the solution is saturated",
                )

        # ----------------------------------------------------------------- #
        # Material balances
        # ----------------------------------------------------------------- #
        @self.Constraint(
            time, prop_aq.component_list, doc="Aqueous component mole balance"
        )
        def aqueous_mole_balance_eqns(blk, t, j):
            return blk.cv_aqueous.properties_out[t].flow_mol_comp[j] == (
                blk.cv_aqueous.properties_in[t].flow_mol_comp[j]
                + sum(
                    prop_aq.stoich_dict.get(r, {}).get(j, 0) * blk.rxn_extent[t, r]
                    for r in blk.merged_rxns
                )
                * blk.cv_aqueous.properties_out[t].flow_vol
            )

        @self.Constraint(time, doc="Volume balance")
        def vol_balance(blk, t):
            return (
                blk.cv_aqueous.properties_out[t].flow_vol
                == blk.cv_aqueous.properties_in[t].flow_vol
            )

        @self.Expression(time, doc="Volume of solution treated per batch")
        def volume(blk, t):
            return pyunits.convert(
                blk.cv_aqueous.properties_in[t].flow_vol * blk.t_batch,
                to_units=pyunits.L,
            )

        if prop_sp is not None:

            @self.Constraint(
                time, prop_sp.component_list, doc="Precipitate component mole balance"
            )
            def precipitate_mole_balance_eqns(blk, t, j):
                return blk.cv_precipitate.properties_out[t].moles_precipitate_comp[
                    j
                ] == blk.cv_precipitate.properties_in[t].moles_precipitate_comp[
                    j
                ] + sum(
                    prop_sp.stoich_dict.get(r, {}).get(j, 0)
                    * blk.rxn_extent[t, r]
                    * blk.cv_aqueous.properties_out[t].flow_vol
                    for r in prop_sp.rxn_set
                )

        # ----------------------------------------------------------------- #
        # Gas phase
        # ----------------------------------------------------------------- #
        if prop_gas is not None:
            self._build_gas_phase()

    def _solid_of_reaction(self, r):
        """Precipitate consumed by dissolution reaction r (negative coefficient)."""
        prop_sp = self.config.property_package_precipitate
        for j, alpha in prop_sp.stoich_dict.get(r, {}).items():
            if alpha < 0:
                return j
        raise ValueError(
            f"{self.name}: precipitation reaction {r} has no precipitate with a "
            "negative stoichiometric coefficient."
        )

    def _build_gas_phase(self):
        """Gas-phase variables, gas-liquid equilibrium, ideal gas law and balances."""
        prop_aq = self.config.property_package_aqueous
        prop_gas = self.config.property_package_gas
        time = self.flowsheet().time

        self.p_ref = pyo.Param(
            initialize=1.0, units=pyunits.bar, doc="Reference pressure"
        )
        self.n_ref = pyo.Param(
            initialize=1.0, units=pyunits.mol / pyunits.s, doc="Reference molar flow"
        )
        if self.config.gas_volume is not None:
            self.gas_volume = pyo.Param(
                initialize=float(self.config.gas_volume),
                mutable=True,
                units=pyunits.L,
                doc="Gas-phase volume",
            )

        self.partial_pressure = pyo.Var(
            time,
            prop_gas.component_list,
            initialize=1.0,
            bounds=(1e-15, 100.0),
            units=pyunits.bar,
            doc="Partial pressure of each gas component",
        )
        self.log_partial_pressure = pyo.Var(
            time,
            prop_gas.component_list,
            initialize=0.0,
            bounds=(-60.0, 5.0),
            units=pyunits.dimensionless,
            doc="ln(P / p_ref) of each gas component",
        )
        self.log_moles_gas_out = pyo.Var(
            time,
            prop_gas.component_list,
            initialize=-5.0,
            bounds=(-60.0, 10.0),
            units=pyunits.dimensionless,
            doc="ln(n_out / n_ref) of each gas component",
        )

        @self.Expression(time, doc="Gas-phase volume used by the ideal gas law")
        def gas_volume_basis(blk, t):
            if hasattr(blk, "gas_volume"):
                return blk.gas_volume
            return blk.volume[t]

        @self.Expression(
            time, prop_gas.component_list, doc="Partial pressure from the ideal gas law"
        )
        def partial_pressure_from_moles(blk, t, j):
            return pyunits.convert(
                blk.cv_gas.properties_out[t].moles_gas_comp[j]
                * blk.t_batch
                * (R_GAS_L_BAR * pyunits.L * pyunits.bar / pyunits.mol / pyunits.K)
                * blk.temperature
                / blk.gas_volume_basis[t],
                to_units=pyunits.bar,
            )

        @self.Constraint(
            time, prop_gas.component_list, doc="Partial pressure from its log"
        )
        def log_pressure_linking_eqns(blk, t, j):
            return (
                blk.partial_pressure[t, j]
                == pyo.exp(blk.log_partial_pressure[t, j]) * blk.p_ref
            )

        @self.Constraint(
            time, prop_gas.component_list, doc="Outlet gas molar flow from its log"
        )
        def log_moles_gas_linking_eqns(blk, t, j):
            return (
                blk.cv_gas.properties_out[t].moles_gas_comp[j]
                == pyo.exp(blk.log_moles_gas_out[t, j]) * blk.n_ref
            )

        ln_rho_over_mw = math.log(prop_gas.rho_solvent / prop_gas.MW_solvent)

        @self.Constraint(
            time, prop_gas.rxn_set, doc="Gas-liquid equilibrium in log form"
        )
        def gas_equilibrium_eqns(blk, t, r):
            stoich_r = prop_gas.stoich_dict[r]
            gas_term = sum(
                stoich_r[j] * (blk.log_partial_pressure[t, j] + ln_rho_over_mw)
                for j in stoich_r
                if j in prop_gas.component_list
            )
            aq_term = sum(
                stoich_r[j] * blk.log_conc_out[t, j]
                for j in stoich_r
                if j in prop_aq.component_list
            )
            return blk.log_k[r] == gas_term + aq_term

        @self.Constraint(time, prop_gas.component_list, doc="Ideal gas law in log form")
        def ideal_gas_eqns(blk, t, j):
            # P = n t R T / V  ->  ln(P/p_ref) - ln(n/n_ref) = ln(n_ref t R T / (V p_ref))
            factor = (
                pyunits.convert(
                    blk.n_ref
                    * blk.t_batch
                    * (R_GAS_L_BAR * pyunits.L * pyunits.bar / pyunits.mol / pyunits.K)
                    * blk.temperature
                    / blk.gas_volume_basis[t],
                    to_units=pyunits.bar,
                )
                / blk.p_ref
            )
            return blk.log_partial_pressure[t, j] - blk.log_moles_gas_out[
                t, j
            ] == pyo.log(factor)

        @self.Constraint(
            time, prop_gas.component_list, doc="Gas component mole balance"
        )
        def gas_mole_balance_eqns(blk, t, j):
            return blk.cv_gas.properties_out[t].moles_gas_comp[j] == (
                blk.cv_gas.properties_in[t].moles_gas_comp[j]
                + sum(
                    prop_gas.stoich_dict.get(r, {}).get(j, 0) * blk.rxn_extent[t, r]
                    for r in prop_gas.rxn_set
                )
                * blk.cv_aqueous.properties_out[t].flow_vol
            )

    def _get_stream_table_contents(self, time_point=0):
        streams = {
            "Aqueous Inlet": self.aqueous_inlet,
            "Aqueous Outlet": self.aqueous_outlet,
        }
        if self.config.property_package_precipitate is not None:
            streams["Precipitate Inlet"] = self.precipitate_inlet
            streams["Precipitate Outlet"] = self.precipitate_outlet
        if self.config.property_package_gas is not None:
            streams["Gas Inlet"] = self.gas_inlet
            streams["Gas Outlet"] = self.gas_outlet
        return create_stream_table_dataframe(streams, time_point=time_point)

    def _get_performance_contents(self, time_point=0):
        var_dict = {"Temperature": self.temperature}
        expr_dict = {
            f"Reaction {r} extent": self.rxn_extent[time_point, r]
            for r in self.merged_rxns
        }
        return {"vars": var_dict, "exprs": expr_dict}
