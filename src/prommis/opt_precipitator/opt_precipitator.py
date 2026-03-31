#####################################################################################################
# "PrOMMiS" was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# ("PrOMMiS") initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
r"""
Optimization-based Precipitator Unit Model
==========================================

Author: Chris Laliwala

This model predicts the equilibrium concentrations of aqueous species and the amounts of
precipitates (and optionally gas-phase species) formed at chemical equilibrium.

It is a migration of the standalone Pyomo precipitator model
(TorresCMULab/precipitator-unit-model, henryslaw branch) to the PrOMMiS/IDAES framework.

Mathematical Formulation
------------------------

All equilibrium expressions are written in natural-log (ln) space for numerical conditioning.
Log-space variables are linked to linear-space variables via exponential constraints.

**Aqueous equilibrium** (for r ∈ N_rxn_aq):

.. math:: \ln(K_r) = \sum_{i} \alpha_{i,r} \ln(C_i^f)

**Precipitation saturation inequality** (for r ∈ N_rxn_sp):

.. math:: \ln(Q_r) = \sum_{i \notin I_{sp}} \alpha_{i,r} \ln(C_i^f) \leq \ln(K_{r,sp})

**Objective** (only active when precipitate reactions are present):

.. math:: \min \sum_{r \in N_{rxn,sp}} (\ln(K_{r,sp}) - \ln(Q_r))^2

**Aqueous mass balance** (CMI convention; r_extent in mol/L):

.. math:: C_i^f = C_i^0 + \sum_r \alpha_{i,r} X_r

**Precipitate mass balance**:

.. math:: n_i^f = n_i^0 + \sum_r \alpha_{i,r} X_r \cdot \dot{V}

where :math:`\dot{V}` is the volumetric flow rate (L/s).

**Gas-liquid equilibrium** (for r ∈ N_rxn_g; requires gas property package):

.. math:: \ln(K_r) = \sum_{i \in I_{gas}} \alpha_{i,r}(\ln P_i + \ln(\rho/MW)) + \sum_{i \notin I_{gas}} \alpha_{i,r} \ln C_i^f

**Ideal gas law** (in log space):

.. math:: \ln P_i - \ln(\dot{n}_i / \dot{V}) = \ln(R T)

where R = 0.08314 L·bar/mol/K.

Property Package Convention
---------------------------

- ``stoich_aq_dict[r][species]``: stoichiometric coefficient of aqueous species in reaction r
  (covers both aqueous AND precipitation reactions for aqueous species).
- ``stoich_sp_dict[r][species]``: stoichiometric coefficient of solid species in precipitation
  reaction r.
- ``stoich_gas_dict[r][species]``: stoichiometric coefficient of species in gas-liquid reaction r
  (all phases included).
- ln(K) values must be pre-computed by the caller (log₁₀ → ln, Van't Hoff if T ≠ 298.15 K).
- H₂O is excluded from equilibrium expressions by omitting it from the stoich dicts.
"""

import pyomo.environ as pyo
from pyomo.common.config import Bool, ConfigBlock, ConfigValue
from pyomo.environ import units as pyunits

from idaes.core import (
    ControlVolume0DBlock,
    UnitModelBlockData,
    declare_process_block_class,
    useDefault,
)
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.tables import create_stream_table_dataframe


@declare_process_block_class("OptPrecipitator")
class OptPrecipitatorData(UnitModelBlockData):
    """
    Optimization-based precipitator unit model.

    Ports
    -----
    aqueous_inlet / aqueous_outlet
        Flow rate (L/s) and molar concentrations (mol/L) of aqueous species.
    precipitate_inlet / precipitate_outlet
        Molar flow rates (mol/s) of solid (precipitate) species.
    gas_inlet / gas_outlet  *(present only when property_package_gas is provided)*
        Molar flow rates (mol/s) of gas-phase species.
    """

    CONFIG = UnitModelBlockData.CONFIG()

    CONFIG.declare(
        "property_package_aqueous",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package for the aqueous phase",
        ),
    )
    CONFIG.declare(
        "property_package_args_aqueous",
        ConfigBlock(
            implicit=True,
            description="Arguments for constructing the aqueous property package",
        ),
    )
    CONFIG.declare(
        "property_package_precipitate",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package for the precipitate (solid) phase",
        ),
    )
    CONFIG.declare(
        "property_package_args_precipitate",
        ConfigBlock(
            implicit=True,
            description="Arguments for constructing the precipitate property package",
        ),
    )
    CONFIG.declare(
        "property_package_gas",
        ConfigValue(
            default=None,
            domain=is_physical_parameter_block,
            description="Property package for the gas phase (optional; omit for no gas)",
        ),
    )
    CONFIG.declare(
        "property_package_args_gas",
        ConfigBlock(
            implicit=True,
            description="Arguments for constructing the gas property package",
        ),
    )
    CONFIG.declare(
        "temperature",
        ConfigValue(
            default=298.15,
            domain=float,
            description="Process temperature (K); used in gas-phase ideal gas law",
        ),
    )
    CONFIG.declare(
        "has_equilibrium_reactions",
        ConfigValue(
            default=True,
            domain=Bool,
            description="Flag for equilibrium reaction construction",
        ),
    )

    def build(self):
        """Build the unit model."""
        super().build()

        prop_aq = self.config.property_package_aqueous
        prop_sp = self.config.property_package_precipitate

        # ------------------------------------------------------------------ #
        # Control volumes
        # ------------------------------------------------------------------ #
        self.cv_aqueous = ControlVolume0DBlock(
            dynamic=False,
            has_holdup=False,
            property_package=prop_aq,
            property_package_args=self.config.property_package_args_aqueous,
        )
        self.cv_precipitate = ControlVolume0DBlock(
            dynamic=False,
            has_holdup=False,
            property_package=prop_sp,
            property_package_args=self.config.property_package_args_precipitate,
        )
        self.cv_aqueous.add_state_blocks(has_phase_equilibrium=False)
        self.cv_precipitate.add_state_blocks(has_phase_equilibrium=False)

        # ------------------------------------------------------------------ #
        # Ports
        # ------------------------------------------------------------------ #
        self.add_inlet_port(block=self.cv_aqueous, name="aqueous_inlet")
        self.add_outlet_port(block=self.cv_aqueous, name="aqueous_outlet")
        self.add_inlet_port(block=self.cv_precipitate, name="precipitate_inlet")
        self.add_outlet_port(block=self.cv_precipitate, name="precipitate_outlet")

        # ------------------------------------------------------------------ #
        # Merged reaction sets and ln(K) parameter
        # ------------------------------------------------------------------ #
        self.merged_ln_k_dict = prop_aq.ln_k_aq_dict | prop_sp.ln_k_sp_dict
        self.merged_rxns = list(self.merged_ln_k_dict.keys())

        self.log_k = pyo.Param(
            self.merged_rxns,
            initialize=lambda m, r: self.merged_ln_k_dict[r],
            within=pyo.Reals,
            doc="ln(K) for each reaction (pre-corrected for temperature)",
        )

        # Reference molality / concentration for log-space scaling (dimensionless)
        self.c_ref = pyo.Param(
            initialize=1,
            units=pyunits.mol / pyunits.L,
            doc="Reference concentration (1 mol/L) to make log terms dimensionless",
        )

        # ------------------------------------------------------------------ #
        # Decision variables
        # ------------------------------------------------------------------ #
        self.rxn_extent = pyo.Var(
            self.merged_rxns,
            initialize=0,
            bounds=(-1e5, 1e5),
            units=pyunits.mol / pyunits.L,
            doc="Extent of reaction r (mol/L); change in concentration per unit volume",
        )

        # Log-space variable for outlet aqueous concentrations (internal)
        self.log_conc_out = pyo.Var(
            prop_aq.component_list,
            initialize=-5,
            bounds=(-100, 10),
            units=pyunits.dimensionless,
            doc="ln(C_out[i] / c_ref) for aqueous species i",
        )

        # log(Q) for precipitation reactions (ion activity product in log space)
        self.log_q_sp = pyo.Var(
            prop_sp.rxn_set,
            initialize=-5,
            bounds=(-100, 10),
            units=pyunits.dimensionless,
            doc="ln(Q_r) for precipitation reaction r (sum of alpha * log_conc_out)",
        )

        # ------------------------------------------------------------------ #
        # Constraints
        # ------------------------------------------------------------------ #

        # Link log variables to linear outlet concentrations
        @self.Constraint(
            self.flowsheet().time,
            prop_aq.component_list,
            doc="log_conc_out[i] = ln(conc_out[i] / c_ref)",
        )
        def log_conc_linking_eqns(blk, t, i):
            return (
                blk.cv_aqueous.properties_out[t].conc_mol_comp[i]
                == pyo.exp(blk.log_conc_out[i]) * self.c_ref
            )

        # Aqueous reaction equilibrium: ln(K) = sum(alpha * log_conc_out)
        @self.Constraint(
            self.flowsheet().time,
            prop_aq.rxn_set,
            doc="Aqueous reaction equilibrium in log space",
        )
        def aqueous_equil_eqns(blk, t, r):
            return blk.log_k[r] == sum(
                prop_aq.stoich_aq_dict[r][i] * blk.log_conc_out[i]
                for i in prop_aq.stoich_aq_dict[r]
            )

        # Precipitation: define log(Q) for each precipitation reaction
        # (sum over aqueous species only; solids have unit activity)
        @self.Constraint(
            self.flowsheet().time,
            prop_sp.rxn_set,
            doc="log(Q) definition for precipitation reactions (aqueous species only)",
        )
        def log_q_precipitate_equilibrium_rxn_eqns(blk, t, r):
            return blk.log_q_sp[r] == sum(
                prop_aq.stoich_aq_dict[r][i] * blk.log_conc_out[i]
                for i in prop_aq.stoich_aq_dict.get(r, {})
            )

        # Precipitation saturation inequality: log(Q) <= ln(K_sp)
        @self.Constraint(
            prop_sp.rxn_set,
            doc="Precipitation saturation: ion activity product <= solubility product",
        )
        def precip_sat_ineq(blk, r):
            return blk.log_q_sp[r] <= blk.log_k[r]

        # Aqueous mole balance: conc_out = conc_in + sum(alpha * rxn_extent)
        # CMI convention: no flow_vol factor; rxn_extent is in mol/L
        @self.Constraint(
            self.flowsheet().time,
            prop_aq.component_list,
            doc="Aqueous species mole balance",
        )
        def aqueous_mole_balance_eqns(blk, t, i):
            return blk.cv_aqueous.properties_out[t].conc_mol_comp[i] == (
                blk.cv_aqueous.properties_in[t].conc_mol_comp[i]
                + sum(
                    prop_aq.stoich_aq_dict.get(r, {}).get(i, 0) * blk.rxn_extent[r]
                    for r in blk.merged_rxns
                )
            )

        # Volume balance: outlet flow = inlet flow
        @self.Constraint(
            self.flowsheet().time,
            doc="Volumetric flow balance (no consumption of solvent)",
        )
        def vol_balance(blk, t):
            return (
                blk.cv_aqueous.properties_out[t].flow_vol
                == blk.cv_aqueous.properties_in[t].flow_vol
            )

        # Precipitate mole balance: moles_out = moles_in + sum(alpha * rxn_extent * flow_vol)
        @self.Constraint(
            self.flowsheet().time,
            prop_sp.component_list,
            doc="Precipitate species mole balance",
        )
        def precipitate_mole_balance_eqns(blk, t, i):
            return blk.cv_precipitate.properties_out[t].moles_precipitate_comp[i] == (
                blk.cv_precipitate.properties_in[t].moles_precipitate_comp[i]
                + sum(
                    prop_sp.stoich_sp_dict.get(r, {}).get(i, 0)
                    * blk.rxn_extent[r]
                    * blk.cv_aqueous.properties_out[t].flow_vol
                    for r in prop_sp.rxn_set
                )
            )

        # ------------------------------------------------------------------ #
        # Objective: minimise sum of squared saturation index residuals
        # min  sum_r ( ln(K_sp[r]) - ln(Q[r]) )^2  for r in N_rxn_sp
        # When no precipitation reactions exist the sum is zero (feasibility).
        # ------------------------------------------------------------------ #
        @self.Objective(
            doc="Minimise sum of squared log(K) - log(Q) residuals for precipitation reactions",
        )
        def min_saturation_index(blk):
            return sum(
                (blk.log_k[r] - blk.log_q_sp[r]) ** 2
                for r in prop_sp.rxn_set
            )

        # ------------------------------------------------------------------ #
        # Gas phase extension (Phase 3) — built conditionally
        # ------------------------------------------------------------------ #
        if self.config.property_package_gas is not None:
            self._build_gas_phase()

    def _build_gas_phase(self):
        """Add gas-phase control volume, ports, and constraints (Phase 3)."""
        prop_aq = self.config.property_package_aqueous
        prop_gas = self.config.property_package_gas

        self.cv_gas = ControlVolume0DBlock(
            dynamic=False,
            has_holdup=False,
            property_package=prop_gas,
            property_package_args=self.config.property_package_args_gas,
        )
        self.cv_gas.add_state_blocks(has_phase_equilibrium=False)
        self.add_inlet_port(block=self.cv_gas, name="gas_inlet")
        self.add_outlet_port(block=self.cv_gas, name="gas_outlet")

        # Add gas-reaction K values to the merged dict and log_k param
        for r, lnk in prop_gas.ln_k_gas_dict.items():
            self.merged_ln_k_dict[r] = lnk
            self.merged_rxns.append(r)

        # Reconstruct log_k to include gas reactions
        self.del_component(self.log_k)
        self.log_k = pyo.Param(
            self.merged_rxns,
            initialize=lambda m, r: self.merged_ln_k_dict[r],
            within=pyo.Reals,
            doc="ln(K) for each reaction including gas reactions",
        )

        # Extend rxn_extent to cover gas reactions
        self.del_component(self.rxn_extent)
        self.rxn_extent = pyo.Var(
            self.merged_rxns,
            initialize=0,
            bounds=(-1e5, 1e5),
            units=pyunits.mol / pyunits.L,
            doc="Extent of reaction r (mol/L)",
        )

        # Internal log-space variables for gas phase
        self.log_partial_pressure = pyo.Var(
            prop_gas.component_list,
            initialize=0,
            bounds=(-100, 10),
            units=pyunits.dimensionless,
            doc="ln(P_i / 1 bar) for gas species i",
        )
        self.partial_pressure = pyo.Var(
            prop_gas.component_list,
            initialize=1,
            bounds=(1e-20, 100),
            units=pyunits.bar,
            doc="Partial pressure of gas species i (bar)",
        )
        self.log_moles_gas_out = pyo.Var(
            prop_gas.component_list,
            initialize=-5,
            bounds=(-100, 10),
            units=pyunits.dimensionless,
            doc="ln(moles_gas_out[i] / (1 mol/s)) for gas species i",
        )

        t0 = self.flowsheet().time.first()
        flow_vol_in = self.cv_aqueous.properties_in[t0].flow_vol

        # Log-linking: P_i = exp(log_P_i)
        @self.Constraint(
            self.flowsheet().time,
            prop_gas.component_list,
            doc="Link log_partial_pressure to partial_pressure",
        )
        def log_pressure_linking_eqns(blk, t, i):
            return blk.partial_pressure[i] == pyo.exp(blk.log_partial_pressure[i])

        # Log-linking: moles_gas_out = exp(log_moles_gas_out)
        @self.Constraint(
            self.flowsheet().time,
            prop_gas.component_list,
            doc="Link log_moles_gas_out to moles_gas_out",
        )
        def log_moles_gas_linking_eqns(blk, t, i):
            return (
                blk.cv_gas.properties_out[t].moles_gas_comp[i]
                == pyo.exp(blk.log_moles_gas_out[i])
            )

        # Gas-liquid Henry's Law equilibrium
        # ln(K) = sum_{gas} alpha*(log_P + ln(rho/MW)) + sum_{aq} alpha*log_conc_out
        rho_over_MW = prop_gas.rho_solvent / prop_gas.MW_solvent  # mol/L

        @self.Constraint(
            self.flowsheet().time,
            prop_gas.rxn_set,
            doc="Gas-liquid equilibrium (Henry's Law, direction-independent)",
        )
        def gas_equil_eqns(blk, t, r):
            stoich_r = prop_gas.stoich_gas_dict[r]
            gas_term = sum(
                stoich_r[i] * (blk.log_partial_pressure[i] + pyo.log(rho_over_MW))
                for i in stoich_r
                if i in prop_gas.component_list
            )
            aq_term = sum(
                stoich_r[i] * blk.log_conc_out[i]
                for i in stoich_r
                if i in prop_aq.component_list
            )
            return blk.log_k[r] == gas_term + aq_term

        # Ideal gas law: ln(P_i) - ln(ng_i/flow_vol) = ln(R*T)
        # i.e. P_i = (ng_i / flow_vol) * R * T   (treating ng/flow_vol as gas concentration)
        R_gas = 0.08314  # L·bar / mol / K

        @self.Constraint(
            self.flowsheet().time,
            prop_gas.component_list,
            doc="Ideal gas law in log space (P = c_gas * R * T)",
        )
        def ideal_gas_eqns(blk, t, i):
            return (
                blk.log_partial_pressure[i] - blk.log_moles_gas_out[i]
                == pyo.log(R_gas * self.config.temperature / flow_vol_in)
            )

        # Gas mole balance: moles_gas_out = moles_gas_in + sum(alpha * rxn_extent * flow_vol)
        @self.Constraint(
            self.flowsheet().time,
            prop_gas.component_list,
            doc="Gas species mole balance",
        )
        def gas_mole_balance_eqns(blk, t, i):
            return blk.cv_gas.properties_out[t].moles_gas_comp[i] == (
                blk.cv_gas.properties_in[t].moles_gas_comp[i]
                + sum(
                    prop_gas.stoich_gas_dict.get(r, {}).get(i, 0)
                    * blk.rxn_extent[r]
                    * blk.cv_aqueous.properties_out[t].flow_vol
                    for r in prop_gas.rxn_set
                )
            )

    def _get_stream_table_contents(self, time_point=0):
        streams = {
            "Aqueous Inlet": self.aqueous_inlet,
            "Aqueous Outlet": self.aqueous_outlet,
            "Precipitate Inlet": self.precipitate_inlet,
            "Precipitate Outlet": self.precipitate_outlet,
        }
        if self.config.property_package_gas is not None:
            streams["Gas Inlet"] = self.gas_inlet
            streams["Gas Outlet"] = self.gas_outlet
        return create_stream_table_dataframe(streams, time_point=time_point)

    def _get_performance_contents(self, time_point=0):
        var_dict = {}
        for r in self.merged_rxns:
            var_dict[f"Reaction {r} extent"] = self.rxn_extent[r]
        return {"vars": var_dict}
