#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
r"""
Aqueous Property Package for the Optimization-based Precipitator
================================================================

Author: Chris Laliwala

Holds the aqueous species, the equilibrium constants and stoichiometry of the reactions
between them, and the aqueous side of the precipitation and gas-liquid reactions of the
:mod:`prommis.cmu_precipitator.opt_based_precipitator` unit model.

Configuration Arguments
-----------------------

``aqueous_comp_list``
    List of aqueous species. The solvent (water) is not a species: it has unit activity
    and is omitted from every stoichiometry dictionary.
``logkeq_dict``
    ``{reaction: log10(K)}`` at 298.15 K for the aqueous (homogeneous) reactions. The
    keys of this dictionary define the set of aqueous reactions, ``rxn_set``.
``stoich_dict``
    ``{reaction: {species: coefficient}}`` with products positive and reactants
    negative. It must contain the aqueous reactions and also the aqueous species of
    every precipitation and gas-liquid reaction declared in the precipitate and gas
    property packages, keyed by the same reaction names.
``dHr_dict``
    Optional ``{reaction: standard enthalpy of reaction (J/mol)}`` for the Van't Hoff
    correction; reactions absent from the dictionary are treated as isothermal.

State Variables
---------------

``flow_vol``
    Volumetric flow rate of solution (L/s).
``flow_mol_comp``
    Molar flow rate of each aqueous species (mol/s), bounded below by 1e-20 mol/s.

The concentration ``conc_mol_comp`` (mol/L) is an expression, ``flow_mol_comp / flow_vol``.
"""

import math

from pyomo.common.config import ConfigValue
from pyomo.environ import Set, Var
from pyomo.environ import units as pyunits

from idaes.core import (
    MaterialBalanceType,
    MaterialFlowBasis,
    Phase,
    PhysicalParameterBlock,
    StateBlock,
    StateBlockData,
    declare_process_block_class,
)
from idaes.core.util.initialization import fix_state_vars

LN10 = math.log(10.0)


@declare_process_block_class("AqueousParameter")
class AqueousParameterData(PhysicalParameterBlock):
    """
    Parameter block for the aqueous species.

    Builds ``component_list``, the aqueous reaction set ``rxn_set``, and the
    dictionaries ``stoich_dict``, ``logkeq_dict`` (log10), ``ln_k_dict`` (natural log)
    and ``dHr_dict`` read by the unit model.
    """

    CONFIG = PhysicalParameterBlock.CONFIG()
    CONFIG.declare(
        "aqueous_comp_list",
        ConfigValue(domain=list, description="List of aqueous components in process"),
    )
    CONFIG.declare(
        "logkeq_dict",
        ConfigValue(
            domain=dict,
            description="Dictionary of aqueous equilibrium constants, log10(K) at 298.15 K",
        ),
    )
    CONFIG.declare(
        "stoich_dict",
        ConfigValue(
            domain=dict,
            description="Dictionary {reaction: {aqueous component: stoichiometric coefficient}}",
        ),
    )
    CONFIG.declare(
        "dHr_dict",
        ConfigValue(
            default={},
            domain=dict,
            description="Dictionary {reaction: standard enthalpy of reaction (J/mol)}; "
            "reactions absent from the dictionary are treated as isothermal",
        ),
    )

    def build(self):
        """
        Callable method for Block construction.
        """
        super().build()

        self.AqueousPhase = Phase()
        self.component_list = self.config.aqueous_comp_list

        # aqueous equilibrium reaction index
        self.rxn_set = Set(initialize=list(self.config.logkeq_dict.keys()))

        # stoichiometry, log10(K), ln(K) and enthalpy of each reaction
        self.stoich_dict = self.config.stoich_dict
        self.logkeq_dict = self.config.logkeq_dict
        self.ln_k_dict = {r: v * LN10 for r, v in self.config.logkeq_dict.items()}
        self.dHr_dict = self.config.dHr_dict

        self._state_block_class = AqueousStateBlock

    @classmethod
    def define_metadata(cls, obj):
        """Define properties supported and units."""
        obj.define_custom_properties(
            {
                "flow_vol": {"method": None},
                "flow_mol_comp": {"method": None},
                "conc_mol_comp": {"method": None},
            }
        )
        obj.add_default_units(
            {
                "time": pyunits.s,
                "length": pyunits.m,
                "mass": pyunits.kg,
                "amount": pyunits.mol,
                "temperature": pyunits.K,
            }
        )


class _AqueousStateBlock(StateBlock):
    def fix_initialization_states(self):
        """
        Fixes state variables for state blocks.

        Returns:
            None
        """
        # Fix state variables
        return fix_state_vars(self)


@declare_process_block_class("AqueousStateBlock", block_class=_AqueousStateBlock)
class AqueousStateBlockData(StateBlockData):
    """
    State block for the aqueous species.

    State variables are the volumetric flow rate (L/s) and the molar flow rate of each
    component (mol/s); the molar concentration (mol/L) is an Expression.
    """

    def build(self):
        super().build()

        self.flow_vol = Var(
            units=pyunits.L / pyunits.s,
            initialize=1,
            bounds=(1e-8, None),
            doc="Volumetric flow rate of the solution",
        )

        self.flow_mol_comp = Var(
            self.component_list,
            units=pyunits.mol / pyunits.s,
            initialize=1e-6,
            bounds=(1e-20, None),
            doc="Molar flow rate of each aqueous component",
        )

        @self.Expression(self.component_list, doc="Molar concentration (mol/L)")
        def conc_mol_comp(b, j):
            return b.flow_mol_comp[j] / b.flow_vol

    def get_material_flow_basis(self):
        return MaterialFlowBasis.molar

    def default_material_balance_type(self):
        return MaterialBalanceType.componentTotal

    def get_material_flow_terms(self, p, j):
        return self.flow_mol_comp[j]

    def define_state_vars(self):
        return {
            "flow_vol": self.flow_vol,
            "flow_mol_comp": self.flow_mol_comp,
        }
