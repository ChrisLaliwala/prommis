#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
r"""
Precipitate Property Package for the Optimization-based Precipitator
====================================================================

Author: Chris Laliwala

Holds the solid species and the solubility products and stoichiometry of the
precipitation/dissolution reactions of the
:mod:`prommis.cmu_precipitator.opt_based_precipitator` unit model.

Configuration Arguments
-----------------------

``precipitate_comp_list``
    List of solid species.
``logkeq_dict``
    ``{reaction: log10(Ksp)}`` at 298.15 K, one entry per dissolution reaction written
    in the direction solid → ions. The keys define the precipitation reaction set,
    ``rxn_set``.
``stoich_dict``
    ``{reaction: {solid: coefficient}}`` over the solid species. The dissolving solid
    carries a negative coefficient; the ions it releases are declared, with positive
    coefficients, under the same reaction name in the aqueous property package.
``dHr_dict``
    Optional ``{reaction: standard enthalpy of dissolution (J/mol)}`` for the Van't
    Hoff correction; reactions absent from the dictionary are treated as isothermal.

State Variables
---------------

``moles_precipitate_comp``
    Molar flow rate of each solid species (mol/s), bounded below by 1e-20 mol/s.
"""

import math

from pyomo.common.config import ConfigValue
from pyomo.environ import Set, Var
from pyomo.environ import units as pyunits

from idaes.core import (
    MaterialFlowBasis,
    Phase,
    PhysicalParameterBlock,
    StateBlock,
    StateBlockData,
    declare_process_block_class,
)
from idaes.core.util.initialization import fix_state_vars

LN10 = math.log(10.0)


@declare_process_block_class("PrecipitateParameter")
class PrecipitateParameterData(PhysicalParameterBlock):
    """
    Parameter block for the solid species.

    Builds ``component_list``, the precipitation reaction set ``rxn_set``, and the
    dictionaries ``stoich_dict``, ``logkeq_dict`` (log10), ``ln_k_dict`` (natural log)
    and ``dHr_dict`` read by the unit model.
    """

    CONFIG = PhysicalParameterBlock.CONFIG()
    CONFIG.declare(
        "precipitate_comp_list",
        ConfigValue(
            domain=list, description="List of precipitate components in process."
        ),
    )
    CONFIG.declare(
        "logkeq_dict",
        ConfigValue(
            domain=dict,
            description="Dictionary of solubility products, log10(Ksp) at 298.15 K",
        ),
    )
    CONFIG.declare(
        "stoich_dict",
        ConfigValue(
            domain=dict,
            description="Dictionary {reaction: {precipitate: stoichiometric coefficient}}",
        ),
    )
    CONFIG.declare(
        "dHr_dict",
        ConfigValue(
            default={},
            domain=dict,
            description="Dictionary {reaction: standard enthalpy of dissolution (J/mol)}; "
            "reactions absent from the dictionary are treated as isothermal",
        ),
    )

    def build(self):
        """
        Callable method for block construction.
        """
        super().build()

        self.SolidPhase = Phase()
        self.component_list = self.config.precipitate_comp_list

        # precipitation equilibrium reaction index
        self.rxn_set = Set(initialize=list(self.config.logkeq_dict.keys()))

        # stoichiometry, log10(Ksp), ln(Ksp) and enthalpy of each reaction
        self.stoich_dict = self.config.stoich_dict
        self.logkeq_dict = self.config.logkeq_dict
        self.ln_k_dict = {r: v * LN10 for r, v in self.config.logkeq_dict.items()}
        self.dHr_dict = self.config.dHr_dict

        self._state_block_class = PrecipitateStateBlock

    @classmethod
    def define_metadata(cls, obj):
        """Define properties supported and units."""
        obj.define_custom_properties(
            {
                "moles_precipitate_comp": {"method": None},
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


class _PrecipitateStateBlock(StateBlock):
    def fix_initialization_states(self):
        """
        Fixes state variables for state blocks.

        Returns:
            None
        """
        # Fix state variables
        return fix_state_vars(self)


@declare_process_block_class(
    "PrecipitateStateBlock", block_class=_PrecipitateStateBlock
)
class PrecipitateStateBlockData(StateBlockData):
    """
    State block for the precipitate species.
    """

    def build(self):
        super().build()

        self.moles_precipitate_comp = Var(
            self.component_list,
            units=pyunits.mol / pyunits.s,
            initialize=1e-20,
            bounds=(1e-20, None),
            doc="Molar flow rate of each precipitate component",
        )

    def get_material_flow_basis(self):
        return MaterialFlowBasis.molar

    def define_state_vars(self):
        return {
            "moles_precipitate_comp": self.moles_precipitate_comp,
        }
