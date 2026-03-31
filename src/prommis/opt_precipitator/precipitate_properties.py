#####################################################################################################
# "PrOMMiS" was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# ("PrOMMiS") initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
Precipitate property package for the optimization-based precipitator unit model.

Authors: Chris Laliwala
"""

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


@declare_process_block_class("PrecipitateParameter")
class PrecipitateParameterData(PhysicalParameterBlock):
    """
    Property package for solid (precipitate) species.

    Requires the user to pass:
      - precipitate_comp_list: list of solid component names
      - ln_k_sp_dict: dict mapping reaction index -> ln(Ksp) (natural log, temperature-corrected)
      - stoich_sp_dict: nested dict {rxn_index: {component: stoich_coeff}} for precipitation reactions
    """

    CONFIG = PhysicalParameterBlock.CONFIG()
    CONFIG.declare(
        "precipitate_comp_list",
        ConfigValue(domain=list, description="List of precipitate components in the system"),
    )
    CONFIG.declare(
        "ln_k_sp_dict",
        ConfigValue(
            domain=dict,
            description="Dictionary mapping reaction index to ln(Ksp) for precipitation reactions",
        ),
    )
    CONFIG.declare(
        "stoich_sp_dict",
        ConfigValue(
            domain=dict,
            description="Nested dict {rxn_index: {component: stoich_coeff}} for precipitation reactions",
        ),
    )

    def build(self):
        super().build()

        self.SolidPhase = Phase()
        self.component_list = self.config.precipitate_comp_list

        self.rxn_set = Set(
            initialize=list(self.config.ln_k_sp_dict.keys()),
            doc="Set of precipitation reaction indices",
        )

        self.ln_k_sp_dict = self.config.ln_k_sp_dict
        self.stoich_sp_dict = self.config.stoich_sp_dict

        self._state_block_class = PrecipitateStateBlock

    @classmethod
    def define_metadata(cls, obj):
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
        return fix_state_vars(self)


@declare_process_block_class("PrecipitateStateBlock", block_class=_PrecipitateStateBlock)
class PrecipitateStateBlockData(StateBlockData):
    """
    State block for precipitate species.

    State variable:
      - moles_precipitate_comp: molar flow of each precipitate component (mol/s)
    """

    def build(self):
        super().build()

        self.moles_precipitate_comp = Var(
            self.component_list,
            units=pyunits.mol / pyunits.s,
            initialize=1e-6,
            bounds=(1e-20, None),
            doc="Molar flow of each precipitate component (mol/s)",
        )

    def get_material_flow_basis(self):
        return MaterialFlowBasis.molar

    def define_state_vars(self):
        return {
            "moles_precipitate_comp": self.moles_precipitate_comp,
        }
