#####################################################################################################
# "PrOMMiS" was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# ("PrOMMiS") initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
Gas-phase property package for the optimization-based precipitator unit model.

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


@declare_process_block_class("GasParameter")
class GasParameterData(PhysicalParameterBlock):
    """
    Property package for gas-phase species.

    Requires the user to pass:
      - gas_comp_list: list of gas component names
      - ln_k_gas_dict: dict mapping reaction index -> ln(K) at T_ref = 298.15 K for gas-liquid reactions
      - stoich_gas_dict: nested dict {rxn_index: {component: stoich_coeff}} for gas reactions
      - rho_solvent: solvent density (g/L); default 1000.0 for water
      - MW_solvent: solvent molecular weight (g/mol); default 18.015 for water
      - dHr_gas_dict: dict mapping reaction index -> ΔHr (J/mol); omit or leave empty for isothermal

    rho_solvent and MW_solvent are used by the unit model to compute the ρ/MW correction
    factor in the gas-liquid Henry's Law equilibrium constraint.
    """

    CONFIG = PhysicalParameterBlock.CONFIG()
    CONFIG.declare(
        "gas_comp_list",
        ConfigValue(domain=list, description="List of gas-phase components in the system"),
    )
    CONFIG.declare(
        "ln_k_gas_dict",
        ConfigValue(
            domain=dict,
            description="Dictionary mapping reaction index to ln(K) for gas-liquid reactions",
        ),
    )
    CONFIG.declare(
        "stoich_gas_dict",
        ConfigValue(
            domain=dict,
            description="Nested dict {rxn_index: {component: stoich_coeff}} for gas reactions",
        ),
    )
    CONFIG.declare(
        "rho_solvent",
        ConfigValue(
            default=1000.0,
            domain=float,
            description="Solvent density (g/L); default 1000.0 for water",
        ),
    )
    CONFIG.declare(
        "MW_solvent",
        ConfigValue(
            default=18.015,
            domain=float,
            description="Solvent molecular weight (g/mol); default 18.015 for water",
        ),
    )
    CONFIG.declare(
        "dHr_gas_dict",
        ConfigValue(
            default={},
            domain=dict,
            description="Dict {rxn_index: ΔHr (J/mol)} for gas-liquid reactions; "
                        "reactions absent from dict are treated as isothermal (ΔHr = 0)",
        ),
    )

    def build(self):
        super().build()

        self.GasPhase = Phase()
        self.component_list = self.config.gas_comp_list

        self.rxn_set = Set(
            initialize=list(self.config.ln_k_gas_dict.keys()),
            doc="Set of gas-liquid reaction indices",
        )

        self.ln_k_gas_dict = self.config.ln_k_gas_dict
        self.stoich_gas_dict = self.config.stoich_gas_dict
        self.rho_solvent = self.config.rho_solvent
        self.MW_solvent = self.config.MW_solvent
        self.dHr_gas_dict = self.config.dHr_gas_dict

        self._state_block_class = GasStateBlock

    @classmethod
    def define_metadata(cls, obj):
        obj.define_custom_properties(
            {
                "moles_gas_comp": {"method": None},
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


class _GasStateBlock(StateBlock):
    def fix_initialization_states(self):
        return fix_state_vars(self)


@declare_process_block_class("GasStateBlock", block_class=_GasStateBlock)
class GasStateBlockData(StateBlockData):
    """
    State block for gas-phase species.

    State variable:
      - moles_gas_comp: molar flow of each gas component (mol/s)

    Partial pressures are internal variables on the unit model (not part of the port),
    computed via the ideal gas law using moles_gas_comp / flow_vol as gas concentration.
    """

    def build(self):
        super().build()

        self.moles_gas_comp = Var(
            self.component_list,
            units=pyunits.mol / pyunits.s,
            initialize=1e-6,
            bounds=(1e-20, None),
            doc="Molar flow of each gas-phase component (mol/s)",
        )

    def get_material_flow_basis(self):
        return MaterialFlowBasis.molar

    def define_state_vars(self):
        return {
            "moles_gas_comp": self.moles_gas_comp,
        }
