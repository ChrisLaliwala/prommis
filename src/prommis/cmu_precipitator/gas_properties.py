#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
Gas-phase property package for the optimization-based precipitator model.

Authors: Chris Laliwala
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


@declare_process_block_class("GasParameter")
class GasParameterData(PhysicalParameterBlock):
    """
    Property package for gas-phase species.

    The user passes a list of gas components (gas_comp_list), a dictionary of the
    gas-liquid equilibrium constants in log10 form (logkeq_dict), a dictionary of the
    stoichiometry of every gas-liquid reaction over both gas and aqueous species
    (stoich_dict), optionally a dictionary of standard reaction enthalpies (dHr_dict,
    J/mol) for the Van't Hoff temperature correction, and the solvent density (g/L) and
    molecular weight (g/mol) used to convert the solvent mole fraction basis of the
    equilibrium constants to concentrations.
    """

    CONFIG = PhysicalParameterBlock.CONFIG()
    CONFIG.declare(
        "gas_comp_list",
        ConfigValue(domain=list, description="List of gas-phase components in process"),
    )
    CONFIG.declare(
        "logkeq_dict",
        ConfigValue(
            domain=dict,
            description="Dictionary of gas-liquid equilibrium constants, log10(K) at 298.15 K",
        ),
    )
    CONFIG.declare(
        "stoich_dict",
        ConfigValue(
            domain=dict,
            description="Dictionary {reaction: {component: stoichiometric coefficient}} "
            "over the gas and aqueous species of each gas-liquid reaction",
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

    def build(self):
        """
        Callable method for block construction.
        """
        super().build()

        self.GasPhase = Phase()
        self.component_list = self.config.gas_comp_list

        # gas-liquid equilibrium reaction index
        self.rxn_set = Set(initialize=list(self.config.logkeq_dict.keys()))

        # stoichiometry, log10(K), ln(K) and enthalpy of each reaction
        self.stoich_dict = self.config.stoich_dict
        self.logkeq_dict = self.config.logkeq_dict
        self.ln_k_dict = {r: v * LN10 for r, v in self.config.logkeq_dict.items()}
        self.dHr_dict = self.config.dHr_dict

        # solvent properties for the mole-fraction to concentration conversion
        self.rho_solvent = self.config.rho_solvent
        self.MW_solvent = self.config.MW_solvent

        self._state_block_class = GasStateBlock

    @classmethod
    def define_metadata(cls, obj):
        """Define properties supported and units."""
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
        """
        Fixes state variables for state blocks.

        Returns:
            None
        """
        # Fix state variables
        return fix_state_vars(self)


@declare_process_block_class("GasStateBlock", block_class=_GasStateBlock)
class GasStateBlockData(StateBlockData):
    """
    State block for the gas-phase species.
    """

    def build(self):
        super().build()

        self.moles_gas_comp = Var(
            self.component_list,
            units=pyunits.mol / pyunits.s,
            initialize=1e-20,
            bounds=(1e-20, None),
            doc="Molar flow rate of each gas-phase component",
        )

    def get_material_flow_basis(self):
        return MaterialFlowBasis.molar

    def define_state_vars(self):
        return {
            "moles_gas_comp": self.moles_gas_comp,
        }
