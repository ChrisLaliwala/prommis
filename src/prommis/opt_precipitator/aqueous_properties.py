#####################################################################################################
# "PrOMMiS" was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# ("PrOMMiS") initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
Aqueous component property package for the optimization-based precipitator unit model.

Authors: Chris Laliwala
"""

from pyomo.common.config import ConfigValue
from pyomo.environ import Expression, Set, Var
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


@declare_process_block_class("AqueousParameter")
class AqueousParameterData(PhysicalParameterBlock):
    """
    Property package for aqueous species.

    Requires the user to pass:
      - aqueous_comp_list: list of aqueous component names
      - ln_k_aq_dict: dict mapping reaction index -> ln(K) at T_ref = 298.15 K (natural log)
      - stoich_aq_dict: nested dict {rxn_index: {component: stoich_coeff}} for aqueous reactions
      - dHr_aq_dict: dict mapping reaction index -> ΔHr (J/mol); omit or leave empty for isothermal
    """

    CONFIG = PhysicalParameterBlock.CONFIG()
    CONFIG.declare(
        "aqueous_comp_list",
        ConfigValue(domain=list, description="List of aqueous components in the system"),
    )
    CONFIG.declare(
        "ln_k_aq_dict",
        ConfigValue(
            domain=dict,
            description="Dictionary mapping reaction index to ln(K) for aqueous reactions",
        ),
    )
    CONFIG.declare(
        "stoich_aq_dict",
        ConfigValue(
            domain=dict,
            description="Nested dict {rxn_index: {component: stoich_coeff}} for aqueous reactions",
        ),
    )
    CONFIG.declare(
        "dHr_aq_dict",
        ConfigValue(
            default={},
            domain=dict,
            description="Dict {rxn_index: ΔHr (J/mol)} for aqueous reactions; "
                        "reactions absent from dict are treated as isothermal (ΔHr = 0)",
        ),
    )

    def build(self):
        super().build()

        self.AqueousPhase = Phase()
        self.component_list = self.config.aqueous_comp_list

        self.rxn_set = Set(
            initialize=list(self.config.ln_k_aq_dict.keys()),
            doc="Set of aqueous reaction indices",
        )

        self.ln_k_aq_dict = self.config.ln_k_aq_dict
        self.stoich_aq_dict = self.config.stoich_aq_dict
        self.dHr_aq_dict = self.config.dHr_aq_dict

        self._state_block_class = AqueousStateBlock

    @classmethod
    def define_metadata(cls, obj):
        obj.add_properties(
            {
                "flow_mol_comp": {"method": None},
                "flow_vol": {"method": None},
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
        return fix_state_vars(self)


@declare_process_block_class("AqueousStateBlock", block_class=_AqueousStateBlock)
class AqueousStateBlockData(StateBlockData):
    """
    State block for aqueous species.

    State variables:
      - flow_vol: volumetric flow rate (L/s)
      - flow_mol_comp: molar flow rate of each component (mol/s)

    Derived quantity:
      - conc_mol_comp: molar concentration (mol/L) = flow_mol_comp / flow_vol
        Defined as an Expression so that IDAES Arcs and Mixers can sum molar
        flows correctly across streams, with concentration computed automatically.
    """

    def build(self):
        super().build()

        self.flow_vol = Var(
            units=pyunits.L / pyunits.s,
            initialize=1,
            bounds=(1e-8, None),
            doc="Volumetric flow rate (L/s)",
        )

        self.flow_mol_comp = Var(
            self.component_list,
            units=pyunits.mol / pyunits.s,
            initialize=1e-6,
            bounds=(1e-20, None),
            doc="Molar flow rate of each aqueous component (mol/s)",
        )

        @self.Expression(
            self.component_list,
            doc="Molar concentration of each aqueous component (mol/L) = flow_mol_comp / flow_vol",
        )
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
