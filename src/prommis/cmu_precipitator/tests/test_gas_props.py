#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
import math

from pyomo.environ import ConcreteModel, Var

from idaes.core import FlowsheetBlock

import pytest

from prommis.cmu_precipitator import gas_properties as gas_thermo_prop_pack


@pytest.mark.unit
def test_build():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # CO2(aq) <-> CO2(g), Henry's law
    gas_comp_list = ["CO2(g)"]
    gas_log_keq_dict = {"G1": 3.245266}
    gas_stoich_dict = {"G1": {"CO2(aq)": -1, "CO2(g)": 1}}

    m.fs.gas_properties = gas_thermo_prop_pack.GasParameter(
        gas_comp_list=gas_comp_list,
        logkeq_dict=gas_log_keq_dict,
        stoich_dict=gas_stoich_dict,
    )

    assert set(m.fs.gas_properties.rxn_set) == {"G1"}
    assert m.fs.gas_properties.ln_k_dict["G1"] == pytest.approx(3.245266 * math.log(10))
    assert m.fs.gas_properties.dHr_dict == {}
    assert m.fs.gas_properties.rho_solvent == pytest.approx(1000.0)
    assert m.fs.gas_properties.MW_solvent == pytest.approx(18.015)

    m.fs.state = m.fs.gas_properties.build_state_block(m.fs.time)
    assert len(m.fs.state) == 1

    assert isinstance(m.fs.state[0].moles_gas_comp, Var)

    for i in m.fs.gas_properties.component_list:
        m.fs.state[0].moles_gas_comp[i].set_value(0.5)

    m.fs.state.fix_initialization_states()

    for i in m.fs.gas_properties.component_list:
        assert m.fs.state[0].moles_gas_comp[i].fixed
