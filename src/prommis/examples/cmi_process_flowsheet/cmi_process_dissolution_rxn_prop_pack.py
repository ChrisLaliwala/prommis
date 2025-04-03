#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2023 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################
"""
Reaction package for Copper(II) Nitrate Dissolution in the Acid-Free Dissolution 
Process of Neodymium Magnets.
"""

from pyomo.environ import units as pyunits

from idaes.models.properties.modular_properties.base.generic_reaction import (
    ConcentrationForm,
)
# from idaes.models.properties.modular_properties.reactions.dh_rxn import constant_dh_rxn

config_dict = {
    "base_units": {
        "time": pyunits.s,
        "length": pyunits.m,
        "mass": pyunits.kg,
        "amount": pyunits.mol,
        "temperature": pyunits.K,
    },
    "rate_reactions": {
        "R1": {
            "stoichiometry": {
                ("Sol", "Nd2Fe14B"): -2,
                ("Aq", "Cu_2+"): -34,
                ("Vap", "O2"): -10.5,
                ("Aq", "Nd_3+"): 4,
                ("Aq", "Fe_2+"): 28,
                ("Sol", "Cu3(BO3)2"): 1,
                ("Sol", "Cu2O"): 15,
                ("Sol", "Cu"): 1,
                ("Aq", "Fe_3+"): 0,
                ("Aq", "NO3_-"): 0,
                ("Aq", "H2C2O4"): 0,
                ("Aq", "C2O4_2-"): 0,
                ("Aq", "NH4_+"): 0,
                ("Aq", "OH_-"): 0,
                ("Liq", "H2O"): 0,
                ("Sol", "Fe(OH)3"): 0,
                ("Sol", "Nd(OH)3"): 0,
                ("Sol", "Nd2(C2O4)3 * 10H2O"): 0,
            },
            # "heat_of_reaction": constant_dh_rxn,
            # "concentration_form": ConcentrationForm.molarity,
            # "parameter_data": {"dh_rxn_ref": (-9.87e6, pyunits.J / pyunits.mol)},
        }, 
        "R2": {
            "stoichiometry": {
                ("Sol", "Nd2Fe14B"): 0,
                ("Aq", "Cu_2+"): 0,
                ("Vap", "O2"): -3,
                ("Aq", "Nd_3+"): 0,
                ("Aq", "Fe_2+"): -12,
                ("Sol", "Cu3(BO3)2"): 0,
                ("Sol", "Cu2O"): 0,
                ("Sol", "Cu"): 0,
                ("Aq", "Fe_3+"): 8,
                ("Aq", "NO3_-"): 0,
                ("Aq", "H2C2O4"): 0,
                ("Aq", "C2O4_2-"): 0,
                ("Aq", "NH4_+"): 0,
                ("Aq", "OH_-"): 0,
                ("Liq", "H2O"): -6,
                ("Sol", "Fe(OH)3"): 4,
                ("Sol", "Nd(OH)3"): 0,
                ("Sol", "Nd2(C2O4)3 * 10H2O"): 0,
            },
            # "heat_of_reaction": constant_dh_rxn,
            # "concentration_form": ConcentrationForm.molarity,
            # "parameter_data": {"dh_rxn_ref": (-9.26e5, pyunits.J / pyunits.mol)},
        },
    },
}
