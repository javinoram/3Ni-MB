import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import h5py

import argparse

import tenpy
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.tools import hdf5_io
from tenpy.tools.process import memory_usage

tenpy.tools.misc.setup_logging(to_stdout="INFO")
sys.path.append("../")
from trimeric_molecule_model import TrimericMoleculeDouble, TrimericMoleculeLinear, TrimericMoleculeParallel, TrimericMoleculeParallelOdd, TrimericMoleculeAlternatedOdd


def main(J, J13, Jinter, hz, hx, L, model="linear"):
    # DMRG parameters
    dmrg_params = {
        "mixer": True,  # setting this to True helps to escape local minima
       # "mixer_params": {"amplitude": 0.01, "decay": 2.},
        
        "max_E_err": 1.0e-12,
        "trunc_params": {
            "svd_min": 1.0e-8,
        },
        "max_sweeps": 36,
        "min_sweeps": 10,
        "chi_list": {0: 50, 4: 100, 8: 200, 12: 400, 16: 800, 20: 1600, 24: 3200},
        "norm_tol": 1e-5,  # default value
    }

    print("\n", "***" * 10)
    print("Preparing dmrg with dmrg_params:")
    print(dmrg_params)

    # Generate model
    model_params = {
        "hz": hz,
        "hx": hx,
        "J": J,
        "J13": J13,
        "Jinter": Jinter,
        "J34": 0.0,
        "L": L,
        "bc": "open",
        "cons_Sz": "None",
    }

    if hz == 0.0:
        model_params["hz"] = 1e-6

    print("\n", "***" * 10)

    if model == "linear":
        model_params["L"] = 2 * L
        fname = f"{model}-L_{L}_Jint_{Jinter:.2f}_hz_{hz:.2f}_hx{hx:.2f}"
        print("Generating TrimericMoleculeDouble with model_params:")
        print(model_params)
        model = TrimericMoleculeLinear(model_params)
    elif model == "alternated" and 2*L%2 == 1.0:
        model_params["L"] = int(model_params["L"]*2)
        fname = f"{model}-L_{L}_Jint_{Jinter:.2f}_hz_{hz:.2f}_hx{hx:.2f}"
        print("Generating TrimericMoleculeAlternatedOdd with model_params:")
        print(model_params)
        model = TrimericMoleculeAlternatedOdd(model_params)
    elif model == "alternated":
        fname = f"{model}-L_{L}_Jint_{Jinter:.2f}_hz_{hz:.2f}_hx{hx:.2f}"
        print("Generating TrimericMoleculeDouble with model_params:")
        print(model_params)
        model = TrimericMoleculeDouble(model_params)
    elif model == "parallel" and 2*L%2 == 1.0:
        model_params["Jnn"] = model_params["Jnnn"] = 0.0
        model_params["L"] = int(model_params["L"]*2)
        fname = f"{model}-L_{L}_Jint_{Jinter:.2f}_hz_{hz:.2f}_hx{hx:.2f}"
        print("Generating TrimericMoleculeParallelOdd with model_params:")
        print(model_params)
        model = TrimericMoleculeParallelOdd(model_params)
    elif model == "parallel":
        model_params["Jnn"] = model_params["Jnnn"] = 0.0
        fname = f"{model}-L_{L}_Jint_{Jinter:.2f}_hz_{hz:.2f}_hx{hx:.2f}"
        print("Generating TrimericMoleculeParallel with model_params:")
        print(model_params)
        model = TrimericMoleculeParallel(model_params)

    N = model.lat.N_sites

    # Initial State
    # this selects a charge sector!

    
    # p_state = ["up"] * len(model.lat.unit_cell) * (model_params["L"])
    p_state = ["up"] * N
    psi_0 = MPS.from_product_state(model.lat.mps_sites(), p_state)

    # shuffle initial state (in case of non-polarized initialization)
    p_state = random.sample(p_state, len(p_state))
    psi_0 = MPS.from_product_state(model.lat.mps_sites(), p_state=p_state)

    # charge sector is the hilbert space sector with a certain conserved quantity (conserved 'charge')
    # in this case Sz is not conserved so the 'charge' value can change
    charge_sector = psi_0.get_total_charge(True)

    print("\n", "***" * 10)
    print("Initial MPS expectations values:")
    print("Total Sz:", psi_0.expectation_value("Sz").sum())
    print("Sz_i:", psi_0.expectation_value("Sz"))
    print("Sx_i:", psi_0.expectation_value("Sx"))
    print("Sy_i:", psi_0.expectation_value("Sy"))
    print("Sz molecules:", psi_0.expectation_value("Sz").reshape(-1, 3).sum(axis=1))
    print("Sx molecules:", psi_0.expectation_value("Sx").reshape(-1, 3).sum(axis=1))
    print("Sy molecules:", psi_0.expectation_value("Sy").reshape(-1, 3).sum(axis=1))

    print("\n", "***" * 10)
    print("Running DMRG. This may take a while...")

    psi = psi_0.copy()
    eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    E, psi = eng.run()  # the main work; modifies psi in place

    print("\n", "***" * 10)
    print("DMRG Finished!")

    with np.printoptions(suppress=True):
        print("\n", "***" * 10)
        print("Final MPS expectations values:")

        print("Total Sz:", psi.expectation_value("Sz").sum())
        print("Sz_i:", psi.expectation_value("Sz"))
        print("Sx_i:", psi.expectation_value("Sx"))
        print("Sy_i:", psi.expectation_value("Sy"))
        print("Sz molecules:", psi.expectation_value("Sz").reshape(-1, 3).sum(axis=1))
        print("Sx molecules:", psi.expectation_value("Sx").reshape(-1, 3).sum(axis=1))
        print("Sy molecules:", psi.expectation_value("Sy").reshape(-1, 3).sum(axis=1))

    data = {
        "psi_0": psi_0,
        "model": model,
        "dmrg_params": dmrg_params,
        "model_params": model_params,
        "sweep_stats": eng.sweep_stats,
        "update_stats": eng.update_stats,
        "memory_usage": memory_usage(),
    }

    with h5py.File(f"non-interacting/datos/data_{fname}.h5", "w") as f:
        hdf5_io.save_to_hdf5(f, data)

    data = {"psi": psi}
    with h5py.File(f"non-interacting/datos/data_psi_{fname}.h5", "w") as f:
        hdf5_io.save_to_hdf5(f, data)

    sweep_stats = eng.sweep_stats
    update_stats = eng.update_stats

    print("\n", "***" * 10)
    print("Run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--hz", type=float, required=True)
    parser.add_argument("--Jinter", type=float, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--hx", type=float, required=True)
    parser.add_argument("--J", type=float, required=True)
    parser.add_argument("--J13", type=float, required=True)
    args = parser.parse_args()

    main(J=args.J, J13=args.J13, Jinter=args.Jinter, hz=args.hz, hx=args.hx, L=args.L, model=args.model)
