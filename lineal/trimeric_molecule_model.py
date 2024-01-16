import numpy as np

from tenpy.tools.params import asConfig

from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel
from tenpy.models import lattice
from tenpy.models.lattice import IrregularLattice

from tenpy.networks.site import SpinSite


class TrimericMolecule(CouplingModel, NearestNeighborModel, MPOModel):
    def __init__(self, model_params):
        # 0) read out/set default parameters
        model_params = asConfig(model_params, "TrimericMolecule")
        L = model_params.get("L", 3)
        J = J12 = J23 = model_params.get("J", 1.49)
        J13 = model_params.get("J13", -0.89)
        J34 = model_params.get("J34", 0.0)
        hz = model_params.get("hz", 0.0)
        hx = model_params.get("hx", 0.0)

        # 1-3):
        # site = SpinSite(S=1, conserve='Sz')
        site = SpinSite(S=1, conserve="None")
        # 4) lattice
        lat = lattice.Lattice(
            Ls=[L],
            unit_cell=[site, site, site],
            order="default",
            bc="open",
            bc_MPS="finite",
            basis=np.array([[3, 0]]),
            positions=np.array([[0, 0], [0.5, 0.1], [1, 0]]),
            pairs=None,
        )  # {'nearest_neighbors': [( 0, 1, 0 ), ( 1, 2, 0.5 )],
        # 'next_nearest_neighbors': [( 0, 1, 1 )]})#)
        lat.pairs = lat.find_coupling_pairs()
        # 5) initialize CouplingModel
        CouplingModel.__init__(self, lat)

        # 6) add terms of the Hamiltonian
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hz, u, "Sz")
            self.add_onsite(-hx, u, "Sx")

        self.distances = list(self.lat.pairs.keys())
        # nearest neighbors: J12 J23
        for u1, u2, dx in self.lat.pairs[self.distances[0]]:
            self.add_coupling(-2 * J, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * J, u1, "Sp", u2, "Sm", dx, plus_hc=True)
        # next nearest neighbors: J13
        for u1, u2, dx in self.lat.pairs[self.distances[1]]:
            self.add_coupling(-2 * J13, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * J13, u1, "Sp", u2, "Sm", dx, plus_hc=True)

        for u1, u2, dx in self.lat.pairs[self.distances[2]]:
            self.add_coupling(-2 * J34, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * J34, u1, "Sp", u2, "Sm", dx, plus_hc=True)

        # 7) initialize H_MPO
        MPOModel.__init__(self, lat, self.calc_H_MPO())
        # 8) initialize H_bond (the order of 7/8 doesn't matter)
        # NearestNeighborModel.__init__(self, lat, self.calc_H_bond())


class TrimericMoleculeLinear(CouplingModel, NearestNeighborModel, MPOModel):
    def __init__(self, model_params):
        # 0) read out/set default parameters
        model_params = asConfig(model_params, "TrimericMolecule")
        L = model_params.get("L", 3)
        bc = model_params.get("bc", "open")
        cons_Sz = model_params.get("cons_Sz", "None")
        hz = model_params.get("hz", 1e-6)
        hx = model_params.get("hx", 0.0)

        J = J12 = J23 = model_params.get("J", 1.49)
        J13 = model_params.get("J13", -0.89)
        J34 = model_params.get("Jinter", 0.0)
        J35 = J24 = model_params.get("J35", 0.0)
        # 1-3):
        site = SpinSite(S=1, conserve=cons_Sz)
        # 4) lattice
        self.lat = lattice.Lattice(
            Ls=[L],
            unit_cell=[site, site, site],
            order="default",
            bc=bc,
            bc_MPS="finite",
            basis=np.array([[3, 0]]),
            positions=np.array([[0, 0], [0.5, 0], [1, 0]]),
            pairs=None,
        )
        # lat.pairs = lat.find_coupling_pairs()

        coupling_pairs = self.lat.find_coupling_pairs()
        pairs_dist = list(coupling_pairs.keys())

        self.lat.pairs["nearest_neighbors"] = coupling_pairs[pairs_dist[0]]
        self.lat.pairs["n_nearest_neighbors"] = coupling_pairs[pairs_dist[1]]
        self.lat.pairs["nn_nearest_neighbors"] = coupling_pairs[pairs_dist[2]]
        self.lat.pairs["nnn_nearest_neighbors"] = coupling_pairs[pairs_dist[3]]

        # 5) initialize CouplingModel
        CouplingModel.__init__(self, self.lat)

        # 6) add terms of the Hamiltonian
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hz, u, "Sz")
            self.add_onsite(-hx, u, "Sx")

        self.distances = list(self.lat.pairs.keys())
        # nearest neighbors: J12 J23 (within molecule, interaction 1-2 and 2-3)
        for u1, u2, dx in self.lat.pairs["nearest_neighbors"]:
            self.add_coupling(-2 * J, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * J, u1, "Sp", u2, "Sm", dx, plus_hc=True)
        # next nearest neighbors: J13 (within molecule, interaction 1-3)
        for u1, u2, dx in self.lat.pairs["n_nearest_neighbors"]:
            self.add_coupling(-2 * J13, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * J13, u1, "Sp", u2, "Sm", dx, plus_hc=True)

        # next-next nearest neighbors: J34 (between molecules)
        for u1, u2, dx in self.lat.pairs["nn_nearest_neighbors"]:
            self.add_coupling(-2 * J34, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * J34, u1, "Sp", u2, "Sm", dx, plus_hc=True)

        # next-next-nex nearest neighbors: J35 J24 (between molecules)
        for u1, u2, dx in self.lat.pairs["nnn_nearest_neighbors"]:
            self.add_coupling(-2 * J35, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * J35, u1, "Sp", u2, "Sm", dx, plus_hc=True)

        # 7) initialize H_MPO
        MPOModel.__init__(self, self.lat, self.calc_H_MPO())
        # 8) initialize H_bond (the order of 7/8 doesn't matter)
        # NearestNeighborModel.__init__(self, lat, self.calc_H_bond())


class TrimericMoleculeDouble(CouplingModel, MPOModel):
    def __init__(self, model_params):
        # 0) read out/set default parameters
        model_params = asConfig(model_params, "TrimericMoleculeAlternated")
        L = model_params.get("L", 2)
        bc = model_params.get("bc", "open")
        bc_MPS = model_params.get("bc_MPS", "finite")
        order = model_params.get("order", "default")

        cons_Sz = model_params.get("cons_Sz", "None")
        hz = model_params.get("hz", 0.0)
        hx = model_params.get("hx", 0.0)

        J = J12 = J23 = model_params.get("J", 1.49)
        J13 = model_params.get("J13", -0.89)
        J35 = model_params.get("Jinter", 0.0)
        J34 = J36 = model_params.get("J34", 0.0)

        # 1-3):
        site = SpinSite(S=1, conserve=cons_Sz)
        # 4) lattice
        self.lat = lattice.Lattice(
            Ls=[L],
            unit_cell=[site, site, site, site, site, site],
            order=order,
            bc=bc,
            bc_MPS=bc_MPS,
            basis=np.array([[6, 0]]),
            positions=np.array(
                [[0, 0], [0.5, 0], [1, 0], [3.5, 0.5], [3.5, 0], [3.5, -0.5]]
            ),
            pairs=None,
        )

        coupling_pairs = self.lat.find_coupling_pairs()
        pairs_dist = list(coupling_pairs.keys())

        self.lat.pairs["nearest_neighbors"] = coupling_pairs[pairs_dist[0]]
        self.lat.pairs["n_nearest_neighbors"] = coupling_pairs[pairs_dist[1]]
        self.lat.pairs["nn_nearest_neighbors"] = coupling_pairs[pairs_dist[2]]
        self.lat.pairs["nnn_nearest_neighbors"] = coupling_pairs[pairs_dist[3]]

        # 5) initialize CouplingModel
        CouplingModel.__init__(self, self.lat)

        # 6) add terms of the Hamiltonian
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hz, u, "Sz")
            self.add_onsite(-hx, u, "Sx")

        self.distances = list(self.lat.pairs.keys())
        # nearest neighbors: J12 J23 (within molecule, interaction 1-2 and 2-3)
        for u1, u2, dx in self.lat.pairs["nearest_neighbors"]:
            self.add_coupling(-2 * J, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * J, u1, "Sp", u2, "Sm", dx, plus_hc=True)
        # next nearest neighbors: J13 (within molecule, interaction 1-3)
        for u1, u2, dx in self.lat.pairs["n_nearest_neighbors"]:
            self.add_coupling(-2 * J13, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * J13, u1, "Sp", u2, "Sm", dx, plus_hc=True)

        # next-next nearest neighbors: J35 (between molecules, edge-spin with center-spin)
        for u1, u2, dx in self.lat.pairs["nn_nearest_neighbors"]:
            self.add_coupling(-2 * J35, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * J35, u1, "Sp", u2, "Sm", dx, plus_hc=True)

        # next-next-nex nearest neighbors: J34 J36 (between molecules, edge-spin with two edge-spins)
        for u1, u2, dx in self.lat.pairs["nnn_nearest_neighbors"]:
            self.add_coupling(-2 * J34, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * J34, u1, "Sp", u2, "Sm", dx, plus_hc=True)

        # 7) initialize H_MPO
        MPOModel.__init__(self, self.lat, self.calc_H_MPO())
        # 8) initialize H_bond (the order of 7/8 doesn't matter)
        # NearestNeighborModel.__init__(self, lat, self.calc_H_bond())
        # CouplingModel.calc_H_bond(self)

class TrimericMoleculeAlternatedOdd(CouplingModel, MPOModel):
    #IRREGULAR LATTICE TENPY!!!
    def __init__(self, model_params):
        # 0) read out/set default parameters
        model_params = asConfig(model_params, "TrimericMoleculeAlternated")
        kind = model_params.get("kind", 0) # 0 or 1, determines the ends of the chain, horizontal (0) or vertical(1)
        L = model_params.get("L", 2)
        bc = model_params.get("bc", "open")
        bc_MPS = model_params.get("bc_MPS", "finite")
        order = model_params.get("order", "default")

        cons_Sz = model_params.get("cons_Sz", "None")
        hz = model_params.get("hz", 0.0)
        hx = model_params.get("hx", 0.0)

        J = J12 = J23 = model_params.get("J", 1.49)
        J13 = model_params.get("J13", -0.89)
        J35 = model_params.get("Jinter", 0.0)
        J34 = J36 = model_params.get("J34", 0.0)

        if kind == 0 or kind == 1:
            #L += 1
            L -=1


        # 1-3):
        site = SpinSite(S=1, conserve=cons_Sz)
        # 4) lattice
        self.lat = lattice.Lattice(
            Ls=[L],
            unit_cell=[site, site, site, site, site, site],
            order=order,
            bc=bc,
            bc_MPS=bc_MPS,
            basis=np.array([[6, 0]]),
            positions=np.array(
                [[0, 0], [0.5, 0], [1, 0], [3.5, 0.5], [3.5, 0], [3.5, -0.5]]
            ),
            pairs=None,
        )
        reg_lat = self.lat.copy()

        # coupling_pairs = self.lat.find_coupling_pairs()
        # pairs_dist = list(coupling_pairs.keys())

        # self.lat.pairs["nearest_neighbors"] = coupling_pairs[pairs_dist[0]]
        # self.lat.pairs["n_nearest_neighbors"] = coupling_pairs[pairs_dist[1]]
        # self.lat.pairs["nn_nearest_neighbors"] = coupling_pairs[pairs_dist[2]]
        # self.lat.pairs["nnn_nearest_neighbors"] = coupling_pairs[pairs_dist[3]]

        if kind == 0:
            self.lat = IrregularLattice(self.lat, remove=[[-1,3],[-1,4],[-1,5]])
            #self.lat = IrregularLattice(self.lat, remove=[[-1,3],[-1,4],[-1,5]], 
            #                                    )
            # self.lat = IrregularLattice(self.lat, add=([[L,0],[L,1],[L,2]], 
            #                             [L*6,L*6+1,L*6+2]),
            #                             add_unit_cell=[site, site, site],
            #                             add_positions=self.lat.unit_cell_positions[:3])
        elif kind == 1:
            #self.lat = IrregularLattice(self.lat, remove=[[0,0],[0,1],[0,2]])
            self.lat = IrregularLattice(self.lat, add=([[-1,3],[-1,4],[-1,5]], None),
                                        add_unit_cell=[site, site, site],
                                        add_positions=self.lat.unit_cell_positions[3:])        # else do nothing. Same as the even lattice

        coupling_pairs = self.lat.find_coupling_pairs()
        pairs_dist = list(coupling_pairs.keys())

        self.lat.pairs["nearest_neighbors"] = coupling_pairs[pairs_dist[0]]
        self.lat.pairs["n_nearest_neighbors"] = coupling_pairs[pairs_dist[1]]
        self.lat.pairs["nn_nearest_neighbors"] = coupling_pairs[pairs_dist[2]]
        self.lat.pairs["nnn_nearest_neighbors"] = coupling_pairs[pairs_dist[3]]

        # 5) initialize CouplingModel
        CouplingModel.__init__(self, self.lat)

        # 6) add terms of the Hamiltonian
        for u in range(len(self.lat.unit_cell)):
            print(u)
            self.add_onsite(-hz, u, "Sz")
            self.add_onsite(-hx, u, "Sx")
        # for u in range(self.lat.N_sites):
        #     self.add_onsite_term(-hz, u, "Sz")
        #     self.add_onsite_term(-hx, u, "Sx")

        self.distances = list(self.lat.pairs.keys())
        # nearest neighbors: J12 J23 (within molecule, interaction 1-2 and 2-3)
        for u1, u2, dx in self.lat.pairs["nearest_neighbors"]:
            self.add_coupling(-2 * J, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * J, u1, "Sp", u2, "Sm", dx, plus_hc=True)
        # next nearest neighbors: J13 (within molecule, interaction 1-3)
        for u1, u2, dx in self.lat.pairs["n_nearest_neighbors"]:
            self.add_coupling(-2 * J13, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * J13, u1, "Sp", u2, "Sm", dx, plus_hc=True)

        # next-next nearest neighbors: J35 (between molecules, edge-spin with center-spin)
        for u1, u2, dx in self.lat.pairs["nn_nearest_neighbors"]:
            self.add_coupling(-2 * J35, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * J35, u1, "Sp", u2, "Sm", dx, plus_hc=True)

        # next-next-nex nearest neighbors: J34 J36 (between molecules, edge-spin with two edge-spins)
        for u1, u2, dx in self.lat.pairs["nnn_nearest_neighbors"]:
            self.add_coupling(-2 * J34, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * J34, u1, "Sp", u2, "Sm", dx, plus_hc=True)

        # 7) initialize H_MPO
        MPOModel.__init__(self, self.lat, self.calc_H_MPO())
        # 8) initialize H_bond (the order of 7/8 doesn't matter)
        # NearestNeighborModel.__init__(self, lat, self.calc_H_bond())
        # CouplingModel.calc_H_bond(self)

class TrimericMoleculeParallel(CouplingModel, MPOModel):
    def __init__(self, model_params):
        # 0) read out/set default parameters
        model_params = asConfig(model_params, "TrimericMoleculeParallel")
        L = model_params.get("L", 2)
        bc = model_params.get("bc", "open")
        bc_MPS = model_params.get("bc_MPS", "finite")
        order = model_params.get("order", "default")

        cons_Sz = model_params.get("cons_Sz", "None")
        hz = model_params.get("hz", 0.0)
        hx = model_params.get("hx", 0.0)

        J = J12 = J23 = model_params.get("J", 1.49)
        J13 = model_params.get("J13", -0.89)
        Jinter = J25 = model_params.get("Jinter", 0.0)
        Jnn = J14 = J36 = model_params.get("Jnn", 0.0)
        Jnnn = model_params.get("Jnnn", 0.0)

        # 1-3):
        site = SpinSite(S=1, conserve=cons_Sz)
        # 4) lattice
        self.lat = lattice.Lattice(
            Ls=[L],
            unit_cell=[site, site, site, site, site, site],
            order=order,
            bc=bc,
            bc_MPS=bc_MPS,
            basis=np.array([[6, 0]]),
            positions=np.array(
                [[0.5, 0.5], [0.5, 0], [0.5, -0.5],
                 [3.5, 0.5], [3.5, 0], [3.5, -0.5]]
            ),
            pairs=None,
        )

        coupling_pairs = self.lat.find_coupling_pairs(max_dx=6)
        pairs_dist = list(coupling_pairs.keys())

        self.lat.pairs["nearest_neighbors"] = coupling_pairs[pairs_dist[0]]
        self.lat.pairs["n_nearest_neighbors"] = coupling_pairs[pairs_dist[1]]
        self.lat.pairs["nn_nearest_neighbors"] = coupling_pairs[pairs_dist[2]]
        self.lat.pairs["nnn_nearest_neighbors"] = coupling_pairs[pairs_dist[3]]

        # 5) initialize CouplingModel
        CouplingModel.__init__(self, self.lat)

        # 6) add terms of the Hamiltonian
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hz, u, "Sz")
            self.add_onsite(-hx, u, "Sx")

        self.distances = list(self.lat.pairs.keys())
        # nearest neighbors: J12 J23 (within molecule, interaction 1-2 and 2-3)
        for u1, u2, dx in self.lat.pairs["nearest_neighbors"]:
            self.add_coupling(-2 * J, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * J, u1, "Sp", u2, "Sm", dx, plus_hc=True)
        # next nearest neighbors: J13 (within molecule, interaction 1-3)
        for u1, u2, dx in self.lat.pairs["n_nearest_neighbors"]:
            self.add_coupling(-2 * J13, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * J13, u1, "Sp", u2, "Sm", dx, plus_hc=True)

        # next-next nearest neighbors: J25 (between molecules, center-spin with center-spin)
        for u1, u2, dx in self.lat.pairs["nn_nearest_neighbors"][2:4]:
            self.add_coupling(-2 * Jinter, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * Jinter, u1, "Sp", u2, "Sm", dx, plus_hc=True)
    
        # next-next nearest neighbors: J14 J36 (between molecules, edge-spin with edge-spin)
        for u1, u2, dx in self.lat.pairs["nn_nearest_neighbors"][0:2] + self.lat.pairs["nn_nearest_neighbors"][4:6]:
            self.add_coupling(-2 * Jnn, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * Jnn, u1, "Sp", u2, "Sm", dx, plus_hc=True)

        # next-next-next nearest neighbors: J15 J24 J26 J35 (between molecules, center-spin with center-spin)
        for u1, u2, dx in self.lat.pairs["nnn_nearest_neighbors"]:
            self.add_coupling(-2 * Jnnn, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * Jnnn, u1, "Sp", u2, "Sm", dx, plus_hc=True)
        
        # 7) initialize H_MPO
        MPOModel.__init__(self, self.lat, self.calc_H_MPO())
        # 8) initialize H_bond (the order of 7/8 doesn't matter)
        # NearestNeighborModel.__init__(self, lat, self.calc_H_bond())
        # CouplingModel.calc_H_bond(self)

class TrimericMoleculeParallelOdd(CouplingModel, MPOModel):
    def __init__(self, model_params):
        # 0) read out/set default parameters
        model_params = asConfig(model_params, "TrimericMoleculeParallel")
        L = model_params.get("L", 3)
        bc = model_params.get("bc", "open")
        bc_MPS = model_params.get("bc_MPS", "finite")
        order = model_params.get("order", "default")

        cons_Sz = model_params.get("cons_Sz", "None")
        hz = model_params.get("hz", 0.0)
        hx = model_params.get("hx", 0.0)

        J = J12 = J23 = model_params.get("J", 1.49)
        J13 = model_params.get("J13", -0.89)
        Jinter = J25 = model_params.get("Jinter", 0.0)
        Jnn = J14 = J36 = model_params.get("Jnn", 0.0)
        Jnnn = model_params.get("Jnnn", 0.0)

        # 1-3):
        site = SpinSite(S=1, conserve=cons_Sz)
        # 4) lattice
        self.lat = lattice.Lattice(
            Ls=[L],
            unit_cell=[site, site, site],
            order=order,
            bc=bc,
            bc_MPS=bc_MPS,
            basis=np.array([[3, 0]]),
            positions=np.array(
                [[0.5, 0.5], [0.5, 0], [0.5, -0.5]]
            ),
            pairs=None,
        )

        coupling_pairs = self.lat.find_coupling_pairs(max_dx=6)
        pairs_dist = list(coupling_pairs.keys())

        self.lat.pairs["nearest_neighbors"] = coupling_pairs[pairs_dist[0]]
        self.lat.pairs["n_nearest_neighbors"] = coupling_pairs[pairs_dist[1]]
        self.lat.pairs["nn_nearest_neighbors"] = coupling_pairs[pairs_dist[2]]
        self.lat.pairs["nnn_nearest_neighbors"] = coupling_pairs[pairs_dist[3]]

        # 5) initialize CouplingModel
        CouplingModel.__init__(self, self.lat)

        # 6) add terms of the Hamiltonian
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hz, u, "Sz")
            self.add_onsite(-hx, u, "Sx")

        self.distances = list(self.lat.pairs.keys())
        # nearest neighbors: J12 J23 (within molecule, interaction 1-2 and 2-3)
        for u1, u2, dx in self.lat.pairs["nearest_neighbors"]:
            self.add_coupling(-2 * J, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * J, u1, "Sp", u2, "Sm", dx, plus_hc=True)
        # next nearest neighbors: J13 (within molecule, interaction 1-3)
        for u1, u2, dx in self.lat.pairs["n_nearest_neighbors"]:
            self.add_coupling(-2 * J13, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * J13, u1, "Sp", u2, "Sm", dx, plus_hc=True)

        # next-next nearest neighbors: J25 (between molecules, center-spin with center-spin)
        for u1, u2, dx in [self.lat.pairs["nn_nearest_neighbors"][1]]:
            self.add_coupling(-2 * Jinter, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * Jinter, u1, "Sp", u2, "Sm", dx, plus_hc=True)
    
        # next-next nearest neighbors: J14 J36 (between molecules, edge-spin with edge-spin)
        for u1, u2, dx in self.lat.pairs["nn_nearest_neighbors"][0:3:2]:
            self.add_coupling(-2 * Jnn, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * Jnn, u1, "Sp", u2, "Sm", dx, plus_hc=True)

        # next-next-next nearest neighbors: J15 J24 J26 J35 (between molecules, center-spin with center-spin)
        for u1, u2, dx in self.lat.pairs["nnn_nearest_neighbors"]:
            self.add_coupling(-2 * Jnnn, u1, "Sz", u2, "Sz", dx, plus_hc=False)
            self.add_coupling(-2 * 0.5 * Jnnn, u1, "Sp", u2, "Sm", dx, plus_hc=True)
        
        # 7) initialize H_MPO
        MPOModel.__init__(self, self.lat, self.calc_H_MPO())
        # 8) initialize H_bond (the order of 7/8 doesn't matter)
        # NearestNeighborModel.__init__(self, lat, self.calc_H_bond())
        # CouplingModel.calc_H_bond(self)