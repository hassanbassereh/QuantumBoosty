import matplotlib.pyplot as plt

class MasterEquation:
    def __init__(self, adj_matrix, num_qubits, dt, T):
        self.adj_matrix = adj_matrix
        self.num_qubits = num_qubits
        self.dt = dt
        self.T = T

    def Hamiltonian(self):
        H = np.zeros((len(self.adj_matrix), len(self.adj_matrix)))
        for i in range(len(self.adj_matrix)):
            for j in range(len(self.adj_matrix)):
                if self.adj_matrix[i][j] == 1:
                    H[i][j] = 1
        return H

    def Lindblad_sink(self):
        L = np.zeros((len(self.adj_matrix), len(self.adj_matrix)))
        for i in range(len(self.adj_matrix)):
            for j in range(len(self.adj_matrix)):
                if self.adj_matrix[i][j] == 1:
                    L[i][j] = -1
        return L

    def master_equation(self, H, L, rho):
        return expm(-1j * (H + L) * self.dt) @ rho @ expm(1j * (H + L) * self.dt)

    def initial_state(self):
        rho = np.zeros((2**self.num_qubits, 2**self.num_qubits), dtype=complex)
        rho[0][0] = 1
        return rho

    def time_evolution(self):
        H = self.Hamiltonian()
        L = self.Lindblad_sink()
        rho = self.initial_state()
        for t in range(self.T):
            rho = self.master_equation(H, L, rho)
        return rho

    def plot_sink_population(self):
        rho = self.time_evolution()
        sink_population = np.real(rho[-1][-1])
        time = np.arange(0, self.T * self.dt, self.dt)
        plt.plot(time, sink_population)
        plt.xlabel('Time (s)')
        plt.ylabel('Sink Population')
        plt.show()

# Example input graph
adj_matrix = [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]]
num_qubits = 4
dt = 0.01
T = 1000

# Solve the system
ME = MasterEquation(adj_matrix, num_qubits, dt, T)
ME.plot_sink_population()
