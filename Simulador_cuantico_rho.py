# %%
import numpy as np  # Importa la biblioteca NumPy para cálculos numéricos. 
import scipy.sparse as sp  # Importa herramientas para trabajar con matrices dispersas.

"""  
Definimos la clase QRegistryDensity para representar un registro cuántico de n qubits
y su matriz de densidad asociada. Esta clase permite la manipulación de estados cuánticos
 y la aplicación de puertas cuánticas, así como la medición de qubits y el colapso del estado.
"""
class QRegistryDensity:
    def __init__(self, n, init_state=None, num_hilos=None):  # Constructor de la clase. Inicializa el registro. # Tiempo O(1) Espacio O(1)
        self.n = n  # Guarda el número de qubits. # Tiempo O(1) Espacio O(1)
        self.dim = 2**n  # Calcula la dimensión del espacio de Hilbert. # Tiempo O(1) Espacio O(1)
        self.rho = self._parse_init_state(init_state)  # Genera la matriz de densidad inicial. # Tiempo O(dim^2) Espacio O(dim^2)

    

    def _parse_init_state(self, init_state):  # Crea la matriz de densidad a partir del estado inicial. # Tiempo O(dim^2) Espacio O(dim^2)
        if init_state is None:
            psi = np.zeros((self.dim, 1), dtype=complex)  # Crea el vector columna |0>. # Tiempo O(dim) Espacio O(dim)
            psi[0, 0] = 1  # Asigna amplitud 1 al primer estado. # Tiempo O(1) Espacio O(1)
            return psi @ psi.conj().T  # Construye ρ = |ψ⟩⟨ψ|. # Tiempo O(dim^2) Espacio O(dim^2)
        elif isinstance(init_state, list):
            rho = np.zeros((self.dim, self.dim), dtype=complex)  # Inicializa la matriz rho vacía. # Tiempo O(dim^2) Espacio O(dim^2)
            for psi, p in init_state:  # Itera sobre los pares (estado, probabilidad). # Tiempo O(k * dim^2) Espacio O(dim^2)
                psi = psi.reshape((self.dim, 1))  # Asegura que sea un vector columna. # Tiempo O(dim) Espacio O(dim)
                rho += p * (psi @ psi.conj().T)  # Suma ponderada de las proyecciones. # Tiempo O(dim^2) Espacio O(dim^2)
            return rho
        else:
            init_state = np.asarray(init_state)  # Convierte a array de NumPy si no lo es. # Tiempo O(dim^2) Espacio O(dim^2)
            if init_state.shape == (self.dim,) or init_state.shape == (self.dim, 1):
                psi = init_state.reshape((self.dim, 1))  # Asegura que sea vector columna. # Tiempo O(dim) Espacio O(dim)
                return psi @ psi.conj().T  # Construye ρ = |ψ⟩⟨ψ|. # Tiempo O(dim^2) Espacio O(dim^2)
            elif init_state.shape == (self.dim, self.dim):
                return init_state  # Ya es matriz de densidad. # Tiempo O(1) Espacio O(dim^2)
            else:
                raise ValueError("init_state inválido.")  # Maneja errores de formato. # Tiempo O(1) Espacio O(1)


    def print_state(self):  # Imprime la matriz de densidad en formato binario. # Tiempo O(dim^2) Espacio O(1)
        print("Matriz de densidad:")  # Mensaje inicial. # Tiempo O(1) Espacio O(1)
        for i in range(self.dim):
            for j in range(self.dim):
                print(f"({i:0{self.n}b},{j:0{self.n}b}): {self.rho[i, j]}")  # Imprime elemento. # Tiempo O(1) Espacio O(1)



    def apply_gate(self, gate, qubits):  # Aplica una puerta cuántica a los qubits especificados. # Tiempo O(dim^3) Espacio O(dim^2)
        n = self.n  # Número de qubits. # Tiempo O(1) Espacio O(1)
        if isinstance(qubits, int):
            qubits = [qubits]  # Convierte a lista si es un único qubit. # Tiempo O(1) Espacio O(1)
        qubits = [(n - 1) - q for q in qubits]  # Ajusta índice de qubit a orden interno. # Tiempo O(k) Espacio O(k)
        qubits.sort()  # Ordena los índices de qubits. # Tiempo O(k log k) Espacio O(1)

        if len(qubits) == 1:
            target = qubits[0]  # Qubit objetivo. # Tiempo O(1) Espacio O(1)
            I_left = sp.eye(2**target) if target > 0 else 1  # Identidad izquierda. # Tiempo O(2^target) Espacio O(2^target)
            I_right = sp.eye(2**(n - target - 1)) if (n - target - 1) > 0 else 1  # Identidad derecha. # Tiempo O(2^(n-target-1)) Espacio O(2^(n-target-1))
            U = sp.kron(I_left, sp.kron(gate, I_right, format='coo'), format='coo')  # Producto de Kronecker. # Tiempo O(nnz) Espacio O(nnz)
        elif len(qubits) == 2:
            q1, q2 = qubits  # Qubits objetivo. # Tiempo O(1) Espacio O(1)
            I_left = sp.eye(2**q1) if q1 > 0 else 1  # Identidad izquierda. # Tiempo O(2^q1) Espacio O(2^q1)
            I_mid = sp.eye(2**(q2 - q1 - 1)) if (q2 - q1 - 1) > 0 else 1  # Identidad intermedia. # Tiempo O(2^(q2-q1-1)) Espacio O(2^(q2-q1-1))
            I_right = sp.eye(2**(n - q2 - 1)) if (n - q2 - 1) > 0 else 1  # Identidad derecha. # Tiempo O(2^(n-q2-1)) Espacio O(2^(n-q2-1))
            U = sp.kron(I_left, sp.kron(gate, sp.kron(I_mid, I_right), format='coo'), format='coo')  # Kronecker completo. # Tiempo O(nnz) Espacio O(nnz)
        else:
            raise ValueError("Solo se soportan puertas de 1 o 2 qubits.")  # Error si más de 2 qubits. # Tiempo O(1) Espacio O(1)

        U = U.toarray()  # Convierte a matriz densa. # Tiempo O(nnz) Espacio O(dim^2)
        self.rho = U @ self.rho @ U.conj().T  # Aplica la operación cuántica completa. # Tiempo O(dim^3) Espacio O(dim^2)
        # Tiempo O(nnz * dim) Espacio O(dim^2)


    def measure_qubit(self, qubit_index):
        """ 
        Esta función mide un qubit específico y devuelve la probabilidad de que esté en el estado |1⟩.
        """
        prob_one = 0.0    # Inicializa la probabilidad acumulada #Tiempo O(1) Espacio O(1)
        for i in range(self.dim):  # Itera sobre todos los índices diagonales #Tiempo O(dim) Espacio O(1)
            if (i >> qubit_index) & 1:   # Comprueba si el qubit está en estado |1⟩ en ese índice #Tiempo O(1) Espacio O(1)
                prob_one += self.rho[i, i].real  # Suma la contribución real del elemento diagonal #Tiempo O(1) Espacio O(1)
        return float(min(prob_one, 1.0))   # Devuelve la probabilidad truncada a máximo 1.0 #Tiempo O(1) Espacio O(1)
        #Tiempo O(2^n) Espacio O(1)


    def _projector(self, qubit_index, value):
        """
        Construye el operador proyector para el colapso de un qubit a un valor concreto (0 o 1).
        """
        n = self.n    # Almacena el número total de qubits #Tiempo O(1) Espacio O(1)
        P0 = np.array([[1, 0], [0, 0]], dtype=complex)   # Matriz proyectora para estado |0⟩ #Tiempo O(1) Espacio O(1)
        P1 = np.array([[0, 0], [0, 1]], dtype=complex)    # Matriz proyectora para estado |1⟩ #Tiempo O(1) Espacio O(1)
        P = P1 if value == 1 else P0     # Selecciona la matriz correspondiente al valor medido #Tiempo O(1) Espacio O(1)
        qubit = (n - 1) - qubit_index   # Ajuste de índice para endianess #Tiempo O(1) Espacio O(1)
        I_left = sp.eye(2**qubit) if qubit > 0 else 1  # Identidad izquierda para producto de Kronecker #Tiempo O(1) Espacio O(log(dim))
        I_right = sp.eye(2**(n - qubit - 1)) if (n - qubit - 1) > 0 else 1    # Identidad derecha para producto de Kronecker #Tiempo O(1) Espacio O(log(dim))
        full_P = sp.kron(I_left, sp.kron(P, I_right), format='coo')  # Producto de Kronecker para construir el proyector completo #Tiempo O(dim log dim) Espacio O(dim)
        return full_P   # Devuelve el proyector disperso #Tiempo O(1) Espacio O(dim)
        #Tiempo O(2^n log(2^n)) Espacio O(2^n)


    def collapse(self, qubit_index, value, prob_one):
        """
        Colapsa el estado cuántico según la medición del qubit.
        """
        P = self._projector(qubit_index, value)  # Construye el proyector del qubit medido #Tiempo O(dim log dim) Espacio O(dim)
        rho_colapsada = P @ self.rho @ P   # Aplica el proyector: PρP #Tiempo O(nnz_P * nnz_rho) Espacio O(nnz)
        normalizador = prob_one if value == 1 else (1 - prob_one)  # Calcula la probabilidad asociada al resultado medido #Tiempo O(1) Espacio O(1)
        self.rho = rho_colapsada / normalizador if normalizador > 0 else rho_colapsada  # Normaliza la nueva rho #Tiempo O(nnz) Espacio O(nnz)
        #Tiempo O(nnz^2) Espacio O(nnz) 


    def measure(self, qubit_index):
        """ 
        Mide un qubit específico y colapsa el estado cuántico en consecuencia.
        """
        prob_one = self.measure_qubit(qubit_index)  # Calcula la probabilidad de obtener |1⟩ #Tiempo O(dim) Espacio O(1)
        value = np.random.choice([0, 1], p=[1 - prob_one, prob_one]) # Realiza una medición aleatoria basada en esa probabilidad #Tiempo O(1) Espacio O(1)
        self.collapse(qubit_index, value, prob_one)  # Colapsa el estado cuántico según el valor medido #Tiempo O(dim^3) Espacio O(dim^2)
        return value  # Devuelve el resultado de la medición #Tiempo O(1) Espacio O(1)
        #Tiempo O(2^n + nnz^2) Espacio O(nnz)


    def partial_trace(self, qubit_index):
        """
        Calcula la traza parcial sobre todos los qubits excepto el indicado.
        Devuelve la matriz de densidad reducida 2x2 para ese qubit.
        """
        if isinstance(self.rho, np.ndarray):   # Comprueba si rho es matriz densa #Tiempo O(1) Espacio O(1)
            rho = sp.coo_matrix(self.rho)  # Convierte a dispersa COO #Tiempo O(dim^2) Espacio O(dim^2)
        else:
            rho = self.rho.tocoo()    # Convierte desde otro formato disperso a COO #Tiempo O(nnz) Espacio O(nnz)
        
        dim = self.dim   # Tamaño total del sistema #Tiempo O(1) Espacio O(1)
        n = self.n   # Número total de qubits #Tiempo O(1) Espacio O(1)
        reduced = np.zeros((2, 2), dtype=complex)  # Inicializa matriz 2x2 para almacenar la traza parcial #Tiempo O(1) Espacio O(1)

        for i, j, val in zip(rho.row, rho.col, rho.data):   # Itera sobre los elementos no nulos de rho #Tiempo O(nnz) Espacio O(1)
            bi = (i >> (n - 1 - qubit_index)) & 1   # Extrae el bit del índice de fila #Tiempo O(1) Espacio O(1)
            bj = (j >> (n - 1 - qubit_index)) & 1    # Extrae el bit del índice de columna #Tiempo O(1) Espacio O(1)
            mask = ~(1 << (n - 1 - qubit_index))  # Máscara para ignorar el qubit medido #Tiempo O(1) Espacio O(1)
            if (i & mask) == (j & mask):   # Solo se considera si los demás qubits son iguales #Tiempo O(1)
                reduced[bi, bj] += val      # Acumula el valor en la posición reducida #Tiempo O(1)
        return reduced     # Devuelve la matriz 2x2 #Tiempo O(1) Espacio O(1)
        #Tiempo O(nnz) Espacio O(1)


    def concurrence(self, qubit_index):
        """
        Calcula la pureza (concurrencia) del qubit mediante la traza parcial.
        """
        rho_reducida = self.partial_trace(qubit_index)   # Calcula la traza parcial #Tiempo O(nnz) Espacio O(1)
        pureza = np.trace(rho_reducida @ rho_reducida).real   # Calcula Tr(rho^2) y toma la parte real #Tiempo O(1) Espacio O(1)
        return pureza     # Devuelve la pureza como medida de concurrencia #Tiempo O(1) Espacio O(1)
        #Tiempo O(nnz) Espacio O(1)


#IMPORTAMOS LAS PUERTAS QUE HEMOS DEFINIDO EN OTRO ARCHIVO LLAMADO puertas.ipynb
from IPython import get_ipython

get_ipython().run_line_magic("run", "puertas.py")


