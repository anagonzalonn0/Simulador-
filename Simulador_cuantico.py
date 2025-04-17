# %% SIMULADOR CUANTICO
# Este código implementa un simulador cuántico básico en Python, utilizando la biblioteca NumPy para cálculos matriciales y SciPy para operaciones con matrices dispersas. El simulador permite crear y manipular qubits, aplicar puertas cuánticas y medir el estado de los qubits. También incluye funciones para calcular la matriz de densidad y los parámetros de la esfera de Bloch.
# El simulador está diseñado para ser eficiente y escalable, utilizando paralelización para mejorar el rendimiento en sistemas cuánticos más grandes. La clase QRegistry representa el registro cuántico y contiene métodos para aplicar puertas cuánticas, medir qubits y colapsar el estado del sistema. El código también incluye funciones para calcular la probabilidad de medir un qubit en el estado |1⟩ y para obtener los parámetros de la esfera de Bloch a partir del vector de estado.

import numpy as np  
import scipy.sparse as sp  
import multiprocessing  

# Función fuera de la clase para calcular probabilidad en un fragmento
def calcular_probabilidad_paralelo(args):
    inicio, fin, qubit_index, vector = args  # Tiempo: O(1), Espacio: O(1)
    prob = 0.0  # Tiempo: O(1), Espacio: O(1)
    for i in range(inicio, fin):  # Tiempo: O(k), Espacio: O(1) donde k = fin - inicio
        if (i >> qubit_index) & 1:  # Tiempo: O(1)
            prob += abs(vector[i]) ** 2  # Tiempo: O(1)
    return prob  # Tiempo: O(1), Espacio: O(1)
    #El coste total es Tiempo: O(2^n/hilos) Espacio: O(1)

class QRegistry:
    def __init__(self, n, num_hilos=None):
        """ Definimos los parametros importantes para la utilizacion de la clase QRegistry """
        #definimos el numero de qubits del cicuito cuántico
        self.n = n  # Tiempo: O(1), Espacio: O(1)
        #definimos el vector de estado del sistema cuántico, que es un vector de 2^n dimensiones
        #inicialmente el vector de estado es |0> = (1, 0, 0, 0, ...)
        self.vector = np.zeros((2**n, 1), dtype=complex)  # Tiempo: O(2^n), Espacio: O(2^n)
        self.vector[0, 0] = 1  # Tiempo: O(1)
        self.state = self.vector  # Tiempo: O(1), Espacio: O(1)
        #definimos el tamaño del vector de estado
        #el tamaño del vector de estado es 2^n, que es el número de estados posibles del sistema cuántico
        self.vector_size = self.vector.size  # Tiempo: O(1), Espacio: O(1)
        #definimos el número de hilos para la paralelización
        #el número de hilos es el número de procesos que se pueden ejecutar en paralelo
        self.num_hilos = num_hilos  # Tiempo: O(1), Espacio: O(1)
        self.pool = multiprocessing.Pool(processes=self.num_hilos)  # Tiempo: O(1), Espacio: O(hilos)
        #El coste total es Tiempo: O(1) Espacio: O(2^n + hilos)


    def print_state(self):
        """ Imprime el vector de estado de manera legible. """
        print("Vector de estado:")  # Tiempo: O(1)
        for i, amplitude in enumerate(self.vector):  # Tiempo: O(2^n)
            print(f"|{i:0{self.n}b}>: {amplitude}")  # Tiempo: O(1) por iteración. Imprimimos las amplitudes de cada estado del vector de estado cuántico.
        #el coste total es Tiempo O(2^n) Espacio O(1)

    def apply_gate(self, gate, qubits):
        """
        Aplica una puerta cuántica al estado del sistema, optimizando la representación de la matriz.
        :param gate: Matriz de la puerta cuántica a aplicar.
        :param qubits: Índices de los qubits a los que se aplica la puerta.
        """
        n = self.n  # Tiempo: O(1)
        if isinstance(qubits, int):  # Tiempo: O(1)
            qubits = [qubits]  # Tiempo: O(1)
        qubits = [(n - 1) - q for q in qubits]  # Tiempo: O(len(qubits)). Ordenamos los qubits de mayor a menor, para que la matriz de la puerta cuántica se aplique correctamente.
        # Esto se hace para que la puerta cuántica se aplique al qubit correcto en el vector de estado.
        qubits.sort()  # Tiempo: O(len(qubits) log len(qubits)).

        if len(qubits) == 1: 
            #Si la puerta a aplicar es de 1 qubit
            target = qubits[0]  # Tiempo: O(1)
            left_identity = sp.eye(2**target) if target > 0 else 1  # Tiempo: O(2^target), Espacio: O(2^target)   eso se usa para generar una matriz identidad  que representa los qubits a la izquierda del objetivo.
            right_identity = sp.eye(2**(n - target - 1)) if (n - target - 1) > 0 else 1  # Tiempo: O(2^(n- target -1)) eso se usa para generar una matriz identidad  que representa los qubits a la derecha del objetivo.
            full_gate = sp.kron(left_identity, sp.kron(gate, right_identity, format='coo'), format='coo')  # Tiempo: O(2^n), Espacio: O(2^n)   eso se usa para generar la operacion correspondiente al qubit objetivo/circuito.
        # Se usa la funcion kron para generar la matriz de la puerta cuántica completa.
        elif len(qubits) == 2: #caso en el que hay dos qubits a los que se les aplica la puerta cuántica
            q1, q2 = qubits  # Tiempo: O(1)
            left_identity = sp.eye(2**q1) if q1 > 0 else 1  # Tiempo: O(2^q1)
            middle_identity = sp.eye(2**(q2 - q1 - 1)) if (q2 - q1 - 1) > 0 else 1  # Tiempo: O(2^(q2 - q1 - 1))
            right_identity = sp.eye(2**(n - q2 - 1)) if (n - q2 - 1) > 0 else 1  # Tiempo: O(2^(n - q2 - 1))
            full_gate = sp.kron(left_identity, sp.kron(gate, sp.kron(middle_identity, right_identity), format='coo'), format='coo')  # Tiempo: O(2^n)
        else:
            raise ValueError("Se soportan únicamente puertas de 1 o 2 qubits.")

        self.vector = full_gate @ self.vector  # Tiempo: O(2^n), Espacio: O(2^n)  #Una vez calculada la puerta completa, lo aplicamos al vector de estado del sistema cuántico.
        #El coste total es Tiempo: O(2^n) Espacio: O(2^n)

    def get_state_probability(self, state):
        """Calcular la probabilidad como el cuadrado del módulo del coeficiente correspondiente"""
        probability = abs(self.vector[state])**2  # Tiempo: O(1), Espacio: O(1). Dado que la probabilidad se define como el cuadrado del módulo del coeficiente correspondiente, se calcula como el cuadrado del módulo del coeficiente correspondiente al estado dado.
        probability = probability[0]  # Tiempo: O(1) # Espacio: O(1). Esto se hace para convertir el resultado a un número real.
        return probability  # Tiempo: O(1) Espacio O(1)
        #El coste total es Tiempo: O(1) Espacio O(1)

    def measure_qubit(self, qubit_index):
        """
        Mide la probabilidad de que un qubit esté en el estado |1⟩.
        """
        prob_one = 0.0  # Tiempo: O(1) Espacio: O(1)
        vector_size = self.vector.size  # Tiempo: O(1) Espacio: O(1)
        for i in range(vector_size):  # Tiempo: O(2^n)
            if (i >> qubit_index) & 1:  # Tiempo: O(1) Si el qubit está en estado |1⟩, sumamos su probabilidad al total.
                # Esto se hace usando el operador de desplazamiento a la derecha y el operador AND para verificar si el qubit está en estado |1⟩.
                prob_one += np.abs(self.vector[i]) ** 2  # Tiempo: O(1)
        return float(min(prob_one, 1.0))  # Tiempo: O(1) Espacio O(1)
        #El coste total es Tiempo: O(2^n) Espacio O(1)
        #La función devuelve la probabilidad de que el qubit esté en el estado |1⟩, asegurándose de que no supere 1.0.

    

    def measure_paralel_qubit(self, qubit_index):
        """
        Mide la probabilidad de que un qubit esté en el estado |1⟩.
        Usa paralelización si se justifica; de lo contrario, calcula secuencialmente.
        """
        if self.num_hilos is None and self.n <= 4:
            # Cálculo secuencial si el sistema es pequeño y el usuario no pidió paralelización
            prob_one = 0.0  # Tiempo: O(1), Espacio: O(1)
            for i in range(self.vector_size):  # Tiempo: O(2^n), Espacio: O(1)
                if (i >> qubit_index) & 1:  # Tiempo: O(1)
                    prob_one += abs(self.vector[i]) ** 2  # Tiempo: O(1)
            return float(min(prob_one, 1.0))  # Tiempo: O(1), Espacio: O(1)
            # Coste total modo secuencial: Tiempo: O(2^n), Espacio: O(1)
        elif self.num_hilos is None and self.n > 4:
            # Cálculo secuencial si el sistema es grande y el usuario no pidió paralelización
            max_hilos = min(multiprocessing.cpu_count(), self.vector_size)  # Tiempo: O(1) 
            self.num_hilos = max_hilos  # Tiempo: O(1), Espacio: O(1) # Ajustamos al máximo posible si no se especifica
        else:
            # Paralelización inteligente. Dejamos hilos sin utilizar si el numero de tareas(tamaño del vector de estado) no es igual o excede el numero de hilos
            max_trabajadores = min(self.num_hilos, self.vector_size)  # Tiempo: O(1), Espacio: O(1)
            chunk_size = self.vector_size // max_trabajadores  # Tiempo: O(1), Espacio: O(1)
            # Definimos el tamaño de cada fragmento para la paralelización
            # Esto se hace para dividir el vector de estado en fragmentos que serán procesados por cada hilo.
            inicios = [i * chunk_size for i in range(max_trabajadores)]  # Tiempo: O(hilos), Espacio: O(hilos)
            finales = [min((i + 1) * chunk_size, self.vector_size) for i in range(max_trabajadores)]  # Tiempo: O(hilos), Espacio: O(hilos)
            # Construimos los argumentos explícitamente para cada hilo
            # Esto se hace para que cada hilo procese su propio fragmento del vector de estado.
            args = [(inicio, fin, qubit_index, self.vector) for inicio, fin in zip(inicios, finales)]  
            # Tiempo: O(hilos), Espacio: O(hilos)
            
            #self.pool = multiprocessing.Pool(processes=self.num_hilos)  # Tiempo: O(1), Espacio: O(hilos)
            resultados = self.pool.map(calcular_probabilidad_paralelo, args)
            
            #Resultados guarda los resultados de cada hilo  
            # Tiempo: O(2^n / hilos), Espacio: O(hilos)

            return min(sum(resultados), 1.0)  # Tiempo: O(hilos), Espacio: O(1)
        # Coste total modo paralelo: Tiempo: O(2^n / hilos + hilos), Espacio: O(hilos)

    def collapse(self, qubit, value, prob_one):
        """
        Colapsa el estado del qubit medido al valor especificado (0 o 1).
        :param qubit: Índice del qubit a colapsar.
        :param value: Valor al que colapsar (0 o 1).
        :param prob_one: Probabilidad de medir el estado |1⟩ antes del colapso.
        """
        if value not in [0, 1]:  # Tiempo: O(1) Espacio: O(1)
            # Verificamos que el valor sea 0 o 1
            raise ValueError("El valor debe ser 0 o 1.")
        norma = np.sqrt(prob_one if value == 1 else 1 - prob_one)  # Tiempo: O(1) Espacio: O(1)
        # Calculamos la norma del vector de estado después de la medición
        for i in range(self.vector.size):  # Tiempo: O(2^n) 
            # Iteramos sobre cada elemento del vector de estado
            if ((i >> qubit) & 1) != value:  # Tiempo: O(1)
                self.vector[i] = 0  # Tiempo: O(1)
            else:
                self.vector[i] /= norma  # Tiempo: O(1) Espacio : O(1)
        # Colapsamos el vector de estado al valor medido, dividiendo por la norma calculada
        #El coste total es Tiempo: O(2^n) Espacio: O(1)

    def measure(self, qubit):
        """
        Mide un qubit y colapsa el estado del sistema.
        :param qubit: Índice del qubit a medir.
        :return: Valor medido (0 o 1).
        """
        prob_one = self.measure_qubit(qubit)  # Tiempo: O(2^n) Espacio : O(1)
        value = np.random.choice([0, 1], p=[1 - prob_one, prob_one])  # Tiempo: O(1) Espacio: O(1)
        self.collapse(qubit, value, prob_one)  # Tiempo: O(2^n) Espacio : O(1)
        # Colapsamos el estado del sistema al valor medido
        return value  # Tiempo: O(1) Espacio: O(1)
        #El coste total es Tiempo: O(2^n) Espacio: O(1)

    def density_matrix(self):
        """
        Calcula la matriz de densidad a partir de un vector de estado.
        :param state_vector: Vector de estado cuántico.
        :return: Matriz de densidad correspondiente.
        """
        return np.outer(self.vector, np.conj(self.vector))  # Tiempo: O(4^n), Espacio: O(4^n)

    def bloch_sphere_parameters(self):
        """
        Calcula los parámetros de la esfera de Bloch para un qubit dado su vector de estado.
        :param state_vector: Vector de estado cuántico (de dimensión 2).
        :return: Tupla (theta, phi) que representa el estado en la esfera de Bloch.
        """
        alpha, beta = self.vector[0], self.vector[1]  # Tiempo: O(1) Espacio : O(1)
        global_phase = np.angle(beta) if alpha == 0 else np.angle(alpha)  # Tiempo: O(1) Espacio : O(1) # Calculamos la fase global del vector de estado
        # Ajustamos el vector de estado para eliminar la fase global
        alpha = alpha * np.exp(-1j * global_phase)  # Tiempo: O(1) 
        beta = beta * np.exp(-1j * global_phase)  # Tiempo: O(1) 
        phi = np.angle(beta)  # Tiempo: O(1)  # Calculamos el ángulo phi en la esfera de Bloch
        beta2 = np.exp(-1j * phi) * beta  # Tiempo: O(1) 
        
        theta = 2 * np.arctan2(beta2.real, alpha.real)  # Tiempo: O(1) Espacio : O(1) Calculamos el ángulo theta en la esfera de Bloch
        # Devolvemos los dos angulos como numeros reales
        return theta[0], phi[0]  # Tiempo: O(1) Espacio : O(1)
        #El coste total es Tiempo: O(1) Espacio : O(1)
    
#IMPORTAMOS LAS PUERTAS QUE HEMOS DEFINIDO EN OTRO ARCHIVO LLAMADO puertas.ipynb
from IPython import get_ipython  # Tiempo: O(1)
get_ipython().run_line_magic("run", "puertas.py")  # Tiempo: Dependiente del contenido de puertas.ipynb



