# %% Definicion de puertas cuanticas para un circuito cuantico
import numpy as np
     
#ahora necesitamos ofrecer las puertas : XYZ, H, Rotacion respecto eje, (Rx(theta), Ry(theta), Rz(theta)), CNOT, SWAP, I(solo para n qubits) 
#funciones sueltas para crear las puertas 
def I():
        """Devuelve la matriz de la puerta identidad"""
        return np.eye(2, dtype=complex)
def X():
        """Devuelve la matriz de la puerta X (NOT)"""
        return np.array([[0, 1], [1, 0]], dtype=complex)
def Y():
        """Devuelve la matriz de la puerta Y"""
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
def Z():
        """Devuelve la matriz de la puerta Z"""
        return np.array([[1, 0], [0, -1]], dtype=complex)
def H():
        """Devuelve la matriz de la puerta Hadamard"""
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def RX(theta):
    """
    Devuelve la matriz de la puerta de rotación Rx(theta).
    """
    return np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)]
    ])

def RY(theta):
    """
    Devuelve la matriz de la puerta de rotación Ry(theta).
    """
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2), np.cos(theta / 2)]
    ])

def RZ(theta):
    """
    Devuelve la matriz de la puerta de rotación Rz(theta).
    """
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ])
def hadamard_n(n):
    """Devuelve la matriz Hadamard para n qubits como producto de Kronecker."""
    H_n = H()
    for _ in range(n - 1):
        H_n = np.kron(H_n, H())  # Producto de Kronecker
    return H_n
def SWAP():
    """Matriz SWAP para dos qubits."""
    return np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]])
def In(n_q):
    """matriz identidad de tamaño 2^n x 2^n para n qubits."""
    return np.eye(2**n_q, dtype=complex) 

def CNOT():
   """Matriz CNOT para dos qubits. El primero de control y el segundo target."""
   return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=complex)
def NOTC():
    """Matriz CNOT para dos qubits. El primero target y el segundo control."""
    return np.array([[1, 0, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0]], dtype=complex)


