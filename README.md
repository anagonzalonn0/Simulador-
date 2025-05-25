# Simulador circuito cuantico con vectores de estado
(En el calculo de costes a veces se escribe dim para referirnos a 2^n, siendo n el numero de qubits)<br>
El archivo Mi_simulador.ipynb es un cuaderno de jupyter notebook que contiene el mismo código que Simulador_cuantico.py, en python. Estos códigos son para trabajar con vectores de estado. <br>
Mientras que los archivos Mi_simulador_rho.ipynb y Simulador_cuantico_rho.py son los análogos pero para matrices de densidad. <br>
El archivo puertas.ipynb contiene las puertas para trabajar con vectores de estado, así como puertas_rho.ipynb contiene las puertas para trabajar con matrices de densidad.<br>

Para el cálculo de operaciones, se puede repartir las operaciones en k hilos para acelerar el cálculo mediante el módulo de multiprocessing.<br>
A continuación se presentan ejemplos de uso.

## Ejemplos de caso de uso
  #### Aplicacion de puertas SWAP
  #PROBAMOS A INTRODUCIR EL PARAMETRO qubit EN LA FUNCION APPLYGATE.  
  
a=QRegistry(12,8) #Creamos el circuito cuántico con 8 hilos, y 12 qubits  

a.apply_gate(X(), 0)  #Estamos aplicando la puerta X al qubit 0.

#SWAP 0 1  

a.apply_gate(SWAP(), [0, 1])  #Estamos aplicando la puerta SWAP al qubit 0 y 1.

#SWAP 1 2  

a.apply_gate(SWAP(), [1, 2])  #Estamos aplicando la puerta SWAP al qubit 1 y 2.

#SWAP 2 3   

a.apply_gate(SWAP(), [2, 3])  #Estamos aplicando la puerta SWAP al qubit 2 y 3.


print(a.print_state())  #Muestra el vector de estado por pantalla  

print(a.get_state_probability(7)) #Calculo de la probabilidad del estado |7>  

print(a.measure_qubit(0)) #Medicion del qubit 0  

#print(a.measure_paralel_qubit(2)) #para utilizar la paralelizacion a la hora de medir 



  # Creamos un estado de Bell   
  
#crear registro y aplico puerta y luego get.state  

a=QRegistry(2)  

 
state = a.apply_gate(np.kron(H(), X()), [0, 1])  #En una sola línea de código hacemos que se aplique la matriz H sobre el qubit 0 y la matriz X sobre el qubit 1.

#state= a.apply_gate( H(), 0  )  

#state= a.apply_gate( X(), 1)  

state= a.apply_gate( CNOT(), [0,1] )  #Aplicamos CNOT, qubit de control 0 y qubit target 1.

print(a.print_state())  #Mostramos estado por pantalla



  # Calculo de parametros de la esfera de Bloch  
  

v=QRegistry(1)  
 
v.apply_gate(X(), 0)  

#print(a.print_state())  

print(v.bloch_sphere_parameters())  #Devuelve los parámetros de la esfera de bloch del estado.


# Simulador circuito cuantico con matrices de densidad  

  ## Probamos a hacer operaciones  
  

#Crear un registro cuántico de 2 qubits  

qr = QRegistryDensity(2, num_hilos=4)  

qr.print_state()  


#Aplicar una puerta Hadamard al primer qubit  

qr.apply_gate(H(), 0)  

#qr.print_state()  

qr.partial_trace(0) #rho_1 = Tr_2(rho)  

#Medir el primer qubit  

resultado = qr.measure(0)  

print(f"Resultado de la medida del primer qubit: {resultado}")  


#Imprimir la matriz de densidad después de la medida  

#qr.print_state()  


  ## Caso en el que le pasamos las probabilidades de estado mezcla  
  
  ## Estado mezcla 50% |0>, 50% |1>  
  
psi0 = np.array([1, 0], dtype=complex) <br>
psi1 = np.array([0, 1], dtype=complex) <br>
qr = QRegistryDensity(n=1, init_state=[(psi0, 0.5), (psi1, 0.5)])<br>

qr.print_state() <br>
qr.apply_gate(H(), 0) <br>
qr.print_state() <br>
resultado = qr.measure_qubit(0) <br>
print("Resultado de la medida:", resultado) <br>
qr.print_state() <br>
