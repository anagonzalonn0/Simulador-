# Simulador-
# Ejemplos de caso de uso
  # Aplicacion de puertas SWAP
  #PROBAMOS A INTRODUCIR EL PARAMETRO qubit EN LA FUNCION APPLYGATE.
a=QRegistry(12,8) #Aquí hemos especificado utilizar 8 hilos, y 12 qubits
a.apply_gate(X(), 0)
#SWAP 0 1
a.apply_gate(SWAP(), [0, 1])
#SWAP 1 2 
a.apply_gate(SWAP(), [1, 2])
#SWAP 2 3 
a.apply_gate(SWAP(), [2, 3])

print(a.print_state())  #Muestra el vector de estado por pantalla
print(a.get_state_probability(7)) #Calculo de la probabilidad del estado |7>
print(a.measure_qubit(0)) #Medicion del qubit 0
#print(a.measure_paralel_qubit(2)) #para utilizar la paralelizacion


  # Creamos un estado de Bell 
#crear registro y aplico puerta y luego get.state
a=QRegistry(2)
#primero, tienes que usar a.apply_gate(puerta)
#no vale trabajar con a.state
#tienes que usar tu método
#Revisar matriz CNOT

state = a.apply_gate(np.kron(H(), X()), [0, 1])
#state= a.apply_gate( H(), 0  )
#state= a.apply_gate( X(), 1)
state= a.apply_gate( CNOT(), [0,1] )
print(a.print_state())


  # Calculo de parametros de la esfera de Bloch

v=QRegistry(1)
v.apply_gate(X(), 0)
#print(a.print_state())
print(v.bloch_sphere_parameters())

# Para el caso de utilizar matrices de densidad:
  # Probamos a hacer operaciones

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

  # Caso en el que le pasamos las probabilidades de estado mezcla
  # Estado mezcla 50% |0>, 50% |1>
psi0 = np.array([1, 0], dtype=complex)
psi1 = np.array([0, 1], dtype=complex)
qr = QRegistryDensity(n=1, init_state=[(psi0, 0.5), (psi1, 0.5)])

qr.print_state()
qr.apply_gate(H(), 0)
qr.print_state()
resultado = qr.measure_qubit(0)
print("Resultado de la medida:", resultado)
qr.print_state()
