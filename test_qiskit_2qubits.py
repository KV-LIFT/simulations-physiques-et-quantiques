"""
test_qiskit_2qubits.py - Intrication quantique : etat de Bell |Phi+>
Execution : python test_qiskit_2qubits.py
"""
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

SHOTS = 1024


def main() -> None:
    # Simulateur local
    simulator = AerSimulator()

    # 1. Circuit : 2 qubits, 2 bits classiques de lecture
    qc = QuantumCircuit(2, 2)

    # 2. Hadamard sur le qubit 0 : superposition
    qc.h(0)

    # 3. CNOT : qubit 0 = controle, qubit 1 = cible -> intrication
    qc.cx(0, 1)

    # 4. Mesure des deux qubits sur les deux bits classiques
    qc.measure([0, 1], [0, 1])

    # 5. Compilation pour le simulateur puis execution sur 1024 tirs
    compiled = transpile(qc, simulator)
    counts = simulator.run(compiled, shots=SHOTS).result().get_counts()

    # 6. Affichage des resultats (etiquette = bit1 bit0, convention Qiskit)
    print(f"Resultats sur {SHOTS} tirs : {counts}")
    for state in ("00", "01", "10", "11"):
        n = counts.get(state, 0)
        print(f"  |{state}> : {n:4d} tirs  ->  P = {n / SHOTS:.3f}")
    correles = counts.get("00", 0) + counts.get("11", 0)
    print(f"Mesures correlees (00 ou 11) : {100 * correles / SHOTS:.1f} %")


if __name__ == "__main__":
    main()
