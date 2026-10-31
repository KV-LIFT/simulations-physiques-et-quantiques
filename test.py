"""
test.py - Premiere superposition quantique (Qiskit + Qiskit Aer)
Execution : python test.py
"""
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

SHOTS = 1024


def main() -> None:
    # 1. Simulateur local
    simulator = AerSimulator()

    # 2. Circuit : 1 qubit, 1 bit classique de lecture
    qc = QuantumCircuit(1, 1)

    # 3. Porte d'Hadamard : |0> -> superposition (|0> + |1>) / sqrt(2)
    qc.h(0)

    # 4. Mesure : le qubit 0 s'effondre sur 0 ou 1, resultat ecrit dans le bit 0
    qc.measure(0, 0)

    # 5. Compilation pour le simulateur puis execution sur 1024 tirs
    compiled = transpile(qc, simulator)
    counts = simulator.run(compiled, shots=SHOTS).result().get_counts()

    # 6. Affichage des resultats et des probabilites mesurees
    print(f"Resultats sur {SHOTS} tirs : {counts}")
    for state in ("0", "1"):
        n = counts.get(state, 0)
        print(f"  |{state}> : {n:4d} tirs  ->  P = {n / SHOTS:.3f}")


if __name__ == "__main__":
    main()
