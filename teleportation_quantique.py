"""
teleportation_quantique.py - Protocole de teleportation quantique (3 qubits)
Q0 = message a envoyer, Q1 = particule d'Alice, Q2 = particule de Bob
Execution : python teleportation_quantique.py

Note : Qiskit 2.x a supprime .c_if(). Les corrections de Bob utilisent son
remplacement officiel, le bloc conditionnel `with qc.if_test(...)`.
"""
import sys

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

SHOTS = 1024

if hasattr(sys.stdout, "reconfigure"):          # console Windows : dessin du circuit
    sys.stdout.reconfigure(errors="replace")


def build_circuit() -> QuantumCircuit:
    # 1. Architecture : 3 qubits, 2 bits pour Alice + 1 bit pour Bob
    q = QuantumRegister(3, "q")                 # q0 message, q1 Alice, q2 Bob
    alice = ClassicalRegister(2, "alice")       # alice[0] <- Q0, alice[1] <- Q1
    bob = ClassicalRegister(1, "bob")           # bob[0]   <- Q2
    qc = QuantumCircuit(q, alice, bob)

    # 2. Initialisation : message |1> sur Q0
    qc.x(q[0])
    qc.barrier()

    # 3. Canal quantique : paire de Bell entre Q1 (Alice) et Q2 (Bob)
    qc.h(q[1])
    qc.cx(q[1], q[2])
    qc.barrier()

    # 4. Protocole d'Alice : CX(Q0 -> Q1), H(Q0), puis mesure de Q0 et Q1
    qc.cx(q[0], q[1])
    qc.h(q[0])
    qc.measure(q[0], alice[0])
    qc.measure(q[1], alice[1])
    qc.barrier()

    # 5. Protocole de Bob : corrections conditionnelles sur Q2
    with qc.if_test((alice[1], 1)):             # bit de Q1 = 1 -> porte X
        qc.x(q[2])
    with qc.if_test((alice[0], 1)):             # bit de Q0 = 1 -> porte Z
        qc.z(q[2])

    # 6. Verification : mesure de Q2 dans le 3e bit classique
    qc.measure(q[2], bob[0])
    return qc


def main() -> None:
    # 7. Execution sur le simulateur local (1024 tirs)
    simulator = AerSimulator()
    qc = build_circuit()
    compiled = transpile(qc, simulator)
    counts = simulator.run(compiled, shots=SHOTS).result().get_counts()

    print(qc.draw(output="text", fold=-1))
    print(f"\nResultats bruts sur {SHOTS} tirs (format 'bob alice') : {counts}\n")

    # Cle Qiskit "b a1a0" : b = Q2 (Bob), a1 = Q1, a0 = Q0 (Alice)
    print("Mesure d'Alice (Q1 Q0)  ->  Bob (Q2)")
    for key, n in sorted(counts.items(), key=lambda kv: kv[0].split()[1]):
        b, a = key.split()
        print(f"        {a[0]} {a[1]}            ->     {b}      : {n:4d} tirs")

    recu = sum(n for key, n in counts.items() if key.split()[0] == "1")
    print(f"\nEtat |1> recu par Bob : {recu}/{SHOTS} tirs = {100 * recu / SHOTS:.1f} %")


if __name__ == "__main__":
    main()
