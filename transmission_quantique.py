#!/usr/bin/env python3
"""
transmission_quantique.py
=========================
Transmission de données (texte, octets, petit fichier) par téléportation
quantique, bit par bit, à travers le circuit à 3 qubits :
    Q0 = qubit message    Q1 = particule d'Alice    Q2 = particule de Bob

Chaîne de bout en bout
    données -> octets -> bits -> 1 téléportation par bit (envoi par paquets)
            -> AerSimulator -> lecture du registre de Bob (vote majoritaire)
            -> bits -> octets -> données

Usage
    python transmission_quantique.py                         # demande le message
    python transmission_quantique.py -m "Bonjour Bob !"
    python transmission_quantique.py -m "Salut" --bruit 0.03 --tirs 15
    python transmission_quantique.py -f photo.png            # petit fichier binaire
    python transmission_quantique.py --schema                # affiche le circuit

Écrit pour : Python 3.11, qiskit 2.6, qiskit-aer 0.18 (env_311)
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

if hasattr(sys.stdout, "reconfigure"):          # console Windows : pas de crash
    sys.stdout.reconfigure(errors="replace")

TAILLE_PAQUET = 1024        # bits (= circuits) envoyés par job de simulation
MAX_OCTETS = 4096           # garde-fou : 1 octet = 8 téléportations


# ════════════════════════════════════════════════════════════════════════════
# 1. Couche données : octets <-> bits
# ════════════════════════════════════════════════════════════════════════════
def iter_bits(data: bytes) -> Iterator[int]:
    """Boucle d'émission : octet par octet, puis du bit de poids fort au plus faible."""
    for octet in data:                          # boucle 1 : octets
        for k in range(7, -1, -1):              # boucle 2 : bits b7 ... b0
            yield (octet >> k) & 1


def bits_to_bytes(bits: Iterable[int]) -> bytes:
    """Boucle de réception : regroupe les bits par 8 pour reformer les octets."""
    bits = list(bits)
    if len(bits) % 8:
        raise ValueError(f"{len(bits)} bits reçus : pas un multiple de 8")
    out = bytearray()
    for i in range(0, len(bits), 8):
        octet = 0
        for b in bits[i:i + 8]:
            octet = (octet << 1) | b
        out.append(octet)
    return bytes(out)


# ════════════════════════════════════════════════════════════════════════════
# 2. Couche quantique : circuit de téléportation d'un bit
# ════════════════════════════════════════════════════════════════════════════
def build_teleportation(bit: int) -> QuantumCircuit:
    """Circuit à 3 qubits qui téléporte |bit> de Q0 (Alice) vers Q2 (Bob).

    Les barrières séparent les phases du protocole : elles interdisent au
    compilateur de fusionner ou de déplacer des portes d'une phase à l'autre.
    La synchronisation physique Alice -> Bob, elle, passe par le canal
    classique (les 2 bits d'Alice qui pilotent les if_test)."""
    q = QuantumRegister(3, "q")
    alice = ClassicalRegister(2, "alice")       # alice[0] <- Q0, alice[1] <- Q1
    bob = ClassicalRegister(1, "bob")           # bob[0]   <- Q2
    qc = QuantumCircuit(q, alice, bob, name=f"teleport_{bit}")

    # Phase 1 - Préparation du message sur Q0
    if bit:
        qc.x(q[0])
    qc.barrier(label="message")

    # Phase 2 - Canal quantique : paire de Bell partagée Q1 (Alice) / Q2 (Bob)
    qc.h(q[1])
    qc.cx(q[1], q[2])
    qc.barrier(label="canal")

    # Phase 3 - Alice : injection de Q0 dans le canal, puis mesure
    qc.cx(q[0], q[1])
    qc.h(q[0])
    qc.measure(q[0], alice[0])
    qc.measure(q[1], alice[1])
    qc.barrier(label="alice")

    # Phase 4 - Bob : corrections pilotées par les 2 bits d'Alice
    #           (if_test remplace .c_if(), supprimé depuis Qiskit 2.0)
    with qc.if_test((alice[1], 1)):
        qc.x(q[2])
    with qc.if_test((alice[0], 1)):
        qc.z(q[2])      # sans effet visible sur un bit 0/1, indispensable pour une superposition

    # Phase 5 - Lecture du bit reçu par Bob
    qc.measure(q[2], bob[0])
    return qc


# ════════════════════════════════════════════════════════════════════════════
# 3. Couche transport : simulateur, bruit, envoi par paquets
# ════════════════════════════════════════════════════════════════════════════
def build_noise_model(p: float) -> NoiseModel | None:
    """Bruit simple : dépolarisation p (portes 1 qubit), 2p (CX), lecture fausse p."""
    if p <= 0:
        return None
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(p, 1), ["x", "h", "z"])
    model.add_all_qubit_quantum_error(depolarizing_error(min(2 * p, 1.0), 2), ["cx"])
    model.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return model


class QuantumLink:
    """Liaison quantique simulée Alice -> Bob."""

    def __init__(self, shots: int = 1, noise: float = 0.0, seed: int | None = None):
        self.shots = shots
        self.backend = AerSimulator(noise_model=build_noise_model(noise), seed_simulator=seed)
        # Optimisation : seulement 2 circuits possibles (bit 0 / bit 1), compilés une fois
        self.templates = {b: transpile(build_teleportation(b), self.backend, optimization_level=1)
                          for b in (0, 1)}

    def send(self, bits: list[int]) -> list[dict[str, int]]:
        """Envoie un paquet : une téléportation par bit, toutes dans un seul job."""
        circuits = [self.templates[b] for b in bits]
        result = self.backend.run(circuits, shots=self.shots).result()
        return [result.get_counts(i) for i in range(len(circuits))]


# ════════════════════════════════════════════════════════════════════════════
# 4. Couche réception : décodage des mesures de Bob
# ════════════════════════════════════════════════════════════════════════════
def split_counts(counts: dict[str, int]) -> tuple[Counter, Counter]:
    """Clé Qiskit 'bob alice' (dernier registre déclaré en premier) -> 2 histogrammes."""
    bob, alice = Counter(), Counter()
    for key, n in counts.items():
        b, a = key.split()
        bob[int(b)] += n
        alice[a] += n                           # a = "a1a0" = (Q1, Q0)
    return bob, alice


def majority(hist: Counter) -> int:
    """Vote majoritaire sur les tirs (égalité -> 0 ; préférer un nombre impair)."""
    return 1 if hist[1] > hist[0] else 0


# ════════════════════════════════════════════════════════════════════════════
# 5. Pipeline de bout en bout
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Report:
    received: bytes = b""
    n_bits: int = 0
    bit_errors: int = 0
    shots_ok: int = 0
    shots_total: int = 0
    alice: Counter = field(default_factory=Counter)
    seconds: float = 0.0


def teleport(data: bytes, link: QuantumLink, packet: int = TAILLE_PAQUET) -> Report:
    """Émission bit par bit (par paquets de circuits), puis décodage côté Bob.
    La comparaison avec les bits envoyés sert uniquement à mesurer la qualité
    du lien : Bob reconstruit le message sans la connaître."""
    sent_bits = list(iter_bits(data))
    received_bits: list[int] = []
    rep = Report(n_bits=len(sent_bits))
    t0 = time.perf_counter()

    for start in range(0, len(sent_bits), packet):          # boucle 3 : paquets
        chunk = sent_bits[start:start + packet]
        for sent, counts in zip(chunk, link.send(chunk)):     # boucle 4 : bits reçus
            bob, alice = split_counts(counts)
            bit = majority(bob)
            received_bits.append(bit)
            rep.bit_errors += bit != sent
            rep.shots_ok += bob[sent]
            rep.shots_total += sum(bob.values())
            rep.alice.update(alice)
        done = min(start + packet, len(sent_bits))
        print(f"  paquet {start // packet + 1} : {done}/{len(sent_bits)} bits téléportés", flush=True)

    rep.received = bits_to_bytes(received_bits)
    rep.seconds = time.perf_counter() - t0
    return rep


def print_report(data: bytes, rep: Report, is_text: bool) -> None:
    print()
    if is_text:
        print(f"Message envoyé : {data.decode('utf-8')!r}")
        print(f"Message reçu   : {rep.received.decode('utf-8', errors='replace')!r}")
    else:
        print(f"Données        : {len(data)} octets envoyés, {len(rep.received)} reçus")
    etat = "identique" if rep.received == data else "ALTÉRÉE"
    print(f"Transmission   : {etat} ({rep.bit_errors} bit(s) faux sur {rep.n_bits})")
    print(f"Tirs corrects  : {100 * rep.shots_ok / rep.shots_total:.2f} % "
          f"sur {rep.shots_total} tirs")
    total = sum(rep.alice.values())
    print("Mesures d'Alice (Q1 Q0) : " + "   ".join(
        f"{k} = {100 * rep.alice.get(k, 0) / total:4.1f} %" for k in ("00", "01", "10", "11")))
    print(f"Durée          : {rep.seconds:.2f} s ({rep.n_bits / max(rep.seconds, 1e-9):.0f} bits/s simulés)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transmission de données par téléportation quantique (3 qubits)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("-m", "--message", help="texte à téléporter (UTF-8)")
    src.add_argument("-f", "--fichier", type=Path, help="petit fichier binaire à téléporter")
    p.add_argument("--tirs", type=int, default=1,
                   help="répétitions par bit + vote majoritaire (1 = protocole physique réel)")
    p.add_argument("--bruit", type=float, default=0.0, help="taux d'erreur par porte, ex. 0.03")
    p.add_argument("--graine", type=int, default=None, help="graine aléatoire (résultats reproductibles)")
    p.add_argument("--paquet", type=int, default=TAILLE_PAQUET, help="bits envoyés par job")
    p.add_argument("--schema", action="store_true", help="affiche le circuit de téléportation et quitte")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.schema:
        print(build_teleportation(1).draw(output="text", fold=-1))
        return
    if args.tirs < 1 or args.paquet < 1 or not 0.0 <= args.bruit <= 0.5:
        sys.exit("Paramètres invalides : --tirs >= 1, --paquet >= 1, 0 <= --bruit <= 0.5")

    if args.fichier:
        data, is_text = args.fichier.read_bytes(), False
    else:
        text = args.message
        if text is None:
            text = input("Message à téléporter (Entrée = « Bonjour Bob ! ») : ").strip() or "Bonjour Bob !"
        data, is_text = text.encode("utf-8"), True
    if not data:
        sys.exit("Rien à envoyer.")
    if len(data) > MAX_OCTETS:
        sys.exit(f"Trop volumineux : {len(data)} octets (maximum {MAX_OCTETS}).")

    print(f"Téléportation de {len(data)} octet(s) = {8 * len(data)} bits | "
          f"tirs/bit = {args.tirs} | bruit = {args.bruit}")
    link = QuantumLink(shots=args.tirs, noise=args.bruit, seed=args.graine)
    rep = teleport(data, link, args.paquet)
    print_report(data, rep, is_text)

    if args.fichier:
        out = args.fichier.with_name("recu_" + args.fichier.name)
        out.write_bytes(rep.received)
        print(f"Fichier reçu   : {out}")


if __name__ == "__main__":
    main()
