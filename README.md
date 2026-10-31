# Recherche physique quantique

Scripts de simulation physique et de calcul quantique en Python 3.11.

| Script | Contenu |
|---|---|
| `test.py` | Superposition quantique : 1 qubit, porte H, 1024 mesures (Qiskit Aer) |
| `test_qiskit_2qubits.py` | Intrication : état de Bell (H + CNOT) sur 2 qubits, 1024 mesures |
| `teleportation_quantique.py` | Téléportation quantique de l'état \|1⟩ sur 3 qubits, corrections de Bob conditionnelles |
| `fluide_taichi.py` | Fluide 3D à 10 000 particules dans un cube fermé (PBF, Taichi) |
| `test_taichi.py` | Vérification de Taichi et du GPU |
| `test_mujoco.py` | Boîte en chute libre avec MuJoCo |
| `lancer.bat` | Menu Windows pour lancer chaque script |

## Installation

```bat
uv venv -p 3.11 env_311
uv pip install --python env_311\Scripts\python.exe -r requirements.txt
```

(ou `py -3.11 -m venv env_311` puis `env_311\Scripts\python -m pip install -r requirements.txt`)

## Lancement

Double-cliquer sur `lancer.bat`, ou :

```bat
env_311\Scripts\python test_qiskit_2qubits.py
```
