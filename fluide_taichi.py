#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fluide_taichi.py
================
Simulation 3D d'un fluide à 10 000 particules dans un cube fermé, avec Taichi.

Méthode : PBF — Position Based Fluids (Macklin & Müller, SIGGRAPH 2013)
  1. Prédiction       : v += g·dt ;  x* = x + v·dt
  2. Voisinage        : grille uniforme (taille de cellule = rayon du noyau h)
  3. Incompressibilité: N itérations — calcul de λ_i puis de Δx_i pour imposer
                        la contrainte ρ_i / ρ0 = 1 (+ terme anti-agrégation s_corr)
  4. Mise à jour      : v = (x* − x) / dt, confinement de vorticité, viscosité XSPH
Les six faces du cube sont des parois solides (projection des positions).

Rendu temps réel avec GGUI (Vulkan), couleur = vitesse. Si GGUI n'est pas
disponible, repli automatique sur l'ancien ti.GUI (projection 2D).

Commandes (fenêtre 3D)
  Clic droit + souris : orienter la caméra     W A S D Q E : déplacer la caméra
  Flèches             : incliner la gravité (faire ballotter le fluide)
  Espace              : pause / reprise        R           : réinitialiser
  Échap               : quitter

Usage
  python fluide_taichi.py                      # GPU si disponible, sinon CPU
  python fluide_taichi.py --arch cpu
  python fluide_taichi.py --headless --frames 600   # sans fenêtre (benchmark)

Écrit pour : Python 3.11, taichi 1.7.4, numpy 2.x  (env_311)
"""

import argparse
import math
import sys
import time

import numpy as np
import taichi as ti

if hasattr(sys.stdout, "reconfigure"):          # console Windows : pas de crash sur ρ, ³…
    sys.stdout.reconfigure(errors="replace")

# ════════════════════════════════════════════════════════════════════════════
# Paramètres
# ════════════════════════════════════════════════════════════════════════════
# --- PARAMS BEGIN
N_X, N_Y, N_Z = 20, 25, 20          # bloc initial : 20 × 25 × 20 = 10 000 particules
N_PARTICLES = N_X * N_Y * N_Z

BOX = 1.0                           # arête du cube fermé (m) — domaine [0, BOX]³
LATTICE = 32                        # le cube contient LATTICE³ sites de réseau
SPACING = BOX / LATTICE             # espacement initial entre particules (m)
H = 2.0 * SPACING                   # rayon d'influence du noyau SPH (m)
PARTICLE_RADIUS = 0.5 * SPACING     # rayon d'une particule (parois + rendu)

FPS = 60                            # images affichées par seconde de simulation
SUBSTEPS = 4                        # sous-pas par image
DT = 1.0 / (FPS * SUBSTEPS)         # pas de temps (s)
SOLVER_ITERS = 4                    # itérations du solveur de densité par sous-pas
GRAVITY = 9.81                      # m/s²
MAX_TILT = math.radians(35.0)       # inclinaison max de la gravité (flèches)

RELAX_FRACTION = 0.01               # ε de λ, en fraction du dénominateur au repos
SCORR_K = 0.002                     # pression artificielle anti-agrégation (s_corr)
SCORR_DQ = 0.3 * H                  # distance de référence Δq de s_corr
XSPH_VISCOSITY = 0.04               # viscosité artificielle XSPH (0 = aucune)
VORTICITY_EPS = 0.02                # confinement de vorticité (0 = désactivé)
V_MAX = 0.5 * H / DT                # vitesse max (garde-fou de type CFL)

# Grille de recherche des voisins
CELL_SIZE = H
GRID_RES = int(math.ceil(BOX / CELL_SIZE))
MAX_PER_CELL = 64
MAX_NEIGHBORS = 96

# Noyaux SPH (Müller et al. 2003) — masse unitaire : seul le rapport ρ/ρ0 compte
H2 = H * H
POLY6_COEF = 315.0 / (64.0 * math.pi * H ** 9)
SPIKY_GRAD_COEF = -45.0 / (math.pi * H ** 6)
W_SELF = POLY6_COEF * H2 ** 3       # contribution d'une particule à sa propre densité


def _rest_state():
    """Densité au repos ρ0 et dénominateur de λ pour une particule entourée
    d'un réseau cubique complet d'espacement SPACING (état initial du bloc).
    Calculer ρ0 ainsi garantit que le bloc initial est exactement au repos."""
    k = np.arange(-3, 4) * SPACING
    r = np.stack(np.meshgrid(k, k, k, indexing="ij"), -1).reshape(-1, 3)
    r = r[np.any(r != 0.0, axis=1)]                     # voisins seulement
    d = np.linalg.norm(r, axis=1)
    inside = d < H
    r, d = r[inside], d[inside]
    rho0 = W_SELF + np.sum(POLY6_COEF * (H2 - d * d) ** 3)
    grad = r * (SPIKY_GRAD_COEF * (H - d) ** 2 / d)[:, None] / rho0
    denom0 = np.sum(grad * grad) + np.sum(grad.sum(0) ** 2)
    return float(rho0), float(denom0)


RHO0, DENOM0 = _rest_state()
INV_RHO0 = 1.0 / RHO0
# Parois : particules-fantômes fixes situées sur le prolongement du réseau
# ((i + ½)·SPACING) à l'extérieur du cube. Elles ne sont pas stockées : on les
# énumère à la volée autour de chaque particule proche d'une paroi.
GHOST_REACH = int(math.ceil(H / SPACING))
RELAXATION = RELAX_FRACTION * DENOM0
SCORR_COEF = SCORR_K / DENOM0       # s_corr exprimé dans les mêmes unités que λ
INV_W_DQ = 1.0 / (POLY6_COEF * (H2 - SCORR_DQ ** 2) ** 3)

WALL_LO = PARTICLE_RADIUS           # positions admissibles : [WALL_LO, WALL_HI]³
WALL_HI = BOX - PARTICLE_RADIUS
WALL_JITTER = 1e-4                  # petit bruit pour éviter l'empilement exact
# --- PARAMS END


# ════════════════════════════════════════════════════════════════════════════
# Initialisation de Taichi
# ════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Fluide PBF 3D (10 000 particules) avec Taichi")
    p.add_argument("--arch", default="gpu", choices=["gpu", "cpu", "cuda", "vulkan"],
                   help="backend Taichi (défaut : gpu, repli automatique sur cpu)")
    p.add_argument("--headless", action="store_true",
                   help="simuler sans fenêtre et afficher des statistiques")
    p.add_argument("--frames", type=int, default=600,
                   help="nombre d'images en mode --headless (défaut : 600 = 10 s)")
    p.add_argument("--legacy-gui", action="store_true",
                   help="forcer l'ancien ti.GUI 2D (si GGUI/Vulkan pose problème)")
    return p.parse_args()


ARGS = parse_args()
ARCHS = {"gpu": ti.gpu, "cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}
ti.init(arch=ARCHS[ARGS.arch], default_fp=ti.f32, random_seed=42)

# ════════════════════════════════════════════════════════════════════════════
# Champs Taichi
# ════════════════════════════════════════════════════════════════════════════
pos = ti.Vector.field(3, ti.f32, N_PARTICLES)       # positions
old_pos = ti.Vector.field(3, ti.f32, N_PARTICLES)   # positions au début du sous-pas
vel = ti.Vector.field(3, ti.f32, N_PARTICLES)       # vitesses
vel_tmp = ti.Vector.field(3, ti.f32, N_PARTICLES)   # tampon (XSPH / vorticité)
dpos = ti.Vector.field(3, ti.f32, N_PARTICLES)      # corrections Δx
omega = ti.Vector.field(3, ti.f32, N_PARTICLES)     # vorticité ω = ∇ × v
lambdas = ti.field(ti.f32, N_PARTICLES)             # multiplicateurs λ
density = ti.field(ti.f32, N_PARTICLES)             # densité SPH
colors = ti.Vector.field(3, ti.f32, N_PARTICLES)    # couleur de rendu

grid_count = ti.field(ti.i32, (GRID_RES, GRID_RES, GRID_RES))
grid_particles = ti.field(ti.i32, (GRID_RES, GRID_RES, GRID_RES, MAX_PER_CELL))
nb_count = ti.field(ti.i32, N_PARTICLES)
nb_list = ti.field(ti.i32, (N_PARTICLES, MAX_NEIGHBORS))

gravity = ti.Vector.field(3, ti.f32, ())            # modifiable en direct (flèches)
xsph_c = ti.field(ti.f32, ())                       # réglables depuis le panneau
vort_eps = ti.field(ti.f32, ())
stats = ti.field(ti.f32, 6)

box_vertices = ti.Vector.field(3, ti.f32, 24)       # 12 arêtes du cube


# ════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires (côté GPU)
# ════════════════════════════════════════════════════════════════════════════
@ti.func
def poly6(r2):
    """Noyau Poly6 évalué à partir de la distance au carré."""
    w = 0.0
    if r2 < H2:
        x = H2 - r2
        w = POLY6_COEF * x * x * x
    return w


@ti.func
def spiky_grad(r):
    """Gradient du noyau Spiky (évite l'agrégation des particules proches)."""
    g = ti.Vector([0.0, 0.0, 0.0])
    r_len = r.norm()
    if r_len > 1e-6 and r_len < H:
        x = H - r_len
        g = r * (SPIKY_GRAD_COEF * x * x / r_len)
    return g


@ti.func
def confine(p):
    """Maintient une position à l'intérieur du cube fermé (parois solides)."""
    for k in ti.static(range(3)):
        if p[k] < WALL_LO:
            p[k] = WALL_LO + WALL_JITTER * ti.random()
        elif p[k] > WALL_HI:
            p[k] = WALL_HI - WALL_JITTER * ti.random()
    return p


@ti.func
def wall_terms(p):
    """Contribution des parois : (∇ρ_paroi.x, .y, .z, ρ_paroi).
    Les parois sont un réseau implicite de particules-fantômes immobiles qui
    complète le voisinage des particules proches des faces, arêtes et coins.
    Sans cela, les couches au contact du fond s'écrasent les unes sur les autres."""
    rho_w = 0.0
    grad_w = ti.Vector([0.0, 0.0, 0.0])
    near = False
    for a in ti.static(range(3)):
        if p[a] < H or p[a] > BOX - H:
            near = True
    if near:
        base = ti.floor(p / SPACING - 0.5, ti.i32)       # site de réseau voisin
        for ox in range(1 - GHOST_REACH, GHOST_REACH + 1):
            for oy in range(1 - GHOST_REACH, GHOST_REACH + 1):
                for oz in range(1 - GHOST_REACH, GHOST_REACH + 1):
                    idx = base + ti.Vector([ox, oy, oz])
                    outside = False
                    for a in ti.static(range(3)):
                        if idx[a] < 0 or idx[a] >= LATTICE:
                            outside = True
                    if outside:                          # site hors du cube = fantôme
                        r = p - (ti.cast(idx, ti.f32) + 0.5) * SPACING
                        r2 = r.norm_sqr()
                        if r2 < H2:
                            rho_w += poly6(r2)
                            grad_w += spiky_grad(r)
    return ti.Vector([grad_w[0], grad_w[1], grad_w[2], rho_w])


@ti.func
def cell_of(p):
    c = ti.cast(p / CELL_SIZE, ti.i32)
    return ti.max(ti.min(c, GRID_RES - 1), 0)


# ════════════════════════════════════════════════════════════════════════════
# Noyaux de calcul
# ════════════════════════════════════════════════════════════════════════════
@ti.kernel
def init_particles():
    """Bloc de fluide (« rupture de barrage ») plaqué dans un coin du cube."""
    for i in pos:
        ix = i % N_X
        iy = (i // N_X) % N_Y
        iz = i // (N_X * N_Y)
        p = ti.cast(ti.Vector([ix, iy, iz]), ti.f32) * SPACING + WALL_LO   # contre les parois
        jitter = (ti.Vector([ti.random(), ti.random(), ti.random()]) - 0.5) * 0.02 * SPACING
        pos[i] = confine(p + jitter)
        old_pos[i] = pos[i]
        vel[i] = ti.Vector([0.0, 0.0, 0.0])
        density[i] = RHO0


@ti.kernel
def predict():
    for i in pos:
        old_pos[i] = pos[i]
        v = vel[i] + DT * gravity[None]
        vel[i] = v
        pos[i] = confine(pos[i] + DT * v)


@ti.kernel
def build_grid():
    for I in ti.grouped(grid_count):
        grid_count[I] = 0
    for i in pos:
        c = cell_of(pos[i])
        slot = ti.atomic_add(grid_count[c[0], c[1], c[2]], 1)
        if slot < MAX_PER_CELL:
            grid_particles[c[0], c[1], c[2], slot] = i


@ti.kernel
def find_neighbors():
    for i in pos:
        pi = pos[i]
        c = cell_of(pi)
        n = 0
        for off in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
            cc = c + off
            if 0 <= cc[0] < GRID_RES and 0 <= cc[1] < GRID_RES and 0 <= cc[2] < GRID_RES:
                cnt = ti.min(grid_count[cc[0], cc[1], cc[2]], MAX_PER_CELL)
                for s in range(cnt):
                    j = grid_particles[cc[0], cc[1], cc[2], s]
                    if j != i and n < MAX_NEIGHBORS and (pos[j] - pi).norm_sqr() < H2:
                        nb_list[i, n] = j
                        n += 1
        nb_count[i] = n


@ti.kernel
def compute_lambdas():
    for i in pos:
        pi = pos[i]
        wall = wall_terms(pi)
        rho = W_SELF + wall[3]
        grad_i = ti.Vector([wall[0], wall[1], wall[2]]) * INV_RHO0
        sum_grad_sq = 0.0
        for k in range(nb_count[i]):
            r = pi - pos[nb_list[i, k]]
            rho += poly6(r.norm_sqr())
            g = spiky_grad(r) * INV_RHO0           # ∇_pj C_i
            grad_i += g
            sum_grad_sq += g.norm_sqr()
        density[i] = rho
        # Contrainte unilatérale : on ne corrige que la compression (évite que
        # la surface libre « colle » et que les particules s'agglutinent).
        C = ti.max(rho * INV_RHO0 - 1.0, 0.0)
        lambdas[i] = -C / (sum_grad_sq + grad_i.norm_sqr() + RELAXATION)


@ti.kernel
def compute_dpos():
    for i in pos:
        pi = pos[i]
        li = lambdas[i]
        wall = wall_terms(pi)
        d = li * ti.Vector([wall[0], wall[1], wall[2]])  # répulsion des parois
        for k in range(nb_count[i]):
            j = nb_list[i, k]
            r = pi - pos[j]
            x = poly6(r.norm_sqr()) * INV_W_DQ
            x2 = x * x
            s_corr = -SCORR_COEF * x2 * x2               # −k (W/W(Δq))⁴
            d += (li + lambdas[j] + s_corr) * spiky_grad(r)
        dpos[i] = d * INV_RHO0


@ti.kernel
def apply_dpos():
    for i in pos:
        pos[i] = confine(pos[i] + dpos[i])


@ti.kernel
def update_velocity():
    for i in pos:
        v = (pos[i] - old_pos[i]) / DT
        speed = v.norm()
        if speed > V_MAX:
            v *= V_MAX / speed
        vel[i] = v


@ti.kernel
def compute_vorticity():
    for i in pos:
        pi = pos[i]
        vi = vel[i]
        w = ti.Vector([0.0, 0.0, 0.0])
        for k in range(nb_count[i]):
            j = nb_list[i, k]
            w += spiky_grad(pi - pos[j]).cross(vel[j] - vi)
        omega[i] = w * INV_RHO0


@ti.kernel
def vorticity_and_xsph():
    for i in pos:
        pi = pos[i]
        vi = vel[i]
        wi = omega[i]
        wi_len = wi.norm()
        eta = ti.Vector([0.0, 0.0, 0.0])     # ∇|ω|
        dv = ti.Vector([0.0, 0.0, 0.0])      # Σ (v_j − v_i) W_ij
        for k in range(nb_count[i]):
            j = nb_list[i, k]
            r = pi - pos[j]
            eta += (omega[j].norm() - wi_len) * spiky_grad(r)
            dv += (vel[j] - vi) * poly6(r.norm_sqr())
        eta *= INV_RHO0
        f = ti.Vector([0.0, 0.0, 0.0])
        eta_len = eta.norm()
        if eta_len > 1e-4:
            f = vort_eps[None] * (eta / eta_len).cross(wi)   # ε (N × ω)
        vel_tmp[i] = vi + DT * f + xsph_c[None] * INV_RHO0 * dv


@ti.kernel
def commit_velocity():
    for i in vel:
        vel[i] = vel_tmp[i]


@ti.kernel
def update_colors(v_ref: ti.f32):
    """Bleu profond (lent) → bleu vif → blanc écume (rapide)."""
    for i in pos:
        deep = ti.Vector([0.04, 0.20, 0.62])
        mid = ti.Vector([0.10, 0.55, 0.95])
        foam = ti.Vector([0.88, 0.96, 1.00])
        s = ti.min(vel[i].norm() / v_ref, 1.0)
        c = deep + (mid - deep) * (2.0 * s)
        if s > 0.5:
            c = mid + (foam - mid) * (2.0 * s - 1.0)
        colors[i] = c


@ti.kernel
def compute_stats():
    """0: v max | 1: ρ/ρ0 max | 2: énergie cinétique | 3: Σ ρ/ρ0 (intérieur)
    4: nb particules intérieures | 5: nb de voisins max"""
    for k in stats:
        stats[k] = 0.0
    for i in pos:
        v2 = vel[i].norm_sqr()
        ti.atomic_max(stats[0], ti.sqrt(v2))
        ti.atomic_max(stats[1], density[i] * INV_RHO0)
        ti.atomic_add(stats[2], 0.5 * v2)
        if nb_count[i] >= 25:
            ti.atomic_add(stats[3], density[i] * INV_RHO0)
            ti.atomic_add(stats[4], 1.0)
        ti.atomic_max(stats[5], ti.cast(nb_count[i], ti.f32))


# ════════════════════════════════════════════════════════════════════════════
# Pilotage de la simulation (côté Python)
# ════════════════════════════════════════════════════════════════════════════
def substep():
    predict()
    build_grid()
    find_neighbors()
    for _ in range(SOLVER_ITERS):
        compute_lambdas()
        compute_dpos()
        apply_dpos()
    update_velocity()
    compute_vorticity()
    vorticity_and_xsph()
    commit_velocity()


def step_frame():
    for _ in range(SUBSTEPS):
        substep()


class Settings:
    """Réglages modifiables en direct (copie Python pour éviter de relire le GPU)."""
    xsph = XSPH_VISCOSITY
    vort = VORTICITY_EPS

    @classmethod
    def push(cls):
        xsph_c[None] = cls.xsph
        vort_eps[None] = cls.vort


def set_gravity(tilt_x, tilt_z):
    """Gravité inclinée de tilt_x (autour de z) et tilt_z (autour de x)."""
    gx, gz = math.sin(tilt_x), math.sin(tilt_z)
    gy = -math.sqrt(max(0.0, 1.0 - gx * gx - gz * gz))
    gravity[None] = [GRAVITY * gx, GRAVITY * gy, GRAVITY * gz]


def reset():
    Settings.xsph, Settings.vort = XSPH_VISCOSITY, VORTICITY_EPS
    Settings.push()
    set_gravity(0.0, 0.0)
    init_particles()


def read_stats():
    compute_stats()
    s = stats.to_numpy()
    return {
        "v_max": float(s[0]),
        "compression_max": 100.0 * (float(s[1]) - 1.0),
        "ec_moy": float(s[2]) / N_PARTICLES,
        "densite_moy": float(s[3] / s[4]) if s[4] > 0 else 0.0,
        "voisins_max": int(s[5]),
    }


def make_box_vertices():
    corners = np.array([[x, y, z] for x in (0.0, BOX) for y in (0.0, BOX) for z in (0.0, BOX)],
                       dtype=np.float32)
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
             if np.count_nonzero(corners[a] != corners[b]) == 1]
    return np.array([corners[k] for e in edges for k in e], dtype=np.float32)   # (24, 3)


def tilt_target(left, right, up, down):
    tx = (MAX_TILT if right else 0.0) - (MAX_TILT if left else 0.0)
    tz = (MAX_TILT if down else 0.0) - (MAX_TILT if up else 0.0)
    return tx, tz


# ════════════════════════════════════════════════════════════════════════════
# Modes d'exécution
# ════════════════════════════════════════════════════════════════════════════
def run_headless(frames):
    print(f"Simulation sans fenêtre : {N_PARTICLES} particules, {frames} images, "
          f"dt = {DT * 1e3:.2f} ms, {SOLVER_ITERS} itérations/sous-pas")
    step_frame()                       # compilation des noyaux (hors chronomètre)
    ti.sync()
    reset()
    t0 = time.perf_counter()
    for f in range(1, frames + 1):
        step_frame()
        if f % FPS == 0 or f == frames:
            s = read_stats()
            p = pos.to_numpy()
            print(f"t = {f / FPS:5.2f} s | v max = {s['v_max']:5.2f} m/s | "
                  f"Ec moy = {s['ec_moy']:.2e} J/kg | ρ/ρ0 intérieur = {s['densite_moy']:.3f} | "
                  f"compression max = {s['compression_max']:5.2f} % | "
                  f"voisins max = {s['voisins_max']:3d} | "
                  f"bornes = [{p.min():.3f}, {p.max():.3f}]")
    ti.sync()
    elapsed = time.perf_counter() - t0
    print(f"{frames} images en {elapsed:.2f} s → {frames / elapsed:.1f} images/s")


def create_ggui_window():
    """Fenêtre 3D GGUI, ou None si Vulkan/GGUI n'est pas disponible."""
    try:
        return ti.ui.Window("Fluide PBF 3D - 10 000 particules (Taichi)", (1280, 800), vsync=True)
    except Exception as exc:
        print(f"[fluide] GGUI indisponible ({exc}) -> repli sur ti.GUI 2D")
        return None


def run_ggui(window):
    canvas = window.get_canvas()
    scene = window.get_scene()
    gui = window.get_gui()
    camera = ti.ui.Camera()
    camera.position(2.15, 1.35, 2.45)
    camera.lookat(0.5, 0.35, 0.5)
    camera.up(0.0, 1.0, 0.0)
    camera.fov(40)

    paused = False
    tilt = [0.0, 0.0]
    sim_time = 0.0
    fps, t_prev = 0.0, time.perf_counter()

    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == ti.ui.SPACE:
                paused = not paused
            elif e.key in ("r", "R"):
                reset()
                sim_time = 0.0

        # Inclinaison progressive de la gravité avec les flèches
        tx, tz = tilt_target(window.is_pressed(ti.ui.LEFT), window.is_pressed(ti.ui.RIGHT),
                             window.is_pressed(ti.ui.UP), window.is_pressed(ti.ui.DOWN))
        tilt[0] += 0.12 * (tx - tilt[0])
        tilt[1] += 0.12 * (tz - tilt[1])
        set_gravity(*tilt)

        if not paused:
            step_frame()
            sim_time += 1.0 / FPS
        update_colors(3.0)

        # Scène 3D
        camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.45, 0.45, 0.50))
        scene.point_light(pos=(0.5, 2.5, 1.5), color=(1.0, 1.0, 1.0))
        scene.point_light(pos=(-1.5, 1.0, -0.5), color=(0.35, 0.35, 0.45))
        scene.particles(pos, radius=PARTICLE_RADIUS, per_vertex_color=colors)
        scene.lines(box_vertices, width=2.0, color=(0.75, 0.80, 0.85))
        canvas.set_background_color((0.07, 0.08, 0.11))
        canvas.scene(scene)

        # Panneau d'information et réglages
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        s = read_stats()
        with gui.sub_window("Fluide PBF", 0.01, 0.01, 0.30, 0.36) as w:
            w.text(f"Particules : {N_PARTICLES}")
            w.text(f"FPS : {fps:5.1f}   temps simulé : {sim_time:6.2f} s")
            w.text(f"Vitesse max : {s['v_max']:.2f} m/s")
            w.text(f"Compression max : {s['compression_max']:.2f} %")
            w.text(f"Gravité inclinée : x {math.degrees(tilt[0]):+5.1f}°  z {math.degrees(tilt[1]):+5.1f}°")
            new_xsph = w.slider_float("Viscosité XSPH", Settings.xsph, 0.0, 0.2)
            new_vort = w.slider_float("Vorticité", Settings.vort, 0.0, 0.1)
            if new_xsph != Settings.xsph or new_vort != Settings.vort:
                Settings.xsph, Settings.vort = new_xsph, new_vort
                Settings.push()
            if w.button("Pause / Reprise"):
                paused = not paused
            if w.button("Réinitialiser"):
                reset()
                sim_time = 0.0
            w.text("Flèches : incliner | Clic droit : caméra")
        window.show()


def run_legacy_gui():
    """Repli sans Vulkan : projection orthographique 2D avec ti.GUI."""
    gui = ti.GUI("Fluide PBF 3D - 10 000 particules (ti.GUI)", res=(900, 900),
                 background_color=0x10131A)
    edges = make_box_vertices().reshape(-1, 2, 3)
    yaw, pitch = 0.65, 0.40
    paused = False
    tilt = [0.0, 0.0]

    def project(p):
        c = p - 0.5 * BOX
        cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        x = c[..., 0] * cy - c[..., 2] * sy
        z = c[..., 0] * sy + c[..., 2] * cy
        y = c[..., 1] * cp - z * sp
        depth = z * cp + c[..., 1] * sp          # plus grand = plus proche
        return np.stack([0.5 + 0.52 * x, 0.47 + 0.52 * y], -1), depth

    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.SPACE:
                paused = not paused
            elif e.key in ("r", "R"):
                reset()
        if gui.is_pressed("a"):
            yaw -= 0.03
        if gui.is_pressed("d"):
            yaw += 0.03
        if gui.is_pressed("w"):
            pitch = min(pitch + 0.02, 1.4)
        if gui.is_pressed("s"):
            pitch = max(pitch - 0.02, -1.4)

        tx, tz = tilt_target(gui.is_pressed(ti.GUI.LEFT), gui.is_pressed(ti.GUI.RIGHT),
                             gui.is_pressed(ti.GUI.UP), gui.is_pressed(ti.GUI.DOWN))
        tilt[0] += 0.12 * (tx - tilt[0])
        tilt[1] += 0.12 * (tz - tilt[1])
        set_gravity(*tilt)

        if not paused:
            step_frame()
        update_colors(3.0)

        p = pos.to_numpy()
        col = (np.clip(colors.to_numpy(), 0.0, 1.0) * 255).astype(np.uint32)
        hexcol = (col[:, 0] << 16) | (col[:, 1] << 8) | col[:, 2]
        xy, depth = project(p)
        order = np.argsort(depth)                # du plus lointain au plus proche
        exy, _ = project(edges)
        gui.lines(exy[:, 0], exy[:, 1], radius=1.5, color=0x8899AA)
        gui.circles(xy[order], radius=2.2, color=hexcol[order])
        gui.text("A/D W/S : camera | Fleches : gravite | Espace : pause | R : reset",
                 (0.02, 0.98), font_size=16, color=0xCCCCCC)
        gui.show()


# ════════════════════════════════════════════════════════════════════════════
def main():
    print(f"[fluide] {N_PARTICLES} particules | cube {BOX} m | h = {H:.3f} m | "
          f"grille {GRID_RES}³ | ρ0 (unités noyau) = {RHO0:.1f}")
    box_vertices.from_numpy(make_box_vertices())
    reset()

    if ARGS.headless:
        run_headless(ARGS.frames)
        return
    window = None if ARGS.legacy_gui else create_ggui_window()
    if window is not None:
        run_ggui(window)
    else:
        run_legacy_gui()


if __name__ == "__main__":
    main()
