"""
Bilan hydrique journalier du riz pluvial (méthode FAO-56 simplifiée, 3 couches de sol).

Sol : 0–10 cm (couche d'évaporation), 10–30 cm, 30–60 cm. Eau utile de chaque couche
= réserve utile (mm/m, SoilGrids + Saxton & Rawls) × épaisseur. Pluie : ruissellement
des fortes pluies, remplissage de haut en bas, l'excédent draine.

Avant le semis (jachère) : évaporation du sol nu depuis 0–10 cm et transpiration
d'une végétation de jachère (Kc 0,3) qui se réduit quand le sol sèche.

Riz (à partir de la germination) :
  - coefficient cultural FAO-56 : Kc 0,5 (initial) → 1,15 (mi-saison) → 0,8 (fin) ;
    durées des phases : 15 %, 25 %, 40 %, 20 % du cycle ;
  - racines : 10 cm à la germination → 60 cm à 50 % du cycle ;
  - stress hydrique (FAO-56) : Ks = 1 tant que l'épuisement de la zone racinaire
    est < p × réserve facilement utilisable (p = 0,2 pour le riz), puis décroît ;
  - ETa = Ks × Kc × ET0 ; indice de satisfaction des besoins en eau
    WRSI = 100 × ΣETa / ΣETc (FAO, FEWS NET).

Critères de réussite d'un semis (riz pluvial, revue de littérature) :
  1. germination : la couche 0–10 cm atteint 60 % de son eau utile dans les 10 jours ;
  2. levée : pas de stress fort (Ks < 0,5) 5 jours de suite dans les 20 jours suivants ;
  3. floraison (cycle − 35 j, de −14 à +7 j) : pas de Ks < 0,5 5 jours de suite,
     et moins de 3 jours à Tmax ≥ 35 °C autour de la floraison (± 5 j) ;
  4. ensemble du cycle : WRSI ≥ WRSI_MIN (75 par défaut).
"""
import os

import numpy as np

# Paramètres par culture (FAO-56, Allen et al. 1998, tableaux 12, 17 et 22).
#   ep : épaisseur des couches de sol (mm) ; kc : (fractions du cycle, valeurs) ; racines : (min, max mm,
#   fraction du cycle où le maximum est atteint) ; p : fraction de l'eau utile consommée sans stress ;
#   flo : jour de floraison (fonction du cycle L) ; fen_flo : fenêtre sensible autour de la floraison.
CULTURES = {
    "riz": {"nom": "riz pluvial", "cycles": (100, 110, 120), "ep": (100.0, 200.0, 300.0),
            "kc": ((0, .15, .40, .80, 1.0), (0.5, 0.5, 1.15, 1.15, 0.8)), "racines": (100.0, 600.0, 0.5),
            "p": 0.20, "flo": lambda L: L - 35, "fen_flo": (-14, 7)},
    "mais": {"nom": "maïs pluvial", "cycles": (90, 105, 120), "ep": (100.0, 200.0, 300.0, 400.0),
             "kc": ((0, .16, .44, .76, 1.0), (0.4, 0.4, 1.20, 1.20, 0.5)), "racines": (100.0, 1000.0, 0.5),
             "p": 0.55, "flo": lambda L: round(0.55 * L), "fen_flo": (-7, 14),
             # floraison femelle très sensible (Ky = 1,5, FAO-33) : un stress modéré suffit
             "ks_fort": 0.7, "jours_stress": 3},
    # Arachide : FAO-56 (Kc 0,40 / 1,15 / 0,60, racines 0,5–1,0 m, p = 0,50). Phase la plus sensible :
    # floraison, mise en place des gynophores et début de formation des gousses (≈ 45 % du cycle).
    "arachide": {"nom": "arachide", "cycles": (90, 105, 120), "ep": (100.0, 200.0, 300.0, 400.0),
                 "kc": ((0, .19, .46, .81, 1.0), (0.4, 0.4, 1.15, 1.15, 0.6)), "racines": (100.0, 800.0, 0.5),
                 "p": 0.50, "flo": lambda L: round(0.45 * L), "fen_flo": (-10, 20),
                 # récolte sous la pluie : gousses qui ne sèchent pas, pourriture, aflatoxines
                 "recolte_humide": {"debut": -5, "fin": 10, "mm": 80.0}},
}
CULTURE = "riz"
EP = np.array(CULTURES["riz"]["ep"])          # épaisseur des couches (mm)
Z_TOP = np.cumsum(EP) - EP                    # profondeur du haut de chaque couche (mm)
P_RIZ = float(os.environ.get("P_RIZ", 0.20))
WRSI_MIN = float(os.environ.get("WRSI_MIN", 75))


def choisir_culture(nom):
    """Règle les paramètres du module pour une culture (« riz » par défaut, « mais »)."""
    global CULTURE, EP, Z_TOP, P_RIZ
    c = CULTURES[nom]
    CULTURE = nom
    EP = np.array(c["ep"]); Z_TOP = np.cumsum(EP) - EP
    P_RIZ = float(os.environ.get("P_RIZ", c["p"])) if nom == "riz" else c["p"]
    return c
KC_JACHERE = 0.30
J_GERM = 10            # jours laissés à la germination
J_LEVEE = 20           # jours de suivi de la levée après germination
RUN_STRESS = 5         # jours consécutifs de stress fort = échec
KS_FORT = 0.5
TMAX_CRIT = 35.0


def capacites(ru_mm_m):
    """Eau utile (mm) de chaque couche ; ru_mm_m : (...,) → (3, ...)."""
    return np.stack([ru_mm_m * e / 1000.0 for e in EP])


def infiltrer(W, cap, pluie):
    """Ruissellement des fortes pluies puis remplissage de haut en bas (in-place)."""
    p = pluie - 0.15 * np.maximum(pluie - 25.0, 0)
    for k in range(len(EP)):
        add = np.minimum(p, cap[k] - W[k])
        W[k] += add
        p = p - add
    return W


def pas_jachere(W, cap, pluie, et0):
    infiltrer(W, cap, pluie)
    r1 = W[0] / cap[0]
    es = np.minimum(1.0 * et0 * np.clip(r1 / 0.6, 0, 1), W[0])      # sol nu, 2 phases
    W[0] -= es
    tot = W.sum(0); captot = cap.sum(0)
    ks = np.clip(tot / (0.5 * captot), 0, 1)
    prof = W[1:].sum(0)
    tr = np.minimum(KC_JACHERE * et0 * ks, np.maximum(prof, 0))
    for k in range(1, len(EP)):
        W[k] -= tr * np.where(prof > 0, W[k] / np.maximum(prof, 1e-9), 0)
    np.maximum(W, 0, out=W)
    return W


def simuler_jachere(pluie, et0, cap, W0=None):
    """Simulation continue de la jachère. pluie, et0 : (T, ...) → états (T, 3, ...) en FIN de journée."""
    W = (0.5 * cap).copy() if W0 is None else W0.copy()
    out = np.empty((pluie.shape[0],) + W.shape, np.float32)
    for t in range(pluie.shape[0]):
        pas_jachere(W, cap, pluie[t], et0[t])
        out[t] = W
    return out


def duree_forcage(L):
    """Nombre de jours de forçage à fournir à simuler_semis pour un cycle de L jours."""
    rh = CULTURES[CULTURE].get("recolte_humide")
    return J_GERM + L + 2 + (rh["fin"] + 1 if rh else 0)


def kc_riz(j, L):
    """Kc FAO-56 de la culture choisie ; j : tableau de jours depuis la germination."""
    fr, val = CULTURES[CULTURE]["kc"]
    return np.interp(j, [f * L for f in fr], val)


def racines(j, L):
    z0, z1, f = CULTURES[CULTURE]["racines"]
    return z0 + (z1 - z0) * np.clip(j / (f * L), 0, 1)     # mm


def simuler_semis(W_semis, cap, pluie_f, et0_f, tmax_f, L):
    """
    Simule des semis à partir de leurs états de sol au jour du semis.
    W_semis : (3, N...) ; pluie_f, et0_f, tmax_f : (J, N...) forçage à partir du jour du semis
    (J ≥ J_GERM + L + 1). Renvoie un dict de tableaux (N...) : reussite (bool), cause (int), wrsi.
    Causes : 0 réussite, 1 pas de germination, 2 levée, 3 floraison (sécheresse),
             4 floraison (chaleur), 5 WRSI insuffisant.
    """
    W = W_semis.copy()
    shp = W.shape[1:]
    germ = np.full(shp, -1, int)
    run = np.zeros(shp, int)
    lev_ko = np.zeros(shp, bool); flo_ko = np.zeros(shp, bool); chaleur = np.zeros(shp, int)
    sum_eta = np.zeros(shp); sum_etc = np.zeros(shp)
    flo = CULTURES[CULTURE]["flo"](L)
    rh = CULTURES[CULTURE].get("recolte_humide")
    pluie_rec = np.zeros(shp)
    ks_fort = CULTURES[CULTURE].get("ks_fort", KS_FORT)
    run_stress = CULTURES[CULTURE].get("jours_stress", RUN_STRESS)
    f0, f1 = CULTURES[CULTURE]["fen_flo"]
    J = pluie_f.shape[0]
    for t in range(J):
        p, e0, tx = pluie_f[t], et0_f[t], tmax_f[t]
        infiltrer(W, cap, p)
        g = germ >= 0
        # germination : couche 0–10 cm suffisamment humide
        new = (~g) & (W[0] / cap[0] >= 0.6) & (t <= J_GERM)
        germ = np.where(new, t, germ)
        g = germ >= 0
        j = np.where(g, t - germ, -1)                       # jours depuis la germination
        # sol nu tant que non germé
        es = np.where(~g, np.minimum(et0_f[t] * np.clip(W[0] / cap[0] / 0.6, 0, 1), W[0]), 0)
        W[0] -= es
        # culture
        if rh is not None:
            jr = np.where(germ >= 0, t - germ, -999)
            pluie_rec += np.where((jr >= L + rh["debut"]) & (jr <= L + rh["fin"]), p, 0)
        if g.any():
            jj = np.clip(j, 0, L)
            kc = kc_riz(jj, L)
            zr = racines(jj, L)
            frac = np.stack([np.clip((zr - Z_TOP[k]) / EP[k], 0, 1) for k in range(len(EP))])
            taw = (frac * cap).sum(0)
            wr = (frac * W).sum(0)
            dr = taw - wr
            raw = P_RIZ * taw
            ks = np.where(dr <= raw, 1.0, np.clip((taw - dr) / np.maximum((1 - P_RIZ) * taw, 1e-9), 0, 1))
            actif = g & (j >= 0) & (j < L)
            etc = kc * e0
            eta = np.minimum(ks * etc, wr)
            share = np.where(wr[None] > 0, frac * W / np.maximum(wr[None], 1e-9), 0)
            W -= np.where(actif[None], share * eta[None], 0)
            np.maximum(W, 0, out=W)
            sum_eta += np.where(actif, eta, 0); sum_etc += np.where(actif, etc, 0)
            fort = actif & (ks < ks_fort)
            run = np.where(fort, run + 1, 0)
            lev_ko |= (run >= run_stress) & (j <= J_LEVEE)
            flo_ko |= (run >= run_stress) & (j >= flo + f0) & (j <= flo + f1)
            chaleur += (actif & (abs(j - flo) <= 5) & (tx >= TMAX_CRIT)).astype(int)
    wrsi = np.where(sum_etc > 0, 100 * sum_eta / np.maximum(sum_etc, 1e-9), 0)
    cause = np.zeros(shp, int)
    if rh is not None:
        cause = np.where(pluie_rec >= rh["mm"], 6, cause)
    cause = np.where(wrsi < WRSI_MIN, 5, cause)
    cause = np.where(chaleur >= 3, 4, cause)
    cause = np.where(flo_ko, 3, cause)
    cause = np.where(lev_ko, 2, cause)
    cause = np.where(germ < 0, 1, cause)
    return {"reussite": cause == 0, "cause": cause, "wrsi": wrsi, "germ": germ}


CAUSES = {1: "pas de germination", 2: "levée", 3: "floraison (sécheresse)",
          4: "floraison (chaleur)", 5: "eau insuffisante sur le cycle", 6: "récolte sous la pluie"}
