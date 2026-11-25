"""
Pondération des scénarios par des prévisions saisonnières en terciles (méthode TAMSAT-ALERT).

Pour une date de semis du mois m, la fenêtre critique est celle des trois mois suivants
(m+1, m+2, m+3 : levée, montaison, floraison). Chaque saison passée est classée sèche, normale ou
humide selon sa pluie TAMSAT (moyenne de zone) dans cette fenêtre, terciles calculés sans la saison
prévue. Une prévision (p_sec, p_normal, p_humide) donne à chaque saison du tercile k le poids
p_k / n_k ; avec (1/3, 1/3, 1/3), on retrouve la climatologie.

Test de « gain maximal », saisons 2006–2025, cycle 110 j, état du sol au 1er février :
  - climatologie : poids égaux ;
  - prévision parfaite : 100 % sur le tercile observé ;
  - prévision réaliste : 50 % sur le tercile observé, 25 % sur chacun des deux autres
    (proche de ce que valent les meilleures prévisions saisonnières en Afrique de l'Ouest) ;
  - prévision « bruitée » : 50 % sur un tercile tiré au hasard (prévision sans qualité).
"""
import datetime as dt
import json
import os
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import bilan_riz as B
import fenetres_riz as F
from validation_retrospective import auc

sys.stdout.reconfigure(encoding="utf-8")
L = 110
SAISONS = list(range(2006, 2026))
rng = np.random.default_rng(1)


def poids_terciles(pluie_fenetre, p):
    """pluie_fenetre : (S,) pluie de chaque saison-scénario ; p : (3,) probabilités sec/normal/humide."""
    t1, t2 = np.percentile(pluie_fenetre, [100 / 3, 200 / 3])
    k = np.where(pluie_fenetre <= t1, 0, np.where(pluie_fenetre <= t2, 1, 2))
    w = np.array([p[c] / max((k == c).sum(), 1) for c in k])
    return w / w.sum(), k


def main():
    dates, pluie, et0, tmax, lat, lon, ru = F.charger()
    cap = B.capacites(ru)
    etats = B.simuler_jachere(pluie, et0, cap)
    idx = {d: i for i, d in enumerate(dates)}
    J = B.J_GERM + L + 2
    zone = pluie.mean(axis=(1, 2))

    def pluie_fen(y, m):
        ms = [(m + q - 1) % 12 + 1 for q in (1, 2, 3)]
        return sum(zone[i] for i, d in enumerate(dates) if d.year == y and d.month in ms)

    # issue réelle de chaque saison (vraie météo)
    obs = {}
    for Y in SAISONS:
        sd = F.dates_semis(Y); i0 = np.array([idx[d] for d in sd]); W0 = np.moveaxis(etats[i0 - 1], 0, 1)
        g = lambda a: a[i0[None, :] + np.arange(J)[:, None]]
        obs[Y] = B.simuler_semis(W0, cap[:, None], g(pluie), g(et0), g(tmax), L)["reussite"]
    mois = np.array([d.month for d in F.dates_semis(2027)])
    P = {"climatologie": [], "parfaite": [], "très bonne (70 %)": [], "réaliste (50 %)": [], "sans qualité": []}
    O = []
    for Y in SAISONS:
        em = dt.date(Y, 2, 1)
        W_now = etats[idx[em] - 1]
        sd = F.dates_semis(Y); ksd = np.array([(d - em).days for d in sd]); H = ksd[-1] + J + 1
        autres = [s for s in range(2005, 2026) if s != Y]
        ii = np.array([idx[em.replace(year=s)] for s in autres])[:, None] + np.arange(H)[None, :]
        Pp, Ee, Tt = pluie[ii], et0[ii], tmax[ii]
        W = np.moveaxis(np.broadcast_to(W_now, (len(autres),) + W_now.shape), 1, 0).copy()
        et_f = np.empty((H,) + W.shape, np.float32)
        for t in range(H):
            B.pas_jachere(W, cap[:, None], Pp[:, t], Ee[:, t]); et_f[t] = W
        W0 = np.moveaxis(np.stack([et_f[k - 1] for k in ksd]), 0, 1)
        sel = ksd[:, None] + np.arange(J)[None, :]
        g = lambda a: np.moveaxis(a[:, sel], [0, 1, 2], [2, 1, 0])
        succ = B.simuler_semis(W0, cap[:, None, None], g(Pp), g(Ee), g(Tt), L)["reussite"]   # (n, S, lat, lon)
        res = {k: np.empty((len(sd), 6, 6)) for k in P}
        for m in (3, 4, 5, 6, 7):
            pf = np.array([pluie_fen(s, m) for s in autres]); vrai = pluie_fen(Y, m)
            t1, t2 = np.percentile(pf, [100 / 3, 200 / 3])
            ko = 0 if vrai <= t1 else (1 if vrai <= t2 else 2)
            hasard = rng.integers(3)
            prevs = {"climatologie": np.ones(3) / 3,
                     "parfaite": np.eye(3)[ko],
                     "réaliste (50 %)": np.where(np.arange(3) == ko, .5, .25),
                     "très bonne (70 %)": np.where(np.arange(3) == ko, .7, .15),
                     "sans qualité": np.where(np.arange(3) == hasard, .5, .25)}
            sel_m = mois == m
            for nom, p in prevs.items():
                w, _ = poids_terciles(pf, p)
                res[nom][sel_m] = np.tensordot(w, succ[sel_m].transpose(1, 0, 2, 3), axes=1)
        for k in P:
            P[k].append(res[k].ravel())
        O.append(obs[Y].ravel().astype(float))
    O = np.concatenate(O)
    bs = lambda p: float(np.mean((p - O) ** 2))
    ref = bs(np.concatenate(P["climatologie"]))
    out = {}
    for k in P:
        p = np.concatenate(P[k])
        out[k] = {"brier": round(bs(p), 4), "BSS": round(1 - bs(p) / ref, 3), "ROC_AUC": round(float(auc(p, O)), 3)}
        print(f"{k:18s} Brier {out[k]['brier']:.4f}  gain {out[k]['BSS']:+.1%}  AUC {out[k]['ROC_AUC']:.3f}")
    json.dump({"emission": "1er février", "saisons": [SAISONS[0], SAISONS[-1]], "cycle": L, "scores": out},
              open(os.path.join(ICI, "validation", "gain_maximal_terciles.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
