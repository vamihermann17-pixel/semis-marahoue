"""
Test de prévision rétrospectif (« hindcast ») du bilan hydrique riz, saisons 2006–2025.

Pour chaque saison Y et chaque date d'émission (1er déc., 1er fév., 1er mars, 1er avr., 1er mai) :
  - état du sol simulé à la date d'émission avec les pluies observées ;
  - scénarios = météo des AUTRES saisons (validation croisée : la saison Y est exclue) ;
  - probabilité prévue de réussite pour chaque date de semis du 1er mars au 31 juillet
    (dates postérieures à l'émission) et chaque maille ;
  - « observé » = réussite simulée avec la vraie météo de la saison Y.

Scores (cycle 110 j) : score de Brier, score de Brier de la climatologie (fréquence de réussite
de la date de semis sur les autres saisons), BSS = 1 − BS/BS_clim, aire sous la courbe ROC,
et fiabilité par classes de probabilité.

Ce test mesure ce que l'état du sol initial apporte par rapport à la climatologie. Il ne valide
pas le modèle de culture lui-même (voir validation_ndvi.py).
"""
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bilan_riz as B
import fenetres_riz as F

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

L = int(os.environ.get("CYCLE_RIZ", 110))
EMISSIONS = [(12, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
SAISONS = list(range(2006, 2026))


def auc(p, o):
    o = o.astype(bool)
    pos, neg = p[o], p[~o]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    # rangs moyens pour les ex aequo
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        ranks[m] = ranks[m].mean()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    dates, pluie, et0, tmax, lat, lon, ru = F.charger()
    cap = B.capacites(ru)
    etats = B.simuler_jachere(pluie, et0, cap)
    idx = {d: i for i, d in enumerate(dates)}
    J = B.J_GERM + L + 2

    # 1. Réussite « observée » (vraie météo) pour chaque saison, date de semis, maille
    obs = {}
    for Y in SAISONS:
        sd = F.dates_semis(Y)
        i0 = np.array([idx[d] for d in sd])
        W0 = np.moveaxis(etats[i0 - 1], 0, 1)
        g = lambda a: a[i0[None, :] + np.arange(J)[:, None]]
        obs[Y] = B.simuler_semis(W0, cap[:, None], g(pluie), g(et0), g(tmax), L)["reussite"]   # (n, lat, lon)
    jours = F.dates_semis(2027)
    res = {}
    for (m, d) in EMISSIONS:
        P, O, C = [], [], []
        for Y in SAISONS:
            em = dt.date(Y - 1 if m >= 8 else Y, m, d)
            i_em = idx[em]
            W_now = etats[i_em - 1]
            sd = [x for x in F.dates_semis(Y) if x >= em]
            ksd = np.array([(x - em).days for x in sd])
            H = ksd[-1] + J + 1
            autres = [s for s in SAISONS + [2005] if s != Y]
            scen = []
            for s in autres:
                e2 = em.replace(year=em.year + (s - Y))
                if e2 in idx and idx[e2] + H <= len(dates):
                    scen.append(idx[e2])
            ii = np.array(scen)[:, None] + np.arange(H)[None, :]
            Pp, Ee, Tt = pluie[ii], et0[ii], tmax[ii]
            W = np.moveaxis(np.broadcast_to(W_now, (len(scen),) + W_now.shape), 1, 0).copy()
            et_f = np.empty((H,) + W.shape, np.float32)
            for t in range(H):
                B.pas_jachere(W, cap[:, None], Pp[:, t], Ee[:, t]); et_f[t] = W
            W0 = np.moveaxis(np.stack([W_now[:, None].repeat(len(scen), 1) if k == 0 else et_f[k - 1] for k in ksd]), 0, 1)
            sel = ksd[:, None] + np.arange(J)[None, :]
            g = lambda a: np.moveaxis(a[:, sel], [0, 1, 2], [2, 1, 0])
            r = B.simuler_semis(W0, cap[:, None, None], g(Pp), g(Ee), g(Tt), L)
            prob = r["reussite"].mean(axis=1)                                    # (n, lat, lon)
            k0 = len(jours) - len(sd)
            o = obs[Y][k0:]
            clim = np.mean([obs[s][k0:] for s in SAISONS if s != Y], axis=0)    # climatologie sans Y
            P.append(prob.ravel()); O.append(o.ravel()); C.append(clim.ravel())
        P, O, C = np.concatenate(P), np.concatenate(O).astype(float), np.concatenate(C)
        bs, bsc = np.mean((P - O) ** 2), np.mean((C - O) ** 2)
        classes = [(0, .5), (.5, .7), (.7, .8), (.8, .9), (.9, 1.01)]
        fiab = [{"classe": f"{a:.0%}–{min(b, 1):.0%}", "n": int(((P >= a) & (P < b)).sum()),
                 "prevu": round(float(P[(P >= a) & (P < b)].mean()), 3) if ((P >= a) & (P < b)).any() else None,
                 "observe": round(float(O[(P >= a) & (P < b)].mean()), 3) if ((P >= a) & (P < b)).any() else None}
                for a, b in classes]
        nom = f"{d:02d}/{m:02d}"
        res[nom] = {"brier": round(float(bs), 4), "brier_climatologie": round(float(bsc), 4),
                    "BSS": round(float(1 - bs / bsc), 3), "ROC_AUC": round(float(auc(P, O)), 3),
                    "ROC_AUC_climatologie": round(float(auc(C, O)), 3), "fiabilite": fiab,
                    "n_cas": int(len(P)), "frequence_reussite": round(float(O.mean()), 3)}
        print(f"Émission {nom} : BSS {res[nom]['BSS']:+.3f}, AUC {res[nom]['ROC_AUC']:.3f} "
              f"(climatologie {res[nom]['ROC_AUC_climatologie']:.3f}), n = {len(P)}")
    json.dump({"cycle": L, "saisons": [SAISONS[0], SAISONS[-1]], "resultats": res},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_retrospective.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
