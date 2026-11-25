"""
Les prévisions de pluie ECMWF améliorent-elles la probabilité de réussite d'un semis ?
Test sur les saisons 2024 et 2025 (seules années archivées par échéance), 36 mailles, cycle 110 j.

Pour chaque date de semis D (1er mars – 31 juillet) et chaque maille :
  - observé : réussite simulée avec la pluie TAMSAT réellement tombée ;
  - A « climatologie » : état du sol réel la veille de D, puis météo des autres saisons
    (= prévision actuelle de la page) ;
  - B « ECMWF » : même chose, mais la pluie des jours D à D+7 est remplacée par la prévision
    ECMWF émise le jour D (déterministe, échéances 0 à 7 jours) ;
  - B' « ECMWF corrigé » : idem, prévision multipliée par le rapport des moyennes TAMSAT / ECMWF
    (calculé sur l'autre saison, pour rester indépendant).
Scores : Brier, gain relatif sur A (BSS), aire ROC ; aussi pour la seule réussite de la levée.
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
NJ = 8          # jours prévus : échéances 0 à 7


def main():
    dates, pluie, et0, tmax, lat, lon, ru = F.charger()
    cap = B.capacites(ru)
    etats = B.simuler_jachere(pluie, et0, cap)
    idx = {d: i for i, d in enumerate(dates)}
    z = np.load(os.path.join(ICI, "validation", "archives_ecmwf_2024_2025.npz"))
    fdates = [dt.date.fromisoformat(d) for d in z["dates"]]
    fidx = {d: i for i, d in enumerate(fdates)}
    fc = z["pluie"].astype(float)                                   # (échéance, jour cible, lat, lon)
    J = B.J_GERM + L + 2
    # biais moyen ECMWF (échéance 1) / TAMSAT par saison, mars–juillet
    ratio = {}
    for y in (2024, 2025):
        k = [i for i, d in enumerate(fdates) if d.year == y and 3 <= d.month <= 7 and d in idx]
        obs = np.nanmean([pluie[idx[fdates[i]]] for i in k]); prev = np.nanmean(fc[1, k])
        ratio[y] = obs / prev
        print(f"{y} : pluie moyenne mars–juillet TAMSAT {obs:.2f} mm/j, ECMWF (J+1) {prev:.2f} mm/j")
    scores = {}
    toutes = {"A": [], "B": [], "Bc": [], "O": [], "Alev": [], "Blev": [], "Olev": []}
    for y in (2024, 2025):
        sd = F.dates_semis(y)
        i0 = np.array([idx[d] for d in sd])
        W0 = np.moveaxis(etats[i0 - 1], 0, 1)                          # (3, n, lat, lon)
        g = lambda a: a[i0[None, :] + np.arange(J)[:, None]]
        r_obs = B.simuler_semis(W0, cap[:, None], g(pluie), g(et0), g(tmax), L)
        obs = r_obs["reussite"]
        lev_obs = ~np.isin(r_obs["cause"], [1, 2])
        # prévision déterministe émise le jour D pour D..D+7
        P_fc = np.full((NJ, len(sd), 6, 6), np.nan)
        for k, d in enumerate(sd):
            for e in range(NJ):
                t = d + dt.timedelta(days=e)
                if t in fidx:
                    P_fc[e, k] = fc[e, fidx[t]]
        ok = np.isfinite(P_fc).all(axis=(0, 2, 3))
        autres = [s for s in range(2005, 2026) if s != y]
        res = {"A": [], "B": [], "Bc": [], "Alev": [], "Blev": []}
        for s in autres:
            j0 = np.array([idx[d.replace(year=s)] for d in sd])
            gs = lambda a: a[j0[None, :] + np.arange(J)[:, None]]
            Ps, Es, Ts = gs(pluie), gs(et0), gs(tmax)
            rA = B.simuler_semis(W0, cap[:, None], Ps, Es, Ts, L)
            PB = Ps.copy(); PB[:NJ] = np.where(np.isfinite(P_fc), P_fc, Ps[:NJ])
            rB = B.simuler_semis(W0, cap[:, None], PB, Es, Ts, L)
            PBc = Ps.copy(); PBc[:NJ] = np.where(np.isfinite(P_fc), P_fc * ratio[4049 - y], Ps[:NJ])
            rBc = B.simuler_semis(W0, cap[:, None], PBc, Es, Ts, L)
            res["A"].append(rA["reussite"]); res["B"].append(rB["reussite"]); res["Bc"].append(rBc["reussite"])
            res["Alev"].append(~np.isin(rA["cause"], [1, 2])); res["Blev"].append(~np.isin(rB["cause"], [1, 2]))
        for k in res:
            res[k] = np.mean(res[k], axis=0)[ok]
        for k in ("A", "B", "Bc", "Alev", "Blev"):
            toutes[k].append(res[k].ravel())
        toutes["O"].append(obs[ok].ravel().astype(float)); toutes["Olev"].append(lev_obs[ok].ravel().astype(float))
    for k in toutes:
        toutes[k] = np.concatenate(toutes[k])
    O, Ol = toutes["O"], toutes["Olev"]
    bs = lambda p, o: float(np.mean((p - o) ** 2))
    for nom, p, o, ref in (("réussite – climatologie (A)", toutes["A"], O, None),
                           ("réussite – ECMWF brut (B)", toutes["B"], O, toutes["A"]),
                           ("réussite – ECMWF corrigé (B')", toutes["Bc"], O, toutes["A"]),
                           ("levée – climatologie (A)", toutes["Alev"], Ol, None),
                           ("levée – ECMWF brut (B)", toutes["Blev"], Ol, toutes["Alev"])):
        s = {"brier": round(bs(p, o), 4), "ROC_AUC": round(float(auc(p, o)), 3), "n": int(len(o)),
             "frequence": round(float(o.mean()), 3)}
        if ref is not None:
            s["BSS_vs_climatologie"] = round(1 - bs(p, o) / bs(ref, o), 3)
        scores[nom] = s
        print(f"{nom:32s} Brier {s['brier']:.4f}  AUC {s['ROC_AUC']:.3f}" +
              (f"  gain {s['BSS_vs_climatologie']:+.1%}" if ref is not None else "") + f"  (fréquence {s['frequence']:.0%}, n = {s['n']})")
    json.dump({"saisons": [2024, 2025], "cycle": L, "jours_prevus": NJ, "rapport_pluie_TAMSAT_sur_ECMWF":
               {str(k): round(v, 2) for k, v in ratio.items()}, "scores": scores},
              open(os.path.join(ICI, "validation", "validation_meteo10j.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
