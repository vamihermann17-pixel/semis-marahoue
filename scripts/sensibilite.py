"""
Analyses de sensibilité des calendriers (cycle médian de chaque culture, 2005–2024, moyenne des 36 mailles).
Une hypothèse change à la fois ; sortie : réussite moyenne par mois de semis, et rang du meilleur mois.
Résultat : validation/sensibilite.json
"""
import copy
import json
import os
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
sys.stdout.reconfigure(encoding="utf-8")

VARIANTES = {
    "riz": [("référence", {}), ("stress Ks<0,5 pendant 3 j", {"ks_fort": 0.5, "jours_stress": 3}),
            ("stress dès Ks<0,7", {"ks_fort": 0.7}), ("WRSI ≥ 85", {"_wrsi": 85})],
    "mais": [("référence", {}), ("stress fort (Ks<0,5, 5 j)", {"ks_fort": 0.5, "jours_stress": 5}),
             ("racines limitées à 0,6 m", {"ep": (100.0, 200.0, 300.0), "racines": (100.0, 600.0, 0.5)}),
             ("WRSI ≥ 85", {"_wrsi": 85})],
    "arachide": [("référence (80 mm)", {}), ("récolte humide 40 mm", {"_rh": 40.0}), ("récolte humide 60 mm", {"_rh": 60.0}),
                 ("récolte humide 100 mm", {"_rh": 100.0}), ("sans critère de récolte", {"_rh": None})],
    "igname": [("référence (10 j)", {}), ("stress 5 j", {"jours_stress": 5}), ("stress 15 j", {"jours_stress": 15}),
               ("WRSI ≥ 85", {"_wrsi": 85})],
}


def main():
    res = {}
    for culture, variantes in VARIANTES.items():
        os.environ["CULTURE"] = culture
        import importlib
        import bilan_riz as B
        import fenetres_riz as F
        importlib.reload(B); importlib.reload(F)
        dates, pluie, et0, tmax, lat, lon, ru = F.charger()
        idx = {d: i for i, d in enumerate(dates)}
        L = F.CYCLES[1]
        base = copy.deepcopy(B.CULTURES[culture])
        jours = F.dates_semis(2027)
        mois = [d.month for d in jours]
        ordre = list(dict.fromkeys(mois))
        annees = list(range(2005 + (1 if F.DEB > F.FIN else 0), 2025))
        res[culture] = {"cycle": L, "mois": ordre, "variantes": {}}
        print(f"## {culture} {L} j — mois {ordre}")
        for nom, chg in variantes:
            B.CULTURES[culture] = copy.deepcopy(base)
            B.WRSI_MIN = 75.0
            for k, v in chg.items():
                if k == "_wrsi":
                    B.WRSI_MIN = float(v)
                elif k == "_rh":
                    if v is None:
                        B.CULTURES[culture].pop("recolte_humide", None)
                    else:
                        B.CULTURES[culture]["recolte_humide"]["mm"] = v
                else:
                    B.CULTURES[culture][k] = v
            B.choisir_culture(culture)
            cap = B.capacites(ru)
            etats = B.simuler_jachere(pluie, et0, cap)
            J = B.duree_forcage(L)
            S = []
            for y in annees:
                sd = F.dates_semis(y)
                i0 = np.array([idx[d] for d in sd])
                W0 = np.moveaxis(etats[i0 - 1], 0, 1)
                g = lambda a: a[i0[None, :] + np.arange(J)[:, None]]
                S.append(B.simuler_semis(W0, cap[:, None], g(pluie), g(et0), g(tmax), L)["reussite"])
            S = np.array(S).mean(axis=(0, 2, 3))
            par_mois = [float(S[[k for k, m in enumerate(mois) if m == mm]].mean()) for mm in ordre]
            res[culture]["variantes"][nom] = [round(v, 3) for v in par_mois]
            print(f"  {nom:32s} " + "  ".join(f"{v:4.0%}" for v in par_mois))
        B.CULTURES[culture] = base
        B.WRSI_MIN = 75.0
    json.dump(res, open(os.path.join(ICI, "validation", "sensibilite.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
