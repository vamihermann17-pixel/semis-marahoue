"""
Analyse des observations de terrain (outil « Suivi des parcelles Marahoué ») contre le modèle.

Entrée : un dossier contenant les exports CSV de l'outil (suivi_parcelles_*.csv,
suivi_observations_*.csv, éventuellement suivi_pluie_*.csv), ou les documents JSON lus dans la base
(sous-dossiers parcelles/, observations/, pluie/).

Pour chaque parcelle récoltée :
  - réussite observée : levée ≥ 70 % des plants attendus et rendement ≥ 50 % de la référence de la
    culture (arachide : gousses non pourries) — mêmes règles que l'outil ;
  - probabilité climatologique du modèle à SA maille, pour SA date de semis et SON cycle (2005–2024) ;
  - issue simulée par le modèle avec la pluie réellement tombée (TAMSAT, ou pluviomètre du village
    s'il est relevé sur toute la saison) et WRSI correspondant.
Scores : score de Brier de la probabilité, tableau de contingence de l'issue simulée, corrélation de
rang rendement / WRSI. Sortie : analyse_terrain.csv et résumé à l'écran.

Usage : python analyse_terrain.py <dossier_des_exports>
"""
import csv
import datetime as dt
import glob
import json
import os
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import bilan_riz as B

REF = {"riz": 2.5, "mais": 3.0, "arachide": 1.2, "igname": 12.0}     # t/ha, à valider avec le CNRA
VILLAGES = {"Bouaflé": (6.99, -5.75), "Sinfra": (6.62, -5.91), "Zuénoula": (7.43, -6.05), "Bonon": (6.92, -6.04),
            "Gohitafla": (7.49, -5.88), "Kononfla": (7.17, -5.94), "Bazré": (6.86, -6.27), "Kouétinfla": (7.06, -6.24)}


def nombre(v):
    try:
        return float(str(v).replace(",", "."))
    except (TypeError, ValueError):
        return None


def lire(dossier):
    P, O, R = [], [], []
    for f in glob.glob(os.path.join(dossier, "suivi_parcelles_*.csv")):
        P += list(csv.DictReader(open(f, encoding="utf-8-sig"), delimiter=";"))
    for f in glob.glob(os.path.join(dossier, "suivi_observations_*.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8-sig"), delimiter=";"):
            r["mesures"] = {k[2:]: v for k, v in r.items() if k.startswith("m_") and v != ""}
            O.append(r)
    for f in glob.glob(os.path.join(dossier, "suivi_pluie_*.csv")):
        R += list(csv.DictReader(open(f, encoding="utf-8-sig"), delimiter=";"))
    for nom, L in (("parcelles", P), ("observations", O), ("pluie", R)):
        for f in glob.glob(os.path.join(dossier, nom, "*.json")):
            d = json.load(open(f, encoding="utf-8"))
            L.append({"id": d.get("id", os.path.basename(f)[:-5]), **d.get("data", d)})
    return P, O, R


def issue_observee(p, obs):
    o_lev = sorted([o for o in obs if o["stade"] == "levee"], key=lambda o: o["date"])
    o_rec = sorted([o for o in obs if o["stade"] == "recolte"], key=lambda o: o["date"])
    if not o_rec:
        return None, None, None
    m = o_rec[-1]["mesures"]
    r = [nombre(m.get(k)) for k in ("r1", "r2", "r3")]
    r = [x for x in r if x is not None]
    if not r:
        return None, None, None
    rdt = np.mean(r) * 2.0                       # kg sur 5 m² → t/ha
    lev = None
    if o_lev:
        ml = o_lev[-1]["mesures"]
        if p["culture"] == "igname":
            b = nombre(ml.get("buttes_levees")); lev = b / 20 if b is not None else None
        else:
            v = [nombre(ml.get(k)) for k in ("p1", "p2", "p3")]
            v = [x for x in v if x is not None]
            d = nombre(p.get("densite_m2"))
            lev = np.mean(v) / d if v and d else None
    ok = (lev is None or lev >= 0.7) and rdt >= 0.5 * REF[p["culture"]]
    if p["culture"] == "arachide" and m.get("gousses") == "pourries ou moisies":
        ok = False
    return ok, rdt, lev


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    P, O, R = lire(sys.argv[1])
    print(f"{len(P)} parcelles, {len(O)} observations, {len(R)} relevés de pluie lus")
    lignes = []
    for culture in ("riz", "mais", "arachide", "igname"):
        Pc = [p for p in P if p.get("culture") == culture]
        if not Pc:
            continue
        os.environ["CULTURE"] = culture
        import importlib
        import fenetres_riz as F
        importlib.reload(F)                       # recharge les paramètres de la culture
        dates, pluie, et0, tmax, lat, lon, ru = F.charger()
        cap = B.capacites(ru)
        etats = B.simuler_jachere(pluie, et0, cap)
        idx = {d: i for i, d in enumerate(dates)}
        for p in Pc:
            obs = [o for o in O if o.get("parcelle") == p["id"]]
            ok, rdt, lev = issue_observee(p, obs)
            la, lo = nombre(p.get("lat")), nombre(p.get("lon"))
            if la is None or lo is None:
                la, lo = VILLAGES.get(p.get("village"), (None, None))
            if la is None or not p.get("date_semis"):
                continue
            i, j = int((la - 6.25) // 0.25), int((lo + 6.75) // 0.25)
            if not (0 <= i < 6 and 0 <= j < 6):
                continue
            L = int(nombre(p.get("cycle_j")) or F.CYCLES[1])
            Lm = min(F.CYCLES, key=lambda c: abs(c - L))
            ds = dt.date.fromisoformat(p["date_semis"])
            J = B.duree_forcage(Lm)
            # probabilité climatologique à la maille (saisons 2005–2024, même jour de l'année)
            succ = []
            saison = ds.year if (ds.month, ds.day) <= F.FIN else ds.year + 1     # igname : décembre → saison suivante
            for y in range(2006, 2025):
                try:
                    d0 = ds.replace(year=ds.year + (y - saison))
                except ValueError:                                              # 29 février
                    continue
                if d0 in idx and idx[d0] + J <= len(dates):
                    k = idx[d0]
                    r = B.simuler_semis(etats[k - 1][:, i:i + 1, j:j + 1], cap[:, i:i + 1, j:j + 1],
                                        pluie[k:k + J, i:i + 1, j:j + 1], et0[k:k + J, i:i + 1, j:j + 1],
                                        tmax[k:k + J, i:i + 1, j:j + 1], Lm)
                    succ.append(bool(r["reussite"].item()))
            proba = float(np.mean(succ)) if succ else None
            # issue simulée avec la pluie réelle de la saison (si les données couvrent tout le cycle)
            sim, wrsi = None, None
            if ds in idx and idx[ds] + J <= len(dates):
                k = idx[ds]
                r = B.simuler_semis(etats[k - 1][:, i:i + 1, j:j + 1], cap[:, i:i + 1, j:j + 1],
                                    pluie[k:k + J, i:i + 1, j:j + 1], et0[k:k + J, i:i + 1, j:j + 1],
                                    tmax[k:k + J, i:i + 1, j:j + 1], Lm)
                sim, wrsi = bool(r["reussite"].item()), float(r["wrsi"].item())
            lignes.append({"parcelle": p.get("code", p["id"]), "culture": culture, "village": p.get("village"),
                           "maille": f"{i}{j}", "date_semis": p["date_semis"], "cycle_j": L,
                           "levee": None if lev is None else round(lev, 2), "rendement_t_ha": None if rdt is None else round(rdt, 2),
                           "reussite_observee": ok, "proba_climatologique": None if proba is None else round(proba, 3),
                           "reussite_simulee_saison": sim, "wrsi_saison": None if wrsi is None else round(wrsi, 1)})
    if not lignes:
        print("Aucune parcelle exploitable (coordonnées ou village, date de semis).")
        return
    out = os.path.join(sys.argv[1], "analyse_terrain.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0]), delimiter=";")
        w.writeheader(); w.writerows(lignes)
    rec = [l for l in lignes if l["reussite_observee"] is not None]
    print(f"{len(lignes)} parcelles analysées, dont {len(rec)} récoltées → {out}")
    if rec:
        o = np.array([l["reussite_observee"] for l in rec], float)
        pr = np.array([l["proba_climatologique"] if l["proba_climatologique"] is not None else np.nan for l in rec])
        m = np.isfinite(pr)
        if m.any():
            bs = np.mean((pr[m] - o[m]) ** 2); bs0 = np.mean((o[m].mean() - o[m]) ** 2)
            print(f"Score de Brier de la probabilité climatologique : {bs:.3f} (référence « fréquence observée » {bs0:.3f})")
        s = [(l["reussite_simulee_saison"], l["reussite_observee"]) for l in rec if l["reussite_simulee_saison"] is not None]
        if s:
            tp = sum(a and b for a, b in s); tn = sum((not a) and (not b) for a, b in s)
            print(f"Issue simulée avec la pluie réelle : {tp + tn}/{len(s)} parcelles bien classées "
                  f"({tp} réussites et {tn} échecs correctement prévus)")
        x = [(l["wrsi_saison"], l["rendement_t_ha"] / REF[l["culture"]]) for l in rec
             if l["wrsi_saison"] is not None and l["rendement_t_ha"] is not None]
        if len(x) >= 5:
            a, b = np.array(x).T
            ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
            print(f"Corrélation de rang WRSI / rendement relatif : {np.corrcoef(ra, rb)[0, 1]:+.2f} (n = {len(x)})")


if __name__ == "__main__":
    main()
