"""
Fenêtres de semis du riz pluvial en Marahoué avec un bilan hydrique propre au riz.

  1. Calendrier climatologique : pour chaque année 2005–2024 et chaque date de semis
     du 1er mars au 31 juillet, le semis réussit-il (bilan_riz.py) ? → probabilité.
  2. Prévision : état du sol d'aujourd'hui (bilan continu sur les pluies TAMSAT
     observées), puis météo de chacune des années passées (méthode TAMSAT-ALERT,
     un scénario par année) → probabilité de réussite pour chaque date de semis.

Entrée : donnees_bilan_marahoue.npz (donnees_bilan.py).
Sorties (OUTDIR, par défaut web/) : calendriers_riz.json, prevision_riz_derniere.json.
"""
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bilan_riz as B

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ICI = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR", os.path.join(ICI, "web"))
CULTURE = os.environ.get("CULTURE", "riz")
PARAM = B.choisir_culture(CULTURE)
CYCLES = PARAM["cycles"]
SUFFIXE = "riz" if CULTURE == "riz" else CULTURE
DEB, FIN = PARAM.get("semis", ((3, 1), (7, 31)))         # période de semis (ou de plantation) évaluée
REGLE_3J, REGLE_10J = 20.0, 40.0     # règle de semis : ≥ 20 mm en 3 jours ou ≥ 40 mm en 10 jours (pluie observée)
AN_MIN, AN_MAX = 2005, 2024          # années complètes servant de climatologie et de scénarios
r3 = lambda a: [None if not np.isfinite(x) else round(float(x), 3) for x in np.ravel(a)]


def charger():
    z = np.load(os.path.join(ICI, "donnees_bilan_marahoue.npz"))
    dates = [dt.date.fromisoformat(d) for d in z["dates"]]
    pluie = np.nan_to_num(z["pluie"].astype(float), nan=0.0)
    doy = np.array([min(d.timetuple().tm_yday, 365) for d in dates])

    def combler(a):
        a = a.astype(float).copy()
        clim = np.array([np.nanmean(a[doy == k], axis=0) for k in range(1, 366)])
        m = np.isnan(a)
        a[m] = clim[doy - 1][m]
        return a
    return dates, pluie, combler(z["et0"]), combler(z["tmax"]), z["lat"], z["lon"], z["ru_mm_m"]


def dates_semis(annee):
    d0, d1 = dt.date(annee, *DEB), dt.date(annee, *FIN)
    jours = [d0 + dt.timedelta(days=k) for k in range((d1 - d0).days + 1)]
    return [d for d in jours if (d.month, d.day) != (2, 29)]      # même nombre de dates chaque année


def faux_departs(pluie, dates, annees):
    """Part des pluies de démarrage (≥ 20 mm en 3 j) suivies de plus de 7 j secs dans le mois."""
    idx = {d: i for i, d in enumerate(dates)}
    jours = dates_semis(2027)
    fs = np.full((len(annees), len(jours)) + pluie.shape[1:], np.nan)
    for a, y in enumerate(annees):
        for k, d in enumerate(dates_semis(y)):
            t = idx[d]
            ev = (pluie[t:t + 3].sum(0) >= 20) & (pluie[t] >= 1)
            sec = pluie[t + 1:t + 31] < 1
            run = np.zeros(pluie.shape[1:], int); mx = np.zeros_like(run)
            for s in sec:
                run = np.where(s, run + 1, 0); mx = np.maximum(mx, run)
            fs[a, k] = np.where(ev, (mx > 7).astype(float), np.nan)
    return np.nanmean(fs, axis=(0, 2, 3))


def calendrier(dates, pluie, et0, tmax, cap, etats):
    idx = {d: i for i, d in enumerate(dates)}
    annees = list(range(AN_MIN, AN_MAX + 1))
    res = {}
    for L in CYCLES:
        J = B.duree_forcage(L)
        succ, causes, wrsi, r3, r10 = [], [], [], [], []
        for y in annees:
            sd = dates_semis(y)
            i0 = np.array([idx[d] for d in sd])
            W0 = np.moveaxis(etats[i0 - 1], 0, 1)                     # (3, n, lat, lon)
            g = lambda a: a[i0[None, :] + np.arange(J)[:, None]]       # (J, n, lat, lon)
            r = B.simuler_semis(W0, cap[:, None], g(pluie), g(et0), g(tmax), L)
            succ.append(r["reussite"]); causes.append(r["cause"]); wrsi.append(r["wrsi"])
            r3.append(np.stack([pluie[i - 3:i].sum(0) for i in i0])); r10.append(np.stack([pluie[i - 10:i].sum(0) for i in i0]))
        succ, causes, wrsi = np.array(succ), np.array(causes), np.array(wrsi)   # (an, n, lat, lon)
        regle = (np.array(r3) >= REGLE_3J) | (np.array(r10) >= REGLE_10J)
        p = succ.mean(axis=(0, 2, 3))
        n = causes.size
        res[L] = {"reussite": p, "wrsi_med": np.median(wrsi, axis=(0, 2, 3)),
                  "causes": {PARAM.get("noms_causes", {}).get(c, B.CAUSES[c]): f"{(causes == c).sum() / n:.0%}" for c in B.CAUSES
                             if (c != 6 or "recolte_humide" in PARAM) and (c != 4 or PARAM.get("chaleur", True))
                             and (c != 1 or PARAM.get("germination", True))},
                  "succ_annee": succ.mean(axis=(2, 3)),
                  "reussite_regle": np.array([succ[:, k][regle[:, k]].mean() if regle[:, k].sum() >= 20 else np.nan
                                              for k in range(succ.shape[1])]),
                  "part_regle": regle.mean(axis=(0, 2, 3))}
        mx = p.max(); im = int(p.argmax()); win = np.where(p >= 0.8 * mx)[0]
        jours = dates_semis(2027)
        print(f"Cycle {L} j : maximum {mx:.0%} le {jours[im]:%d/%m}, fenêtre {jours[win[0]]:%d/%m} → "
              f"{jours[win[-1]]:%d/%m} ; causes {res[L]['causes']}")
        for m in sorted({d.month for d in jours}):
            ks = [k for k, d in enumerate(jours) if d.month == m]
            print(f"   {m:02d} : réussite {p[ks].mean():.0%}, WRSI médian {np.median(res[L]['wrsi_med'][ks]):.0f}")
    return res, annees


def prevision(dates, pluie, et0, tmax, cap, etats):
    idx = {d: i for i, d in enumerate(dates)}
    emis = dates[-1] + dt.timedelta(days=1)                       # 1er jour sans pluie observée
    W_now = etats[-1]
    saison = emis.year if (emis.month, emis.day) <= FIN else emis.year + 1
    sd = [d for d in dates_semis(saison) if d >= emis]
    fin = sd[-1] + dt.timedelta(days=B.duree_forcage(max(CYCLES)) + 1)
    H = (fin - emis).days
    # scénario « année de saison Y » : la météo reprend au même jour calendaire, Y − (saison − émission) ans plus tôt
    decal = saison - emis.year
    scen = []
    for ys_ in range(AN_MIN, dates[-1].year + 1):     # toutes les saisons complètes disponibles
        y = ys_ - decal
        d_eq = emis.replace(year=y, day=28) if (emis.month, emis.day) == (2, 29) else emis.replace(year=y)
        i_eq = idx.get(d_eq)
        if i_eq is not None and i_eq + H <= len(dates):
            scen.append((ys_, i_eq))
    ys = [y for y, _ in scen]
    ii = np.array([i for _, i in scen])[:, None] + np.arange(H)[None, :]   # (S, H)
    P, E, T = pluie[ii], et0[ii], tmax[ii]                                    # (S, H, lat, lon)
    # jachère depuis aujourd'hui dans chaque scénario
    W = np.broadcast_to(W_now, (len(ys),) + W_now.shape).copy()               # (S, 3, lat, lon)
    W = np.moveaxis(W, 1, 0).copy()                                           # (3, S, lat, lon)
    etats_f = np.empty((H,) + W.shape, np.float32)
    capS = cap[:, None]
    for t in range(H):
        B.pas_jachere(W, capS, P[:, t], E[:, t])
        etats_f[t] = W
    jours = [emis + dt.timedelta(days=k) for k in range(H)]
    rel30 = (etats_f[:, 0] + etats_f[:, 1]) / (cap[0] + cap[1])               # (H, S, lat, lon)
    k_sd = np.array([(d - emis).days for d in sd])
    W0 = np.moveaxis(np.concatenate([W_now[None, :, None].repeat(len(ys), 2) if k == 0 else
                                     etats_f[k - 1][None] for k in k_sd]), 0, 1)        # (3, n, S, lat, lon)
    out = {"date_prevision": str(emis), "scenarios": ys, "jours": [str(d) for d in jours],
           "humidite_membres": [r3(rel30[:, s].mean(axis=(1, 2))) for s in range(len(ys))],
           "pluie_obs_30j": round(float(pluie[-30:].sum(0).mean()), 1),
           "pluie_recente": {"jusqu_au": str(dates[-1]), "mm_3j": [r3(x) for x in pluie[-3:].sum(0)],
                             "mm_10j": [r3(x) for x in pluie[-10:].sum(0)],
                             "jours": [str(d) for d in dates[-10:]], "zone_mm_jour": r3(pluie[-10:].mean(axis=(1, 2)))},
           "jours_eval": [str(d) for d in sd], "lat": r3(LAT), "lon": r3(LON), "cycles": {}}
    for L in CYCLES:
        J = B.duree_forcage(L)
        sel = k_sd[:, None] + np.arange(J)[None, :]                           # (n, J)
        g = lambda a: np.moveaxis(a[:, sel], [0, 1, 2], [2, 1, 0])            # (J, n, S, lat, lon)
        r = B.simuler_semis(W0, cap[:, None, None], g(P), g(E), g(T), L)
        p_pix = r["reussite"].mean(axis=1)                                     # (n, lat, lon)
        out["cycles"][str(L)] = {"proba_zone": r3(p_pix.mean(axis=(1, 2))),
                                 "wrsi_med": r3(np.median(r["wrsi"], axis=(1, 2, 3))),
                                 "proba_pixels": [[r3(p_pix[k, i]) for i in range(p_pix.shape[1])]
                                                  for k in range(len(sd))]}
        pz = p_pix.mean(axis=(1, 2)); b = int(pz.argmax())
        print(f"Prévision cycle {L} j : maximum {pz[b]:.0%} le {sd[b]:%d/%m/%Y}")
    return out


def main():
    global LAT, LON
    dates, pluie, et0, tmax, LAT, LON, ru = charger()
    cap = B.capacites(ru)
    print(f"Données : {dates[0]} → {dates[-1]} ; réserve utile 0–60 cm : "
          f"{np.round(ru.mean() * 0.6):.0f} mm en moyenne")
    etats = B.simuler_jachere(pluie, et0, cap)
    cal, annees = calendrier(dates, pluie, et0, tmax, cap, etats)
    fs = faux_departs(pluie, dates, annees)
    os.makedirs(OUT, exist_ok=True)
    jours = dates_semis(2027)
    cj = {"zone": "Marahoué (6,25–7,75° N ; 6,75–5,25° O)", "annees": f"{annees[0]}–{annees[-1]}",
          "culture": CULTURE, "culture_nom": PARAM["nom"], "cycles_disponibles": list(CYCLES),
          "methode": f"bilan hydrique {PARAM['nom']} (FAO-56, WRSI)",
          "regle_semis": {"mm_3j": REGLE_3J, "mm_10j": REGLE_10J},
          "parametres": {"p": B.P_RIZ, "profondeur_racines_cm": PARAM["racines"][1] / 10, "WRSI_MIN": B.WRSI_MIN, "ks_fort": PARAM.get("ks_fort", B.KS_FORT), "jours_stress": PARAM.get("jours_stress", B.RUN_STRESS),
                         "tmax_crit": B.TMAX_CRIT, "ru_mm_m_moy": round(float(ru.mean()), 0)},
          "cycles": {}}
    for L in CYCLES:
        c = cal[L]
        cj["cycles"][str(L)] = {"dates": [f"{d:%d/%m}" for d in jours], "reussite": r3(c["reussite"]),
                                "faux_depart": r3(fs), "wrsi_med": r3(c["wrsi_med"]), "causes": c["causes"],
                                "par_annee": {str(y): round(float(v.mean()), 3) for y, v in zip(annees, c["succ_annee"])},
                                "reussite_regle": r3(c["reussite_regle"]), "part_regle": r3(c["part_regle"])}
    json.dump(cj, open(os.path.join(OUT, f"calendriers_{SUFFIXE}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    fc = prevision(dates, pluie, et0, tmax, cap, etats)
    json.dump(fc, open(os.path.join(OUT, f"prevision_{SUFFIXE}_derniere.json"), "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    print("Écrit :", OUT)


if __name__ == "__main__":
    main()
