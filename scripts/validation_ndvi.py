"""
Validation indépendante par satellite : le bilan hydrique riz explique-t-il les variations d'une
année à l'autre de la végétation non arborée (champs, jachères, savanes) mesurée par MODIS ?

Réponse observée (indépendante des modèles) : anomalie d'EVI août–octobre (et septembre–novembre),
tendance linéaire retirée maille par maille (dérive du capteur, changements d'occupation du sol).

Indicateurs comparés, par maille de 0,25° et par année 2005–2024 :
  - modèle riz : part des semis réussis et WRSI médian, semis du 15 mai au 30 juin (pratique
    paysanne du Centre-Ouest, revue de littérature) et semis du 1er mars au 31 juillet ;
  - modèle riz : eau utile 0–30 cm d'un sol non cultivé, août–octobre ;
  - beta TAMSAT-ALERT (graminée C4, 1 m), moyenne août–octobre ;
  - pluie TAMSAT juin–septembre et pluie de juillet (petite saison sèche).

Scores : corrélation de rang de Spearman sur les anomalies (720 maille-années) et sur la moyenne
de zone (20 années, test par permutation).
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

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ANS = np.arange(2005, 2025)
rng = np.random.default_rng(0)


def lire_evi():
    out = np.full((4, len(ANS), 6, 6), np.nan)           # evi_jj, evi_aso, evi_son, ndvi_aso
    for ligne in open(os.path.join(ICI, "validation", "ndvi_saison_compact.txt"), encoding="utf-8"):
        if ligne.startswith("#") or ":" not in ligne:
            continue
        cle, val = ligne.strip().split(":", 1)
        i, j = int(cle[0]), int(cle[1])
        for a, bloc in enumerate(val.split(";")):
            for k, v in enumerate(bloc.strip().split(" ") if bloc.strip() else []):
                pass
            parts = bloc.split(" ")
            parts = (["" for _ in range(4 - len(parts))] + parts) if len(parts) < 4 else parts
            for k, v in enumerate(parts[-4:]):
                if v.strip():
                    out[k, a, i, j] = int(v) / 1000
    return out


def anomalies(x):
    """Retire la tendance linéaire et la moyenne, maille par maille ; x : (an, lat, lon)."""
    y = np.full_like(x, np.nan)
    t = ANS - ANS.mean()
    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            v = x[:, i, j]; m = np.isfinite(v)
            if m.sum() >= 10:
                p = np.polyfit(t[m], v[m], 1)
                y[m, i, j] = v[m] - np.polyval(p, t[m])
    return y


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1]), int(m.sum())


def p_perm(a, b, n=5000):
    r0, _ = spearman(a, b)
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    c = sum(abs(spearman(rng.permutation(a), b)[0]) >= abs(r0) for _ in range(n))
    return (c + 1) / (n + 1)


def main():
    evi = lire_evi()
    dates, pluie, et0, tmax, lat, lon, ru = F.charger()
    cap = B.capacites(ru)
    etats = B.simuler_jachere(pluie, et0, cap)
    idx = {d: i for i, d in enumerate(dates)}
    L = 110; J = B.J_GERM + L + 2
    ind = {}
    succ_paysan = np.full((len(ANS), 6, 6), np.nan); wrsi_paysan = np.full_like(succ_paysan, np.nan)
    succ_tout = np.full_like(succ_paysan, np.nan)
    for a, y in enumerate(ANS):
        sd = F.dates_semis(y)
        i0 = np.array([idx[d] for d in sd])
        W0 = np.moveaxis(etats[i0 - 1], 0, 1)
        g = lambda x: x[i0[None, :] + np.arange(J)[:, None]]
        r = B.simuler_semis(W0, cap[:, None], g(pluie), g(et0), g(tmax), L)
        pay = np.array([(d.month == 5 and d.day >= 15) or d.month == 6 for d in sd])
        succ_paysan[a] = r["reussite"][pay].mean(0); wrsi_paysan[a] = np.median(r["wrsi"][pay], 0)
        succ_tout[a] = r["reussite"].mean(0)
    ind["riz : réussite, semis 15 mai–30 juin"] = succ_paysan
    ind["riz : WRSI médian, semis 15 mai–30 juin"] = wrsi_paysan
    ind["riz : réussite, semis 1er mars–31 juil."] = succ_tout

    def moy_periode(x, m0, m1):
        out = np.full((len(ANS), 6, 6), np.nan)
        for a, y in enumerate(ANS):
            k = [i for i, d in enumerate(dates) if d.year == y and m0 <= d.month <= m1]
            out[a] = x[k].mean(0)
        return out
    rel30 = (etats[:, 0] + etats[:, 1]) / (cap[0] + cap[1])
    ind["riz : eau utile 0–30 cm, août–oct."] = moy_periode(rel30, 8, 10)
    zb = np.load(os.path.join(ICI, "historique_beta_marahoue.npz"))
    db = [dt.date.fromisoformat(d) for d in zb["dates"]]
    beta = zb["beta"].astype(float)
    if np.nanmax(beta) > 1.5:
        beta = beta / 100
    ib = np.argsort(zb["lat"]); jb = np.argsort(zb["lon"])
    beta = beta[:, ib][:, :, jb]
    bo = np.full((len(ANS), 6, 6), np.nan)
    for a, y in enumerate(ANS):
        k = [i for i, d in enumerate(db) if d.year == y and 8 <= d.month <= 10]
        bo[a] = np.nanmean(beta[k], 0)
    ind["beta TAMSAT-ALERT (C4, 1 m), août–oct."] = bo
    ps = np.full((len(ANS), 6, 6), np.nan); pj = np.full_like(ps, np.nan)
    for a, y in enumerate(ANS):
        ps[a] = pluie[[i for i, d in enumerate(dates) if d.year == y and 6 <= d.month <= 9]].sum(0)
        pj[a] = pluie[[i for i, d in enumerate(dates) if d.year == y and d.month == 7]].sum(0)
    ind["pluie TAMSAT juin–sept."] = ps
    ind["pluie TAMSAT juillet"] = pj

    res = {}
    for nom_cible, k in (("EVI août–oct.", 1), ("EVI sept.–nov.", 2)):
        cible = anomalies(evi[k])
        res[nom_cible] = {}
        print(f"\nAnomalies d'{nom_cible} (tendance retirée), Spearman :")
        for nom, x in ind.items():
            ax = anomalies(x)
            r_pool, n = spearman(ax.ravel(), cible.ravel())
            zx, zc = np.nanmean(ax, axis=(1, 2)), np.nanmean(cible, axis=(1, 2))
            r_zone, _ = spearman(zx, zc)
            p = p_perm(zx, zc)
            res[nom_cible][nom] = {"r_mailles": round(r_pool, 3), "n": n, "r_zone": round(r_zone, 3), "p_zone": round(p, 4)}
            print(f"  {nom:45s} mailles r = {r_pool:+.2f} (n = {n}) | zone r = {r_zone:+.2f} (p = {p:.3f})")
    json.dump(res, open(os.path.join(ICI, "validation", "validation_ndvi.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
