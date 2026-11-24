"""
Prévision probabiliste des conditions de semis du riz pluvial dans la Marahoué
à partir des prévisions opérationnelles TAMSAT-ALERT (Université de Reading).

Les prévisions TAMSAT-ALERT (fichiers alert_AAAAMMJJ_ens.daily.nc, Afrique entière,
0,25°, 15 membres = météo des années 2005–2019, 150 jours) sont lues à distance
par lectures partielles HTTP : seule la zone d'étude est téléchargée (~quelques Mo),
au lieu des 3,4 Go du fichier.

Critère « jour favorable au semis » (riz pluvial, INDICATIF, à calibrer) :
  - beta(j) ≥ BETA_SEMIS : sol assez humide le jour du semis ;
  - moyenne de beta sur les J_SUIVI jours suivants ≥ BETA_SUIVI : humidité durable ;
  - aucune moyenne glissante sur 7 jours < BETA_SECHE dans ces J_SUIVI jours :
    pas de poche sèche longue après le semis (risque d'échec de levée).
beta = facteur de disponibilité en eau (0 = flétrissement, 1 = pas de contrainte).

La probabilité d'un jour = part des 15 membres d'ensemble qui satisfont le critère
(pondération égale = pas de prévision saisonnière en terciles appliquée).

Usage :
    python talert_marahoue.py                 # dernière prévision disponible
    python talert_marahoue.py 20261119        # prévision d'une date donnée
Dépendances : numpy, h5py, fsspec, aiohttp, matplotlib
Données : TAMSAT, University of Reading, CC BY 4.0.
"""
import datetime as dt
import os
import re
import sys

import fsspec
import h5py
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

BASE = "https://gws-access.jasmin.ac.uk/public/tamsat/tamsat_alert/forecasts/"
# Marahoué : boîte englobante (Bouaflé, Sinfra, Zuénoula, Bonon, Gohitafla)
LAT_MIN, LAT_MAX = 6.25, 7.75
LON_MIN, LON_MAX = -6.75, -5.25

BETA_SEMIS = 0.70   # sol assez humide le jour du semis
BETA_SUIVI = 0.60   # humidité moyenne minimale sur les jours suivants
BETA_SECHE = 0.40   # seuil de poche sèche (moyenne glissante 7 jours)
J_SUIVI = 21        # jours de suivi après semis (germination, levée, installation)
P_SEUIL = 0.60      # probabilité jugée suffisante pour recommander de semer
SEMIS_DEBUT = (3, 1)  # (mois, jour) : début de la période de semis évaluée (grande saison
                      # des pluies). Avant, l'humidité résiduelle de la petite saison
                      # (sept.–nov.) donnerait de fausses fenêtres de semis.

OUT = os.path.dirname(os.path.abspath(__file__))

# Seuils calés sur l'historique : de préférence calendrier_riz.py (réussite du riz selon
# la revue de littérature, humidité du sol), sinon calibrer_semis_riz.py (démarrage des pluies)
_CAL = os.path.join(OUT, "calibration_humidite_riz.json")
if not os.path.exists(_CAL):
    _CAL = os.path.join(OUT, "calibration_semis_riz.json")
CALIBRE = False
if os.path.exists(_CAL):
    import json as _json
    with open(_CAL, encoding="utf-8") as _f:
        _s = _json.load(_f)["seuils"]
    BETA_SEMIS, BETA_SUIVI, BETA_SECHE, J_SUIVI = (_s["BETA_SEMIS"], _s["BETA_SUIVI"],
                                                     _s["BETA_SECHE"], _s["J_SUIVI"])
    CALIBRE = True


def latest_forecast_date():
    html = fsspec.filesystem("https").cat(BASE).decode("utf-8", "replace")
    dates = sorted(set(re.findall(r"alert_(\d{8})_ens\.daily\.nc", html)))
    if not dates:
        sys.exit("Aucune prévision trouvée sur le serveur TAMSAT.")
    return dates[-1]


def fetch_ranges(fs, url, starts, ends, batch=200, tries=6):
    """Lectures partielles par lots, avec nouvelles tentatives si le serveur refuse."""
    import time
    out = [None] * len(starts)
    todo = list(range(len(starts)))
    for attempt in range(tries):
        failed = []
        for b in range(0, len(todo), batch):
            idx = todo[b:b + batch]
            res = fs.cat_ranges([url] * len(idx), [starts[i] for i in idx], [ends[i] for i in idx],
                                on_error="return")
            for i, r in zip(idx, res):
                if isinstance(r, (bytes, bytearray)):
                    out[i] = r
                else:
                    failed.append(i)
        if not failed:
            return out
        todo = failed
        time.sleep(3 * (attempt + 1))
    sys.exit(f"{len(todo)} lectures partielles ont échoué malgré {tries} tentatives.")


def read_box(fs, url, h, name, layer=None):
    """Lit var[ens, time, (layer,) lat0:lat1, lon0:lon1] par lectures partielles."""
    v = h[name]
    shape = v.shape
    off = v.id.get_offset()
    if off is None:
        sys.exit(f"{name} : stockage non contigu, lecture partielle impossible.")
    lat = h["latitude"][:]
    lon = h["longitude"][:]
    r = np.where((lat >= LAT_MIN) & (lat <= LAT_MAX))[0]
    c = np.where((lon >= LON_MIN) & (lon <= LON_MAX))[0]
    r0, r1, c0, c1 = r[0], r[-1] + 1, c[0], c[-1] + 1
    ne, nt = shape[0], shape[1]
    nlat, nlon = shape[-2], shape[-1]
    isz = v.dtype.itemsize
    starts, ends = [], []
    for e in range(ne):
        for t in range(nt):
            if layer is None:
                base = (e * nt + t) * nlat
            else:
                base = ((e * nt + t) * shape[2] + layer) * nlat
            s = off + (base + r0) * nlon * isz
            starts.append(s)
            ends.append(s + (r1 - r0) * nlon * isz)
    blobs = fetch_ranges(fs, url, starts, ends)
    arr = np.empty((ne, nt, r1 - r0, c1 - c0), dtype=np.float32)
    k = 0
    for e in range(ne):
        for t in range(nt):
            band = np.frombuffer(blobs[k], dtype=v.dtype).reshape(r1 - r0, nlon)
            arr[e, t] = band[:, c0:c1]
            k += 1
    fill = v.attrs.get("_FillValue", [-9999])[0]
    scale = float(v.attrs.get("scale_factor", [1.0])[0])
    arr[arr == fill] = np.nan
    return arr * scale, lat[r0:r1], lon[c0:c1]


def favourable(beta):
    """beta : (ens, time, ...) → booléen (ens, time_eval, ...) jour favorable au semis."""
    ne, nt = beta.shape[:2]
    nev = nt - J_SUIVI
    ok = np.zeros((ne, nev) + beta.shape[2:], dtype=bool)
    k = np.ones(7) / 7
    for d in range(nev):
        win = beta[:, d:d + J_SUIVI + 1]
        mean_win = np.nanmean(win[:, 1:], axis=1)
        # moyenne glissante 7 jours dans la fenêtre de suivi
        roll = np.apply_along_axis(lambda x: np.convolve(x, k, mode="valid"), 1, win[:, 1:])
        ok[:, d] = (win[:, 0] >= BETA_SEMIS) & (mean_win >= BETA_SUIVI) & (np.nanmin(roll, axis=1) >= BETA_SECHE)
    return ok


def main():
    fdate = sys.argv[1] if len(sys.argv) > 1 else latest_forecast_date()
    url = f"{BASE}alert_{fdate}_ens.daily.nc"
    print(f"Prévision TAMSAT-ALERT du {fdate} : {url}")
    fs = fsspec.filesystem("https")
    h = h5py.File(fs.open(url, block_size=2**16), "r")
    t_units = h["time"].attrs["units"].decode() if hasattr(h["time"].attrs["units"], "decode") else h["time"].attrs["units"]
    t0 = dt.date.fromisoformat(t_units.split("since ")[1][:10])
    days = [t0 + dt.timedelta(days=int(x)) for x in h["time"][:]]
    years = h["ens_year"][:]

    beta, lat, lon = read_box(fs, url, h, "beta_c4grass")
    if np.nanmax(beta) > 1.5:   # fourni de 0 à 100 après facteur d'échelle → 0 à 1
        beta = beta / 100.0
    sm0, _, _ = read_box(fs, url, h, "smcl", layer=0)
    print(f"Zone : {len(lat)} × {len(lon)} pixels de 0,25°, {beta.shape[0]} membres, {beta.shape[1]} jours "
          f"({days[0]} → {days[-1]})")

    ok = favourable(beta)                          # (ens, d, lat, lon)
    # Pas de semis évalué avant le début de la période de semis de la grande saison
    for d in range(ok.shape[1]):
        dd = days[d]
        if (dd.month, dd.day) < SEMIS_DEBUT and dd.month >= 1 and not (dd.month >= 8):
            ok[:, d] = False
        if dd.month >= 8:                          # août–décembre : hors grande saison
            ok[:, d] = False
    eval_mask = np.array([(days[d].month, days[d].day) >= SEMIS_DEBUT and days[d].month < 8
                          for d in range(ok.shape[1])])
    if not eval_mask.any():
        print("La période de semis évaluée n'est pas couverte par cette prévision "
              "(horizon de 150 jours). Relancez avec une prévision plus récente.")
    p_pix = ok.mean(axis=0)                        # probabilité par pixel et par jour
    p_zone = np.nanmean(p_pix, axis=(1, 2))        # moyenne zonale
    beta_zone = np.nanmean(beta, axis=(2, 3))      # (ens, time)
    sm_zone = np.nanmean(sm0, axis=(2, 3))

    # Première date où la probabilité atteint P_SEUIL, par pixel
    first = np.full(p_pix.shape[1:], np.nan)
    for i in range(p_pix.shape[1]):
        for j in range(p_pix.shape[2]):
            idx = np.where(p_pix[:, i, j] >= P_SEUIL)[0]
            if len(idx):
                first[i, j] = idx[0]

    tag = f"{fdate}"
    os.makedirs(OUT, exist_ok=True)
    csv = os.path.join(OUT, f"proba_semis_riz_marahoue_{tag}.csv")
    with open(csv, "w", encoding="utf-8") as f:
        f.write("date;proba_semis_favorable;beta_mediane;beta_p10;beta_p90;sm_0_10cm_mediane_m3m3\n")
        for d in range(len(days)):
            pz = f"{p_zone[d]:.3f}" if d < len(p_zone) and eval_mask[d] else ""
            b = beta_zone[:, d]
            f.write(f"{days[d]};{pz};{np.nanmedian(b):.3f};{np.nanpercentile(b, 10):.3f};"
                    f"{np.nanpercentile(b, 90):.3f};{np.nanmedian(sm_zone[:, d]):.4f}\n")
    print("CSV :", csv)

    # Export JSON pour la page web interactive
    import json
    r = lambda a, n=3: [None if not np.isfinite(x) else round(float(x), n) for x in np.ravel(a)]
    sel_d = [d for d in range(len(p_zone)) if eval_mask[d]]
    web = {
        "date_prevision": f"{fdate[:4]}-{fdate[4:6]}-{fdate[6:]}",
        "annees_meteo": [int(years.min()), int(years.max())],
        "membres": int(beta.shape[0]),
        "seuils": {"BETA_SEMIS": BETA_SEMIS, "BETA_SUIVI": BETA_SUIVI, "BETA_SECHE": BETA_SECHE,
                   "J_SUIVI": J_SUIVI, "P_SEUIL": P_SEUIL},
        "jours": [str(d) for d in days],
        "beta_membres": [r(beta_zone[e]) for e in range(beta_zone.shape[0])],
        "sm_mediane": r(np.nanmedian(sm_zone, axis=0), 4),
        "proba_zone": [r([p_zone[d]])[0] if (d < len(p_zone) and eval_mask[d]) else None for d in range(len(days))],
        "lat": r(lat, 3), "lon": r(lon, 3),
        "jours_eval": [str(days[d]) for d in sel_d],
        "proba_pixels": [[r(p_pix[d, i], 3) for i in range(p_pix.shape[1])] for d in sel_d],
    }
    for name in (f"prevision_riz_{tag}.json", "prevision_riz_derniere.json"):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            json.dump(web, f, ensure_ascii=False, separators=(",", ":"))
    print("JSON :", os.path.join(OUT, f"prevision_riz_{tag}.json"))

    # Graphiques
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for e in range(beta_zone.shape[0]):
        a1.plot(days, beta_zone[e], color="#9ab", lw=0.8, alpha=0.8)
    a1.fill_between(days, np.nanpercentile(beta_zone, 10, axis=0), np.nanpercentile(beta_zone, 90, axis=0),
                    color="#2f6b4f", alpha=0.2, label="10e–90e centile")
    a1.plot(days, np.nanmedian(beta_zone, axis=0), color="#2f6b4f", lw=2, label="Médiane de l'ensemble")
    a1.axhline(BETA_SEMIS, color="#d98a2b", ls="--", lw=1, label=f"Seuil de semis (beta = {BETA_SEMIS})")
    a1.set_ylabel("Disponibilité en eau du sol (beta)")
    a1.set_ylim(0, 1.02)
    a1.set_title(f"Marahoué — prévision TAMSAT-ALERT du {fdate[6:]}/{fdate[4:6]}/{fdate[:4]} "
                 f"(15 scénarios météo {years.min()}–{years.max()})")
    a1.legend(loc="upper left", fontsize=8)
    d_eval = days[:len(p_zone)]
    sel = [d for d in range(len(p_zone)) if eval_mask[d]]
    a2.bar([d_eval[d] for d in sel], [p_zone[d] * 100 for d in sel], width=1,
           color=["#3f8f5f" if p_zone[d] >= P_SEUIL else "#c9d6cf" for d in sel])
    if sel:
        a2.axvspan(d_eval[sel[0]], d_eval[sel[-1]], color="#f3efe0", zorder=0, label="Période de semis évaluée")
        a2.legend(loc="upper left", fontsize=8)
    a2.axhline(P_SEUIL * 100, color="#7d1f2a", ls="--", lw=1)
    a2.set_ylabel("Probabilité de semis\nfavorable (%)")
    a2.set_ylim(0, 100)
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    a2.set_xlabel(f"Riz pluvial — critère : beta ≥ {BETA_SEMIS} au semis, moyenne ≥ {BETA_SUIVI} sur "
                  f"{J_SUIVI} j, pas de semaine < {BETA_SECHE}")
    fig.tight_layout()
    png1 = os.path.join(OUT, f"proba_semis_riz_marahoue_{tag}.png")
    fig.savefig(png1, dpi=130)
    print("Graphique :", png1)

    fig, ax = plt.subplots(figsize=(6, 5))
    first_days = np.where(np.isnan(first), np.nan, first)
    im = ax.pcolormesh(lon, lat, first_days, cmap="viridis_r", shading="nearest",
                       vmin=0, vmax=max(1, len(d_eval) - 1))
    cb = fig.colorbar(im, ax=ax)
    ticks = cb.get_ticks()
    cb.set_ticks(ticks)
    cb.set_ticklabels([(days[0] + dt.timedelta(days=int(t))).strftime("%d/%m") for t in ticks])
    cb.set_label(f"Première date avec P ≥ {int(P_SEUIL * 100)} %")
    ax.set_title("Première date de semis favorable (riz pluvial)\n"
                 "blanc = seuil non atteint dans l'horizon de prévision", fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    png2 = os.path.join(OUT, f"carte_premiere_date_semis_riz_{tag}.png")
    fig.savefig(png2, dpi=130)
    print("Carte :", png2)

    idx = [d for d in sel if p_zone[d] >= P_SEUIL]
    print("Période de semis évaluée :", f"{d_eval[sel[0]]} → {d_eval[sel[-1]]}" if sel else "aucune")
    print("Première date (moyenne zonale) avec P ≥ seuil :", d_eval[idx[0]] if idx else "aucune dans l'horizon")
    if sel:
        best = max(sel, key=lambda d: p_zone[d])
        print("Probabilité maximale :", f"{p_zone[best]:.0%}", "le", d_eval[best])
    print("beta médian (zone) :", ", ".join(f"{days[d]}={np.nanmedian(beta_zone[:, d]):.2f}"
                                             for d in range(0, len(days), 15)))


if __name__ == "__main__":
    main()
