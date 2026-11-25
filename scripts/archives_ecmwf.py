"""
Prévisions de pluie ECMWF IFS (0,25°) archivées par échéance, saisons 2024 et 2025 (Open-Meteo,
« Previous Runs API ») : pour chaque jour cible, la pluie prévue 0, 1, …, 7 jours avant.
Sortie : validation/archives_ecmwf_2024_2025.npz  pluie[echeance 0–7, jour, lat, lon] (mm/j), dates.
Données ECMWF (CC BY 4.0) via Open-Meteo (CC BY 4.0).
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
LAT = np.arange(6.375, 7.75, 0.25)
LON = np.arange(-6.625, -5.25, 0.25)
PERIODES = [("2024-02-15", "2024-12-15"), ("2025-02-15", "2025-12-15")]
VARS = ["precipitation"] + [f"precipitation_previous_day{k}" for k in range(1, 8)]
URL = ("https://previous-runs-api.open-meteo.com/v1/forecast?latitude={la}&longitude={lo}&hourly={v}"
       "&models=ecmwf_ifs025&start_date={a}&end_date={b}&timezone=GMT")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pts = [(i, j) for i in range(len(LAT)) for j in range(len(LON))]
    blocs, dates = [], []
    for a, b in PERIODES:
        jours = None
        arr = None
        for k in range(0, len(pts), 6):
            grp = pts[k:k + 6]
            u = URL.format(la=",".join(f"{LAT[i]:.3f}" for i, _ in grp), lo=",".join(f"{LON[j]:.3f}" for _, j in grp),
                           v=",".join(VARS), a=a, b=b)
            for essai in range(5):
                try:
                    d = json.load(urllib.request.urlopen(u, timeout=180)); break
                except Exception as e:
                    print("  nouvel essai :", type(e).__name__); time.sleep(20 * (essai + 1))
            else:
                raise RuntimeError("Open-Meteo indisponible")
            d = d if isinstance(d, list) else [d]
            for (i, j), loc in zip(grp, d):
                h = loc["hourly"]
                t = h["time"]; n = len(t) // 24
                if jours is None:
                    jours = [t[24 * q][:10] for q in range(n)]
                    arr = np.full((8, n, len(LAT), len(LON)), np.nan, np.float32)
                for e, v in enumerate(VARS):
                    x = np.array([np.nan if z is None else z for z in h[v]], float)[:24 * n].reshape(n, 24)
                    arr[e, :, i, j] = np.where(np.isnan(x).any(1), np.nan, x.sum(1))
            print(f"{a[:4]} : mailles {k + len(grp)}/{len(pts)}")
            time.sleep(1)
        blocs.append(arr); dates += jours
    out = np.concatenate(blocs, axis=1)
    os.makedirs(os.path.join(ICI, "validation"), exist_ok=True)
    np.savez_compressed(os.path.join(ICI, "validation", "archives_ecmwf_2024_2025.npz"), pluie=out, dates=np.array(dates))
    print("Écrit", out.shape, "manquants", int(np.isnan(out).sum()))


if __name__ == "__main__":
    main()
