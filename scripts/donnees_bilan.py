"""
Données d'entrée du bilan hydrique du riz pluvial (Marahoué, grille TAMSAT 0,25°).

  1. Pluie journalière TAMSAT v3.2 (source des prévisions TAMSAT-ALERT), 2005 → hier :
     lecture partielle HTTP des seuls blocs journaliers, décompression locale.
  2. Météo journalière NASA POWER (MERRA-2) → ET0 FAO-56 Penman-Monteith et Tmax.
  3. Sol SoilGrids 2.0 (sable, argile, carbone organique 0–60 cm) → réserve utile
     (fonctions de pédotransfert de Saxton et Rawls, 2006).

Cache par année dans cache_bilan/ : seule l'année en cours est retéléchargée.
Sortie : donnees_bilan_marahoue.npz (pluie, et0, tmax, dates, lat, lon, ru_mm_m).

Usage : python donnees_bilan.py [--sans-sol]
"""
import datetime as dt
import json
import os
import sys
import time
import urllib.request
import zlib

import fsspec
import h5py
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ICI = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ICI, "cache_bilan")
RFE = ("https://gws-access.jasmin.ac.uk/public/tamsat/tamsat_alert_forcing_data/subset/"
       "Africa/0.25/yearly_files/{y}/prate_tamsat_{y}_sub.v3.2.nc")
LAT = np.arange(6.375, 7.75, 0.25)          # centres des mailles 0,25° de la zone
LON = np.arange(-6.625, -5.25, 0.25)
AN0 = 2005


# ---------------------------------------------------------------- pluie TAMSAT
def _unshuffle(raw, itemsize):
    a = np.frombuffer(raw, np.uint8)
    return a.reshape(itemsize, -1).T.copy().ravel()


def pluie_annee(y, fs):
    url = RFE.format(y=y)
    h = h5py.File(fs.open(url, block_size=2**16), "r")
    v = h["rfe"]
    lat, lon = h["latitude"][:], h["longitude"][:]
    i0 = int(np.argmin(abs(lat - LAT[0]))); j0 = int(np.argmin(abs(lon - LON[0])))
    assert abs(lat[i0] - LAT[0]) < 1e-3 and abs(lon[j0] - LON[0]) < 1e-3
    nt, ny, nx = v.shape
    assert v.chunks == (1, ny, nx) and v.compression == "gzip"
    shuf = v.shuffle
    infos = [v.id.get_chunk_info(k) for k in range(v.id.get_num_chunks())]
    infos.sort(key=lambda c: c.chunk_offset[0])
    starts = [c.byte_offset for c in infos]; ends = [c.byte_offset + c.size for c in infos]
    out = np.full((nt, len(LAT), len(LON)), np.nan, np.float32)
    B = 30
    for b in range(0, nt, B):
        raws = [None] * len(starts[b:b + B])
        for essai in range(8):
            todo = [k for k, r in enumerate(raws) if not isinstance(r, (bytes, bytearray))]
            if not todo:
                break
            try:
                got = fs.cat_ranges([url] * len(todo), [starts[b + k] for k in todo],
                                    [ends[b + k] for k in todo], on_error="return")
            except Exception as e:
                got = [e] * len(todo)
            for k, r in zip(todo, got):
                raws[k] = r
            if any(not isinstance(r, (bytes, bytearray)) for r in raws):
                time.sleep(3 * (essai + 1))   # coupures réseau du serveur JASMIN
        else:
            if any(not isinstance(r, (bytes, bytearray)) for r in raws):
                raise RuntimeError(f"Lecture impossible : {url} (jours {b}–{b + B})")
        for k, raw in enumerate(raws):
            d = zlib.decompress(raw)
            if shuf:
                d = _unshuffle(d, 4)
            arr = np.frombuffer(d, np.float32).reshape(ny, nx)
            out[b + k] = arr[i0:i0 + len(LAT), j0:j0 + len(LON)]
    out[(out < 0) | (out > 500)] = np.nan
    t0 = dt.date(y, 1, 1)
    dates = [str(t0 + dt.timedelta(days=int(x))) for x in h["time"][:]]
    return out, dates


# ---------------------------------------------------------------- météo NASA POWER
POWER = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters="
         "T2M_MAX,T2M_MIN,T2MDEW,WS2M,ALLSKY_SFC_SW_DWN,PS&community=AG"
         "&longitude={lon}&latitude={lat}&start={a}&end={b}&format=JSON&time-standard=LST")


def et0_fao56(tmax, tmin, tdew, u2, rs, ps_kpa, lat_deg, doy, z=250.0):
    """ET0 FAO-56 Penman-Monteith journalière (mm/j)."""
    tmean = (tmax + tmin) / 2
    es = lambda t: 0.6108 * np.exp(17.27 * t / (t + 237.3))
    ea = es(tdew)
    es_m = (es(tmax) + es(tmin)) / 2
    delta = 4098 * es(tmean) / (tmean + 237.3) ** 2
    gamma = 0.000665 * ps_kpa
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    ra = 24 * 60 / np.pi * 0.082 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    rso = (0.75 + 2e-5 * z) * ra
    rns = 0.77 * rs
    sig = 4.903e-9
    rnl = sig * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2 * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))) \
        * (1.35 * np.clip(rs / np.maximum(rso, 1e-6), 0.25, 1.0) - 0.35)
    rn = rns - rnl
    et0 = (0.408 * delta * rn + gamma * 900 / (tmean + 273) * u2 * (es_m - ea)) / (delta + gamma * (1 + 0.34 * u2))
    return np.maximum(et0, 0)


def meteo_point(lat, lon, a, b):
    url = POWER.format(lat=lat, lon=lon, a=a, b=b)
    for essai in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                p = json.load(r)["properties"]["parameter"]
            break
        except Exception as e:
            print(f"  POWER {lat},{lon}: {type(e).__name__}, nouvel essai"); time.sleep(10 * (essai + 1))
    else:
        raise RuntimeError("NASA POWER indisponible")
    jours = sorted(p["T2M_MAX"].keys())
    g = lambda k: np.array([p[k][d] for d in jours], float)
    tx, tn, td, u2, rs, ps = (g(k) for k in ("T2M_MAX", "T2M_MIN", "T2MDEW", "WS2M", "ALLSKY_SFC_SW_DWN", "PS"))
    bad = (tx < -900) | (tn < -900) | (td < -900) | (u2 < -900) | (rs < -900) | (ps < -900)
    doy = np.array([dt.date(int(d[:4]), int(d[4:6]), int(d[6:])).timetuple().tm_yday for d in jours])
    et0 = et0_fao56(tx, tn, td, u2, rs, ps, lat, doy)
    et0[bad] = np.nan; tx = np.where(tx < -900, np.nan, tx)
    return [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in jours], et0, tx


# ---------------------------------------------------------------- sol SoilGrids
SG = ("https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}"
      "&property=sand&property=clay&property=soc&depth=0-5cm&depth=5-15cm&depth=15-30cm"
      "&depth=30-60cm&value=mean")


def saxton_rawls(sand, clay, om):
    """Réserve utile (m3/m3) : θ33 − θ1500, Saxton & Rawls (2006). sand, clay en fraction, om en %."""
    t1500t = -0.024 * sand + 0.487 * clay + 0.006 * om + 0.005 * sand * om - 0.013 * clay * om + 0.068 * sand * clay + 0.031
    t1500 = t1500t + (0.14 * t1500t - 0.02)
    t33t = -0.251 * sand + 0.195 * clay + 0.011 * om + 0.006 * sand * om - 0.027 * clay * om + 0.452 * sand * clay + 0.299
    t33 = t33t + (1.283 * t33t ** 2 - 0.374 * t33t - 0.015)
    return max(t33 - t1500, 0.03)


def sol_point(lat, lon):
    for essai in range(5):
        try:
            with urllib.request.urlopen(SG.format(lat=lat, lon=lon), timeout=60) as r:
                layers = json.load(r)["properties"]["layers"]
            break
        except Exception as e:
            print(f"  SoilGrids {lat},{lon}: {type(e).__name__}, nouvel essai"); time.sleep(15 * (essai + 1))
    else:
        return None
    v = {L["name"]: {d["label"]: d["values"]["mean"] for d in L["depths"]} for L in layers}
    ep = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15, "30-60cm": 30}
    tot = 0.0
    for d, e in ep.items():
        s, c, o = v["sand"][d], v["clay"][d], v["soc"][d]
        if s is None or c is None or o is None:
            return None
        tot += saxton_rawls(s / 1000, c / 1000, o / 100 * 1.724) * e   # soc dg/kg → % MO
    return tot / 60 * 1000          # mm d'eau utile par mètre de sol


def main():
    os.makedirs(CACHE, exist_ok=True)
    fs = fsspec.filesystem("https")
    today = dt.date.today()
    pr_all, d_all = [], []
    for y in range(AN0, today.year + 1):
        f = os.path.join(CACHE, f"pluie_{y}.npz")
        z = np.load(f) if os.path.exists(f) else None
        if z is not None and y < today.year and str(z["dates"][-1]).endswith("12-31"):
            pr, dates = z["pr"], list(z["dates"])
        else:
            print(f"Pluie TAMSAT {y}…")
            for essai in range(6):
                try:
                    pr, dates = pluie_annee(y, fs); break
                except Exception as e:          # coupure réseau ou DNS : on réessaie
                    print(f"  {y}: {type(e).__name__}, nouvel essai dans {30 * (essai + 1)} s")
                    time.sleep(30 * (essai + 1)); fs = fsspec.filesystem("https", skip_instance_cache=True)
            else:
                raise RuntimeError(f"Pluie TAMSAT {y} indisponible")
            np.savez_compressed(f, pr=pr, dates=np.array(dates))
        pr_all.append(pr); d_all += dates
    pluie = np.concatenate(pr_all); dates = np.array(d_all)
    print(f"Pluie : {dates[0]} → {dates[-1]} ({len(dates)} j)")

    # Météo : un appel par maille (toute la période), cache sauf année en cours
    fmet = os.path.join(CACHE, "meteo_power.npz")
    a = f"{AN0}0101"; b = (today - dt.timedelta(days=1)).strftime("%Y%m%d")
    et0 = np.full((len(dates), len(LAT), len(LON)), np.nan, np.float32)
    tmax = np.full_like(et0, np.nan)
    old = dict(np.load(fmet)) if os.path.exists(fmet) else None   # chargé une seule fois en mémoire
    reprise = None
    if old is not None:      # reprise au 1er janvier de la dernière année en cache
        ok = [d for k, d in enumerate(old["dates"]) if np.isfinite(old["et0"][k]).all()]
        reprise = f"{ok[-1][:4]}0101" if ok else a
    idx = {d: k for k, d in enumerate(dates)}
    for i, la in enumerate(LAT):
        for j, lo in enumerate(LON):
            debut = reprise or a
            jours, e, t = meteo_point(round(la, 3), round(lo, 3), debut, b)
            if old is not None:
                od = list(old["dates"])
                for k, d in enumerate(od):
                    if d < f"{reprise[:4]}-01-01" and d in idx:
                        et0[idx[d], i, j] = old["et0"][k, i, j]; tmax[idx[d], i, j] = old["tmax"][k, i, j]
            for k, d in enumerate(jours):
                if d in idx:
                    et0[idx[d], i, j] = e[k]; tmax[idx[d], i, j] = t[k]
        print(f"  météo POWER : ligne {i + 1}/{len(LAT)}")
    np.savez_compressed(fmet, et0=et0, tmax=tmax, dates=dates)

    # Sol (fixe) : mis en cache une fois
    fsol = os.path.join(CACHE, "sol_soilgrids.json")
    if os.path.exists(fsol):
        ru = np.array(json.load(open(fsol))["ru_mm_m"], float)
    elif "--sans-sol" in sys.argv:
        ru = np.full((len(LAT), len(LON)), 120.0)
    else:
        ru = np.full((len(LAT), len(LON)), np.nan)
        for i, la in enumerate(LAT):
            for j, lo in enumerate(LON):
                r = sol_point(round(la, 3), round(lo, 3))
                ru[i, j] = np.nan if r is None else r
                time.sleep(13)       # limite SoilGrids : 5 requêtes par minute
            print(f"  sol SoilGrids : ligne {i + 1}/{len(LAT)}")
        miss = np.isnan(ru)
        if miss.all():
            ru[:] = 120.0; src = "valeur par défaut 120 mm/m (SoilGrids indisponible)"
        else:
            ru[miss] = np.nanmedian(ru); src = "SoilGrids 2.0, Saxton & Rawls 2006"
        json.dump({"ru_mm_m": ru.round(1).tolist(), "source": src}, open(fsol, "w"))
    print("Réserve utile (mm/m) :", np.round(ru, 0).tolist())

    np.savez_compressed(os.path.join(ICI, "donnees_bilan_marahoue.npz"), pluie=pluie, et0=et0, tmax=tmax,
                        dates=dates, lat=LAT, lon=LON, ru_mm_m=ru)
    print("Écrit : donnees_bilan_marahoue.npz")


if __name__ == "__main__":
    main()
