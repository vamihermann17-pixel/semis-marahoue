"""
Figures 1–5 de l'article (anglais), à partir des sorties existantes du modèle et des validations.
Sortie : article/figures/fig{1..5}.png (300 dpi) et .pdf (vectoriel).
Lisibles en noir et blanc : styles de trait et niveaux de gris, pas de code couleur seul.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ICI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ICI, "article", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 300})
MOIS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
STYLES = ["-", "--", ":"]
GRIS = ["0.0", "0.35", "0.55"]


def lire(chemin):
    return json.load(open(os.path.join(ICI, chemin), encoding="utf-8"))


def sauver(fig, nom):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{nom}.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ---------- Fig. 3 : calendriers ----------
def fig3():
    cultures = [("riz", "(a) Upland rice"), ("mais", "(b) Maize"), ("arachide", "(c) Groundnut"), ("igname", "(d) Yam")]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2), sharey=True)
    for ax, (c, titre) in zip(axes.ravel(), cultures):
        d = lire(f"web/calendriers_{c}.json")
        cycles = sorted(d["cycles"], key=int)
        dates = d["cycles"][cycles[0]]["dates"]
        x = np.arange(len(dates))
        if c != "igname":   # faux départs : sans objet pour une plantation d'igname en saison sèche
            fd = np.array([np.nan if v is None else v for v in d["cycles"][cycles[1]]["faux_depart"]], dtype=float)
            fl = np.array([np.nanmean(fd[max(0, i - 3):i + 4]) for i in range(len(fd))])   # moyenne glissante 7 j
            ax.fill_between(x, 0, fl * 100, step="mid", color="0.86", lw=0)
        for k, cy in enumerate(cycles):
            ax.plot(x, np.array(d["cycles"][cy]["reussite"]) * 100, ls=STYLES[k], color=GRIS[k], lw=1.1)
        debuts = [i for i, s in enumerate(dates) if s.startswith("01/")]
        ax.set_xticks(debuts, [MOIS[int(dates[i][3:]) - 1] for i in debuts])
        ax.set_xlim(-0.5, len(dates) - 0.5)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", lw=0.3, color="0.8")
        ax.set_title(f"{titre} — cycles {' / '.join(cycles)} d", loc="left", fontsize=8.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Probability (%)")
    axes[1, 0].set_xlabel("Sowing date"); axes[1, 1].set_xlabel("Planting date")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    poignees = [Line2D([], [], ls=STYLES[k], color=GRIS[k], lw=1.1) for k in range(3)] + [Patch(color="0.86")]
    fig.legend(poignees, ["Success, shortest cycle", "Success, median cycle", "Success, longest cycle",
                          "False-start frequency (7-day mean; a–c)"],
               loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    sauver(fig, "fig3_calendars")


# ---------- Fig. 4 : EVI ----------
def fig4():
    s = lire("validation/ndvi_series_zone.json")["EVI août–oct."]
    evi = np.array(s["EVI"])   # anomalies, EVI x1000 (unités du fichier source)
    panneaux = [("riz : eau utile 0–30 cm, août–oct.", "(a) Simulated soil water, 0–0.3 m|Anomaly, Aug–Oct (fraction of AWC)", 0.79),
                ("beta TAMSAT-ALERT (C4, 1 m), août–oct.", "(b) TAMSAT-ALERT β (C4 grass)|Anomaly, Aug–Oct", 0.59),
                ("pluie TAMSAT juin–sept.", "(c) TAMSAT rainfall|Anomaly, Jun–Sep (mm)", 0.60)]
    ans = lire("validation/ndvi_series_zone.json")["annees"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5), sharey=True)
    for ax, (cle, lab, rho) in zip(axes, panneaux):
        x = np.array(s[cle])
        ax.scatter(x, evi, s=14, facecolor="white", edgecolor="black", lw=0.7)
        p = np.polyfit(x, evi, 1)
        xx = np.linspace(x.min(), x.max(), 10)
        ax.plot(xx, np.polyval(p, xx), color="0.4", lw=0.8, ls="--")
        for a, xi, yi in zip(ans, x, evi):
            if a in (2005, 2015, 2020):
                ax.annotate(str(a), (xi, yi), fontsize=5.5, xytext=(3, 2), textcoords="offset points", color="0.3")
        ax.set_xlabel(lab.split("|")[1])
        ax.set_title(lab.split("|")[0], loc="left", fontsize=8)
        ax.text(0.97, 0.05, f"ρ = {rho:.2f}", transform=ax.transAxes, ha="right", fontsize=8)
        ax.axhline(0, lw=0.3, color="0.7"); ax.axvline(0, lw=0.3, color="0.7")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("EVI anomaly, Aug–Oct")
    fig.tight_layout()
    sauver(fig, "fig4_evi")


# ---------- Fig. 5 : valeur des sources d'information ----------
def fig5():
    lignes = []   # (groupe, étiquette, BSS en %)
    noms = {"riz": "Rice", "mais": "Maize", "arachide": "Groundnut", "igname": "Yam"}
    for c in ("riz", "mais", "arachide", "igname"):
        f = "validation_retrospective.json" if c == "riz" else f"validation/validation_retrospective_{c}.json"
        r = lire(f)["resultats"]
        v = [r[k]["BSS"] * 100 for k in r]
        lignes.append(("A", f"{noms[c]} (5 issue dates)", v))
    m = lire("validation/validation_meteo10j.json")["scores"]
    lignes.append(("B", "Rice success, raw", [m["réussite – ECMWF brut (B)"]["BSS_vs_climatologie"] * 100]))
    lignes.append(("B", "Rice success, bias-corrected", [m["réussite – ECMWF corrigé (B')"]["BSS_vs_climatologie"] * 100]))
    lignes.append(("B", "Rice establishment, raw", [m["levée – ECMWF brut (B)"]["BSS_vs_climatologie"] * 100]))
    t = lire("validation/gain_maximal_terciles.json")["scores"]
    for k, lab in (("parfaite", "Perfect (1.00)"), ("très bonne (70 %)", "Very good (0.70)"),
                   ("réaliste (50 %)", "Realistic (0.50)"), ("sans qualité", "No skill (random 0.50)")):
        lignes.append(("C", lab, [t[k]["BSS"] * 100]))
    groupes = {"A": "Initial soil state\n(hindcasts 2006–2025)", "B": "ECMWF 0–7-day forecast\n(rice, 2024–2025)",
               "C": "Seasonal tercile oracle\n(rice, 2006–2025)"}
    fig, ax = plt.subplots(figsize=(7.1, 3.6))
    y = 0; yt, yl = [], []
    XMIN = -16
    for g in "ABC":
        for gg, lab, v in lignes:
            if gg != g:
                continue
            v = np.array(v)
            vc = np.clip(v, XMIN + 0.6, None)
            if len(v) > 1:
                ax.plot([v.min(), v.max()], [y, y], color="0.5", lw=2.5, solid_capstyle="butt")
            ax.scatter(vc, [y] * len(v), s=18, facecolor="black" if g != "C" else "white", edgecolor="black", zorder=3)
            for vi in v[v < XMIN]:
                ax.annotate(f"◄ {vi:.0f} % (off scale)".replace("-", "−"), (XMIN + 0.6, y), xytext=(6, -3),
                            textcoords="offset points", fontsize=7)
            yt.append(y); yl.append(lab); y -= 1
        ax.text(5.3, y + 1 + (sum(1 for l in lignes if l[0] == g) - 1) / 2, groupes[g], fontsize=7.5, va="center")
        y -= 0.6
    ax.axvline(0, color="black", lw=0.8)
    ax.text(0.25, y + 0.5, "= climatology", fontsize=6.5, va="bottom")
    ax.set_yticks(yt, yl)
    ax.set_xlim(XMIN, 5); ax.set_ylim(y + 0.3, 0.8)
    ax.set_xlabel("Brier skill score relative to climatology (%)")
    ax.grid(axis="x", lw=0.3, color="0.85")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.subplots_adjust(right=0.72, left=0.3)
    sauver(fig, "fig5_skill")


# ---------- Fig. 1 : zone d'étude (grille de calcul) ----------
def fig1():
    z = np.load(os.path.join(ICI, "donnees_bilan_marahoue.npz"))
    lat, lon, ru = z["lat"], z["lon"], z["ru_mm_m"]
    dates = z["dates"]
    ans = np.array([int(str(d)[:4]) for d in dates])
    pl = z["pluie"].astype(float)
    pa = np.array([np.nansum(pl[ans == a], 0) for a in range(2005, 2025)]).mean(0)
    io, jo = np.argsort(lat), np.argsort(lon)
    pa, ru = pa[io][:, jo], ru[io][:, jo]
    lat, lon = lat[io], lon[jo]
    bords_lat = np.r_[lat - 0.125, lat[-1] + 0.125]
    bords_lon = np.r_[lon - 0.125, lon[-1] + 0.125]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3))
    for ax, champ, titre, cmap, unite in ((axes[0], pa, "(a) Mean annual rainfall 2005–2024", "Greys", "mm"),
                                         (axes[1], ru, "(b) Available water capacity (SoilGrids)", "Greys", "mm m⁻¹")):
        im = ax.pcolormesh(bords_lon, bords_lat, champ, cmap=cmap, edgecolors="white", lw=0.6,
                           vmin=champ.min() - 0.4 * np.ptp(champ), vmax=champ.max())
        for i in range(6):
            for j in range(6):
                v = champ[i, j]
                ax.text(lon[j], lat[i], f"{v:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if v > champ.min() + 0.55 * np.ptp(champ) else "black")
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.12); cb.set_label(unite)
        ax.set_aspect(1 / np.cos(np.deg2rad(7.0)))
        ax.set_xticks(np.arange(-6.75, -5.24, 0.5), [f"{abs(v):.2f}°W" for v in np.arange(-6.75, -5.24, 0.5)])
        ax.set_yticks(np.arange(6.25, 7.76, 0.5), [f"{v:.2f}°N" for v in np.arange(6.25, 7.76, 0.5)])
        ax.set_title(titre, loc="left", fontsize=8)
        # nord et échelle
        ax.annotate("N", xy=(-5.05, 7.75), xytext=(-5.05, 7.55), ha="center", fontsize=8, fontweight="bold",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8))
        km = 50 / (111.32 * np.cos(np.deg2rad(7.0)))
        ax.plot([-5.25 - km, -5.25], [5.99, 5.99], color="black", lw=2, clip_on=False)
        ax.text(-5.25 - km / 2, 5.96, "50 km", ha="center", va="top", fontsize=6.5, clip_on=False)
        ax.set_ylim(6.25, 7.75); ax.set_xlim(-6.75, -5.25)
    fig.text(0.01, -0.02, "Geographic coordinates, WGS 84 (EPSG:4326). Cells: 0.25° TAMSAT-ALERT grid (36 cells).",
             fontsize=6.5)
    fig.tight_layout()
    sauver(fig, "fig1_study_area_grid")


# ---------- Fig. 2 : schéma de la méthode ----------
def fig2():
    fig, ax = plt.subplots(figsize=(7.1, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis("off")

    def boite(x, y, w, h, txt, gras=False, fond="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fond, ec="black", lw=0.7))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7, fontweight="bold" if gras else "normal")

    def fleche(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black"))
    boite(1, 54, 28, 7, "TAMSAT v3.2 rainfall\n(daily, 0.25°)")
    boite(36, 54, 28, 7, "NASA POWER\n→ ET₀ (FAO-56), Tmax")
    boite(71, 54, 28, 7, "SoilGrids 2.0\n→ available water capacity")
    boite(18, 40, 64, 7, "Daily layered water balance under fallow, 2005 → today\n→ soil water state on any day", gras=True, fond="0.92")
    for x in (15, 50, 85):
        fleche(x, 53.6, x if x == 50 else (30 if x == 15 else 70), 47.6)
    boite(18, 27, 64, 7, "Sowing simulation for each crop and cycle (Table 1)\n→ success or failure cause, WRSI", gras=True, fond="0.92")
    fleche(50, 39.6, 50, 34.6)
    boite(1, 12, 30, 9, "20 past seasons\n→ climatological\nsowing calendars (3.5)")
    boite(35, 12, 30, 9, "Today's soil state\n+ 20 rainfall scenarios\n→ weekly forecast (3.6)")
    boite(69, 12, 30, 9, "MODIS EVI anomalies\n→ independent\nsatellite check (3.7)")
    for x in (16, 50, 84):
        fleche(50 if x == 50 else (35 if x == 16 else 65), 26.6, x, 21.6)
    boite(1, 0.5, 98, 7, "Exp. 1  initial soil state  ·  Exp. 2  ECMWF 0–7-day forecasts  ·  Exp. 3  seasonal tercile oracle\n"
                         "Exp. 4  observed-rainfall sowing rule  ·  sensitivity analyses (17 variants)")
    fleche(50, 11.6, 50, 8.0)
    sauver(fig, "fig2_workflow")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5):
        f()
        print("ok", f.__name__)
