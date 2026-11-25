# Fenêtres de semis et de plantation — riz, maïs, arachide, igname — Marahoué (Côte d'Ivoire)

Page web : https://vamihermann17-pixel.github.io/semis-marahoue/

Probabilité qu'un semis de riz ou de maïs pluvial réussisse selon la date (1er mars – 31 juillet) et la
durée du cycle (riz : 100, 110, 120 j ; maïs : 90, 105, 120 j), calculée avec un **bilan hydrique
journalier propre à chaque culture** (`CULTURE=riz` ou `CULTURE=mais`).

Maïs (FAO-56) : 0–100 cm, Kc 0,4 → 1,2 → 0,5, racines de 10 à 100 cm, p = 0,55 ; échec si Ks < 0,7
3 jours de suite à la levée ou autour de la sortie des soies (55 % du cycle, −7 à +14 j).

Arachide (FAO-56) : 0–100 cm, Kc 0,4 → 1,15 → 0,6, racines de 10 à 80 cm, p = 0,50, cycles 90/105/120 j ;
échec si Ks < 0,5 5 jours de suite à la levée ou pendant floraison et gousses (45 % du cycle, −10 à +20 j),
ou si plus de 80 mm de pluie tombent dans les 15 jours autour de la maturité (récolte sous la pluie).

Igname : plantation du 1er décembre au 31 mai, cycles 210/240/270 j comptés depuis la sortie des pousses,
0–60 cm, Kc 0,4 → 1,05 → 0,6, p = 0,50. La semenceau reste en dormance : pousses au plus tôt 30 j après la
plantation, quand le sol 0–30 cm est à moitié plein, au plus tard 100 j ; échec si Ks < 0,5 10 jours de suite
sur les jeunes pousses ou pendant la tubérisation (40–75 % du cycle), ou WRSI < 75.

## Méthode
- `scripts/bilan_riz.py` : bilan hydrique FAO-56 sur 0–60 cm (3 couches). Kc du riz 0,5 → 1,15 → 0,8,
  racines de 10 à 60 cm, stress dès que 20 % de l'eau utile des racines est consommée (p = 0,2),
  indice de satisfaction des besoins en eau WRSI. Réussite : germination dans les 10 jours, pas de
  stress fort (Ks < 0,5) 5 jours de suite à la levée ni autour de la floraison (cycle − 35 j),
  moins de 3 jours à Tmax ≥ 35 °C à la floraison, WRSI ≥ 75.
- `scripts/fenetres_riz.py` : calendrier climatologique 2005–2024 et prévision. La prévision part de
  l'état du sol d'aujourd'hui (bilan continu sur les pluies observées), puis rejoue la météo de chaque
  saison passée (un scénario par année, principe de TAMSAT-ALERT).
- `scripts/donnees_bilan.py` : pluie TAMSAT v3.2 (Université de Reading), ET0 Penman-Monteith et Tmax
  à partir de NASA POWER, réserve utile du sol SoilGrids 2.0 (ISRIC) et Saxton & Rawls (2006).
  Les données déjà téléchargées sont gardées dans `scripts/cache_bilan/`.

## Mise à jour
`.github/workflows/mise-a-jour.yml` : chaque lundi à 5 h UTC, téléchargement des nouvelles données,
calcul et republication. Lancement manuel : onglet **Actions** → « Mise à jour de la prévision » →
**Run workflow**.

## Limites
Paramètres du riz issus de la littérature, non calés au champ ; mailles de 0,25° (≈ 28 km) ;
scénarios non pondérés par les prévisions saisonnières. Résultats indicatifs.
