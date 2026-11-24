# Fenêtres de semis du riz pluvial — Marahoué (Côte d'Ivoire)

Page web : probabilité de réussite d'un semis de riz pluvial selon la date, la durée du cycle
(100, 110 ou 120 jours) et la dernière prévision d'humidité du sol TAMSAT-ALERT.

- `docs/` : la page (`index.html`) et ses données, publiées par GitHub Pages.
  - `calendriers_riz.json` : climatologie 2005–2024 (fixe).
  - `prevision_riz_derniere.json` : dernière prévision, remplacée chaque lundi.
- `scripts/talert_marahoue.py` : lit la prévision TAMSAT-ALERT (Afrique, 0,25°, 15 scénarios,
  150 jours) par requêtes HTTP partielles sur la seule zone d'étude et calcule les probabilités.
- `scripts/calibration_humidite_riz.json` : seuils d'humidité du sol calibrés sur 2005–2024.
- `.github/workflows/mise-a-jour.yml` : chaque lundi à 5 h UTC, télécharge la prévision,
  enregistre les données et republie la page. Lancement manuel : onglet **Actions** →
  « Mise à jour de la prévision » → **Run workflow**.

Données : TAMSAT, Université de Reading (CC BY 4.0) ; CHIRPS (UCSB-CHG).
Résultats indicatifs : à valider par des essais au champ.
