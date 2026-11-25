# Validation du bilan hydrique du riz pluvial — Marahoué

État au 24 novembre 2026. Scripts : `scripts/validation_retrospective.py`, `scripts/validation_ndvi.py`, `scripts/validation_meteo10j.py`.

## 1. Test de prévision rétrospectif (2006–2025)

Pour chaque saison, on refait la prévision telle qu'elle aurait été émise le 1er décembre, 1er février,
1er mars, 1er avril et 1er mai : état du sol simulé à cette date, puis météo des **autres** saisons
(validation croisée). On compare à la réussite simulée avec la vraie météo de la saison. Cycle 110 j,
36 mailles, toutes les dates de semis du 1er mars au 31 juillet postérieures à l'émission.

| Émission | Gain sur la climatologie (BSS) | Aire ROC prévision | Aire ROC climatologie |
|---|---|---|---|
| 1er décembre | −0,001 | 0,736 | 0,737 |
| 1er février | +0,001 | 0,738 | 0,737 |
| 1er mars | +0,003 | 0,740 | 0,737 |
| 1er avril | 0,000 | 0,724 | 0,722 |
| 1er mai | +0,001 | 0,677 | 0,672 |

**Conclusion.** L'état du sol au moment de l'émission n'apporte pas d'information utile au-delà de la
climatologie : la réserve de 0–60 cm (environ 70 mm) se vide ou se remplit en quelques semaines.
La « prévision » actuelle équivaut donc à la climatologie de chaque date de semis. Un gain réel
viendra d'informations sur la météo à venir : prévisions saisonnières (pondération des scénarios
par les terciles PRESASS ou C3S) et prévisions à 1–2 semaines au moment de semer.

## 2. Comparaison avec la végétation observée par satellite (2005–2024)

Réponse indépendante des modèles : anomalies d'EVI MODIS (MOD13Q1, pixels de bonne qualité) sur la
végétation non arborée (champs, jachères, savanes : ESA WorldCover classes 20–40), août–octobre,
tendance retirée par maille. Corrélation de rang de Spearman.

| Indicateur | 720 mailles-années | Moyenne de zone (20 ans) | p (zone) |
|---|---|---|---|
| **Bilan riz : eau utile 0–30 cm, août–oct.** | **+0,46** | **+0,79** | < 0,001 |
| Bilan riz : WRSI médian, semis 15 mai–30 juin | +0,36 | +0,60 | 0,006 |
| Bilan riz : part de semis réussis, 15 mai–30 juin | +0,17 | +0,39 | 0,09 |
| beta TAMSAT-ALERT (C4, 1 m), août–oct. | +0,41 | +0,59 | 0,007 |
| Pluie TAMSAT juin–sept. | +0,39 | +0,60 | 0,006 |
| Pluie TAMSAT juillet | +0,36 | +0,62 | 0,004 |

**Conclusion.** Le bilan hydrique du riz reproduit mieux les bonnes et mauvaises années de la
végétation que beta ou la pluie seule. Le WRSI (continu) suit l'EVI ; la réussite oui/non, plus
grossière, beaucoup moins. **Limite :** l'EVI mesure toute la végétation non arborée, pas le riz ;
ce test valide la dynamique de l'eau du sol, pas les seuils de réussite du riz (germination,
floraison, WRSI ≥ 75). Seules des observations au champ peuvent valider ces seuils.

## 4. Prévisions météo à 10 jours (ECMWF) et règle de semis

**Test.** Saisons 2024 et 2025 (seules années où Open-Meteo archive les prévisions ECMWF IFS 0,25° par
échéance), 36 mailles, cycle 110 j. Pour chaque date de semis, la pluie des jours J à J+7 des
scénarios est remplacée par la prévision ECMWF émise le jour du semis. Script :
`scripts/validation_meteo10j.py`.

| Prévision | Score de Brier | Aire ROC | Gain sur la climatologie |
|---|---|---|---|
| Réussite, climatologie (prévision actuelle) | 0,196 | 0,750 | — |
| Réussite, avec ECMWF brut | 0,219 | 0,669 | −12 % |
| Réussite, avec ECMWF corrigé du biais | 0,219 | 0,681 | −12 % |
| Levée, climatologie | 0,047 | 0,875 | — |
| Levée, avec ECMWF brut | 0,070 | 0,546 | −49 % |

Pluie cumulée prévue contre pluie TAMSAT, corrélation de rang : 0,31 (J0–J1), 0,18 (J0–J3),
0,08 (J0–J7) à la maille ; ECMWF sous-estime la pluie de mars–juillet de 20 à 25 %.

**Conclusion.** Sur la Marahoué, la prévision de pluie au-delà de 1–2 jours n'a presque pas de
qualité, comme l'ont montré Vogel et al. (2018, 2020) pour l'Afrique tropicale du Nord. L'intégrer
dégraderait les probabilités : elle n'est pas utilisée. Une prévision d'ensemble post-traitée pourrait
apporter un léger gain ; à retester quand des archives d'ensemble seront disponibles.

**Règle de semis sur la pluie observée.** Semer seulement après ≥ 20 mm en 3 jours ou ≥ 40 mm en
10 jours (2005–2024, cycle 110 j) :

| Mois | Semer à date fixe | Après ≥ 20 mm en 3 j | Après ≥ 40 mm en 10 j |
|---|---|---|---|
| mars | 90 % | 94 % | 96 % |
| avril | 95 % | 95 % | 96 % |
| mai | 73 % | 76 % | 74 % |
| juin | 78 % | 78 % | 78 % |
| juillet | 86 % | 93 % | 88 % |

En mars, les échecs de levée passent de 6 % à 2 %. En mai–juin, la règle n'aide pas : les échecs
viennent de la sécheresse à la floraison. La page affiche chaque semaine si la règle est remplie,
maille par maille, avec la pluie TAMSAT des 10 derniers jours.

## 5. Protocole de validation au champ (saisons 2027 et 2028)

**Objectif.** Vérifier, sur parcelles paysannes, que la probabilité de réussite calculée pour une date
de semis correspond à la réussite observée, et caler les seuils du modèle (stress à la levée et à la
floraison, WRSI minimal), en particulier pour les semis de mars et de mai–juin.

**Dispositif.**
- 30 parcelles de riz pluvial (plateau, hors bas-fonds) réparties dans 6 à 8 mailles de 28 km :
  Bouaflé, Sinfra, Zuénoula, Bonon, Gohitafla, avec l'ANADER et le CNRA.
- Dans chaque village, 3 dates de semis échelonnées si possible (par exemple mi-mars, mi-avril, début juin),
  même variété (NERICA de 100 ou 110 j), même conduite.
- Un pluviomètre manuel par village, relevé chaque matin.

**Observations par parcelle.**
| Stade | À relever |
|---|---|
| Semis | date, variété, densité, humidité du sol au toucher (sec / frais / humide) |
| Levée | date de levée, plants levés sur 3 placettes de 1 m² à 15 jours, ressemis éventuel |
| Végétation | tous les 15 jours : flétrissement (oui / non), hauteur, photo |
| Floraison | date de 50 % de floraison ; jours de flétrissement dans les 2 semaines autour |
| Récolte | date ; rendement grain sur 3 placettes de 5 m² ; taux de grains vides |
| Sol | une fois : texture (test au toucher ou analyse), profondeur utile |

**Analyse.**
1. Réussite observée = levée ≥ 70 % des plants attendus et rendement ≥ 50 % du rendement de
   référence de la variété (à fixer avec le CNRA).
2. Comparer à la probabilité du modèle pour la même maille et la même date de semis : score de
   Brier, courbe ROC, diagramme de fiabilité.
3. Relancer le modèle avec la pluie du pluviomètre et le sol observé, puis caler les seuils
   (stress à la levée, à la floraison, WRSI minimal) par validation croisée entre villages.
4. Comparer le rendement observé au WRSI simulé (relation attendue croissante).

**Calendrier.** Recrutement des parcelles en janvier–février 2027 ; observations de mars à
décembre 2027 ; analyse en janvier 2028 ; seconde saison en 2028 pour confirmer.
