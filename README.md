# Comparative Telemetry Analysis System (V-A-G)

Système d'analyse comparative de télémétrie F1 entre deux pilotes : extraction FastF1, resynchronisation spatiale, calculs physiques (Gx, Gy, delta temps) et visualisation Streamlit + Plotly.

## Installation

```bash
pip install -r requirements.txt
```

## Cache FastF1

Le cache est stocké dans le dossier `data/` à la racine du projet. Il est créé automatiquement au premier chargement de session. Conserver ce dossier évite de solliciter l'API à chaque exécution.

## Lancement du dashboard

Depuis la racine du projet :

```bash
streamlit run app/dashboard.py
```

## Architecture

- **data/** : cache local FastF1
- **src/data_loader.py** : connexion API FastF1, cache, extraction des tours les plus rapides et normalisation du schéma
- **src/physics.py** : calculs purs (Gx, Gy, filtrage Butterworth)
- **src/processor.py** : resynchronisation sur la distance, calcul du time delta
- **app/dashboard.py** : point d'entrée Streamlit (sélecteurs, graphiques synchronisés, cercle de Kamm)
- **app/components.py** : composants Plotly réutilisables

Flux : **Extraction** → **Resync + Physique** → **Visualisation**
