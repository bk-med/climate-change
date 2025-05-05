# Module de Création de Personas

Ce module fait partie du framework d'ateliers d'adaptation climatique et est responsable de la génération de personas réalistes et scientifiquement précis pour les simulations d'ateliers.

## Fonctionnalités

- Génération de personas basée sur des profils démographiques et de localisation
- Intégration avec Azure OpenAI pour la génération de contenu
- Sauvegarde structurée des personas avec métadonnées et biographies
- Validation des personas générés

## Structure des Données

Chaque persona généré contient :

- **Profil Personnel**
  - Nom
  - Âge
  - Occupation
  - Localisation

- **Facteurs de Vulnérabilité**
  - Exposition physique
  - Sensibilité
  - Capacité d'adaptation

- **Impacts Climatiques**
  - Types d'impacts
  - Descriptions
  - Niveaux de sévérité

- **Contexte d'Adaptation**
  - Facteurs facilitateurs
  - Barrières
  - Options d'adaptation

- **Biographie et Citations**
  - Histoire personnelle
  - Citations représentatives

## Utilisation

```python
from persona_development.persona_generator import PersonaGenerator

# Initialiser le générateur
generator = PersonaGenerator()

# Définir les profils cibles
target_profiles = [
    {"demographic": "elderly", "location": "coastal", "count": 2},
    {"demographic": "low_income", "location": "flood_zone", "count": 2},
    {"demographic": "fishing_industry", "location": "coastal", "count": 2}
]

# Générer les personas
personas = generator.generate_persona_set(target_profiles)

# Sauvegarder les personas
generator.save_personas(personas, "data/personas")
```

## Configuration

Le module utilise Azure OpenAI pour la génération de contenu. Les variables d'environnement suivantes doivent être configurées :

```bash
AZURE_OPENAI_API_KEY=votre_clé_api
AZURE_OPENAI_ENDPOINT=votre_endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=nom_de_votre_déploiement
AZURE_OPENAI_API_VERSION=version_api
```

## Tests

Les tests peuvent être exécutés avec pytest :

```bash
pytest tests/test_persona_generator.py
```

## Structure des Fichiers

```
persona_development/
├── persona_generator.py    # Générateur principal de personas
├── README.md              # Documentation
└── __init__.py           # Initialisation du module
```

## Intégration

Ce module est conçu pour s'intégrer avec les autres composants du framework :
- Fournit des personas pour le module de simulation de dialogue
- Utilise les configurations d'atelier du module d'initialisation
- Contribue aux évaluations du framework d'évaluation 