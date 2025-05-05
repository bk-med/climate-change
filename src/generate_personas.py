from persona_development.persona_generator import PersonaGenerator

def main():
    """Génère les personas pour l'atelier."""
    # Définir les profils des personas souhaités
    target_profiles = [
        {
            "demographic": "personne âgée vivant seule",
            "location": "zone côtière vulnérable",
            "type": "human"
        },
        {
            "demographic": "famille monoparentale avec trois enfants",
            "location": "quartier sujet aux inondations",
            "type": "human"
        },
        {
            "demographic": "pêcheur traditionnel",
            "location": "port de pêche historique",
            "type": "human"
        },
        {
            "demographic": "récif corallien et écosystème marin",
            "location": "zone côtière protégée",
            "type": "non_human"
        }
    ]
    
    # Créer le générateur et générer les personas
    generator = PersonaGenerator(output_dir="workshop_output/personas")
    personas = generator.generate_persona_set(target_profiles)
    generator.save_index(personas)
    
    print(f"\nPersonas générés avec succès : {len(personas)} personas créés")
    for persona in personas:
        print(f"- {persona['nom']}")

if __name__ == "__main__":
    main() 