import pytest
from src.persona_development.persona_generator import PersonaGenerator, Persona

def test_persona_generator_initialization():
    """Teste l'initialisation du générateur de personas."""
    generator = PersonaGenerator()
    assert generator is not None
    assert hasattr(generator, 'system_prompt')

def test_persona_generation():
    """Teste la génération d'un persona."""
    generator = PersonaGenerator()
    persona = generator.generate_persona("elderly", "coastal")
    
    # Vérifier que le persona a toutes les propriétés requises
    assert isinstance(persona, Persona)
    assert persona.id is not None
    assert persona.name is not None
    assert isinstance(persona.age, int)
    assert persona.occupation is not None
    assert persona.location is not None
    assert isinstance(persona.vulnerability_factors, dict)
    assert isinstance(persona.climate_impacts, list)
    assert isinstance(persona.adaptation_context, dict)
    assert persona.biography is not None
    assert isinstance(persona.quotes, list)

def test_persona_set_generation():
    """Teste la génération d'un ensemble de personas."""
    generator = PersonaGenerator()
    target_profiles = [
        {"demographic": "elderly", "location": "coastal", "count": 2},
        {"demographic": "low_income", "location": "flood_zone", "count": 1}
    ]
    
    personas = generator.generate_persona_set(target_profiles)
    
    # Vérifier le nombre de personas générés
    assert len(personas) == 3
    
    # Vérifier que chaque persona est unique
    persona_ids = [p.id for p in personas]
    assert len(set(persona_ids)) == len(persona_ids)

def test_persona_saving(tmp_path):
    """Teste la sauvegarde des personas."""
    generator = PersonaGenerator()
    target_profiles = [
        {"demographic": "elderly", "location": "coastal", "count": 1}
    ]
    
    personas = generator.generate_persona_set(target_profiles)
    output_dir = tmp_path / "personas"
    
    # Sauvegarder les personas
    generator.save_personas(personas, str(output_dir))
    
    # Vérifier que les fichiers ont été créés
    assert output_dir.exists()
    assert (output_dir / "persona_index.json").exists()
    
    persona_dir = output_dir / personas[0].id
    assert persona_dir.exists()
    assert (persona_dir / "metadata.json").exists()
    assert (persona_dir / "biography.md").exists() 