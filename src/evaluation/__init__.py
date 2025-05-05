"""Evaluation module for the climate adaptation workshop.

This module provides tools to evaluate different aspects of the workshop:
- Thematic evolution
- Topic coherence
- Persona fidelity
"""

from .thematic_evolution import ThematicEvolutionTracker
from .topic_coherence import TopicCoherenceScorer
from .argument_structure import ArgumentStructureAnalyzer
from .persona_fidelity import PersonaFidelityVerifier

__all__ = [
    'ThematicEvolutionTracker',
    'TopicCoherenceScorer',
    'ArgumentStructureAnalyzer',
    'PersonaFidelityVerifier'
] 