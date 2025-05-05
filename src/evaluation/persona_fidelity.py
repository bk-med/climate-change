from typing import List, Dict, Any
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class PersonaFidelityVerifier:
    """Vérifie la fidélité des personas dans les discussions de l'atelier."""
    
    def __init__(self):
        """Initialise le vérificateur de fidélité des personas."""
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.characteristic_weights = {
            "age": 0.15,
            "occupation": 0.2,
            "location": 0.15,
            "vulnerabilite": 0.25,
            "adaptation": 0.25
        }
    
    def verify_fidelity(self, personas: List[Dict[str, Any]], dialogue_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Vérifie la fidélité des personas dans le dialogue.
        
        Args:
            personas: Liste des personas définis
            dialogue_history: Historique complet du dialogue
            
        Returns:
            Résultats de la vérification
        """
        verification = {
            "date_verification": datetime.now().isoformat(),
            "coherence_individuelle": self._evaluate_individual_coherence(personas, dialogue_history),
            "fidelite_caracteristiques": self._verify_characteristic_fidelity(personas, dialogue_history),
            "evolution_roles": self._analyze_role_evolution(personas, dialogue_history),
            "interactions": self._analyze_persona_interactions(personas, dialogue_history)
        }
        
        return verification
    
    def _evaluate_individual_coherence(self, personas: List[Dict[str, Any]], 
                                     dialogue_history: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Évalue la cohérence individuelle de chaque persona."""
        coherence = {}
        
        for persona in personas:
            persona_name = persona["name"]
            persona_interventions = self._extract_persona_interventions(persona_name, dialogue_history)
            
            if persona_interventions:
                # Calculer la cohérence lexicale
                try:
                    tfidf_matrix = self.vectorizer.fit_transform(persona_interventions)
                    similarities = cosine_similarity(tfidf_matrix)
                    np.fill_diagonal(similarities, 0)
                    lexical_coherence = float(similarities.mean())
                except:
                    lexical_coherence = 0.5
                
                # Calculer la cohérence thématique
                thematic_coherence = self._calculate_thematic_coherence(
                    persona_interventions,
                    persona["vulnerabilite"] + " " + persona["adaptation"]
                )
                
                coherence[persona_name] = {
                    "coherence_lexicale": lexical_coherence,
                    "coherence_thematique": thematic_coherence,
                    "score_global": (lexical_coherence + thematic_coherence) / 2
                }
            else:
                coherence[persona_name] = {
                    "coherence_lexicale": 0.0,
                    "coherence_thematique": 0.0,
                    "score_global": 0.0
                }
        
        return coherence
    
    def _verify_characteristic_fidelity(self, personas: List[Dict[str, Any]], 
                                      dialogue_history: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Vérifie la fidélité aux caractéristiques définies des personas."""
        fidelity = {}
        
        for persona in personas:
            persona_name = persona["name"]
            interventions = self._extract_persona_interventions(persona_name, dialogue_history)
            
            if interventions:
                # Calculer les scores de fidélité pour chaque caractéristique
                characteristic_scores = {}
                
                for characteristic, weight in self.characteristic_weights.items():
                    if characteristic in persona:
                        score = self._calculate_characteristic_fidelity(
                            interventions,
                            str(persona[characteristic])
                        )
                        characteristic_scores[characteristic] = score
                
                # Calculer le score global pondéré
                if characteristic_scores:
                    total_weight = sum(self.characteristic_weights[c] for c in characteristic_scores)
                    weighted_score = sum(
                        score * self.characteristic_weights[char] / total_weight
                        for char, score in characteristic_scores.items()
                    )
                    
                    fidelity[persona_name] = {
                        "scores_caracteristiques": characteristic_scores,
                        "score_global": weighted_score
                    }
                else:
                    fidelity[persona_name] = {
                        "scores_caracteristiques": {},
                        "score_global": 0.0
                    }
            else:
                fidelity[persona_name] = {
                    "scores_caracteristiques": {},
                    "score_global": 0.0
                }
        
        return fidelity
    
    def _analyze_role_evolution(self, personas: List[Dict[str, Any]], 
                              dialogue_history: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Analyse l'évolution des rôles des personas au fil du dialogue."""
        evolution = {}
        
        for persona in personas:
            persona_name = persona["name"]
            stage_analysis = []
            
            for i, stage in enumerate(dialogue_history):
                interventions = self._extract_persona_interventions(persona_name, [stage])
                
                if interventions:
                    # Analyser la participation
                    participation = len(interventions)
                    
                    # Analyser la pertinence thématique
                    thematic_relevance = self._calculate_thematic_coherence(
                        interventions,
                        persona["vulnerabilite"] + " " + persona["adaptation"]
                    )
                    
                    # Analyser le niveau d'engagement
                    engagement = self._calculate_engagement_level(interventions)
                    
                    stage_analysis.append({
                        "etape": f"S{i+1}",
                        "participation": participation,
                        "pertinence": thematic_relevance,
                        "engagement": engagement
                    })
                else:
                    stage_analysis.append({
                        "etape": f"S{i+1}",
                        "participation": 0,
                        "pertinence": 0.0,
                        "engagement": 0.0
                    })
            
            evolution[persona_name] = stage_analysis
        
        return evolution
    
    def _analyze_persona_interactions(self, personas: List[Dict[str, Any]], 
                                    dialogue_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse les interactions entre personas."""
        interactions = {
            "matrice_interactions": self._create_interaction_matrix(personas, dialogue_history),
            "patterns_interaction": self._identify_interaction_patterns(personas, dialogue_history)
        }
        
        return interactions
    
    def _extract_persona_interventions(self, persona_name: str, 
                                     dialogue_history: List[Dict[str, Any]]) -> List[str]:
        """Extrait les interventions d'un persona spécifique."""
        interventions = []
        
        for stage in dialogue_history:
            dialogue = stage["dialogue"]
            lines = dialogue.split("\n")
            
            current_intervention = []
            is_persona_speaking = False
            
            for line in lines:
                if line.startswith(f"[{persona_name}]"):
                    is_persona_speaking = True
                    current_intervention = [line[line.index(":")+1:].strip()]
                elif line.startswith("[") and "]:" in line:
                    if is_persona_speaking and current_intervention:
                        interventions.append(" ".join(current_intervention))
                    is_persona_speaking = False
                    current_intervention = []
                elif is_persona_speaking and line.strip():
                    current_intervention.append(line.strip())
            
            if is_persona_speaking and current_intervention:
                interventions.append(" ".join(current_intervention))
        
        return interventions
    
    def _calculate_thematic_coherence(self, interventions: List[str], reference: str) -> float:
        """Calcule la cohérence thématique entre les interventions et une référence."""
        try:
            # Ajouter la référence aux textes à vectoriser
            texts = interventions + [reference]
            
            # Vectoriser les textes
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculer la similarité moyenne avec la référence
            similarities = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1:])
            coherence = float(similarities.mean())
            
            return coherence
        except:
            return 0.5  # Valeur par défaut en cas d'erreur
    
    def _calculate_characteristic_fidelity(self, interventions: List[str], 
                                         characteristic: str) -> float:
        """Calcule la fidélité à une caractéristique spécifique."""
        try:
            # Vectoriser les interventions et la caractéristique
            texts = interventions + [characteristic]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculer la similarité moyenne avec la caractéristique
            similarities = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1:])
            fidelity = float(similarities.mean())
            
            return fidelity
        except:
            return 0.5  # Valeur par défaut en cas d'erreur
    
    def _calculate_engagement_level(self, interventions: List[str]) -> float:
        """Calcule le niveau d'engagement basé sur les interventions."""
        if not interventions:
            return 0.0
        
        total_words = sum(len(intervention.split()) for intervention in interventions)
        avg_words = total_words / len(interventions)
        
        # Normaliser le score (supposer qu'une intervention moyenne a 20 mots)
        engagement = min(avg_words / 20, 1.0)
        
        return engagement
    
    def _create_interaction_matrix(self, personas: List[Dict[str, Any]], 
                                 dialogue_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Crée une matrice des interactions entre personas."""
        persona_names = [p["name"] for p in personas]
        n_personas = len(persona_names)
        
        # Initialiser la matrice d'interactions
        interaction_matrix = np.zeros((n_personas, n_personas))
        
        # Analyser les interactions
        for stage in dialogue_history:
            dialogue = stage["dialogue"]
            lines = dialogue.split("\n")
            
            last_speaker = None
            for line in lines:
                if line.startswith("[") and "]:" in line:
                    current_speaker = line[1:line.index("]")]
                    
                    if last_speaker and last_speaker in persona_names and current_speaker in persona_names:
                        i = persona_names.index(last_speaker)
                        j = persona_names.index(current_speaker)
                        interaction_matrix[i][j] += 1
                    
                    last_speaker = current_speaker
        
        return {
            "personas": persona_names,
            "matrix": interaction_matrix.tolist()
        }
    
    def _identify_interaction_patterns(self, personas: List[Dict[str, Any]], 
                                    dialogue_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identifie les patterns d'interaction récurrents entre personas."""
        patterns = []
        
        # Extraire les séquences d'interactions
        interaction_sequences = []
        current_sequence = []
        
        for stage in dialogue_history:
            dialogue = stage["dialogue"]
            lines = dialogue.split("\n")
            
            for line in lines:
                if line.startswith("[") and "]:" in line:
                    speaker = line[1:line.index("]")]
                    if speaker not in ["FACILITATEUR", "Facilitateur"]:
                        current_sequence.append(speaker)
                        
                        if len(current_sequence) >= 3:
                            pattern = {
                                "sequence": current_sequence[-3:],
                                "frequence": 1
                            }
                            
                            # Vérifier si le pattern existe déjà
                            existing = next(
                                (p for p in patterns if p["sequence"] == pattern["sequence"]),
                                None
                            )
                            
                            if existing:
                                existing["frequence"] += 1
                            else:
                                patterns.append(pattern)
        
        return sorted(patterns, key=lambda x: x["frequence"], reverse=True)
    
    def generate_visualizations(self, verification: Dict[str, Any], output_dir: str) -> None:
        """Génère des visualisations pour la vérification de fidélité."""
        # 1. Graphique de la cohérence individuelle
        self._plot_individual_coherence(
            verification["coherence_individuelle"],
            output_dir
        )
        
        # 2. Graphique de la fidélité aux caractéristiques
        self._plot_characteristic_fidelity(
            verification["fidelite_caracteristiques"],
            output_dir
        )
        
        # 3. Graphique de l'évolution des rôles
        self._plot_role_evolution(
            verification["evolution_roles"],
            output_dir
        )
        
        # 4. Graphique des interactions
        self._plot_interaction_matrix(
            verification["interactions"]["matrice_interactions"],
            output_dir
        )
    
    def _plot_individual_coherence(self, coherence: Dict[str, Dict[str, float]], 
                                 output_dir: str) -> None:
        """Crée un graphique de la cohérence individuelle des personas."""
        plt.figure(figsize=(12, 6))
        
        personas = list(coherence.keys())
        lexical_scores = [c["coherence_lexicale"] for c in coherence.values()]
        thematic_scores = [c["coherence_thematique"] for c in coherence.values()]
        global_scores = [c["score_global"] for c in coherence.values()]
        
        x = np.arange(len(personas))
        width = 0.25
        
        plt.bar(x - width, lexical_scores, width, label='Cohérence Lexicale')
        plt.bar(x, thematic_scores, width, label='Cohérence Thématique')
        plt.bar(x + width, global_scores, width, label='Score Global')
        
        plt.xlabel('Personas')
        plt.ylabel('Scores')
        plt.title('Cohérence Individuelle des Personas')
        plt.xticks(x, personas, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/individual_coherence.png")
        plt.close()
    
    def _plot_characteristic_fidelity(self, fidelity: Dict[str, Dict[str, Any]], 
                                    output_dir: str) -> None:
        """Crée un graphique de la fidélité aux caractéristiques."""
        plt.figure(figsize=(12, 6))
        
        personas = list(fidelity.keys())
        characteristics = list(self.characteristic_weights.keys())
        
        data = np.array([[
            fidelity[p]["scores_caracteristiques"].get(c, 0)
            for c in characteristics
        ] for p in personas])
        
        im = plt.imshow(data, cmap='YlOrRd')
        
        plt.colorbar(im)
        plt.xticks(range(len(characteristics)), characteristics, rotation=45)
        plt.yticks(range(len(personas)), personas)
        
        plt.title('Fidélité aux Caractéristiques par Persona')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/characteristic_fidelity.png")
        plt.close()
    
    def _plot_role_evolution(self, evolution: Dict[str, List[Dict[str, Any]]], 
                           output_dir: str) -> None:
        """Crée un graphique de l'évolution des rôles."""
        plt.figure(figsize=(12, 6))
        
        personas = list(evolution.keys())
        stages = [stage["etape"] for stage in evolution[personas[0]]]
        
        for persona in personas:
            engagement = [stage["engagement"] for stage in evolution[persona]]
            plt.plot(stages, engagement, marker='o', label=persona)
        
        plt.xlabel('Étapes')
        plt.ylabel('Niveau d\'Engagement')
        plt.title('Évolution de l\'Engagement des Personas')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/role_evolution.png")
        plt.close()
    
    def _plot_interaction_matrix(self, interaction_data: Dict[str, Any], 
                               output_dir: str) -> None:
        """Crée une visualisation de la matrice d'interactions."""
        plt.figure(figsize=(10, 8))
        
        personas = interaction_data["personas"]
        matrix = np.array(interaction_data["matrix"])
        
        im = plt.imshow(matrix, cmap='YlOrRd')
        
        plt.colorbar(im)
        plt.xticks(range(len(personas)), personas, rotation=45)
        plt.yticks(range(len(personas)), personas)
        
        plt.title('Matrice des Interactions entre Personas')
        
        # Ajouter les valeurs dans les cellules
        for i in range(len(personas)):
            for j in range(len(personas)):
                text = plt.text(j, i, int(matrix[i, j]),
                              ha="center", va="center", color="black") 