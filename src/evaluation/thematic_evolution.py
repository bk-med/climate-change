import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

class ThematicEvolutionTracker:
    """Analyzes the evolution of themes throughout the workshop."""
    
    def __init__(self):
        """Initializes the thematic tracker."""
        self.themes_principaux = {
            "adaptation": ["adapter", "adaptation", "ajuster", "modifier", "changement"],
            "vulnerabilite": ["vulnerable", "risque", "menace", "danger", "exposition"],
            "resilience": ["resilient", "resistant", "fort", "capable", "rebondir"],
            "collaboration": ["ensemble", "communaute", "collectif", "partager", "cooperation"],
            "education": ["apprendre", "former", "eduquer", "comprendre", "sensibiliser"],
            "financement": ["cout", "financer", "budget", "ressources", "investir"],
            "environnement": ["ecosysteme", "biodiversite", "nature", "habitat", "preservation"],
            "social": ["equite", "justice", "inclusion", "solidarite", "participation"]
        }
        self.vectorizer = TfidfVectorizer(max_features=1000)
    
    def analyze_evolution(self, dialogue_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes the thematic evolution across the workshop stages.
        
        Args:
            dialogue_history: Dialogue history by stage
        Returns:
            Thematic analysis results
        """
        evolution = {
            "date_analyse": datetime.now().isoformat(),
            "themes_par_etape": self._analyze_themes_by_stage(dialogue_history),
            "flux_thematique": self._calculate_theme_flow(dialogue_history),
            "coherence_thematique": self._evaluate_thematic_coherence(dialogue_history),
            "emergence_themes": self._detect_emerging_themes(dialogue_history)
        }
        
        return evolution
    
    def _analyze_themes_by_stage(self, dialogue_history: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyzes the presence of themes in each stage."""
        themes_by_stage = {}
        
        for stage in dialogue_history:
            stage_id = stage["stage"]
            dialogue = stage["dialogue"].lower()
            
            # Calculate the frequency of themes
            theme_frequencies = {}
            for theme, keywords in self.themes_principaux.items():
                frequency = sum(dialogue.count(keyword) for keyword in keywords)
                theme_frequencies[theme] = frequency
            
            # Normalize frequencies
            total = sum(theme_frequencies.values()) or 1
            themes_by_stage[stage_id] = {
                theme: freq/total for theme, freq in theme_frequencies.items()
            }
        
        return themes_by_stage
    
    def _calculate_theme_flow(self, dialogue_history: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Calculates the evolution of themes over time."""
        theme_flow = defaultdict(list)
        
        for stage in dialogue_history:
            dialogue = stage["dialogue"].lower()
            for theme, keywords in self.themes_principaux.items():
                theme_strength = sum(dialogue.count(keyword) for keyword in keywords)
                theme_flow[theme].append(theme_strength)
        
        return dict(theme_flow)
    
    def _evaluate_thematic_coherence(self, dialogue_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluates thematic coherence between stages."""
        coherence = {}
        dialogues = [stage["dialogue"].lower() for stage in dialogue_history]
        
        # Vectorize dialogues
        try:
            tfidf_matrix = self.vectorizer.fit_transform(dialogues)
            
            # Calculate cosine similarity between consecutive steps
            for i in range(len(dialogues)-1):
                stage_id = f"S{i+1}_S{i+2}"
                similarity = np.dot(tfidf_matrix[i].toarray(), tfidf_matrix[i+1].toarray().T)[0][0]
                coherence[stage_id] = float(similarity)
        except:
            # Fallback if vectorization fails
            coherence = {f"S{i+1}_S{i+2}": 0.5 for i in range(len(dialogues)-1)}
        
        return coherence
    
    def _detect_emerging_themes(self, dialogue_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detects unexpected emerging themes."""
        emerging_themes = []
        all_text = " ".join(stage["dialogue"].lower() for stage in dialogue_history)
        
        # List of words to exclude (main themes and common words)
        exclude_words = set()
        for keywords in self.themes_principaux.values():
            exclude_words.update(keywords)
        
        # Vectorize and identify frequent terms
        try:
            vectorizer = TfidfVectorizer(max_features=20, stop_words='french')
            tfidf_matrix = vectorizer.fit_transform([all_text])
            
            # Retrieve the most important terms
            feature_array = np.array(vectorizer.get_feature_names_out())
            tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
            
            # Filter and format emerging themes
            for idx in tfidf_sorting[:5]:  # Top 5 terms
                term = feature_array[idx]
                if term not in exclude_words:
                    emerging_themes.append({
                        "terme": term,
                        "importance": float(tfidf_matrix.toarray()[0][idx])
                    })
        except:
            # Fallback if detection fails
            emerging_themes = []
        
        return emerging_themes
    
    def generate_visualizations(self, evolution: Dict[str, Any], output_dir: str) -> None:
        """Generates visualizations of thematic evolution."""
        # 1. Theme evolution by step
        self._plot_theme_evolution(evolution["themes_par_etape"], output_dir)
        
        # 2. Thematic flow
        self._plot_theme_flow(evolution["flux_thematique"], output_dir)
        
        # 3. Thematic coherence
        self._plot_coherence(evolution["coherence_thematique"], output_dir)
    
    def _plot_theme_evolution(self, themes_by_stage: Dict[str, Dict[str, float]], output_dir: str) -> None:
        """Creates a plot of theme evolution by stage."""
        plt.figure(figsize=(12, 6))
        stages = list(themes_by_stage.keys())
        themes = list(self.themes_principaux.keys())
        
        for theme in themes:
            values = [themes_by_stage[stage][theme] for stage in stages]
            plt.plot(stages, values, marker='o', label=theme)
        
        plt.title("Theme Evolution by Step")
        plt.xlabel("Stages")
        plt.ylabel("Relative Importance")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(f"{output_dir}/theme_evolution.png", bbox_inches='tight')
        plt.close()
    
    def _plot_theme_flow(self, theme_flow: Dict[str, List[float]], output_dir: str) -> None:
        """Creates a plot of thematic flow."""
        plt.figure(figsize=(12, 6))
        
        for theme, values in theme_flow.items():
            plt.plot(range(1, len(values) + 1), values, marker='o', label=theme)
        
        plt.title("Thematic Flow")
        plt.xlabel("Workshop Progression")
        plt.ylabel("Intensity")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(f"{output_dir}/theme_flow.png", bbox_inches='tight')
        plt.close()
    
    def _plot_coherence(self, coherence: Dict[str, float], output_dir: str) -> None:
        """Creates a plot of thematic coherence."""
        plt.figure(figsize=(8, 6))
        
        transitions = list(coherence.keys())
        values = list(coherence.values())
        
        plt.bar(transitions, values)
        plt.title("Thematic Coherence between Steps")
        plt.xlabel("Transitions")
        plt.ylabel("Coherence Score")
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/theme_coherence.png")
        plt.close() 