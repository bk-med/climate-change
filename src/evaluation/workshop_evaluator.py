import os
import json
from typing import Dict, List, Any
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class WorkshopEvaluator:
    """Evaluator for workshops on climate adaptation."""
    
    def __init__(self, output_dir: str = "workshop_output"):
        """Initializes the evaluator."""
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, "evaluations"), exist_ok=True)
    
    def evaluate_dialogue(self, dialogue_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates the workshop dialogue.
        Returns:
            Evaluation results
        """
        evaluation = {
            "date_evaluation": datetime.now().isoformat(),
            "metrics": {
                "participation": self._evaluate_participation(dialogue_history),
                "themes": self._analyze_themes(dialogue_history),
                "progression": self._evaluate_progression(dialogue_history)
            },
            "recommendations": self._generate_recommendations(dialogue_history)
        }
        
        return evaluation
    
    def _evaluate_participation(self, dialogue_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluates participation of different personas."""
        participation = {}
        total_interventions = 0
        
        # Count interventions per persona
        for stage in dialogue_history:
            dialogue = stage["dialogue"]
            for line in dialogue.split("\n"):
                if line.startswith("[") and "]:" in line:
                    persona = line[1:line.index("]")]
                    if persona.upper() == "FACILITATEUR" or persona.lower() == "facilitator":
                        persona = "MODERATOR"
                    if persona != "MODERATOR":
                        participation[persona] = participation.get(persona, 0) + 1
                        total_interventions += 1
                    else:
                        participation[persona] = participation.get(persona, 0) + 1
        
        # Calculate percentages
        if total_interventions > 0:
            for persona in participation:
                participation[persona] = (participation[persona] / total_interventions) * 100
                
        return participation
    
    def _analyze_themes(self, dialogue_history: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyzes the themes discussed in the dialogue."""
        themes = {
            "adaptation": 0,
            "vulnerabilite": 0,
            "resilience": 0,
            "collaboration": 0,
            "education": 0,
            "financement": 0
        }
        
        keywords = {
            "adaptation": ["adapter", "adaptation", "ajuster", "modifier"],
            "vulnerabilite": ["vulnerable", "risque", "menace", "danger"],
            "resilience": ["resilient", "resistant", "fort", "capable"],
            "collaboration": ["ensemble", "communaute", "collectif", "partager"],
            "education": ["apprendre", "former", "eduquer", "comprendre"],
            "financement": ["cout", "financer", "budget", "ressources"]
        }
        
        for stage in dialogue_history:
            dialogue = stage["dialogue"].lower()
            for theme, words in keywords.items():
                for word in words:
                    themes[theme] += dialogue.count(word)
        
        return themes
    
    def _evaluate_progression(self, dialogue_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluates progression through the workshop stages."""
        progression = {
            "completion": len(dialogue_history) / 4 * 100,  # 4 expected steps
            "stage_completion": {}
        }
        
        expected_topics = {
            "S1": ["presentation", "background"],
            "S2": ["impacts", "vulnerabilites"],
            "S3": ["solutions", "barrieres"],
            "S4": ["evaluation", "priorites"]
        }
        
        for stage in dialogue_history:
            stage_id = stage["stage"]
            dialogue = stage["dialogue"].lower()
            topics_covered = sum(1 for topic in expected_topics[stage_id] 
                               if any(topic in line.lower() for line in dialogue.split("\n")))
            progression["stage_completion"][stage_id] = (topics_covered / len(expected_topics[stage_id])) * 100
            
        return progression
    
    def _generate_recommendations(self, dialogue_history: List[Dict[str, Any]]) -> List[str]:
        """Generates recommendations based on dialogue analysis."""
        recommendations = []
        participation = self._evaluate_participation(dialogue_history)
        themes = self._analyze_themes(dialogue_history)
        
        # Recommendations based on participation
        min_participation = min(participation.values())
        if min_participation < 15:  # Less than 15% participation
            recommendations.append(
                "Encourage more balanced participation. Some participants "
                "might benefit from additional support to express themselves."
            )
        
        # Recommendations based on themes
        if themes["financement"] < 5:
            recommendations.append(
                "Deepen the discussion on financial aspects and resources "
                "needed to implement the proposed solutions."
            )
        
        if themes["education"] < 5:
            recommendations.append(
                "Strengthen the educational component and awareness in future sessions."
            )
            
        return recommendations
    
    def generate_report(self, evaluation: Dict[str, Any]) -> None:
        """Generates a visual evaluation report."""
        # Create visualizations
        self._plot_participation(evaluation["metrics"]["participation"])
        self._plot_themes(evaluation["metrics"]["themes"])
        self._plot_progression(evaluation["metrics"]["progression"])
        
        # Save the complete report
        report_path = os.path.join(self.output_dir, "evaluations", "rapport_evaluation.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=2)
            
        print(f"Evaluation report saved in {report_path}")
    
    def _plot_participation(self, participation: Dict[str, float]) -> None:
        """Creates a participation chart with MODERATOR in a different color."""
        plt.figure(figsize=(10, 6))
        personas = list(participation.keys())
        values = list(participation.values())
        colors = ["orange" if p == "MODERATOR" else "steelblue" for p in personas]
        plt.bar(personas, values, color=colors)
        plt.title("Participation by Persona")
        plt.ylabel("Participation Percentage")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "evaluations", "participation.png"))
        plt.close()
    
    def _plot_themes(self, themes: Dict[str, int]) -> None:
        """Creates a chart of discussed themes."""
        plt.figure(figsize=(10, 6))
        theme_names = list(themes.keys())
        counts = list(themes.values())
        
        plt.bar(theme_names, counts)
        plt.title("Theme Frequency")
        plt.ylabel("Occurrence Count")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "evaluations", "themes.png"))
        plt.close()
    
    def _plot_progression(self, progression: Dict[str, Any]) -> None:
        """Creates a chart of workshop progression."""
        plt.figure(figsize=(10, 6))
        stages = list(progression["stage_completion"].keys())
        completion = list(progression["stage_completion"].values())
        
        plt.plot(stages, completion, marker='o')
        plt.title("Progression by Step")
        plt.ylabel("Completion Percentage")
        plt.ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "evaluations", "progression.png"))
        plt.close() 