import re
from typing import List, Dict, Any
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer

class ArgumentStructureAnalyzer:
    """Analyse la structure des arguments dans les discussions de l'atelier."""
    
    def __init__(self):
        """Initialise l'analyseur de structure argumentative."""
        self.argument_markers = {
            "proposition": ["je propose", "nous devrions", "il faudrait", "suggère"],
            "support": ["parce que", "car", "puisque", "en effet"],
            "opposition": ["mais", "cependant", "toutefois", "néanmoins"],
            "conclusion": ["donc", "ainsi", "par conséquent", "en conclusion"],
            "exemple": ["par exemple", "notamment", "comme", "tel que"]
        }
        self.vectorizer = CountVectorizer(max_features=1000)
    
    def analyze_structure(self, dialogue_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyse la structure argumentative du dialogue.
        
        Args:
            dialogue_history: Historique complet du dialogue
            
        Returns:
            Résultats de l'analyse structurelle
        """
        analysis = {
            "date_analyse": datetime.now().isoformat(),
            "structure_globale": self._analyze_global_structure(dialogue_history),
            "chaines_argumentatives": self._identify_argument_chains(dialogue_history),
            "qualite_arguments": self._evaluate_argument_quality(dialogue_history),
            "interactions": self._analyze_interactions(dialogue_history)
        }
        
        return analysis
    
    def _analyze_global_structure(self, dialogue_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse la structure globale de l'argumentation."""
        structure = {
            "distribution_marqueurs": defaultdict(int),
            "complexite_argumentative": [],
            "equilibre_discussion": {}
        }
        
        total_interventions = 0
        persona_interventions = defaultdict(int)
        
        for stage in dialogue_history:
            dialogue = stage["dialogue"]
            lines = dialogue.split("\n")
            
            for line in lines:
                # Compter les interventions par persona
                if line.startswith("[") and "]:" in line:
                    persona = line[1:line.index("]")]
                    if persona not in ["FACILITATEUR", "Facilitateur"]:
                        persona_interventions[persona] += 1
                        total_interventions += 1
                
                # Analyser les marqueurs argumentatifs
                for type_arg, markers in self.argument_markers.items():
                    for marker in markers:
                        if marker.lower() in line.lower():
                            structure["distribution_marqueurs"][type_arg] += 1
                
                # Évaluer la complexité argumentative
                complexity = self._calculate_argument_complexity(line)
                structure["complexite_argumentative"].append(complexity)
        
        # Calculer l'équilibre des interventions
        if total_interventions > 0:
            structure["equilibre_discussion"] = {
                persona: count/total_interventions 
                for persona, count in persona_interventions.items()
            }
        
        return structure
    
    def _identify_argument_chains(self, dialogue_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identifie les chaînes argumentatives dans le dialogue."""
        chains = []
        current_chain = None
        
        for stage in dialogue_history:
            dialogue = stage["dialogue"]
            lines = dialogue.split("\n")
            
            for line in lines:
                # Détecter le début d'un nouvel argument
                if any(marker in line.lower() for marker in self.argument_markers["proposition"]):
                    if current_chain:
                        chains.append(current_chain)
                    current_chain = {
                        "debut": line,
                        "supports": [],
                        "oppositions": [],
                        "conclusion": None
                    }
                
                # Ajouter les éléments à la chaîne courante
                if current_chain:
                    if any(marker in line.lower() for marker in self.argument_markers["support"]):
                        current_chain["supports"].append(line)
                    elif any(marker in line.lower() for marker in self.argument_markers["opposition"]):
                        current_chain["oppositions"].append(line)
                    elif any(marker in line.lower() for marker in self.argument_markers["conclusion"]):
                        current_chain["conclusion"] = line
                        chains.append(current_chain)
                        current_chain = None
        
        # Ajouter la dernière chaîne si elle existe
        if current_chain:
            chains.append(current_chain)
        
        return chains
    
    def _evaluate_argument_quality(self, dialogue_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Évalue la qualité des arguments présentés."""
        quality = {
            "completude": [],  # Score de complétude pour chaque argument
            "coherence": [],   # Score de cohérence pour chaque argument
            "support": []      # Niveau de support pour chaque argument
        }
        
        for stage in dialogue_history:
            dialogue = stage["dialogue"]
            arguments = self._extract_arguments(dialogue)
            
            for arg in arguments:
                # Évaluer la complétude (présence des éléments clés)
                completude = self._calculate_completeness(arg)
                quality["completude"].append(completude)
                
                # Évaluer la cohérence
                coherence = self._calculate_coherence(arg)
                quality["coherence"].append(coherence)
                
                # Évaluer le niveau de support
                support = self._calculate_support_level(arg)
                quality["support"].append(support)
        
        return quality
    
    def _analyze_interactions(self, dialogue_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse les interactions entre les arguments."""
        interactions = {
            "graphe_interactions": self._create_interaction_graph(dialogue_history),
            "patterns": self._identify_interaction_patterns(dialogue_history)
        }
        
        return interactions
    
    def _calculate_argument_complexity(self, text: str) -> float:
        """Calcule la complexité d'un argument."""
        complexity = 0.0
        
        # Nombre de marqueurs argumentatifs différents
        marker_types = sum(1 for markers in self.argument_markers.values() 
                         if any(marker in text.lower() for marker in markers))
        
        # Longueur relative de l'argument
        words = len(text.split())
        
        # Formule de complexité
        complexity = (marker_types * 0.6 + min(words/50, 1) * 0.4)
        
        return complexity
    
    def _extract_arguments(self, text: str) -> List[Dict[str, str]]:
        """Extrait les arguments d'un texte."""
        arguments = []
        lines = text.split("\n")
        current_arg = None
        
        for line in lines:
            if any(marker in line.lower() for marker in self.argument_markers["proposition"]):
                if current_arg:
                    arguments.append(current_arg)
                current_arg = {"proposition": line, "support": [], "opposition": [], "conclusion": None}
            elif current_arg:
                if any(marker in line.lower() for marker in self.argument_markers["support"]):
                    current_arg["support"].append(line)
                elif any(marker in line.lower() for marker in self.argument_markers["opposition"]):
                    current_arg["opposition"].append(line)
                elif any(marker in line.lower() for marker in self.argument_markers["conclusion"]):
                    current_arg["conclusion"] = line
                    arguments.append(current_arg)
                    current_arg = None
        
        if current_arg:
            arguments.append(current_arg)
        
        return arguments
    
    def _calculate_completeness(self, argument: Dict[str, Any]) -> float:
        """Calcule le score de complétude d'un argument."""
        score = 0.0
        
        # Vérifier la présence des éléments essentiels
        if "proposition" in argument and argument["proposition"]:
            score += 0.4
        
        if "support" in argument and argument["support"]:
            score += 0.3
        
        if "conclusion" in argument and argument["conclusion"]:
            score += 0.3
        
        return score
    
    def _calculate_coherence(self, argument: Dict[str, Any]) -> float:
        """Calcule le score de cohérence d'un argument."""
        score = 0.0
        
        try:
            # Vectoriser les éléments de l'argument
            elements = [argument["proposition"]]
            if argument["support"]:
                elements.extend(argument["support"])
            if argument["conclusion"]:
                elements.append(argument["conclusion"])
            
            # Calculer la cohérence lexicale
            matrix = self.vectorizer.fit_transform(elements)
            words = self.vectorizer.get_feature_names_out()
            
            # Calculer le recouvrement lexical moyen
            total_overlap = 0
            comparisons = 0
            
            for i in range(len(elements)):
                for j in range(i+1, len(elements)):
                    overlap = len(set(elements[i].split()) & set(elements[j].split()))
                    total_overlap += overlap
                    comparisons += 1
            
            if comparisons > 0:
                score = min(total_overlap / (comparisons * 5), 1.0)  # Normaliser à 1
        except:
            score = 0.5  # Valeur par défaut en cas d'erreur
        
        return score
    
    def _calculate_support_level(self, argument: Dict[str, Any]) -> float:
        """Calcule le niveau de support d'un argument."""
        score = 0.0
        
        # Nombre de supports
        if "support" in argument:
            num_supports = len(argument["support"])
            score += min(num_supports * 0.3, 0.6)  # Maximum 0.6 pour les supports
        
        # Présence d'exemples
        has_example = any(
            any(marker in support.lower() for marker in self.argument_markers["exemple"])
            for support in argument.get("support", [])
        )
        if has_example:
            score += 0.2
        
        # Traitement des oppositions
        if "opposition" in argument and argument["opposition"]:
            score += 0.2  # Bonus pour la considération des contre-arguments
        
        return min(score, 1.0)
    
    def _create_interaction_graph(self, dialogue_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Crée un graphe des interactions entre arguments."""
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        # Extraire tous les arguments
        all_arguments = []
        for stage in dialogue_history:
            arguments = self._extract_arguments(stage["dialogue"])
            all_arguments.extend(arguments)
        
        # Créer les nœuds
        for i, arg in enumerate(all_arguments):
            node_id = f"A{i+1}"
            graph_data["nodes"].append({
                "id": node_id,
                "type": "proposition",
                "content": arg["proposition"][:50] + "..."
            })
        
        # Créer les arêtes
        for i, arg1 in enumerate(all_arguments):
            for j, arg2 in enumerate(all_arguments[i+1:], i+1):
                # Détecter les relations
                relation = self._detect_argument_relation(arg1, arg2)
                if relation:
                    graph_data["edges"].append({
                        "source": f"A{i+1}",
                        "target": f"A{j+1}",
                        "type": relation
                    })
        
        return graph_data
    
    def _detect_argument_relation(self, arg1: Dict[str, Any], arg2: Dict[str, Any]) -> str:
        """Détecte la relation entre deux arguments."""
        # Vérifier le support
        if any(support in arg2["proposition"].lower() for support in self.argument_markers["support"]):
            return "support"
        
        # Vérifier l'opposition
        if any(opp in arg2["proposition"].lower() for opp in self.argument_markers["opposition"]):
            return "opposition"
        
        # Vérifier la conclusion
        if any(conc in arg2["proposition"].lower() for conc in self.argument_markers["conclusion"]):
            return "conclusion"
        
        return None
    
    def _identify_interaction_patterns(self, dialogue_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identifie les patterns d'interaction récurrents."""
        patterns = []
        
        # Extraire les séquences d'interactions
        for stage in dialogue_history:
            dialogue = stage["dialogue"]
            lines = dialogue.split("\n")
            
            current_pattern = []
            for line in lines:
                pattern_type = None
                
                # Identifier le type d'intervention
                for type_arg, markers in self.argument_markers.items():
                    if any(marker in line.lower() for marker in markers):
                        pattern_type = type_arg
                        break
                
                if pattern_type:
                    current_pattern.append(pattern_type)
                    
                    # Détecter les patterns de longueur 3
                    if len(current_pattern) >= 3:
                        pattern = {
                            "sequence": current_pattern[-3:],
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
    
    def generate_visualizations(self, analysis: Dict[str, Any], output_dir: str) -> None:
        """Génère des visualisations pour l'analyse de structure."""
        # 1. Distribution des marqueurs argumentatifs
        self._plot_marker_distribution(
            analysis["structure_globale"]["distribution_marqueurs"], 
            output_dir
        )
        
        # 2. Complexité argumentative
        self._plot_argument_complexity(
            analysis["structure_globale"]["complexite_argumentative"],
            output_dir
        )
        
        # 3. Graphe des interactions
        self._plot_interaction_graph(
            analysis["interactions"]["graphe_interactions"],
            output_dir
        )
    
    def _plot_marker_distribution(self, distribution: Dict[str, int], output_dir: str) -> None:
        """Crée un graphique de la distribution des marqueurs argumentatifs."""
        plt.figure(figsize=(10, 6))
        
        categories = list(distribution.keys())
        counts = list(distribution.values())
        
        plt.bar(categories, counts)
        plt.title("Distribution des Marqueurs Argumentatifs")
        plt.xlabel("Types de Marqueurs")
        plt.ylabel("Nombre d'Occurrences")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/marker_distribution.png")
        plt.close()
    
    def _plot_argument_complexity(self, complexity_scores: List[float], output_dir: str) -> None:
        """Crée un graphique de la complexité argumentative."""
        plt.figure(figsize=(10, 6))
        
        plt.hist(complexity_scores, bins=20, edgecolor='black')
        plt.title("Distribution de la Complexité Argumentative")
        plt.xlabel("Score de Complexité")
        plt.ylabel("Fréquence")
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/argument_complexity.png")
        plt.close()
    
    def _plot_interaction_graph(self, graph_data: Dict[str, Any], output_dir: str) -> None:
        """Crée une visualisation du graphe d'interactions."""
        plt.figure(figsize=(12, 8))
        
        G = nx.DiGraph()
        
        # Ajouter les nœuds
        for node in graph_data["nodes"]:
            G.add_node(node["id"], type=node["type"])
        
        # Ajouter les arêtes
        edge_colors = []
        for edge in graph_data["edges"]:
            G.add_edge(edge["source"], edge["target"])
            color = {
                "support": "green",
                "opposition": "red",
                "conclusion": "blue"
            }.get(edge["type"], "gray")
            edge_colors.append(color)
        
        # Dessiner le graphe
        pos = nx.spring_layout(G)
        
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                             node_size=1000, alpha=0.6)
        
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                             width=2, alpha=0.6)
        
        nx.draw_networkx_labels(G, pos)
        
        plt.title("Graphe des Interactions Argumentatives")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/interaction_graph.png")
        plt.close()

if __name__ == "__main__":
    # Example usage for testing
    analyzer = ArgumentStructureAnalyzer()
    # Example dialogue history (replace with real data as needed)
    dialogue_history = [
        {"dialogue": "[Alice]: I propose we build a wall.\n[Bob]: But it is expensive.\n[Alice]: Because it protects us.\n[Bob]: So, we should consider alternatives."}
    ]
    result = analyzer.analyze_structure(dialogue_history)
    print(result) 