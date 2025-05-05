import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

class TopicCoherenceScorer:
    """Evaluates topic coherence in workshop discussions."""
    
    def __init__(self):
        """Initializes the coherence scorer."""
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='french')
        self.topic_categories = {
            "impacts": ["impact", "effet", "consequence", "resultat"],
            "solutions": ["solution", "action", "mesure", "intervention"],
            "barrieres": ["barriere", "obstacle", "difficulte", "contrainte"],
            "ressources": ["ressource", "moyen", "outil", "capacite"]
        }
    
    def evaluate_coherence(self, dialogue_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates topic coherence in the entire dialogue.
        Returns:
            Coherence evaluation results
        """
        evaluation = {
            "date_evaluation": datetime.now().isoformat(),
            "coherence_globale": self._calculate_global_coherence(dialogue_history),
            "coherence_thematique": self._evaluate_thematic_coherence(dialogue_history),
            "flux_discussion": self._analyze_discussion_flow(dialogue_history),
            "transitions": self._evaluate_transitions(dialogue_history)
        }
        
        return evaluation
    
    def _calculate_global_coherence(self, dialogue_history: List[Dict[str, Any]]) -> float:
        """Calculates the overall coherence of the discussions."""
        try:
            # Extract all dialogues
            dialogues = [stage["dialogue"] for stage in dialogue_history]
            
            # Vectorize dialogues
            tfidf_matrix = self.vectorizer.fit_transform(dialogues)
            
            # Calculate average similarity between all segment pairs
            similarities = cosine_similarity(tfidf_matrix)
            
            # Exclude the diagonal (self-similarity)
            np.fill_diagonal(similarities, 0)
            
            # Calculate the average similarity
            coherence = float(similarities.mean())
            
            return coherence
        except:
            return 0.5  # Default value in case of error
    
    def _evaluate_thematic_coherence(self, dialogue_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluates coherence by thematic category."""
        coherence_scores = {}
        
        for category, keywords in self.topic_categories.items():
            relevant_segments = []
            
            # Collect relevant segments for each category
            for stage in dialogue_history:
                dialogue = stage["dialogue"].lower()
                if any(keyword in dialogue for keyword in keywords):
                    relevant_segments.append(dialogue)
            
            if len(relevant_segments) > 1:
                try:
                    # Vectorize relevant segments
                    tfidf_matrix = self.vectorizer.fit_transform(relevant_segments)
                    similarities = cosine_similarity(tfidf_matrix)
                    np.fill_diagonal(similarities, 0)
                    coherence_scores[category] = float(similarities.mean())
                except:
                    coherence_scores[category] = 0.5
            else:
                coherence_scores[category] = 0.0
        
        return coherence_scores
    
    def _analyze_discussion_flow(self, dialogue_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyzes the flow and progression of the discussion."""
        flow_analysis = {
            "progression": [],
            "topic_shifts": []
        }
        
        try:
            # Vectorize all segments
            dialogues = [stage["dialogue"] for stage in dialogue_history]
            tfidf_matrix = self.vectorizer.fit_transform(dialogues)
            
            # Calculate similarities between consecutive segments
            for i in range(len(dialogues)-1):
                similarity = cosine_similarity(
                    tfidf_matrix[i:i+1], 
                    tfidf_matrix[i+1:i+2]
                )[0][0]
                
                flow_analysis["progression"].append({
                    "transition": f"S{i+1}_S{i+2}",
                    "smoothness": float(similarity)
                })
                
                # Detect abrupt topic changes
                if similarity < 0.3:  # Arbitrary threshold for significant changes
                    flow_analysis["topic_shifts"].append({
                        "position": f"S{i+1}_S{i+2}",
                        "severity": float(1 - similarity)
                    })
        except:
            flow_analysis["progression"] = []
            flow_analysis["topic_shifts"] = []
        
        return flow_analysis
    
    def _evaluate_transitions(self, dialogue_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluates the quality of topic transitions."""
        transitions = {
            "qualite_transitions": [],
            "graphe_transitions": self._create_transition_graph(dialogue_history)
        }
        
        try:
            # Analyze transitions between consecutive segments
            for i in range(len(dialogue_history)-1):
                current = dialogue_history[i]["dialogue"]
                next_segment = dialogue_history[i+1]["dialogue"]
                
                # Vectorize segments
                segments = [current, next_segment]
                tfidf_matrix = self.vectorizer.fit_transform(segments)
                similarity = cosine_similarity(tfidf_matrix)[0][1]
                
                transitions["qualite_transitions"].append({
                    "transition": f"S{i+1}_S{i+2}",
                    "score": float(similarity),
                    "qualite": "fluide" if similarity > 0.5 else "abrupte"
                })
        except:
            transitions["qualite_transitions"] = []
        
        return transitions
    
    def _create_transition_graph(self, dialogue_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Creates a graph of topic transitions."""
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        try:
            # Create a directed graph
            G = nx.DiGraph()
            
            # Add nodes (dialogue segments)
            for i, stage in enumerate(dialogue_history):
                node_id = f"S{i+1}"
                G.add_node(node_id)
                graph_data["nodes"].append({
                    "id": node_id,
                    "content": stage["dialogue"][:50] + "..."  # Content preview
                })
                
                # Add edges (transitions)
                if i > 0:
                    prev_id = f"S{i}"
                    # Vectorize and calculate similarity
                    segments = [dialogue_history[i-1]["dialogue"], stage["dialogue"]]
                    tfidf_matrix = self.vectorizer.fit_transform(segments)
                    weight = float(cosine_similarity(tfidf_matrix)[0][1])
                    
                    G.add_edge(prev_id, node_id, weight=weight)
                    graph_data["edges"].append({
                        "source": prev_id,
                        "target": node_id,
                        "weight": weight
                    })
        except:
            graph_data["nodes"] = []
            graph_data["edges"] = []
        
        return graph_data
    
    def generate_visualizations(self, evaluation: Dict[str, Any], output_dir: str) -> None:
        """Generates visualizations for coherence evaluation."""
        # 1. Thematic coherence graph
        self._plot_thematic_coherence(evaluation["coherence_thematique"], output_dir)
        
        # 2. Discussion flow graph
        self._plot_discussion_flow(evaluation["flux_discussion"], output_dir)
        
        # 3. Transition graph
        self._plot_transition_graph(evaluation["transitions"]["graphe_transitions"], output_dir)
    
    def _plot_thematic_coherence(self, coherence_scores: Dict[str, float], output_dir: str) -> None:
        """Creates a plot of thematic coherence."""
        plt.figure(figsize=(10, 6))
        
        categories = list(coherence_scores.keys())
        scores = list(coherence_scores.values())
        
        plt.bar(categories, scores)
        plt.title("Thematic Coherence by Category")
        plt.xlabel("Categories")
        plt.ylabel("Coherence Score")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/thematic_coherence.png")
        plt.close()
    
    def _plot_discussion_flow(self, flow_analysis: Dict[str, Any], output_dir: str) -> None:
        """Creates a plot of discussion flow."""
        plt.figure(figsize=(12, 6))
        
        transitions = [prog["transition"] for prog in flow_analysis["progression"]]
        smoothness = [prog["smoothness"] for prog in flow_analysis["progression"]]
        
        plt.plot(transitions, smoothness, marker='o')
        plt.title("Transition Smoothness")
        plt.xlabel("Transitions")
        plt.ylabel("Smoothness Score")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        # Mark abrupt changes
        for shift in flow_analysis["topic_shifts"]:
            plt.axvline(x=transitions.index(shift["position"]), color='r', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/discussion_flow.png")
        plt.close()
    
    def _plot_transition_graph(self, graph_data: Dict[str, Any], output_dir: str) -> None:
        """Creates a visualization of the transition graph."""
        plt.figure(figsize=(12, 8))
        
        G = nx.DiGraph()
        
        # Add nodes and edges
        for node in graph_data["nodes"]:
            G.add_node(node["id"])
        
        for edge in graph_data["edges"]:
            G.add_edge(edge["source"], edge["target"], weight=edge["weight"])
        
        # Draw the graph
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=1000, alpha=0.6)
        
        # Draw edges with colors based on weights
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, edge_color=weights, 
                             edge_cmap=plt.cm.YlOrRd, width=2)
        
        # Add labels
        nx.draw_networkx_labels(G, pos)
        
        plt.title("Graphe des Transitions")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/transition_graph.png")
        plt.close() 