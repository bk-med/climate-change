import os
from pathlib import Path
from typing import Dict, Any
import json
import yaml
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

class WorkshopInitializer:
    """Class to initialize and configure a climate adaptation workshop."""
    
    def __init__(self, output_dir: str = "workshop_output"):
        """
        Initializes the workshop manager.
        
        Args:
            output_dir: Output directory for workshop files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_workshop_config(self, workshop_input: Dict[str, str]) -> Dict[str, Any]:
        """
        Creates the workshop configuration from user input.
        
        Args:
            workshop_input: Dictionary containing workshop parameters
        Returns:
            Structured workshop configuration
        """
        # Basic workshop configuration
        workshop_config = {
            "workshop_parameters": {
                "purpose": {
                    "context": workshop_input["context"],
                    "objectives": workshop_input["objectives"],
                    "outputs_expected": workshop_input["outputs"]
                },
                "stages": [
                    {
                        "id": "S1",
                        "name": "Introduction",
                        "description": "Establish persona identities and initial climate perspectives",
                        "interaction_pattern": "linear",
                        "required_topics": ["personal_background", "climate_concerns"],
                        "duration_points": 5
                    },
                    {
                        "id": "S2",
                        "name": "Impact Exploration",
                        "description": "Identify climate risks and differential impacts",
                        "interaction_pattern": "spoke_wheel",
                        "required_topics": ["climate_risks", "vulnerable_groups", "current_coping"],
                        "duration_points": 8
                    },
                    {
                        "id": "S3",
                        "name": "Adaptation Brainstorming",
                        "description": "Generate adaptation options and identify facilitators/barriers",
                        "interaction_pattern": "mesh_network",
                        "required_topics": ["adaptation_options", "barriers", "enablers"],
                        "duration_points": 12
                    },
                    {
                        "id": "S4",
                        "name": "Strategy Evaluation",
                        "description": "Evaluate adaptation options according to multiple criteria",
                        "interaction_pattern": "bipartite_evaluation",
                        "required_topics": ["feasibility", "equity", "effectiveness"],
                        "duration_points": 8
                    }
                ],
                "outputs": [
                    {
                        "id": "DN_01",
                        "class": "DN",
                        "format": "structured_narrative",
                        "fields": ["climate_impacts", "vulnerable_groups", "differential_impacts"],
                        "quality": {"detail_level": "high", "scientific_accuracy": True},
                        "stage": "S2"
                    },
                    {
                        "id": "AO_01",
                        "class": "AO",
                        "format": "structured_list",
                        "fields": ["description", "beneficiaries", "implementation_challenges"],
                        "quality": {"feasibility_min": 0.7, "equity_required": True},
                        "stage": "S3"
                    },
                    {
                        "id": "IC_01",
                        "class": "IC",
                        "format": "stakeholder_matrix",
                        "fields": ["stakeholder", "role", "engagement_strategy"],
                        "quality": {"comprehensiveness": "medium", "actionability": True},
                        "stage": "S4"
                    }
                ]
            }
        }
        
        return workshop_config
    
    def save_config(self, config: Dict[str, Any], filename: str = "workshop_config") -> None:
        """
        Saves the workshop configuration.
        
        Args:
            config: Workshop configuration
            filename: File name (without extension)
        """
        # Save as JSON
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        # Save as YAML for readability
        yaml_path = self.output_dir / f"{filename}.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def main():
    """Main function to create a new workshop."""
    # Example workshop entries
    workshop_input = {
        "context": "Our coastal community faces increasing challenges related to sea level rise and extreme weather events. Traditional fishing areas are threatened, and low-income residents are particularly vulnerable to coastal flooding.",
        "objectives": "1. Identify the differential impacts of climate change on different groups\n2. Develop inclusive and equitable adaptation strategies\n3. Create a resilient community action plan",
        "outputs": "- Mapping of climate vulnerabilities by demographic group\n- Prioritized list of adaptation options\n- Stakeholder engagement plan\n- Recommendations for strengthening community resilience"
    }
    
    # Initialize and create the workshop
    initializer = WorkshopInitializer()
    config = initializer.create_workshop_config(workshop_input)
    initializer.save_config(config)
    
    print("Workshop configuration created successfully!")
    print(f"Files saved in: {initializer.output_dir}")

if __name__ == "__main__":
    main() 