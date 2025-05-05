import os
from typing import List, Dict, Any
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

class DialogueManager:
    """Dialogue manager for the climate adaptation workshop."""
    
    def __init__(self, workshop_config: Dict[str, Any], personas: List[Dict[str, Any]]):
        """
        Initializes the dialogue manager.
        
        Args:
            workshop_config: Workshop configuration
            personas: List of participating personas
        """
        self.config = workshop_config
        self.personas = personas
        self.current_stage = "S1"
        self.dialogue_history = []
        
    def start_dialogue(self) -> None:
        """Starts the workshop dialogue."""
        print("\n=== Start of the Climate Adaptation Workshop ===\n")
        
        # Iterate through each workshop step
        for stage in self.config["workshop_parameters"]["stages"]:
            self.current_stage = stage["id"]
            print(f"\n--- {stage['name']} ---")
            print(f"Description: {stage['description']}\n")
            
            # Generate dialogue for the current step
            self._generate_stage_dialogue(stage)
            
        print("\n=== End of the Workshop ===\n")
        
    def _generate_stage_dialogue(self, stage: Dict[str, Any]) -> None:
        """
        Generates the dialogue for a specific stage.
        
        Args:
            stage: Stage configuration
        """
        # Build the prompt for the dialogue
        prompt = self._build_stage_prompt(stage)
        
        try:
            # Generate dialogue with Azure OpenAI
            response = client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Display the generated dialogue
            dialogue = response.choices[0].message.content
            print(dialogue)
            
            # Save to history
            self.dialogue_history.append({
                "stage": stage["id"],
                "dialogue": dialogue
            })
            
        except Exception as e:
            print(f"Error generating dialogue: {str(e)}")
    
    def _build_stage_prompt(self, stage: Dict[str, Any]) -> str:
        """
        Builds the prompt for a workshop stage.
        
        Args:
            stage: Stage configuration
        Returns:
            Prompt for dialogue generation
        """
        # Workshop context
        context = self.config["workshop_parameters"]["purpose"]["context"]
        
        # Step objectives
        topics = ", ".join(stage["required_topics"])
        
        # Build the prompt
        prompt = f"""
        Workshop context:
        {context}

        Current step: {stage['name']}
        Description: {stage['description']}
        Topics to cover: {topics}

        Participants:
        {self._format_personas()}

        Generate a natural dialogue between participants who:
        1. Cover all required topics
        2. Respect the interaction pattern "{stage['interaction_pattern']}"
        3. Reflect the unique perspectives and concerns of each persona
        4. Progress toward workshop objectives
        5. Integrate perspectives of non-human personas in a creative and relevant way

        Dialogue format:
        [Persona Name]: Message
        """
        
        return prompt
    
    def _format_personas(self) -> str:
        """
        Formats persona information for the prompt.
        
        Returns:
            Formatted persona descriptions
        """
        persona_descriptions = []
        for persona in self.personas:
            if "description" in persona:  # Non-human persona
                description = f"- {persona['nom']} (Non-human): {persona['description'][:200]}..."
            else:  # Human persona
                description = f"- {persona['nom']}: {persona['profession']}, {persona['age']} years"
                description += f"\n  Situation: {persona['situation_familiale']}"
                description += f"\n  Vulnerabilities: {', '.join(persona['vulnerabilites'][:3])}"
            
            persona_descriptions.append(description)
        
        return "\n".join(persona_descriptions)
    
    def _get_system_prompt(self) -> str:
        """
        Returns the system prompt for the LLM.
        
        Returns:
            System prompt
        """
        return ("You are an expert facilitator in climate adaptation workshops. "
                "Your role is to generate realistic dialogues between participants, including "
                "non-human perspectives (ecosystems, infrastructures) represented in a metaphorical "
                "but relevant way. Dialogues should be natural, informative, and constructive, "
                "while coherently integrating the different perspectives.")

def main():
    """Main function to test the dialogue manager."""
    # Example of minimal configuration
    workshop_config = {
        "workshop_parameters": {
            "purpose": {
                "context": "Workshop on adaptation to coastal flooding"
            },
            "stages": [
                {
                    "id": "S1",
                    "name": "Introduction",
                    "description": "Presentation of participants",
                    "interaction_pattern": "linear",
                    "required_topics": ["background", "concerns"]
                }
            ]
        }
    }
    
    # Example of personas
    personas = [
        {
            "name": "Marie Dubois",
            "age": 65,
            "occupation": "Retired",
            "biography": "Long-time resident of the coastal neighborhood."
        },
        {
            "name": "Jean Martin",
            "age": 45,
            "occupation": "Fisherman",
            "biography": "Traditional fisherman concerned about the future of his activity."
        }
    ]
    
    # Create and run the dialogue
    manager = DialogueManager(workshop_config, personas)
    manager.start_dialogue()

if __name__ == "__main__":
    main() 