import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from openai import AzureOpenAI
from dotenv import load_dotenv
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

@dataclass
class HumanPersona:
    """Class representing a human persona."""
    nom: str
    age: int
    profession: str
    situation_familiale: str
    revenu: str
    logement: str
    vulnerabilites: List[str]
    impacts_actuels: List[str]
    impacts_projetes: List[str]
    capacite_adaptation: str
    obstacles: List[str]
    options_adaptation: List[str]
    citations: List[str]

class PersonaGenerator:
    """Generator of personas for climate adaptation workshops."""
    
    def __init__(self, output_dir: str = "workshop_autpout/personas"):
        """Initialize the persona generator."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.system_prompt = "You are an expert in creating personas for climate adaptation. Respond only in valid JSON format."
    
    def generate_persona(self, demographic: str, location: str) -> Dict:
        """Generate a persona with specific characteristics."""
        
        prompt = f"""Create a detailed persona for a climate adaptation workshop.

Demographics: {demographic}
Location: {location}

The persona must include:
1. Name and age
2. Profession and family situation
3. Income level and housing type
4. Climate change specific vulnerabilities
5. Current and projected climate impacts
6. Adaptation capacity and obstacles
7. Relevant adaptation options
8. Authentic quotes reflecting their concerns

Respond only with a valid JSON object containing these fields:
{{
    "nom": "string",
    "age": number,
    "profession": "string",
    "situation_familiale": "string",
    "revenu": "string",
    "logement": "string",
    "vulnerabilites": ["string"],
    "impacts_actuels": ["string"],
    "impacts_projetes": ["string"],
    "capacite_adaptation": "string",
    "obstacles": ["string"],
    "options_adaptation": ["string"],
    "citations": ["string"]
}}"""

        try:
            response = client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            persona_data = json.loads(response.choices[0].message.content)
            
            # Validate required fields
            required_fields = ["nom", "age", "profession", "situation_familiale", "revenu", 
                             "logement", "vulnerabilites", "impacts_actuels", "impacts_projetes",
                             "capacite_adaptation", "obstacles", "options_adaptation", "citations"]
            
            for field in required_fields:
                if field not in persona_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Add metadata
            persona_data["metadata"] = {
                "date_creation": datetime.now().isoformat(),
                "demographic": demographic,
                "location": location,
                "version": "1.0"
            }
            
            return persona_data
            
        except json.JSONDecodeError as e:
            print(f"JSON decoding error for persona {demographic} at {location}: {str(e)}")
            return None
        except Exception as e:
            print(f"Error generating persona: {str(e)}")
            return None

    def generate_persona_set(self, target_profiles: List[Dict[str, Any]]) -> List[Dict]:
        """Generate a set of personas based on target profiles."""
        personas = []
        failed_profiles = []
        
        for profile in target_profiles:
            print(f"Generating persona: {profile['demographic']} at {profile['location']}")
            persona = self.generate_persona(
                profile["demographic"],
                profile["location"]
            )
            
            if persona:
                personas.append(persona)
                # Save persona individually
                safe_name = persona["nom"].replace(" ", "_").lower()
                filename = f"{self.output_dir}/{safe_name}.json"
                
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(persona, f, ensure_ascii=False, indent=2)
                    print(f"Persona saved in {filename}")
                except Exception as e:
                    print(f"Error saving persona {safe_name}: {str(e)}")
            else:
                failed_profiles.append(profile)
                print(f"Failed to generate persona for {profile['demographic']} at {profile['location']}")
        
        if failed_profiles:
            print("\nFailed profiles:")
            for profile in failed_profiles:
                print(f"- {profile['demographic']} at {profile['location']}")
        
        return personas

    def save_index(self, personas: List[Dict]) -> None:
        """Save the personas index with detailed metadata."""
        index = {
            "date_creation": datetime.now().isoformat(),
            "nombre_personas": len(personas),
            "statistiques": {
                "par_demographie": {},
                "par_location": {},
                "par_age": {
                    "jeunes": 0,
                    "adultes": 0,
                    "seniors": 0
                }
            },
            "personas": personas
        }
        
        # Calculate statistics
        for persona in personas:
            # Demographics
            demo = persona.get("metadata", {}).get("demographic", "unknown")
            index["statistiques"]["par_demographie"][demo] = index["statistiques"]["par_demographie"].get(demo, 0) + 1
            
            # Location
            loc = persona.get("metadata", {}).get("location", "unknown")
            index["statistiques"]["par_location"][loc] = index["statistiques"]["par_location"].get(loc, 0) + 1
            
            # Age groups
            age = persona.get("age", 0)
            if age < 30:
                index["statistiques"]["par_age"]["jeunes"] += 1
            elif age < 65:
                index["statistiques"]["par_age"]["adultes"] += 1
            else:
                index["statistiques"]["par_age"]["seniors"] += 1
        
        try:
            with open(f"{self.output_dir}/index.json", 'w', encoding='utf-8') as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
            print(f"\nIndex saved with {len(personas)} personas and detailed statistics")
        except Exception as e:
            print(f"Error saving index: {str(e)}")

def main():
    """Main function to test the persona generator."""
    # Ask for the number of personas to generate
    try:
        nombre_personas = int(input("How many personas would you like to generate? (4-20): "))
        nombre_personas = max(4, min(20, nombre_personas))  # Limit between 4 and 20
    except ValueError:
        print("Invalid value. Generating 10 personas by default.")
        nombre_personas = 10

    # Available profiles
    available_profiles = [
        # Coastal area profiles
        {"demographic": "elderly person", "location": "coastal area"},
        {"demographic": "fisherman", "location": "coastal area"},
        {"demographic": "coastal tourism worker", "location": "coastal area"},
        {"demographic": "coastal property owner", "location": "coastal area"},
        
        # Flood zone profiles
        {"demographic": "low-income family", "location": "flood zone"},
        {"demographic": "single parent", "location": "flood zone"},
        {"demographic": "renting family", "location": "flood zone"},
        {"demographic": "flood zone resident", "location": "flood zone"},
        
        # Urban area profiles
        {"demographic": "young adult", "location": "urban area"},
        {"demographic": "student", "location": "urban area"},
        {"demographic": "urban professional", "location": "urban area"},
        {"demographic": "urban service worker", "location": "urban area"},
        
        # Downtown profiles
        {"demographic": "business owner", "location": "downtown"},
        {"demographic": "restaurant owner", "location": "downtown"},
        {"demographic": "shop owner", "location": "downtown"},
        {"demographic": "office worker", "location": "downtown"},
        
        # Rural area profiles
        {"demographic": "farmer", "location": "rural area"},
        {"demographic": "rural resident", "location": "rural area"},
        {"demographic": "agricultural worker", "location": "rural area"},
        {"demographic": "rural business owner", "location": "rural area"}
    ]
    
    # Randomly select the requested number of profiles
    import random
    target_profiles = random.sample(available_profiles, nombre_personas)
    
    generator = PersonaGenerator()
    personas = []
    attempts = 0
    max_attempts = nombre_personas * 3  # Allow more retries
    
    print(f"\nStarting generation of {nombre_personas} personas...")
    
    while len(personas) < nombre_personas and attempts < max_attempts:
        remaining_profiles = target_profiles[len(personas):]
        new_personas = generator.generate_persona_set(remaining_profiles)
        if new_personas:  # Only extend if we got valid personas
            personas.extend(new_personas)
            print(f"Progress: {len(personas)}/{nombre_personas} personas generated")
        attempts += 1
    
    if len(personas) < nombre_personas:
        print(f"\nError: Could only generate {len(personas)} personas out of {nombre_personas} requested.")
        print("Please try running the script again.")
        return
    
    generator.save_index(personas)
    print(f"\nSuccess! Generated exactly {len(personas)} personas in the {generator.output_dir} folder")

if __name__ == "__main__":
    main() 