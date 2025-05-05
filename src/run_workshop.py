import os
from pathlib import Path
from workshop_initialization.workshop_init import WorkshopInitializer
from workshop_initialization.dialogue_manager import DialogueManager
from persona_development.persona_generator import PersonaGenerator
from evaluation.workshop_evaluator import WorkshopEvaluator

def run_workshop():
    """Runs the complete climate adaptation workshop."""
    print("\n=== Climate Adaptation Workshop Configuration ===\n")
    
    # 1. Define workshop parameters
    workshop_input = {
        "context": "Our coastal community of Bayshore faces increasing challenges related to sea level rise and extreme weather events. Traditional fishing areas are threatened, and low-income residents are particularly vulnerable to coastal flooding. The community has about 5,000 residents, 40% of whom live in high-risk areas.",
        "objectives": "1. Identify the differential impacts of climate change on different groups\n2. Develop inclusive and equitable adaptation strategies\n3. Create a resilient community action plan\n4. Establish support mechanisms for vulnerable groups",
        "outputs": "- Detailed mapping of climate vulnerabilities by demographic group\n- Prioritized list of adaptation options with cost-benefit analysis\n- Stakeholder engagement plan with roles and responsibilities\n- Recommendations for strengthening community resilience\n- Financing and implementation strategies"
    }
    
    # 2. Initialize the workshop
    print("Initializing the workshop...")
    initializer = WorkshopInitializer()
    workshop_config = initializer.create_workshop_config(workshop_input)
    initializer.save_config(workshop_config)
    print("Workshop configuration created and saved.")
    
    # 3. Generate personas
    print("\nGenerating personas...")
    try:
        # Ask for the number of personas to generate
        nombre_personas = int(input("How many personas would you like to generate? (4-20): "))
        nombre_personas = max(4, min(20, nombre_personas))
    except ValueError:
        print("Invalid value. Generating 10 personas by default.")
        nombre_personas = 10

    generator = PersonaGenerator(output_dir="workshop_autpout/personas")
    
    # Define available profiles
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
    
    print(f"\nGenerating {nombre_personas} personas...")
    personas = generator.generate_persona_set(target_profiles)
    
    if not personas:
        print("Error: No personas were generated. Exiting workshop.")
        return
        
    print(f"Successfully generated {len(personas)} personas.")
    
    # 4. Launch the workshop dialogue
    print("\nStarting the workshop dialogue...")
    dialogue_manager = DialogueManager(workshop_config, personas)
    dialogue_manager.start_dialogue()
    
    # 5. Evaluate the workshop
    print("\nEvaluating the workshop...")
    evaluator = WorkshopEvaluator()
    evaluation = evaluator.evaluate_dialogue(dialogue_manager.dialogue_history)
    evaluator.generate_report(evaluation)
    
    # Display recommendations
    print("\nRecommendations for future workshops:")
    for i, rec in enumerate(evaluation["recommendations"], 1):
        print(f"{i}. {rec}")
    
    print("\n=== Workshop Completed ===")
    print("Results have been saved in the 'workshop_autpout' folder")

if __name__ == "__main__":
    run_workshop() 