# Technical Documentation
## Module 2: Persona Development System

**Version:** 1.0  
**Last Updated:** April 28, 2025  
**Author:** Climate Adaptation Team

## Table of Contents
1. [Module Overview](#module-overview)
2. [Input and Output Specification](#input-and-output-specification)
3. [LLM Integration](#llm-integration)
4. [Implementation Details](#implementation-details)
5. [Usage Example](#usage-example)
6. [Google Colab Implementation](#google-colab-implementation)
7. [Dependencies](#dependencies)
8. [References](#references)

## Module Overview

The Persona Development System is the second module in our LLM-simulated climate adaptation workshop framework. This module transforms scientific climate data and vulnerability assessments into richly detailed, ethically constructed personas that represent diverse stakeholders in climate adaptation contexts. These personas serve as the synthetic participants in the simulated focus group discussions.

### Purpose

This module establishes the foundation for authentic dialogue simulation by:
- Integrating climate science with demographic and socioeconomic realities
- Ensuring representative diversity with special attention to vulnerable populations
- Creating scientifically accurate yet narratively engaging personas
- Mitigating bias and stereotyping through systematic validation
- Bridging workshop objectives from Module 1 with dialogue simulation in Module 3

### Process Flow

The module follows this sequential workflow:
1. **Climate Risk and Vulnerability Assessment (CRVA)**
   - Collect and integrate climate projections, historical data, and demographic profiles
   - Analyze hazards, exposure patterns, and vulnerability factors
   - Map spatial distribution of climate risks and social vulnerability

2. **Persona Template Creation**
   - Structure data into standardized templates with three sections:
     * Personal Profile (demographics, values, challenges)
     * Climate Impacts (hazards, temporal dimensions, manifestations)
     * Adaptation Context (enablers, barriers, options)

3. **LLM-Enhanced Narrative Generation**
   - Transform structured data into engaging biographies using specialized LLM prompting
   - Generate authentic quoted speech in appropriate vernacular
   - Maintain scientific accuracy while adding narrative depth

4. **Validation and Refinement**
   - Technical validation by climate science experts
   - Ethical validation by community representatives
   - Iterative refinement through guided LLM regeneration

5. **Finalization and Integration**
   - Assemble complete persona profiles with all components
   - Prepare metadata for dialogue simulation engine
   - Define persona-specific parameters for LLM agents

## Input and Output Specification

### Input

The module accepts multiple data sources that inform persona development:

| Input Category | Description | Format | Required |
|----------------|-------------|--------|----------|
| Climate Projections | Downscaled climate projections for target region | CSV/NetCDF/JSON | Yes |
| Historical Climate Data | Observed climate trends and events | CSV/JSON | Yes |
| Demographic Data | Community demographic and socioeconomic profiles | CSV/JSON | Yes |
| Spatial Hazard Data | Georeferenced hazard zones (flood maps, etc.) | GeoJSON/Shapefile | Yes |
| Workshop Configuration | Output from Module 1 | JSON | Yes |
| Local Knowledge | Context-specific information about the community | Text | No |
| Existing Personas | Templates or examples for reference | JSON/YAML | No |

#### Climate Projections Example:
```json
{
  "region": "Coastal County X",
  "scenarios": {
    "RCP4.5": {
      "2030": {"temperature_change": 0.8, "precipitation_change": 5, "sea_level_rise": 0.15},
      "2050": {"temperature_change": 1.4, "precipitation_change": 8, "sea_level_rise": 0.30},
      "2070": {"temperature_change": 1.9, "precipitation_change": 12, "sea_level_rise": 0.45}
    },
    "RCP8.5": {
      "2030": {"temperature_change": 1.0, "precipitation_change": 6, "sea_level_rise": 0.18},
      "2050": {"temperature_change": 2.1, "precipitation_change": 11, "sea_level_rise": 0.40},
      "2070": {"temperature_change": 3.4, "precipitation_change": 18, "sea_level_rise": 0.72}
    }
  },
  "hazards": ["coastal_flooding", "storm_surge", "extreme_heat", "drought"]
}
```

#### Demographic Data Example:
```json
{
  "total_population": 35000,
  "demographics": {
    "age_distribution": {
      "under_18": 0.21, "18_to_25": 0.08, "26_to_45": 0.26, 
      "46_to_65": 0.27, "over_65": 0.18
    },
    "income_distribution": {
      "below_poverty": 0.18, "low_income": 0.24, "middle_income": 0.42, 
      "high_income": 0.16
    },
    "housing": {
      "renters": 0.38, "owners": 0.62, "mobile_homes": 0.07,
      "flood_zone_residents": 0.31
    },
    "employment": {
      "fishing_industry": 0.12, "tourism": 0.23, "service": 0.32,
      "professional": 0.18, "retired": 0.15
    }
  },
  "vulnerable_groups": [
    {"name": "Low-income renters in flood zones", "population_percentage": 0.09},
    {"name": "Elderly with limited mobility", "population_percentage": 0.06},
    {"name": "Fishing families", "population_percentage": 0.12},
    {"name": "Mobile home residents", "population_percentage": 0.07}
  ]
}
```

### Output

The module produces a set of personas representing diverse community members:

```
personas
├── human_personas  
│   ├── persona_1
│   │   ├── metadata.json  
│   │   ├── biography.md
│   │   └── image.png (optional)
│   ├── persona_2
│   │   ├── ...
│   └── ... (typically 9-11 human personas)
├── non_human_personas
│   ├── ecological_persona
│   │   ├── ...
│   └── infrastructure_persona (optional)
│       ├── ...
└── persona_index.json
```

#### Individual Persona Structure:

Each persona includes the following components:

```json
{
  "id": "persona_coastal_elder_1",
  "type": "human",
  "name": "Eleanor Martinez",
  "age": 73,
  "metadata": {
    "demographics": {
      "gender": "female",
      "income_level": "low_to_middle",
      "housing_type": "owned_single_family",
      "occupation": "retired_teacher",
      "location": "coastal_zone_A"
    },
    "vulnerability_factors": {
      "physical_exposure": 0.8,
      "sensitivity": 0.7,
      "adaptive_capacity": 0.4
    },
    "value_priorities": [
      {"value": "security", "weight": 0.8},
      {"value": "tradition", "weight": 0.7},
      {"value": "universalism", "weight": 0.6},
      {"value": "self_determination", "weight": 0.4},
      {"value": "achievement", "weight": 0.3}
    ],
    "workshop_relevance": {
      "stakeholder_group": "elderly residents",
      "priority_level": "high",
      "perspective_contribution": "historic knowledge, limited mobility challenges"
    }
  },
  "climate_impacts": {
    "current": [
      {"hazard": "coastal_flooding", "description": "Yard floods during king tides", "severity": "moderate"},
      {"hazard": "extreme_heat", "description": "Health impacts during summer heat waves", "severity": "high"}
    ],
    "projected_2050": [
      {"hazard": "coastal_flooding", "description": "Home likely inundated during major storms", "severity": "severe"},
      {"hazard": "extreme_heat", "description": "Lethal heat conditions without intervention", "severity": "extreme"}
    ]
  },
  "adaptation_context": {
    "enablers": [
      {"factor": "home_ownership", "description": "Owns home outright, no mortgage pressure"},
      {"factor": "community_connections", "description": "Strong local support network"}
    ],
    "barriers": [
      {"factor": "fixed_income", "description": "Limited financial resources for major renovations or relocation"},
      {"factor": "physical_mobility", "description": "Difficulty evacuating without assistance"}
    ],
    "adaptation_options": [
      {"option": "home_elevation", "feasibility": "low", "preference": "low"},
      {"option": "community_resilience_hub", "feasibility": "high", "preference": "high"},
      {"option": "assisted_evacuation_program", "feasibility": "high", "preference": "medium"}
    ]
  },
  "biography": "Eleanor Martinez has lived in her coastal home for over 40 years...",
  "quotes": [
    {"text": "I've seen the water getting closer every year. The tide comes up into my garden now, which never happened when we first moved here.", "context": "discussing observed changes"},
    {"text": "At my age, starting over somewhere new just isn't realistic. This is my home, all my memories are here.", "context": "discussing retreat options"}
  ],
  "linguistic_parameters": {
    "formality_level": 0.7,
    "technical_vocabulary": 0.3,
    "regional_markers": 0.4,
    "speech_patterns": ["reflective", "historically_oriented", "direct"]
  }
}
```

## LLM Integration

The Persona Development System leverages Large Language Models throughout the persona creation process. This approach combines the scientific rigor of structured data with the narrative richness enabled by LLMs.

### Key LLM Integration Points

1. **Vulnerability Factor Extraction and Correlation**
   - LLM analyzes demographic and hazard data to identify key vulnerability factors
   - Connects abstract vulnerability concepts with concrete personal circumstances
   - Correlates socioeconomic factors with specific climate risks

2. **Biography Generation**
   - Transforms structured template data into engaging narrative profiles
   - Maintains scientific accuracy while adding human dimension
   - Creates relatable backstories that explain vulnerability context

3. **Direct Quote Generation**
   - Produces authentic speech patterns appropriate to the persona's background
   - Varies linguistic style according to demographic parameters
   - Captures emotional responses to climate impacts

4. **Adaptation Options Analysis**
   - Generates persona-specific adaptation preferences and barriers
   - Ensures adaptation options are scientifically feasible and contextually appropriate
   - Balances abstract adaptation concepts with personal priorities

### Prompt Templates

The module uses specialized prompt templates for each LLM task. For example:

#### Biography Generation Prompt
```
You are an expert climate adaptation writer crafting realistic personas for climate planning workshops.

TEMPLATE DATA:
Name: {name}
Age: {age}
Demographics: {demographics_json}
Climate Impacts: {impacts_json}
Adaptation Context: {adaptation_json}

Using ONLY the information above, create a 300-word biography for this persona that:
1. Shows their personal background and connection to the area
2. Illustrates how they personally experience the specified climate impacts
3. Describes their relationship to the identified adaptation enablers and barriers
4. Avoids stereotypes while remaining authentic to their specific context
5. Uses the third person perspective throughout

The biography should be factually accurate to the template data but narratively engaging.
```

#### Quote Generation Prompt
```
You are generating authentic quotes for a climate adaptation persona.

PERSONA BACKGROUND:
{biography_summary}

LINGUISTIC PARAMETERS:
- Formality Level (0-1): {formality}
- Technical Vocabulary (0-1): {technical}
- Regional Markers (0-1): {regional}
- Speech Patterns: {patterns}

QUOTE CONTEXT:
The persona is discussing: {context}

Generate 2-3 direct quotes (30-50 words each) that this persona might realistically say about the topic.
The quotes should:
- Reflect the persona's values and concerns
- Use language appropriate to their background and linguistic parameters
- Avoid stereotypes while capturing authentic speech patterns
- Include specific references to their lived climate experience

Format as a JSON array of quote objects:
[
  {"text": "Quote text here...", "topic": "brief topic label"},
  ...
]
```

## Implementation Details

### Key Components

1. **VulnerabilityAssessment**: Processes climate and demographic data to generate structured vulnerability profiles
2. **TemplateManager**: Creates and validates persona templates with required fields
3. **LLMPersonaGenerator**: Transforms templates into fully-developed personas using LLM
4. **PersonaValidator**: Verifies scientific accuracy and ethical representation
5. **PersonaLibrary**: Manages the collection of personas and provides access methods

### Design Considerations

- **Scientific Accuracy**: Rigorous validation steps ensure climate impacts are factually correct
- **Representational Ethics**: Systematic bias detection prevents stereotyping and misrepresentation
- **Modularity**: Components interact through clean interfaces for maintainability
- **Extensibility**: Architecture supports additional persona types (non-human, institutional)
- **LLM Provider Flexibility**: Abstraction allows use of different LLM providers

### Key Algorithms

#### Vulnerability Factor Calculation

The system uses a multi-factor vulnerability scoring algorithm based on established frameworks:

```python
def calculate_vulnerability(demographic_factors, hazard_exposure, adaptive_capacity):
    """
    Calculate vulnerability scores using weighted factors.
    
    Args:
        demographic_factors: Dictionary of demographic vulnerability indicators
        hazard_exposure: Dictionary of exposure levels to different hazards
        adaptive_capacity: Dictionary of factors affecting adaptation ability
        
    Returns:
        Dictionary of vulnerability scores by hazard type
    """
    # Define weights for the vulnerability equation: Vulnerability = (Exposure * Sensitivity) / Adaptive Capacity
    sensitivity_weights = {
        "age_over_65": 0.7,
        "age_under_5": 0.7,
        "low_income": 0.8,
        "disability": 0.6,
        "english_proficiency": 0.4,
        "vehicle_access": 0.5,
        "population_density": 0.3
    }
    
    # Calculate sensitivity score based on demographic factors
    sensitivity = sum(sensitivity_weights.get(factor, 0) * value 
                     for factor, value in demographic_factors.items())
    
    # Normalize sensitivity
    sensitivity = min(max(sensitivity, 0), 1)
    
    # Calculate vulnerability for each hazard type
    vulnerability_scores = {}
    for hazard, exposure in hazard_exposure.items():
        # Get adaptive capacity for this hazard
        capacity = adaptive_capacity.get(hazard, 0.5)  # Default to medium capacity if not specified
        
        # Ensure capacity is never zero to avoid division by zero
        capacity = max(capacity, 0.1)
        
        # Calculate vulnerability score using the vulnerability equation
        vulnerability = (exposure * sensitivity) / capacity
        
        # Normalize to 0-1 scale
        vulnerability = min(max(vulnerability / 3.0, 0), 1)
        
        vulnerability_scores[hazard] = vulnerability
    
    return vulnerability_scores
```

#### Stereotype Detection

The Stereotype Checklist system implements systematic bias detection:

```python
def detect_stereotypes(persona_text, demographic_category):
    """
    Check for common stereotypes in persona text based on demographic category.
    
    Args:
        persona_text: The generated persona text to check
        demographic_category: Category to check (e.g., "elderly", "low_income")
        
    Returns:
        List of detected stereotype flags with explanations
    """
    # Load stereotype patterns for the specific demographic category
    stereotype_patterns = STEREOTYPE_PATTERNS.get(demographic_category, [])
    
    detected_stereotypes = []
    
    for pattern in stereotype_patterns:
        if re.search(pattern["regex"], persona_text, re.IGNORECASE):
            detected_stereotypes.append({
                "type": pattern["type"],
                "description": pattern["description"],
                "mitigation": pattern["mitigation_guidance"]
            })
    
    # Check for general stereotyping patterns
    general_patterns = [
        {
            "regex": r"\ball\s+(?:of\s+)?them\s+are\b|\bthey\s+all\s+(?:are|have|do)\b",
            "type": "generalization",
            "description": "Overgeneralization about an entire group",
            "mitigation_guidance": "Replace with specific experiences of this individual"
        },
        {
            "regex": r"\balways\b|\bnever\b",
            "type": "absolutist_language",
            "description": "Absolutist language that oversimplifies behavior",
            "mitigation_guidance": "Use more nuanced language like 'tends to' or 'often'"
        }
    ]
    
    for pattern in general_patterns:
        if re.search(pattern["regex"], persona_text, re.IGNORECASE):
            detected_stereotypes.append({
                "type": pattern["type"],
                "description": pattern["description"],
                "mitigation": pattern["mitigation_guidance"]
            })
    
    return detected_stereotypes
```

## Usage Example

```python
from persona_development import PersonaDevelopmentSystem, VulnerabilityData, ClimateData

# Load workshop configuration from Module 1
with open("workshop_config.json", "r") as f:
    workshop_config = json.load(f)

# Load vulnerability data
vulnerability_data = VulnerabilityData.from_files(
    demographic_file="data/demographic_profile.csv",
    hazard_file="data/hazard_exposure.geojson"
)

# Load climate data
climate_data = ClimateData.from_files(
    historical_file="data/historical_climate.csv",
    projection_file="data/climate_projections.nc",
    scenario="RCP8.5",
    time_horizons=[2030, 2050, 2070]
)

# Initialize the system
persona_system = PersonaDevelopmentSystem(
    llm_provider="anthropic",  # Using Claude for narrative generation
    api_key="your_api_key",
    output_dir="./personas"
)

# Define target persona profiles based on workshop needs
target_profiles = [
    {"demographic": "elderly", "location": "coastal", "count": 2},
    {"demographic": "low_income", "location": "flood_zone", "count": 2},
    {"demographic": "fishing_industry", "location": "any", "count": 2},
    {"demographic": "parent", "location": "urban", "count": 1},
    {"demographic": "business_owner", "location": "downtown", "count": 1},
    {"demographic": "youth", "location": "any", "count": 1},
    {"demographic": "non_human", "type": "coastal_ecosystem", "count": 1}
]

# Generate personas based on workshop parameters
personas = persona_system.generate_personas(
    workshop_config=workshop_config,
    vulnerability_data=vulnerability_data,
    climate_data=climate_data,
    target_profiles=target_profiles,
    validate=True  # Enable validation
)

# Print summary
print(f"Generated {len(personas)} personas:")
for persona in personas:
    print(f"- {persona.name}: {persona.metadata['demographics']['occupation']} from {persona.metadata['demographics']['location']}")
    print(f"  Primary climate concerns: {', '.join(impact['hazard'] for impact in persona.climate_impacts['current'])}")
    print()

# Save all personas to the output directory
persona_system.save_personas(personas)
```

## Google Colab Implementation

The code below can be run directly in Google Colab. It implements a simplified version of the Persona Development System that demonstrates the key functionality.

```python
# persona_development.py for Google Colab
# ==========================================
# This implementation ensures compatibility with Google Colab environment
# and demonstrates the core persona generation functionality.

import json
import os
import yaml
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import random

# Install required dependencies
!pip install -q openai anthropic jsonschema pyyaml matplotlib

try:
    from openai import OpenAI
except ImportError:
    import openai

try:
    import anthropic
except ImportError:
    !pip install -q anthropic

# Define stereotype patterns for checking
STEREOTYPE_PATTERNS = {
    "elderly": [
        {
            "regex": r"\btechnologically\s+illiterate\b|\bcan'?t\s+use\s+technology\b|\bstruggles?\s+with\s+all\s+technology\b",
            "type": "tech_illiteracy",
            "description": "Portraying all elderly people as unable to use technology",
            "mitigation_guidance": "Specify particular technologies they find challenging, if relevant"
        },
        {
            "regex": r"\bset\s+in\s+their\s+ways\b|\brefuses?\s+to\s+change\b|\bunwilling\s+to\s+adapt\b|\bstubborn\b",
            "type": "resistance_to_change",
            "description": "Portraying elderly as universally resistant to change",
            "mitigation_guidance": "Focus on specific concerns rather than general resistance"
        }
    ],
    "low_income": [
        {
            "regex": r"\blazy\b|\bwon'?t\s+work\b|\bunmotivated\b|\bdon'?t\s+try\b",
            "type": "work_ethic",
            "description": "Suggesting low-income individuals lack work ethic",
            "mitigation_guidance": "Focus on structural barriers rather than personal attributes"
        },
        {
            "regex": r"\bmakes?\s+bad\s+decisions\b|\bpoor\s+choices\b|\birresponsible\b",
            "type": "bad_decisions",
            "description": "Attributing poverty to poor decision making",
            "mitigation_guidance": "Consider systemic factors and limited options"
        }
    ]
}

class LLMClient:
    """Client for interacting with Large Language Models."""
    
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize LLM client.
        
        Args:
            provider: LLM provider ("openai", "anthropic", etc.)
            api_key: API key for the provider (defaults to environment variable)
            model: Model to use (defaults to provider's recommended model)
        """
        self.provider = provider.lower()
        
        # Setup API key
        if api_key:
            self.api_key = api_key
        else:
            # Try to get from environment
            env_var = f"{provider.upper()}_API_KEY"
            self.api_key = os.environ.get(env_var)
            if not self.api_key:
                raise ValueError(f"API key not provided and {env_var} not found in environment")
        
        # Set default models
        if not model:
            if provider == "openai":
                self.model = "gpt-4-turbo"
            elif provider == "anthropic":
                self.model = "claude-3-opus-20240229"
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        else:
            self.model = model
        
        # Initialize client
        if provider == "openai":
            self.client = OpenAI(api_key=self.api_key)
        elif provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 temperature: float = 0.7, max_tokens: int = 2000,
                 response_format: Optional[Dict[str, str]] = None) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt for context
            temperature: Controls randomness (0-1)
            max_tokens: Maximum tokens to generate
            response_format: Optional format specification (e.g., {"type": "json_object"})
            
        Returns:
            Generated text from the LLM
        """
        try:
            if self.provider == "openai":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format:
                    kwargs["response_format"] = response_format
                
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
                
            elif self.provider == "anthropic":
                if system_prompt:
                    system = system_prompt
                else:
                    system = "You are a helpful AI assistant."
                    
                response = self.client.messages.create(
                    model=self.model,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.content[0].text
                
        except Exception as e:
            logging.error(f"Error generating text with {self.provider}: {str(e)}")
            raise

class PromptTemplates:
    """Collection of prompt templates for persona development."""
    
    # System prompt for all tasks
    SYSTEM_PROMPT = """You are an expert in climate adaptation planning and persona development.
Your task is to create scientifically accurate and ethically sensitive personas representing diverse 
community members in climate adaptation contexts. Ensure all content is factual, based on the 
provided data, and avoids stereotypes while remaining authentic to the specific local context."""
    
    # Template for generating a persona biography
    BIOGRAPHY_GENERATION = """
Create a detailed biography for a climate adaptation persona using ONLY the information provided below.

DEMOGRAPHIC PROFILE:
Name: {name}
Age: {age}
Occupation: {occupation}
Location: {location_description}
Income Level: {income_level}
Housing: {housing_situation}
Family: {family_situation}
Additional Demographics: {additional_demographics}

VULNERABILITY PROFILE:
Physical Exposure: {physical_exposure}/10
Sensitivity: {sensitivity}/10
Adaptive Capacity: {adaptive_capacity}/10
Key Vulnerability Factors: {vulnerability_factors}

CLIMATE IMPACTS:
Current Impacts: {current_impacts}
Projected Impacts (2050): {projected_impacts}

ADAPTATION CONTEXT:
Enabling Factors: {enablers}
Barriers: {barriers}
Potential Adaptation Options: {adaptation_options}

Using ONLY the information above, write a 300-word biography for this persona that:
1. Shows their personal background and connection to the area
2. Illustrates how they personally experience the specified climate impacts
3. Describes their relationship to the identified adaptation enablers and barriers
4. Avoids stereotypes while remaining authentic to their specific context
5. Uses the third person perspective throughout

The biography should be factually accurate to the template data but narratively engaging.
"""

    # Template for generating quotes
    QUOTE_GENERATION = """
Generate authentic quotes that this climate adaptation persona might say about their experiences.

PERSONA SUMMARY:
{biography_summary}

DEMOGRAPHIC BACKGROUND:
Age: {age}
Occupation: {occupation}
Location: {location}
Key Vulnerability Factors: {vulnerability_factors}

CLIMATE IMPACTS EXPERIENCED:
{climate_impacts}

LINGUISTIC PARAMETERS:
Formality Level (0-10): {formality}
Technical Knowledge (0-10): {technical}
Local Dialect Markers (0-10): {regional}
Common Phrases/Patterns: {patterns}

Generate 3 direct quotes (20-40 words each) that this persona might realistically say about:
1. Their personal experience with current climate impacts
2. Their concerns about future climate projections
3. Their perspective on a specific adaptation option relevant to them

The quotes should:
- Use first-person perspective ("I" or "we")
- Reflect their specific demographic background and values
- Use language appropriate to their formality level and technical knowledge
- Include authentic speech patterns without resorting to stereotypes
- Connect directly to their specific climate experiences, not generic statements

Format your response as a JSON array of quote objects:
[
  {"topic": "current_impacts", "text": "Quote text here..."},
  {"topic": "future_concerns", "text": "Quote text here..."},
  {"topic": "adaptation_perspective", "text": "Quote text here..."}
]
"""

    # Template for validating scientific accuracy
    SCIENTIFIC_VALIDATION = """
Evaluate the scientific accuracy of this climate adaptation persona.

PERSONA BIOGRAPHY:
{biography}

PERSONA QUOTES:
{quotes}

REFERENCE CLIMATE DATA:
{climate_data}

Please assess the following aspects of scientific accuracy:
1. Are the described climate impacts plausible for the specified location and timeframe?
2. Are the vulnerability factors realistically portrayed?
3. Are the adaptation options technically feasible and appropriate for this context?
4. Are there any scientific inaccuracies or implausible elements?
5. Does the persona correctly reflect interactions between their personal circumstances and climate risks?

Format your response as a JSON object:
{
  "accuracy_score": 0-10,
  "strengths": ["Scientifically accurate element 1", "Scientifically accurate element 2"...],
  "issues": ["Scientific issue 1", "Scientific issue 2"...],
  "recommendations": ["Recommendation 1", "Recommendation 2"...]
}
"""

@dataclass
class ClimateData:
    """Container for climate projection and historical data."""
    region: str
    projections: Dict[str, Dict[str, Dict[str, float]]]
    hazards: List[str]
    historical_trends: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_files(cls, projection_file: str, historical_file: Optional[str] = None,
                  scenario: str = "RCP8.5", time_horizons: List[int] = [2050]):
        """
        Load climate data from files.
        
        This is a simplified example - in production, this would handle various file formats
        and perform more sophisticated data processing.
        """
        # For demo purposes, we'll just create some example data
        projections = {
            "RCP4.5": {
                "2030": {"temperature_change": 0.8, "precipitation_change": 5, "sea_level_rise": 0.15},
                "2050": {"temperature_change": 1.4, "precipitation_change": 8, "sea_level_rise": 0.30},
                "2070": {"temperature_change": 1.9, "precipitation_change": 12, "sea_level_rise": 0.45}
            },
            "RCP8.5": {
                "2030": {"temperature_change": 1.0, "precipitation_change": 6, "sea_level_rise": 0.18},
                "2050": {"temperature_change": 2.1, "precipitation_change": 11, "sea_level_rise": 0.40},
                "2070": {"temperature_change": 3.4, "precipitation_change": 18, "sea_level_rise": 0.72}
            }
        }
        
        hazards = ["coastal_flooding", "extreme_heat", "drought", "storms"]
        
        historical_trends = {
            "temperature": {"change_per_decade": 0.2, "trend_start_year": 1980},
            "sea_level": {"total_rise_cm": 15, "measurement_period": "1990-2020"},
            "extreme_events": [
                {"type": "flooding", "year": 2012, "impact": "moderate"},
                {"type": "hurricane", "year": 2018, "impact": "severe"}
            ]
        }
        
        return cls(
            region="Coastal County",
            projections=projections,
            hazards=hazards,
            historical_trends=historical_trends
        )

@dataclass
class VulnerabilityData:
    """Container for demographic and vulnerability data."""
    total_population: int
    demographics: Dict[str, Dict[str, float]]
    vulnerable_groups: List[Dict[str, Any]]
    hazard_exposure: Dict[str, Dict[str, float]]
    
    @classmethod
    def from_files(cls, demographic_file: str, hazard_file: str):
        """
        Load vulnerability data from files.
        
        This is a simplified example - in production, this would handle various file formats
        and perform more sophisticated data processing.
        """
        # For demo purposes, we'll just create some example data
        demographics = {
            "age_distribution": {
                "under_18": 0.21, "18_to_25": 0.08, "26_to_45": 0.26, 
                "46_to_65": 0.27, "over_65": 0.18
            },
            "income_distribution": {
                "below_poverty": 0.18, "low_income": 0.24, "middle_income": 0.42, 
                "high_income": 0.16
            },
            "housing": {
                "renters": 0.38, "owners": 0.62, "mobile_homes": 0.07,
                "flood_zone_residents": 0.31
            },
            "employment": {
                "fishing_industry": 0.12, "tourism": 0.23, "service": 0.32,
                "professional": 0.18, "retired": 0.15
            }
        }
        
        vulnerable_groups = [
            {"name": "Low-income renters in flood zones", "population_percentage": 0.09},
            {"name": "Elderly with limited mobility", "population_percentage": 0.06},
            {"name": "Fishing families", "population_percentage": 0.12},
            {"name": "Mobile home residents", "population_percentage": 0.07}
        ]
        
        hazard_exposure = {
            "coastal_zone": {
                "flooding": 0.8,
                "storm_surge": 0.9,
                "erosion": 0.7
            },
            "urban_center": {
                "flooding": 0.5,
                "extreme_heat": 0.8,
                "water_scarcity": 0.4
            },
            "rural_inland": {
                "flooding": 0.3,
                "drought": 0.7,
                "wildfire": 0.5
            }
        }
        
        return cls(
            total_population=35000,
            demographics=demographics,
            vulnerable_groups=vulnerable_groups,
            hazard_exposure=hazard_exposure
        )

@dataclass
class Persona:
    """Represents a fully developed climate adaptation persona."""
    id: str
    type: str  # "human" or "non_human"
    name: str
    metadata: Dict[str, Any]
    climate_impacts: Dict[str, List[Dict[str, Any]]]
    adaptation_context: Dict[str, List[Dict[str, Any]]]
    biography: str
    quotes: List[Dict[str, str]]
    linguistic_parameters: Dict[str, Any]
    image_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the persona to a dictionary representation."""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "metadata": self.metadata,
            "climate_impacts": self.climate_impacts,
            "adaptation_context": self.adaptation_context,
            "biography": self.biography,
            "quotes": self.quotes,
            "linguistic_parameters": self.linguistic_parameters,
            "image_path": self.image_path
        }
    
    def save(self, output_dir: Path) -> None:
        """Save the persona to disk."""
        persona_dir = output_dir / self.id
        persona_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        with open(persona_dir / "metadata.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        # Save biography
        with open(persona_dir / "biography.md", "w") as f:
            f.write(f"# {self.name}\n\n{self.biography}\n\n## Quotes\n\n")
            for quote in self.quotes:
                f.write(f"> *{quote['text']}*\n>\n> — {self.name}, on {quote['topic']}\n\n")

class PersonaDevelopmentSystem:
    """
    System for developing climate adaptation personas based on vulnerability data.
    """
    
    def __init__(self, llm_provider: str = "openai", api_key: Optional[str] = None, 
                 output_dir: Optional[str] = None, logger=None):
        """
        Initialize the Persona Development System.
        
        Args:
            llm_provider: LLM provider to use ("openai", "anthropic", etc.)
            api_key: API key for the LLM provider
            output_dir: Directory to save generated personas
            logger: Optional logger instance
        """
        # Setup LLM client
        self.llm = LLMClient(provider=llm_provider, api_key=api_key)
        
        # Setup output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Default to current directory
            self.output_dir = Path.cwd() / "personas"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logger or self._setup_logger()
        self.logger.info(f"Persona Development System initialized with {llm_provider} provider")
        
        # Store prompt templates
        self.prompts = PromptTemplates()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the system."""
        logger = logging.getLogger("persona_development")
        logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def generate_personas(self, workshop_config: Dict[str, Any], 
                        vulnerability_data: VulnerabilityData,
                        climate_data: ClimateData,
                        target_profiles: List[Dict[str, Any]],
                        validate: bool = True) -> List[Persona]:
        """
        Generate a set of personas based on workshop configuration and vulnerability data.
        
        Args:
            workshop_config: Workshop configuration from Module 1
            vulnerability_data: Demographic and vulnerability data
            climate_data: Climate projection and historical data
            target_profiles: List of target persona profiles to generate
            validate: Whether to validate the generated personas
            
        Returns:
            List of generated Persona objects
        """
        personas = []
        
        # Extract key information from workshop config
        purpose = workshop_config.get("workshop_parameters", {}).get("purpose", {})
        constraints = workshop_config.get("workshop_parameters", {}).get("constraint_sets", {})
        
        self.logger.info(f"Generating {sum(profile['count'] for profile in target_profiles)} personas")
        
        # Generate personas for each target profile
        for profile in target_profiles:
            for i in range(profile.get("count", 1)):
                self.logger.info(f"Generating {profile['demographic']} persona in {profile['location']}")
                
                # Generate template data for this persona
                template_data = self._generate_template_data(
                    profile=profile,
                    vulnerability_data=vulnerability_data,
                    climate_data=climate_data
                )
                
                # Generate persona ID
                persona_id = f"persona_{profile['demographic']}_{profile['location']}_{i+1}"
                
                # Generate biography using LLM
                biography = self._generate_biography(template_data)
                
                # Check for stereotypes
                stereotypes = self._check_stereotypes(biography, profile['demographic'])
                if stereotypes:
                    self.logger.warning(f"Detected stereotypes in {persona_id}: {stereotypes}")
                    # Regenerate with more explicit guidance
                    biography = self._regenerate_biography(template_data, stereotypes)
                
                # Generate quotes
                quotes = self._generate_quotes(template_data, biography)
                
                # Create persona object
                persona = Persona(
                    id=persona_id,
                    type="human" if profile.get("type", "human") == "human" else "non_human",
                    name=template_data["name"],
                    metadata={
                        "demographics": {
                            "age": template_data["age"],
                            "gender": template_data.get("gender", ""),
                            "occupation": template_data["occupation"],
                            "income_level": template_data["income_level"],
                            "housing_type": template_data["housing_situation"],
                            "location": template_data["location_description"]
                        },
                        "vulnerability_factors": {
                            "physical_exposure": template_data["physical_exposure"] / 10,
                            "sensitivity": template_data["sensitivity"] / 10,
                            "adaptive_capacity": template_data["adaptive_capacity"] / 10
                        },
                        "value_priorities": self._generate_values(template_data),
                        "workshop_relevance": {
                            "stakeholder_group": profile["demographic"],
                            "priority_level": "high" if "priority" in profile and profile["priority"] else "medium",
                            "perspective_contribution": template_data.get("perspective_contribution", "")
                        }
                    },
                    climate_impacts={
                        "current": self._format_climate_impacts(template_data["current_impacts"]),
                        "projected_2050": self._format_climate_impacts(template_data["projected_impacts"])
                    },
                    adaptation_context={
                        "enablers": self._format_adaptation_factors(template_data["enablers"]),
                        "barriers": self._format_adaptation_factors(template_data["barriers"]),
                        "adaptation_options": self._format_adaptation_options(template_data["adaptation_options"])
                    },
                    biography=biography,
                    quotes=quotes,
                    linguistic_parameters={
                        "formality_level": template_data["formality"] / 10,
                        "technical_vocabulary": template_data["technical"] / 10,
                        "regional_markers": template_data["regional"] / 10,
                        "speech_patterns": template_data["patterns"].split(", ")
                    }
                )
                
                # Validate if requested
                if validate:
                    self._validate_persona(persona, climate_data)
                
                personas.append(persona)
        
        self.logger.info(f"Generated {len(personas)} personas successfully")
        return personas
    
    def _generate_template_data(self, profile: Dict[str, Any], 
                              vulnerability_data: VulnerabilityData,
                              climate_data: ClimateData) -> Dict[str, Any]:
        """
        Generate template data for a persona based on the profile and vulnerability data.
        
        Args:
            profile: Target profile for the persona
            vulnerability_data: Demographic and vulnerability data
            climate_data: Climate projection and historical data
            
        Returns:
            Dictionary of template data for generating the persona
        """
        # This would normally involve sophisticated data processing
        # For demo purposes, we'll use a simplified approach with some randomization
        
        # Generate basic demographic information
        demographic = profile["demographic"]
        location = profile["location"]
        
        # Define name based on demographic (in a real system, this would use diverse name databases)
        if demographic == "elderly":
            names = [("Eleanor Martinez", "female"), ("Robert Chen", "male"), ("Gloria Washington", "female")]
        elif demographic == "fishing_industry":
            names = [("Miguel Rodriguez", "male"), ("Sarah Lowe", "female"), ("James Nguyen", "male")]
        elif demographic == "low_income":
            names = [("Tamika Johnson", "female"), ("Luis Morales", "male"), ("Deshawn Williams", "male")]
        elif demographic == "youth":
            names = [("Zoe Chen", "female"), ("Jamal Brown", "male"), ("Tyler Smith", "male")]
        elif demographic == "business_owner":
            names = [("Priya Patel", "female"), ("Marcus Jackson", "male"), ("Sophia Lee", "female")]
        elif demographic == "parent":
            names = [("Maria Gonzalez", "female"), ("David Kim", "male"), ("Aisha Abdullah", "female")]
        else:
            names = [("Alex Rivera", "non-binary"), ("Sam Taylor", "male"), ("Jordan Casey", "female")]
        
        # Select a name randomly
        name, gender = random.choice(names)
        
        # Define age based on demographic
        if demographic == "elderly":
            age = random.randint(68, 85)
        elif demographic == "youth":
            age = random.randint(16, 24)
        else:
            age = random.randint(30, 55)
        
        # Define occupation based on demographic
        if demographic == "elderly":
            occupations = ["retired teacher", "retired postal worker", "retired small business owner"]
        elif demographic == "fishing_industry":
            occupations = ["commercial fisher", "seafood processor", "boat repair technician"]
        elif demographic == "business_owner":
            occupations = ["restaurant owner", "retail shop owner", "tour company operator"]
        elif demographic == "low_income":
            occupations = ["service worker", "part-time cashier", "home health aide"]
        elif demographic == "youth":
            occupations = ["student", "part-time retail worker", "apprentice"]
        elif demographic == "parent":
            occupations = ["teacher", "nurse", "office administrator"]
        else:
            occupations = ["local government employee", "construction worker", "healthcare provider"]
        
        occupation = random.choice(occupations)
        
        # Define location description based on location
        if location == "coastal":
            locations = ["waterfront property", "coastal neighborhood", "fishing dock area"]
        elif location == "flood_zone":
            locations = ["riverside community", "low-lying neighborhood", "flood-prone area"]
        elif location == "urban":
            locations = ["downtown apartment", "urban residential district", "city center"]
        elif location == "downtown":
            locations = ["main street storefront", "downtown business district", "city marketplace"]
        else:
            locations = ["suburban neighborhood", "mixed-use district", "residential area"]
        
        location_description = random.choice(locations)
        
        # Define income level based on demographic
        if demographic == "low_income":
            income_level = "low"
        elif demographic == "business_owner":
            income_level = "middle_to_high"
        elif demographic == "elderly":
            income_level = "fixed_retirement"
        elif demographic == "fishing_industry":
            income_level = "moderate_seasonal"
        else:
            income_levels = ["low", "low_to_middle", "middle", "middle_to_high"]
            income_level = random.choice(income_levels)
        
        # Define housing situation
        if income_level in ["low", "low_to_middle"]:
            if random.random() > 0.3:
                housing_situation = "rental"
            else:
                housing_situation = "mobile_home"
        else:
            if random.random() > 0.4:
                housing_situation = "owned_single_family"
            else:
                housing_situation = "owned_condominium"
        
        # Define family situation
        if demographic == "elderly":
            family_situations = ["widowed, lives alone", "lives with adult child", "married, empty nest"]
        elif demographic == "parent":
            family_situations = ["single parent with children", "married with children", "extended family household"]
        elif demographic == "youth":
            family_situations = ["lives with parents", "shares apartment with roommates", "single, lives alone"]
        else:
            family_situations = ["single", "married without children", "married with children", "divorced"]
        
        family_situation = random.choice(family_situations)
        
        # Generate vulnerability scores
        if location in ["coastal", "flood_zone"]:
            physical_exposure = random.randint(7, 9)
        else:
            physical_exposure = random.randint(4, 7)
        
        # Sensitivity based on demographic
        if demographic in ["elderly", "low_income"]:
            sensitivity = random.randint(7, 9)
        else:
            sensitivity = random.randint(3, 7)
        
        # Adaptive capacity
        if income_level in ["low", "fixed_retirement"]:
            adaptive_capacity = random.randint(2, 5)
        else:
            adaptive_capacity = random.randint(5, 8)
        
        # Vulnerability factors
        vulnerability_factors = []
        if demographic == "elderly":
            vulnerability_factors.append("age-related health concerns")
            vulnerability_factors.append("limited mobility")
            if adaptive_capacity < 5:
                vulnerability_factors.append("fixed income constraints")
        
        if demographic == "low_income":
            vulnerability_factors.append("financial constraints")
            vulnerability_factors.append("limited transportation options")
            if housing_situation == "rental":
                vulnerability_factors.append("housing insecurity")
        
        if location in ["coastal", "flood_zone"]:
            vulnerability_factors.append("direct exposure to flooding")
            
        if demographic == "fishing_industry":
            vulnerability_factors.append("livelihood dependent on threatened ecosystem")
            vulnerability_factors.append("seasonal income fluctuations")
            
        # Climate impacts
        current_impacts = []
        projected_impacts = []
        
        if location == "coastal":
            current_impacts.append("Experiences periodic flooding during king tides")
            current_impacts.append("Has noticed shoreline erosion near property")
            projected_impacts.append("Property at risk of regular inundation by 2050")
            projected_impacts.append("Access road likely to be compromised by sea level rise")
            
        elif location == "flood_zone":
            current_impacts.append("Neighborhood floods during heavy rain events")
            current_impacts.append("Has experienced property damage from past floods")
            projected_impacts.append("Flood insurance becoming increasingly expensive")
            projected_impacts.append("Models suggest property may become uninsurable by 2050")
            
        elif location == "urban" or location == "downtown":
            current_impacts.append("Experiences urban heat island effect during summer")
            current_impacts.append("Occasional disruption to utilities during extreme weather")
            projected_impacts.append("Increasing cooling costs as heat waves intensify")
            projected_impacts.append("Infrastructure stress during more frequent extreme weather events")
            
        # Add general impacts
        current_impacts.append("Rising utility costs related to climate events")
        projected_impacts.append("Health risks from changing disease patterns")
        
        # Adaptation enablers and barriers
        enablers = []
        barriers = []
        
        # Enablers based on situation
        if housing_situation.startswith("owned"):
            enablers.append("Home ownership provides adaptation decision authority")
        
        if demographic == "business_owner":
            enablers.append("Business provides motivation for long-term planning")
            
        if income_level in ["middle", "middle_to_high"]:
            enablers.append("Financial resources for some adaptation measures")
            
        if occupation in ["local government employee"]:
            enablers.append("Professional knowledge of local planning processes")
            
        # Additional random enablers
        potential_enablers = [
            "Strong community connections",
            "Previous experience with climate events",
            "Access to reliable information sources",
            "Technical skills relevant to adaptation",
            "Family support network"
        ]
        enablers.append(random.choice(potential_enablers))
        
        # Barriers based on situation
        if demographic == "elderly":
            barriers.append("Physical limitations for implementing some measures")
            
        if income_level in ["low", "low_to_middle", "fixed_retirement"]:
            barriers.append("Financial constraints for major adaptations")
            
        if housing_situation == "rental":
            barriers.append("Limited authority to modify property")
            
        if demographic == "fishing_industry":
            barriers.append("Occupation tied to specific location")
            
        # Additional random barriers
        potential_barriers = [
            "Limited awareness of available programs",
            "Competing daily priorities",
            "Uncertainty about future climate conditions",
            "Lack of technical knowledge",
            "Distrust of government initiatives"
        ]
        barriers.append(random.choice(potential_barriers))
        
        # Adaptation options
        adaptation_options = []
        
        # Location-specific options
        if location in ["coastal", "flood_zone"]:
            adaptation_options.append({"option": "Home elevation", "feasibility": "medium" if income_level not in ["low"] else "low", "preference": "medium"})
            adaptation_options.append({"option": "Flood-proofing measures", "feasibility": "high", "preference": "high"})
            adaptation_options.append({"option": "Managed retreat", "feasibility": "low", "preference": "low"})
            
        if location in ["urban", "downtown"]:
            adaptation_options.append({"option": "Heat-resistant building upgrades", "feasibility": "medium", "preference": "high"})
            adaptation_options.append({"option": "Community cooling centers", "feasibility": "high", "preference": "high"})
            
        # General options
        adaptation_options.append({"option": "Emergency preparedness planning", "feasibility": "high", "preference": "high"})
        adaptation_options.append({"option": "Community resilience initiatives", "feasibility": "medium", "preference": "medium"})
        
        # Linguistic parameters
        if demographic == "elderly":
            formality = random.randint(6, 9)
            technical = random.randint(3, 7)
            regional = random.randint(5, 8)
            patterns = "historical references, measured pace, traditional phrases"
        elif demographic == "youth":
            formality = random.randint(3, 6)
            technical = random.randint(5, 8)
            regional = random.randint(4, 7)
            patterns = "contemporary references, direct, technology-aware"
        elif demographic == "fishing_industry":
            formality = random.randint(4, 7)
            technical = random.randint(6, 9)
            regional = random.randint(7, 9)
            patterns = "industry terminology, practical focus, weather-aware"
        elif demographic == "business_owner":
            formality = random.randint(5, 8)
            technical = random.randint(5, 8)
            regional = random.randint(3, 6)
            patterns = "cost-benefit perspective, community-minded, practical"
        else:
            formality = random.randint(4, 8)
            technical = random.randint(4, 7)
            regional = random.randint(4, 7)
            patterns = "pragmatic, family-focused, community-oriented"
        
        # Perspective contribution
        if demographic == "elderly":
            perspective_contribution = "historical knowledge, long-term observation of changes"
        elif demographic == "fishing_industry":
            perspective_contribution = "direct observation of marine ecosystem changes, livelihood impacts"
        elif demographic == "low_income":
            perspective_contribution = "resource constraint challenges, vulnerabilities in existing systems"
        elif demographic == "business_owner":
            perspective_contribution = "economic perspective, community investment viewpoint"
        elif demographic == "youth":
            perspective_contribution = "future-oriented perspective, technological solutions"
        elif demographic == "parent":
            perspective_contribution = "intergenerational concerns, family adaptation needs"
        else:
            perspective_contribution = "unique local knowledge, community connections"
        
        # Additional demographics that might influence vulnerability or perspective
        additional_demographics = ""
        if random.random() > 0.7:
            options = [
                f"First-generation immigrant from {random.choice(['Central America', 'Southeast Asia', 'Caribbean'])}",
                "Volunteer with local emergency response team",
                "Active in local environmental advocacy",
                "Has lived in the area for over 30 years",
                "Recently moved to the area",
                "Caregiver for a family member with a disability"
            ]
            additional_demographics = random.choice(options)
        
        # Compile all template data
        template_data = {
            "name": name,
            "gender": gender,
            "age": age,
            "occupation": occupation,
            "location_description": location_description,
            "income_level": income_level,
            "housing_situation": housing_situation,
            "family_situation": family_situation,
            "additional_demographics": additional_demographics,
            "physical_exposure": physical_exposure,
            "sensitivity": sensitivity,
            "adaptive_capacity": adaptive_capacity,
            "vulnerability_factors": ", ".join(vulnerability_factors),
            "current_impacts": "; ".join(current_impacts),
            "projected_impacts": "; ".join(projected_impacts),
            "enablers": "; ".join(enablers),
            "barriers": "; ".join(barriers),
            "adaptation_options": adaptation_options,
            "formality": formality,
            "technical": technical,
            "regional": regional,
            "patterns": patterns,
            "perspective_contribution": perspective_contribution
        }
        
        return template_data
    
    def _generate_biography(self, template_data: Dict[str, Any]) -> str:
        """
        Generate a biography for a persona using the LLM.
        
        Args:
            template_data: Template data for the persona
            
        Returns:
            Generated biography text
        """
        prompt = self.prompts.BIOGRAPHY_GENERATION.format(
            name=template_data["name"],
            age=template_data["age"],
            occupation=template_data["occupation"],
            location_description=template_data["location_description"],
            income_level=template_data["income_level"],
            housing_situation=template_data["housing_situation"],
            family_situation=template_data["family_situation"],
            additional_demographics=template_data["additional_demographics"],
            physical_exposure=template_data["physical_exposure"],
            sensitivity=template_data["sensitivity"],
            adaptive_capacity=template_data["adaptive_capacity"],
            vulnerability_factors=template_data["vulnerability_factors"],
            current_impacts=template_data["current_impacts"],
            projected_impacts=template_data["projected_impacts"],
            enablers=template_data["enablers"],
            barriers=template_data["barriers"],
            adaptation_options=", ".join(f"{o['option']} (feasibility: {o['feasibility']})" for o in template_data["adaptation_options"])
        )
        
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.prompts.SYSTEM_PROMPT
        )
        
        return response.strip()
    
    def _check_stereotypes(self, biography: str, demographic: str) -> List[Dict[str, str]]:
        """
        Check a generated biography for stereotypes.
        
        Args:
            biography: Generated biography text
            demographic: Demographic category to check against
            
        Returns:
            List of detected stereotypes
        """
        detected_stereotypes = []
        
        # Get stereotype patterns for this demographic
        stereotype_patterns = STEREOTYPE_PATTERNS.get(demographic, [])
        
        # Check for stereotypes
        for pattern in stereotype_patterns:
            if re.search(pattern["regex"], biography, re.IGNORECASE):
                detected_stereotypes.append({
                    "type": pattern["type"],
                    "description": pattern["description"],
                    "mitigation": pattern["mitigation_guidance"]
                })
        
        # Check for general stereotyping patterns
        general_patterns = [
            {
                "regex": r"\ball\s+(?:of\s+)?them\s+are\b|\bthey\s+all\s+(?:are|have|do)\b",
                "type": "generalization",
                "description": "Overgeneralization about an entire group",
                "mitigation_guidance": "Replace with specific experiences of this individual"
            },
            {
                "regex": r"\balways\b|\bnever\b",
                "type": "absolutist_language",
                "description": "Absolutist language that oversimplifies behavior",
                "mitigation_guidance": "Use more nuanced language like 'tends to' or 'often'"
            }
        ]
        
        for pattern in general_patterns:
            if re.search(pattern["regex"], biography, re.IGNORECASE):
                detected_stereotypes.append({
                    "type": pattern["type"],
                    "description": pattern["description"],
                    "mitigation": pattern["mitigation_guidance"]
                })
        
        return detected_stereotypes
    
    def _regenerate_biography(self, template_data: Dict[str, Any], 
                           stereotypes: List[Dict[str, str]]) -> str:
        """
        Regenerate a biography with guidance to avoid detected stereotypes.
        
        Args:
            template_data: Template data for the persona
            stereotypes: Detected stereotypes to avoid
            
        Returns:
            Regenerated biography text
        """
        # Create a more specific prompt with stereotype mitigation guidance
        mitigation_guidance = "\n".join([f"- Avoid {s['type']}: {s['description']}. {s['mitigation']}" for s in stereotypes])
        
        prompt = self.prompts.BIOGRAPHY_GENERATION.format(
            name=template_data["name"],
            age=template_data["age"],
            occupation=template_data["occupation"],
            location_description=template_data["location_description"],
            income_level=template_data["income_level"],
            housing_situation=template_data["housing_situation"],
            family_situation=template_data["family_situation"],
            additional_demographics=template_data["additional_demographics"],
            physical_exposure=template_data["physical_exposure"],
            sensitivity=template_data["sensitivity"],
            adaptive_capacity=template_data["adaptive_capacity"],
            vulnerability_factors=template_data["vulnerability_factors"],
            current_impacts=template_data["current_impacts"],
            projected_impacts=template_data["projected_impacts"],
            enablers=template_data["enablers"],
            barriers=template_data["barriers"],
            adaptation_options=", ".join(f"{o['option']} (feasibility: {o['feasibility']})" for o in template_data["adaptation_options"])
        )
        
        # Add stereotype mitigation guidance
        prompt += f"\n\nIMPORTANT: Avoid the following stereotypes that were detected in a previous version:\n{mitigation_guidance}"
        
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.prompts.SYSTEM_PROMPT
        )
        
        return response.strip()
    
    def _generate_quotes(self, template_data: Dict[str, Any], biography: str) -> List[Dict[str, str]]:
        """
        Generate quotes for a persona using the LLM.
        
        Args:
            template_data: Template data for the persona
            biography: Generated biography text
            
        Returns:
            List of generated quotes
        """
        # Create a summary of the biography for context
        biography_summary = biography[:300] + "..." if len(biography) > 300 else biography
        
        prompt = self.prompts.QUOTE_GENERATION.format(
            biography_summary=biography_summary,
            age=template_data["age"],
            occupation=template_data["occupation"],
            location=template_data["location_description"],
            vulnerability_factors=template_data["vulnerability_factors"],
            climate_impacts=template_data["current_impacts"] + "; " + template_data["projected_impacts"],
            formality=template_data["formality"],
            technical=template_data["technical"],
            regional=template_data["regional"],
            patterns=template_data["patterns"]
        )
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.prompts.SYSTEM_PROMPT,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            quotes = json.loads(response)
            
            # Ensure we have a list
            if not isinstance(quotes, list):
                self.logger.warning(f"Unexpected format in quote generation for {template_data['name']}")
                quotes = []
                
            return quotes
            
        except Exception as e:
            self.logger.error(f"Error generating quotes: {str(e)}")
            # Fallback to basic quotes
            return [
                {"topic": "current_impacts", "text": f"I've noticed changes in our area over the years. {template_data['current_impacts'].split(';')[0]}"},
                {"topic": "future_concerns", "text": f"I worry about what the future holds. {template_data['projected_impacts'].split(';')[0]}"},
                {"topic": "adaptation_perspective", "text": f"We need to think about how we can adapt. {template_data['adaptation_options'][0]['option']} might work for people like me."}
            ]
    
    def _validate_persona(self, persona: Persona, climate_data: ClimateData) -> None:
        """
        Validate a persona for scientific accuracy.
        
        Args:
            persona: The persona to validate
            climate_data: Climate data for reference
            
        Returns:
            None, but logs validation results
        """
        # This would normally involve more detailed validation
        # For demo purposes, we'll just do a basic check
        
        # Convert climate data to a string for the prompt
        climate_data_str = f"Region: {climate_data.region}\n"
        climate_data_str += "Projections:\n"
        for scenario, horizons in climate_data.projections.items():
            climate_data_str += f"  {scenario}:\n"
            for year, values in horizons.items():
                climate_data_str += f"    {year}: {values}\n"
        
        climate_data_str += f"Hazards: {', '.join(climate_data.hazards)}\n"
        
        if climate_data.historical_trends:
            climate_data_str += "Historical Trends:\n"
            for category, data in climate_data.historical_trends.items():
                if category == "extreme_events":
                    climate_data_str += f"  {category}: {len(data)} recorded events\n"
                else:
                    climate_data_str += f"  {category}: {data}\n"
        
        prompt = self.prompts.SCIENTIFIC_VALIDATION.format(
            biography=persona.biography,
            quotes="\n".join([f"- \"{q['text']}\" (on {q['topic']})" for q in persona.quotes]),
            climate_data=climate_data_str
        )
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.prompts.SYSTEM_PROMPT,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            validation_results = json.loads(response)
            
            # Log validation results
            self.logger.info(f"Validation results for {persona.name}: Score: {validation_results.get('accuracy_score', 'N/A')}/10")
            
            if "issues" in validation_results and validation_results["issues"]:
                self.logger.warning(f"Scientific issues detected in {persona.name}: {validation_results['issues']}")
                
            if "recommendations" in validation_results and validation_results["recommendations"]:
                self.logger.info(f"Recommendations for {persona.name}: {validation_results['recommendations']}")
                
        except Exception as e:
            self.logger.error(f"Error validating persona: {str(e)}")
    
    def _generate_values(self, template_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate value priorities for a persona based on template data.
        
        Args:
            template_data: Template data for the persona
            
        Returns:
            List of value priorities
        """
        # This is a simplified implementation
        # In a full system, this would be more sophisticated
        
        values = [
            {"value": "security", "weight": 0.0},
            {"value": "tradition", "weight": 0.0},
            {"value": "universalism", "weight": 0.0},
            {"value": "self_determination", "weight": 0.0},
            {"value": "achievement", "weight": 0.0}
        ]
        
        # Adjust weights based on demographics
        if template_data["age"] > 65:
            values[0]["weight"] += 0.3  # security
            values[1]["weight"] += 0.3  # tradition
        elif template_data["age"] < 30:
            values[3]["weight"] += 0.3  # self_determination
            values[4]["weight"] += 0.2  # achievement
            
        if "low" in template_data["income_level"]:
            values[0]["weight"] += 0.3  # security
        elif "high" in template_data["income_level"]:
            values[4]["weight"] += 0.3  # achievement
            values[3]["weight"] += 0.2  # self_determination
            
        if template_data["occupation"] in ["teacher", "nurse", "social worker"]:
            values[2]["weight"] += 0.3  # universalism
            
        if "family" in template_data["family_situation"].lower():
            values[0]["weight"] += 0.2  # security
            
        # Add random variation
        for value in values:
            value["weight"] += random.uniform(0, 0.3)
            
        # Normalize weights
        total = sum(value["weight"] for value in values)
        for value in values:
            value["weight"] = round(value["weight"] / total, 2)
            
        # Sort by weight
        values.sort(key=lambda x: x["weight"], reverse=True)
        
        return values
    
    def _format_climate_impacts(self, impacts_str: str) -> List[Dict[str, Any]]:
        """
        Format climate impacts string into structured list.
        
        Args:
            impacts_str: Semicolon-separated string of impacts
            
        Returns:
            List of impact dictionaries
        """
        impact_list = []
        
        # Split the impacts string
        impacts = [imp.strip() for imp in impacts_str.split(";") if imp.strip()]
        
        # Map to common hazards
        hazard_keywords = {
            "flood": "coastal_flooding",
            "inundation": "coastal_flooding",
            "storm": "storm_surge",
            "erosion": "coastal_erosion",
            "heat": "extreme_heat",
            "drought": "drought",
            "water": "water_scarcity",
            "utility": "infrastructure_disruption",
            "infrastructure": "infrastructure_disruption",
            "health": "health_impacts",
            "disease": "health_impacts",
            "cost": "economic_impacts",
            "insurance": "economic_impacts"
        }
        
        # Process each impact
        for impact in impacts:
            # Determine severity
            if any(term in impact.lower() for term in ["severe", "extreme", "significant", "major"]):
                severity = "severe"
            elif any(term in impact.lower() for term in ["moderate", "occasional", "some"]):
                severity = "moderate"
            else:
                severity = "mild"
                
            # Determine hazard type
            hazard = "other"
            for keyword, hazard_type in hazard_keywords.items():
                if keyword in impact.lower():
                    hazard = hazard_type
                    break
            
            impact_list.append({
                "hazard": hazard,
                "description": impact,
                "severity": severity
            })
        
        return impact_list
    
    def _format_adaptation_factors(self, factors_str: str) -> List[Dict[str, Any]]:
        """
        Format adaptation factors string into structured list.
        
        Args:
            factors_str: Semicolon-separated string of factors
            
        Returns:
            List of factor dictionaries
        """
        factor_list = []
        
        # Split the factors string
        factors = [factor.strip() for factor in factors_str.split(";") if factor.strip()]
        
        # Process each factor
        for factor in factors:
            # Determine factor type
            if any(term in factor.lower() for term in ["financial", "economic", "income", "cost", "resource"]):
                factor_type = "financial"
            elif any(term in factor.lower() for term in ["knowledge", "information", "aware", "technical"]):
                factor_type = "knowledge"
            elif any(term in factor.lower() for term in ["social", "community", "network", "support"]):
                factor_type = "social"
            elif any(term in factor.lower() for term in ["physical", "health", "mobility"]):
                factor_type = "physical"
            elif any(term in factor.lower() for term in ["governance", "policy", "regulation", "authority"]):
                factor_type = "governance"
            else:
                factor_type = "other"
            
            factor_list.append({
                "factor": factor_type,
                "description": factor
            })
        
        return factor_list
    
    def _format_adaptation_options(self, options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format adaptation options into structured list.
        
        Args:
            options: List of adaptation option dictionaries
            
        Returns:
            Formatted list of option dictionaries
        """
        option_list = []
        
        for option in options:
            option_list.append({
                "option": option["option"],
                "feasibility": option["feasibility"],
                "preference": option["preference"]
            })
        
        return option_list
    
    def save_personas(self, personas: List[Persona]) -> None:
        """
        Save all personas to the output directory.
        
        Args:
            personas: List of personas to save
            
        Returns:
            None
        """
        # Create index file
        index = {
            "count": len(personas),
            "personas": []
        }
        
        # Save each persona
        for persona in personas:
            persona.save(self.output_dir)
            
            # Add to index
            index["personas"].append({
                "id": persona.id,
                "name": persona.name,
                "type": persona.type,
                "demographic": persona.metadata["workshop_relevance"]["stakeholder_group"]
            })
        
        # Save index file
        with open(self.output_dir / "persona_index.json", "w") as f:
            json.dump(index, f, indent=2)
            
        self.logger.info(f"Saved {len(personas)} personas to {self.output_dir}")

# Demo implementation
def run_demo():
    """Run a demonstration of the Persona Development System."""
    print("\n=== Persona Development System Demo ===\n")
    
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not found in environment variables.")
        api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Load example workshop configuration from Module 1
    workshop_config = {
        "workshop_parameters": {
            "purpose": {
                "context": "Bayshore coastal community faces amplified flood risk under RCP-8.5 scenario with projected 0.5m sea level rise by 2050. Vulnerability assessment indicates 40% of residents are in high-risk zones, with disproportionate exposure among low-income renters (62%), small-scale fishers dependent on vulnerable infrastructure, and elderly residents (30% of exposed population).",
                "objectives": "Surface specific barriers faced by low-income renters and small-scale fishers; identify adaptation options that address intersectional vulnerability; co-produce actionable strategies with implementation pathways.",
                "outputs_expected": "Generate ≥10 ranked adaptation options (AO) with feasibility assessments, equity considerations, and resource requirements. Document vulnerable group impacts (DN) and create stakeholder implementation considerations (IC)."
            },
            "constraint_sets": {
                "C_purpose": [
                    "#BOUNDARY Content must remain within coastal flood adaptation context for Bayshore community",
                    "#PERSPECTIVE Special attention to low-income renters and small-scale fishers required",
                    "#GOAL Discussions must contribute to actionable adaptation strategies"
                ],
                "C_equity": [
                    "#REQUIRE Adaptation options must explicitly consider distributional impacts",
                    "#REQUIRE Implementation considerations must address access barriers"
                ]
            }
        }
    }
    
    # Load example data
    vulnerability_data = VulnerabilityData.from_files(
        demographic_file="data/demographic_profile.csv",
        hazard_file="data/hazard_exposure.geojson"
    )
    
    climate_data = ClimateData.from_files(
        projection_file="data/climate_projections.nc",
        historical_file="data/historical_climate.csv"
    )
    
    print("Loaded workshop configuration and data.")
    print(f"Workshop context: {workshop_config['workshop_parameters']['purpose']['context'][:100]}...")
    
    # Define target personas based on workshop focus
    target_profiles = [
        {"demographic": "elderly", "location": "coastal", "count": 1},
        {"demographic": "low_income", "location": "flood_zone", "count": 1},
        {"demographic": "fishing_industry", "location": "coastal", "count": 1}
    ]
    
    print(f"\nGenerating {sum(p['count'] for p in target_profiles)} personas...")
    
    # Create temporary directory for output
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize the system
        persona_system = PersonaDevelopmentSystem(
            llm_provider="openai",
            api_key=api_key,
            output_dir=tmpdir
        )
        
        # Generate personas
        personas = persona_system.generate_personas(
            workshop_config=workshop_config,
            vulnerability_data=vulnerability_data,
            climate_data=climate_data,
            target_profiles=target_profiles,
            validate=True
        )
        
        # Display summary
        print("\nGenerated Personas:")
        for i, persona in enumerate(personas, 1):
            print(f"\n--- Persona {i}: {persona.name} ---")
            print(f"Stakeholder Group: {persona.metadata['workshop_relevance']['stakeholder_group']}")
            print(f"Location: {persona.metadata['demographics']['location']}")
            print(f"Age: {persona.metadata['demographics']['age']}")
            print(f"Occupation: {persona.metadata['demographics']['occupation']}")
            
            print("\nVulnerability Profile:")
            print(f"Physical Exposure: {persona.metadata['vulnerability_factors']['physical_exposure']:.2f}")
            print(f"Sensitivity: {persona.metadata['vulnerability_factors']['sensitivity']:.2f}")
            print(f"Adaptive Capacity: {persona.metadata['vulnerability_factors']['adaptive_capacity']:.2f}")
            
            print("\nClimate Impacts:")
            for impact in persona.climate_impacts["current"]:
                print(f"- {impact['hazard']} ({impact['severity']}): {impact['description']}")
            
            print("\nAdaptation Options:")
            for option in persona.adaptation_context["adaptation_options"]:
                print(f"- {option['option']} (Feasibility: {option['feasibility']}, Preference: {option['preference']})")
            
            print("\nBiography Excerpt:")
            print(f"{persona.biography[:200]}...")
            
            print("\nSample Quotes:")
            for quote in persona.quotes[:2]:
                print(f'"{quote["text"]}" (re: {quote["topic"]})')
            
            print(f"\nPersona saved to {tmpdir}/{persona.id}/")
        
        # Save all personas
        persona_system.save_personas(personas)
        
        print(f"\nAll personas saved to {tmpdir}")
        print("Demo complete!\n")
        
        # Store the personas for display in visualizations
        global demo_personas
        demo_personas = personas

# Run the demo
run_demo()
```

## Examining the Results

Let's visualize some aspects of the generated personas:

```python
import matplotlib.pyplot as plt
import numpy as np

# Visualize vulnerability factors
if 'demo_personas' in globals():
    # Extract vulnerability data
    names = [p.name for p in demo_personas]
    exposure = [p.metadata['vulnerability_factors']['physical_exposure'] for p in demo_personas]
    sensitivity = [p.metadata['vulnerability_factors']['sensitivity'] for p in demo_personas]
    adaptive_capacity = [p.metadata['vulnerability_factors']['adaptive_capacity'] for p in demo_personas]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set the positions of the bars on the x-axis
    bar_width = 0.25
    r1 = np.arange(len(names))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create the bars
    ax.bar(r1, exposure, width=bar_width, label='Physical Exposure', color='#ff9999')
    ax.bar(r2, sensitivity, width=bar_width, label='Sensitivity', color='#66b3ff')
    ax.bar(r3, adaptive_capacity, width=bar_width, label='Adaptive Capacity', color='#99ff99')
    
    # Add labels and title
    ax.set_xlabel('Personas')
    ax.set_ylabel('Vulnerability Score (0-1)')
    ax.set_title('Vulnerability Factors by Persona')
    ax.set_xticks([r + bar_width for r in range(len(names))])
    ax.set_xticklabels(names)
    ax.legend()
    
    # Display the plot
    plt.tight_layout()
    plt.show()
    
    # Create a radar chart for adaptation options
    fig = plt.figure(figsize=(12, 8))
    
    for i, persona in enumerate(demo_personas, 1):
        ax = fig.add_subplot(1, len(demo_personas), i, polar=True)
        
        # Get adaptation options
        options = [option['option'] for option in persona.adaptation_context['adaptation_options']]
        feasibility = [0.3 if option['feasibility'] == 'low' else 0.6 if option['feasibility'] == 'medium' else 0.9 for option in persona.adaptation_context['adaptation_options']]
        preference = [0.3 if option['preference'] == 'low' else 0.6 if option['preference'] == 'medium' else 0.9 for option in persona.adaptation_context['adaptation_options']]
        
        # Number of variables
        N = len(options)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add the first point again to close the loop
        feasibility += feasibility[:1]
        preference += preference[:1]
        
        # Draw the axes and labels
        plt.xticks(angles[:-1], options, size=8)
        
        # Draw the feasibility line
        ax.plot(angles, feasibility, 'b-', linewidth=1, label='Feasibility')
        ax.fill(angles, feasibility, 'b', alpha=0.1)
        
        # Draw the preference line
        ax.plot(angles, preference, 'r-', linewidth=1, label='Preference')
        ax.fill(angles, preference, 'r', alpha=0.1)
        
        # Set the title
        ax.set_title(persona.name, size=11)
        
        # Add legend
        if i == 1:
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.show()
else:
    print("Please run the demo first to generate visualization data")
```

## Conclusion

The Persona Development System transforms abstract climate data and vulnerability assessments into richly detailed, scientifically accurate, and ethically constructed personas. These personas serve as the foundation for simulated focus group discussions in climate adaptation planning.

Key capabilities demonstrated include:

1. **Integration of climate science with narrative development**: The system combines rigorous vulnerability assessment with engaging biographical narratives.

2. **Ethical representation of diverse stakeholders**: Built-in stereotype detection and validation mechanisms ensure respectful portrayal of vulnerable populations.

3. **LLM-enhanced persona creation**: Large Language Models transform structured data into rich, contextually relevant persona profiles while maintaining scientific accuracy.

4. **Comprehensive persona profiles**: Each persona includes detailed metadata, climate impacts, adaptation context, biography, quotes, and linguistic parameters needed for realistic dialogue simulation.

This approach creates a diverse "virtual community" that can explore climate impacts and adaptation options from multiple perspectives, enhancing the inclusivity and effectiveness of climate adaptation planning.
