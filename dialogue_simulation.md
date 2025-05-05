# Technical Documentation
## Module 3: Multi-Agent Dialogue Simulation

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

The Multi-Agent Dialogue Simulation module is the third component in our LLM-simulated climate adaptation workshop framework. This module orchestrates realistic focus group interactions between the diverse personas created in Module 2, following the workshop structure defined in Module 1.

### Purpose

This module creates synthetic yet authentic dialogue that simulates a climate adaptation workshop by:
- Implementing a multi-agent architecture with moderator and persona agents
- Modeling realistic group dynamics including agreement, contention, and opinion shifts
- Orchestrating a structured four-stage workshop flow
- Managing contextual memory to maintain coherent discussions despite token limitations
- Enabling cross-referencing and building upon ideas between synthetic participants
- Generating actionable adaptation insights from diverse stakeholder perspectives

### Process Flow

The module follows this sequential workflow:
1. **Agent Initialization**
   - Load workshop configuration from Module 1
   - Initialize persona agents using profiles from Module 2
   - Configure moderator agent with facilitation parameters
   - Establish agent communication channels and memory structure

2. **Workshop Execution**
   - Sequentially process each of the four workshop stages:
     * S1: Introduction (linear pattern)
     * S2: Impact Exploration (spoke-and-wheel pattern) 
     * S3: Adaptation Brainstorm (mesh network pattern)
     * S4: Strategy Evaluation (bipartite evaluation pattern)
   - For each stage, the moderator agent manages turn selection, topic coverage, and contextual interventions

3. **Group Dynamics Simulation**
   - Implement peer-awareness prompting to enable cross-referencing
   - Deploy interaction controllers for agreement and contention
   - Calculate stance evolution and opinion shifts through dialogue
   - Trigger experience-sharing for concrete climate impact examples

4. **Dialogue Management**
   - Handle contextual retrieval to maintain discussion coherence
   - Detect and correct semantic drift, elaboration needs, or ambiguity
   - Generate appropriate moderator interventions to guide discussion
   - Maintain topic coverage tracking to ensure key areas are addressed

5. **Output Generation**
   - Produce complete workshop transcript with all interactions
   - Generate stage summaries capturing key insights
   - Organize identified adaptation options and implementation considerations
   - Record stance evolution and group convergence data

## Input and Output Specification

### Input

The module accepts the structured outputs from the previous modules plus LLM configuration:

| Input Category | Description | Format | Required |
|----------------|-------------|--------|----------|
| Workshop Configuration | Structured output from Module 1 | JSON | Yes |
| Persona Profiles | Set of personas from Module 2 | JSON directory | Yes |
| LLM Provider Settings | API credentials and model selection | Dict | Yes |
| Interaction Controls | Optional parameters for fine-tuning interaction patterns | JSON | No |
| Memory Parameters | Settings for the buffer-retrieval system | Dict | No |
| Seed Statements | Optional predefined statements to guide discussion | JSON | No |

#### Workshop Configuration Example:
```json
{
  "workshop_parameters": {
    "purpose": {
      "context": "Bayshore coastal community faces amplified flood risk under RCP-8.5 scenario with projected 0.5m sea level rise by 2050...",
      "objectives": "Surface specific barriers faced by low-income renters and small-scale fishers...",
      "outputs_expected": "Generate ≥10 ranked adaptation options (AO) with feasibility assessments..."
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
      // Additional stages...
    ],
    "outputs": [
      {
        "id": "AO_01",
        "class": "AO",
        "format": "structured_list",
        "fields": ["description", "primary_beneficiaries", "implementation_challenges", "resource_needs"],
        "quality": {"feasibility_min": 0.70, "equity_required": true, "cobenefit_note": true},
        "stage": "S3"
      },
      // Additional outputs...
    ],
    "constraint_sets": {
      "C_purpose": [
        "#BOUNDARY Content must remain within coastal flood adaptation context for Bayshore community",
        // Additional constraints...
      ]
    }
  }
}
```

#### Persona Profiles Example:
```json
{
  "id": "persona_elderly_coastal_1",
  "type": "human",
  "name": "Eleanor Martinez",
  "metadata": {
    "demographics": {
      "gender": "female",
      "age": 73,
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
    ]
  },
  "climate_impacts": {
    "current": [
      {"hazard": "coastal_flooding", "description": "Yard floods during king tides", "severity": "moderate"},
      {"hazard": "extreme_heat", "description": "Health impacts during summer heat waves", "severity": "high"}
    ],
    "projected_2050": [
      {"hazard": "coastal_flooding", "description": "Home likely inundated during major storms", "severity": "severe"}
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
    {"text": "I've seen the water getting closer every year. The tide comes up into my garden now, which never happened when we first moved here.", "topic": "observed_changes"},
    {"text": "At my age, starting over somewhere new just isn't realistic. This is my home, all my memories are here.", "topic": "retreat_options"}
  ],
  "linguistic_parameters": {
    "formality_level": 0.7,
    "technical_vocabulary": 0.3,
    "regional_markers": 0.4,
    "speech_patterns": ["reflective", "historically_oriented", "direct"]
  }
}
```

### Output

The module produces a comprehensive workshop simulation:

```
workshop_simulation
├── transcript.md               # Complete dialogue transcript
├── transcript.json             # Structured dialogue data
├── stage_summaries             
│   ├── S1_summary.md          # Stage 1 summary
│   ├── S2_summary.md          # Stage 2 summary
│   ├── S3_summary.md          # Stage 3 summary
│   └── S4_summary.md          # Stage 4 summary
├── workshop_outputs           
│   ├── AO_01.json             # Adaptation Options output
│   ├── DN_01.json             # Descriptive Narrative output
│   └── IC_01.json             # Implementation Considerations output
├── analysis                   
│   ├── stance_evolution.json  # Tracked opinion changes
│   ├── topic_coverage.json    # Coverage of required topics
│   └── interaction_data.json  # Participation and dynamics metrics
└── metadata.json              # Simulation metadata
```

#### Dialogue Transcript Structure:

The transcript.json file contains a structured representation of the entire simulation:

```json
{
  "workshop_id": "coastal_adaptation_20250428_001",
  "timestamp": "2025-04-28T14:30:00Z",
  "personas": ["Eleanor_Martinez", "Miguel_Rodriguez", "Tamika_Johnson"],
  "stages": [
    {
      "id": "S1",
      "name": "Introduction",
      "exchanges": [
        {
          "id": "S1_T1",
          "speaker": "moderator",
          "content": "Welcome everyone to our discussion on coastal flooding adaptation in Bayshore. Let's start by introducing ourselves and sharing your personal experiences with climate impacts.",
          "topics": ["workshop_introduction"],
          "timestamp": "2025-04-28T14:30:05Z"
        },
        {
          "id": "S1_T2",
          "speaker": "Eleanor_Martinez",
          "content": "Hello everyone, I'm Eleanor. I've lived in my home near the shore for over 40 years. I've seen the water getting closer every year. The tide comes up into my garden now, which never happened when we first moved here. At my age, starting over somewhere new just isn't realistic. This is my home, all my memories are here.",
          "topics": ["personal_background", "observed_changes", "emotional_connection"],
          "referring_to": null,
          "stance_indicators": {
            "retreat": -0.8,
            "community_resilience": 0.7
          },
          "timestamp": "2025-04-28T14:31:10Z"
        },
        // Additional exchanges...
      ]
    },
    // Additional stages...
  ],
  "outputs": {
    "AO_01": {
      "items": [
        {
          "description": "Community resilience hub with backup power and cooling",
          "primary_beneficiaries": "Elderly residents, low-income families without transportation, fishing community for emergency equipment storage",
          "implementation_challenges": "Funding for construction and ongoing maintenance, site selection that's accessible but protected from flooding",
          "resource_needs": "Grant funding ($750K-1.2M), dedicated staff position for operations, community volunteers"
        },
        // Additional adaptation options...
      ]
    },
    // Additional outputs...
  }
}
```

## LLM Integration

The Multi-Agent Dialogue Simulation system uses Large Language Models extensively throughout the simulation process, creating a dynamic yet controlled communication environment.

### Key LLM Integration Points

1. **Moderator Agent**
   - Manages the overall workshop flow, stage transitions, and turn allocation
   - Detects discussion quality issues and generates appropriate interventions
   - Creates summaries to maintain contextual awareness across the simulation
   - Ensures complete topic coverage across all required areas

2. **Persona Agents**
   - Generate dialogue contributions reflecting the persona's demographics, values, and experiences
   - Process and respond to other personas' contributions through peer-awareness
   - Adapt stance vectors based on the dialogue context
   - Express emotion and share personal climate experiences

3. **Memory and Context Management**
   - Implements sliding buffer-retrieval systems to maintain coherent discussion
   - Encodes and retrieves relevant context despite token limitations
   - Tracks semantic drift to ensure discussion remains on-topic
   - Maintains character consistency throughout the simulation

4. **Group Dynamics Modeling**
   - Simulates agreement dynamics between similar persona viewpoints
   - Generates appropriate disagreement when personas have conflicting values or priorities
   - Models opinion shifts and convergence over the course of the workshop
   - Implements experience-sharing triggers to maintain concrete discussion

### Prompt Templates

The module uses specialized prompt templates for each agent role and interaction type. For example:

#### Moderator Introduction Prompt
```
You are a skilled workshop facilitator leading a climate adaptation focus group.

WORKSHOP CONTEXT:
{workshop_purpose}

CURRENT STAGE:
Stage: {stage_name}
Objectives: {stage_objectives}
Required Topics: {required_topics}

Craft an introduction to the {stage_name} that:
1. Clearly explains the purpose of this stage
2. Makes participants feel comfortable sharing their perspectives
3. Provides any necessary context about climate impacts relevant to this discussion
4. Asks an open-ended question to start the conversation

Keep your introduction concise (100-150 words) and conversational in tone.
```

#### Persona Response Template
```
You are roleplaying as {persona_name}, speaking in a climate adaptation workshop.

PERSONA DETAILS:
{persona_summary}
Values: {value_priorities}
Climate Impacts: {climate_impacts}
Speaking Style: Formality ({formality_level}/10), Technical ({technical_level}/10), Regional ({regional_level}/10)
Speech Patterns: {speech_patterns}

WORKSHOP CONTEXT:
{current_stage_description}
Current Topic: {current_topic}

RECENT DISCUSSION:
{recent_exchanges}

You've been asked to respond to:
{moderator_question}

INSTRUCTIONS:
1. Stay in character as {persona_name} with their unique perspective and speaking style
2. Directly address the question/topic
3. Draw on your persona's specific experiences with climate impacts
4. Express opinions aligned with your persona's values and adaptation context
5. Keep your response to 100-150 words maximum

SPECIAL DIRECTIVES:
{agreement_directive}
{contention_directive}
{experience_sharing_directive}
```

#### Contextual Intervention Template
```
You are the moderator noticing an issue in the climate adaptation workshop discussion.

ISSUE TYPE: {issue_type} (semantic_drift | shallow_engagement | ambiguity | dominated_discussion)

CONTEXT:
{relevant_context}

WORKSHOP OBJECTIVE:
{workshop_objective}

TARGET TOPIC:
{target_topic}

Generate an appropriate moderator intervention that:
1. Addresses the specific issue without criticizing participants
2. Gently redirects the conversation toward the target topic
3. Encourages deeper exploration or clarification as needed
4. Uses an engaging, supportive facilitation style

Your intervention should be 2-3 sentences (40-60 words) and conversational in tone.
```

## Implementation Details

### Key Components

1. **AgentArchitecture**: Manages the multi-agent system including moderator and persona agents
2. **DialogueManager**: Controls the flow of conversation and agent interaction patterns
3. **BufferRetrievalSystem**: Implements the memory layer for maintaining context
4. **GroupDynamicsEngine**: Models social interactions between agents
5. **OutputProcessor**: Transforms dialogue into structured workshop outputs

### Design Considerations

- **Deterministic Reproducibility**: Implementation ensures identical results when run with the same seed
- **Stance Evolution Modeling**: Systematic tracking of opinion shifts through dialogue
- **Interaction Pattern Implementation**: Specific topologies for different workshop stages
- **Token Efficiency**: Careful memory management to maximize contextual information within token constraints
- **Facilitation Authenticity**: Moderator interventions designed based on real workshop facilitation protocols

### Key Algorithms

#### Buffer-Retrieval Facilitation Loop

The core dialogue management algorithm implements a buffer-retrieval facilitation loop:

```python
def buffer_retrieval_facilitation_loop(workshop_config, personas, max_turns_per_stage, llm_client):
    """
    Implements the buffer-retrieval facilitation loop for workshop simulation.
    
    Args:
        workshop_config: Workshop configuration from Module 1
        personas: List of personas from Module 2
        max_turns_per_stage: Maximum dialogue turns per stage
        llm_client: Configured LLM client
        
    Returns:
        Complete workshop transcript and structured outputs
    """
    # Initialize workshop components
    memory = VectorMemoryBuffer(buffer_size=20)
    moderator = ModeratorAgent(workshop_config, llm_client)
    persona_agents = {p.id: PersonaAgent(p, llm_client) for p in personas}
    
    # Initialize tracking
    participation_tracker = {p_id: 0 for p_id in persona_agents.keys()}
    topic_coverage = {topic: 0 for stage in workshop_config["stages"] 
                     for topic in stage["required_topics"]}
    stance_vectors = {p_id: initialize_stance_vector(p) for p_id, p in persona_agents.items()}
    
    # Workshop transcript
    transcript = []
    
    # Process each stage
    for stage in workshop_config["stages"]:
        # Stage initialization
        stage_brief = moderator.initialize_stage(stage)
        memory.reset_for_new_stage()
        transcript.append({"speaker": "moderator", "content": stage_brief})
        memory.add(stage_brief, speaker="moderator")
        
        # Stage dialogue
        turns_completed = 0
        while turns_completed < max_turns_per_stage:
            # Check if all required topics are covered
            uncovered_topics = [t for t, count in topic_coverage.items() 
                               if t in stage["required_topics"] and count < 2]
            
            # Stage complete when all topics covered and minimum turns reached
            if len(uncovered_topics) == 0 and turns_completed >= stage["duration_points"]:
                break
                
            # Select next speaker (inverse frequency sampling)
            if uncovered_topics:
                # Select persona who might best address an uncovered topic
                next_speaker = select_persona_for_topic(uncovered_topics, persona_agents, 
                                                     participation_tracker)
                # Prepare a prompt targeting the uncovered topic
                question = moderator.generate_topic_question(uncovered_topics[0], 
                                                          memory.get_context())
            else:
                # Select persona with least participation
                weights = {p_id: 1.0/(count + 1) for p_id, count in participation_tracker.items()}
                total = sum(weights.values())
                probs = {p_id: w/total for p_id, w in weights.items()}
                next_speaker = random.choices(list(probs.keys()), 
                                          weights=list(probs.values()), k=1)[0]
                # Generate follow-up question based on recent exchange
                question = moderator.generate_followup_question(memory.get_context())
            
            # Update participation tracker
            participation_tracker[next_speaker] += 1
            
            # Generate peer context (what other personas have said)
            peer_context = memory.get_peer_context(exclude_speaker=next_speaker)
            
            # Check for agreement/contention triggers
            agreement_directive = check_agreement_trigger(next_speaker, peer_context, 
                                                        stance_vectors)
            contention_directive = check_contention_trigger(next_speaker, peer_context, 
                                                         stance_vectors)
            
            # Check for experience sharing trigger
            concrete_ratio = calculate_concrete_ratio(memory.get_recent_exchanges(5))
            experience_directive = "Share a specific personal experience" if concrete_ratio < 0.3 else ""
            
            # Generate persona response
            response = persona_agents[next_speaker].generate_response(
                question=question,
                peer_context=peer_context,
                stage=stage,
                agreement_directive=agreement_directive,
                contention_directive=contention_directive,
                experience_directive=experience_directive
            )
            
            # Add to transcript
            transcript.append({"speaker": next_speaker, "content": response})
            
            # Update memory
            memory.add(response, speaker=next_speaker)
            
            # Update topic coverage
            detected_topics = detect_topics(response, stage["required_topics"])
            for topic in detected_topics:
                topic_coverage[topic] = topic_coverage.get(topic, 0) + 1
                
            # Update stance vectors based on exchange
            update_stance_vectors(stance_vectors, next_speaker, response, peer_context)
            
            # Check for intervention needs
            intervention_type = detect_intervention_need(memory.get_recent_exchanges(3), 
                                                      stage["required_topics"])
            if intervention_type:
                intervention = moderator.generate_intervention(
                    intervention_type=intervention_type,
                    context=memory.get_context(),
                    target_topics=uncovered_topics
                )
                transcript.append({"speaker": "moderator", "content": intervention})
                memory.add(intervention, speaker="moderator")
            
            turns_completed += 1
        
        # Stage closure
        summary = moderator.generate_stage_summary(memory.get_context(), stage)
        transcript.append({"speaker": "moderator", "content": summary})
        memory.add_stage_summary(summary)
    
    # Process outputs
    outputs = process_outputs(transcript, workshop_config["outputs"])
    
    return {
        "transcript": transcript,
        "outputs": outputs,
        "analysis": {
            "stance_evolution": stance_vectors,
            "topic_coverage": topic_coverage,
            "participation": participation_tracker
        }
    }
```

#### Stance Vector Evolution

The system tracks changing opinions and positions through stance vector evolution:

```python
def update_stance_vectors(stance_vectors, speaker, current_response, peer_context):
    """
    Updates stance vectors based on dialogue exchange.
    
    Args:
        stance_vectors: Current stance vectors for all personas
        speaker: Current speaker ID
        current_response: The response just generated
        peer_context: Recent exchanges from other participants
        
    Returns:
        Updated stance vectors
    """
    # Define key stance dimensions based on adaptation issues
    stance_dimensions = [
        "retreat_vs_protect", 
        "individual_vs_collective", 
        "engineered_vs_nature_based",
        "incremental_vs_transformative",
        "present_vs_future_focused"
    ]
    
    # Extract stance signals from current response
    current_signals = {}
    for dimension in stance_dimensions:
        # Analyze response for signals about this stance dimension
        signal = extract_stance_signal(current_response, dimension)
        if signal is not None:  # Signal detected
            current_signals[dimension] = signal
    
    # If no signals detected, return unchanged
    if not current_signals:
        return stance_vectors
    
    # Update speaker's own stance (reinforcement effect)
    for dimension, signal in current_signals.items():
        if dimension in stance_vectors[speaker]:
            # Reinforce existing stance (with some regression toward the signal)
            current = stance_vectors[speaker][dimension]
            # Weighted average with 0.8 weight to current stance
            stance_vectors[speaker][dimension] = (0.8 * current) + (0.2 * signal)
        else:
            # Initialize stance dimension 
            stance_vectors[speaker][dimension] = signal
    
    # Update other personas' stances (influence effect)
    if peer_context:
        # Extract which personas have engaged recently
        recent_speakers = [exchange["speaker"] for exchange in peer_context 
                         if exchange["speaker"] != "moderator"]
        
        for other_speaker in recent_speakers:
            # Skip self-influence
            if other_speaker == speaker:
                continue
                
            # Calculate influence strength based on factors like:
            # - Value alignment between personas
            # - Prior stance alignment
            # - Power dynamics
            influence_factor = calculate_influence_factor(speaker, other_speaker, stance_vectors)
            
            # Apply influence to each dimension where current speaker expressed a position
            for dimension, signal in current_signals.items():
                if dimension in stance_vectors[other_speaker]:
                    # Current stance of the other persona
                    other_current = stance_vectors[other_speaker][dimension]
                    
                    # Calculate stance distance (how much they disagree)
                    stance_distance = abs(other_current - signal)
                    
                    # Apply bounded confidence model - personas are only influenced
                    # if the difference isn't too extreme
                    if stance_distance < 0.5:  # Threshold for influence
                        # Calculate new stance with appropriate influence
                        new_stance = other_current + (influence_factor * (signal - other_current))
                        stance_vectors[other_speaker][dimension] = new_stance
    
    return stance_vectors
```

#### Agreement Controller

The agreement controller monitors semantic similarity to identify potential agreement opportunities:

```python
def check_agreement_trigger(speaker_id, peer_context, stance_vectors, threshold=0.78):
    """
    Checks whether to trigger an agreement response based on semantic similarity.
    
    Args:
        speaker_id: ID of the current speaker
        peer_context: Recent exchanges from other participants
        stance_vectors: Current stance positions
        threshold: Similarity threshold for triggering agreement
        
    Returns:
        Agreement directive or empty string
    """
    if not peer_context or len(peer_context) < 2:
        return ""  # Not enough context to find agreement
    
    # Get most recent exchange that isn't from moderator
    recent_peer_exchanges = [ex for ex in peer_context if ex["speaker"] != "moderator"]
    if not recent_peer_exchanges:
        return ""
        
    recent_exchange = recent_peer_exchanges[0]
    recent_speaker = recent_exchange["speaker"]
    
    # Skip if same speaker (can't agree with yourself)
    if recent_speaker == speaker_id:
        return ""
    
    # Calculate semantic similarity between personas' stance vectors
    similarity = calculate_stance_similarity(
        stance_vectors.get(speaker_id, {}),
        stance_vectors.get(recent_speaker, {})
    )
    
    # If similarity exceeds threshold, trigger agreement
    if similarity > threshold:
        # Identify topic to agree on based on the recent exchange
        topics = recent_exchange.get("topics", [])
        if not topics:
            topic = "this point"
        else:
            topic = topics[0].replace("_", " ")
            
        # Generate appropriate agreement directive
        return f"Express agreement with {recent_speaker}'s point about {topic} before adding your own perspective"
    
    return ""
```

## Usage Example

```python
from dialogue_simulation import WorkshopSimulation, ModeratorAgent, PersonaAgent

# Load workshop configuration from Module 1
with open("workshop_config.json", "r") as f:
    workshop_config = json.load(f)

# Load personas from Module 2
persona_lib = PersonaLibrary.from_directory("personas/")
personas = persona_lib.get_all_personas()

# Initialize the simulation
simulation = WorkshopSimulation(
    llm_provider="anthropic",
    api_key="your_api_key",
    output_dir="./simulation_results"
)

# Run the simulation
results = simulation.run_workshop(
    workshop_config=workshop_config,
    personas=personas,
    max_turns_per_stage=15,
    save_outputs=True,
    visualize_interactions=True
)

# Access simulation results
transcript = results["transcript"]
outputs = results["outputs"]
stance_evolution = results["analysis"]["stance_evolution"]

# Print summary statistics
print(f"Generated workshop with {len(transcript)} total exchanges")
print(f"Produced {len(outputs)} structured outputs")

# Print a sample exchange
for exchange in transcript[:5]:
    print(f"[{exchange['speaker']}]: {exchange['content'][:100]}...")

# Visualize stance evolution
simulation.visualize_stance_evolution(stance_evolution)
```

## Google Colab Implementation

The code below can be run directly in Google Colab. It implements a simplified version of the Multi-Agent Dialogue Simulation that demonstrates the key functionality.

```python
# dialogue_simulation.py for Google Colab
# ==========================================
# This implementation ensures compatibility with Google Colab environment
# and demonstrates the core simulation functionality.

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
import matplotlib.pyplot as plt
from collections import Counter

# Install required dependencies
!pip install -q openai tiktoken matplotlib pandas numpy scipy

try:
    from openai import OpenAI
except ImportError:
    import openai

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
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install the 'anthropic' package to use the Anthropic provider")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 temperature: float = 0.7, max_tokens: int = 1000,
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
                try:
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
                    raise ValueError(f"Error with Anthropic provider: {str(e)}")
                
        except Exception as e:
            logging.error(f"Error generating text with {self.provider}: {str(e)}")
            raise

class PromptTemplates:
    """Collection of prompt templates for dialogue simulation."""
    
    # System prompts for different agent types
    MODERATOR_SYSTEM = """You are an expert climate adaptation workshop facilitator with years of experience 
running focus groups. Your role is to guide the conversation, ensure all topics are covered, and help the 
group generate valuable insights while ensuring everyone participates equally."""

    PERSONA_SYSTEM = """You are roleplaying as a specific persona in a climate adaptation workshop.
You must maintain consistent character traits, values, and speaking style throughout the conversation.
Your responses should reflect your persona's unique perspective, experiences, and concerns about climate impacts."""
    
    # Moderator introduction template
    STAGE_INTRODUCTION = """
You are facilitating a climate adaptation workshop in {community_name}. 

WORKSHOP CONTEXT:
{workshop_purpose}

CURRENT STAGE:
Stage: {stage_name} ({stage_id})
Description: {stage_description}
Required Topics: {required_topics}

PARTICIPANTS:
{participant_list}

Craft an introduction to Stage {stage_id}: {stage_name} that:
1. Clearly explains the purpose of this stage
2. Makes participants feel comfortable sharing their perspectives
3. Provides any necessary context about climate impacts relevant to this discussion
4. Asks an open-ended question to start the conversation about one of the required topics

Keep your introduction concise (100-150 words) and conversational in tone.
"""
    
    # Topic question template
    TOPIC_QUESTION = """
You are facilitating a climate adaptation workshop and need to ask a question about a specific topic.

CURRENT STAGE:
Stage: {stage_name}
Required Topics: {required_topics}

TARGET TOPIC:
{topic}

RECENT DISCUSSION:
{recent_exchanges}

Craft a question that:
1. Focuses specifically on the target topic ({topic})
2. Builds naturally from the recent discussion
3. Encourages participants to share personal experiences and perspectives
4. Is open-ended rather than yes/no
5. Uses accessible, non-technical language

Keep your question concise (1-2 sentences) and conversational in tone.
"""

    # Follow-up question template
    FOLLOWUP_QUESTION = """
You are facilitating a climate adaptation workshop and want to ask a follow-up question.

CURRENT STAGE:
{stage_description}

RECENT EXCHANGES:
{recent_exchanges}

Craft a follow-up question that:
1. Builds naturally on the most recent comment
2. Encourages deeper exploration of the point just raised
3. Helps connect personal experiences to broader adaptation challenges
4. Invites specific examples or elaboration
5. Is open-ended rather than yes/no

Keep your question concise (1-2 sentences) and conversational in tone.
"""

    # Persona response template
    PERSONA_RESPONSE = """
You are roleplaying as {persona_name}, speaking in a climate adaptation workshop in {community_name}.

PERSONA DETAILS:
Age: {age}
Occupation: {occupation}
Location: {location}
Background: {background_summary}
Key climate concerns: {climate_concerns}
Values: {value_priorities}
Speaking Style: Formality ({formality_level}/10), Technical ({technical_level}/10), Regional ({regional_level}/10)
Speech Patterns: {speech_patterns}

WORKSHOP CONTEXT:
Current Stage: {stage_name} - {stage_description}
Current Topic: {current_topic}

RECENT DISCUSSION:
{recent_exchanges}

You've been asked to respond to:
{moderator_question}

RESPONSE INSTRUCTIONS:
1. Stay in character as {persona_name} with their unique perspective and speaking style
2. Directly address the question/topic
3. Draw on your persona's specific experiences with climate impacts
4. Express opinions aligned with your persona's values
5. Keep your response to 100-150 words maximum
6. Use first-person perspective ("I think..." rather than "{persona_name} thinks...")

SPECIAL DIRECTIVES:
{agreement_directive}
{contention_directive}
{experience_sharing_directive}

Your response:
"""

    # Intervention template
    INTERVENTION = """
You are the moderator noticing an issue in the climate adaptation workshop discussion.

ISSUE TYPE: {issue_type}

CONTEXT:
{relevant_context}

WORKSHOP OBJECTIVE:
{workshop_objective}

TARGET TOPIC:
{target_topic}

Generate an appropriate moderator intervention that:
1. Addresses the specific issue without criticizing participants
2. Gently redirects the conversation toward the target topic if needed
3. Encourages deeper exploration or clarification as needed
4. Uses an engaging, supportive facilitation style

Your intervention should be 2-3 sentences (40-60 words) and conversational in tone.
"""

    # Stage summary template
    STAGE_SUMMARY = """
You are summarizing the key points from a stage of a climate adaptation workshop.

WORKSHOP CONTEXT:
{workshop_purpose}

STAGE INFORMATION:
Stage: {stage_name}
Objectives: {stage_description}
Required Topics: {required_topics}

FULL DISCUSSION TRANSCRIPT:
{stage_transcript}

Create a concise summary (150-200 words) that:
1. Identifies the main themes and insights that emerged
2. Highlights points of consensus and any significant differences in perspective
3. Notes specific climate impacts or adaptation challenges discussed
4. Captures personal experiences shared by participants that illustrate key points
5. Maintains a neutral, balanced representation of all perspectives shared

Format your summary with appropriate paragraph breaks and bullet points where relevant.
"""

    # Output extraction template
    OUTPUT_EXTRACTION = """
Extract the required information from the workshop discussion to create a structured output.

OUTPUT SPECIFICATION:
ID: {output_id}
Class: {output_class} - {output_class_description}
Format: {output_format}
Required Fields: {output_fields}
Quality Requirements: {quality_metrics}

RELEVANT DISCUSSION:
{relevant_transcript}

Based on the discussion transcript, create a structured output that:
1. Follows the specified format ({output_format})
2. Includes all required fields
3. Draws directly from participants' contributions
4. Meets all quality requirements
5. Synthesizes multiple perspectives where appropriate

Format your response as a JSON object following this exact structure:
{output_structure_example}

Ensure all fields have meaningful content derived from the discussion.
"""

# ... [Additional classes and implementation code would go here] ...

def run_demo():
    """Run a demonstration of the Multi-Agent Dialogue Simulation."""
    print("\n=== Multi-Agent Dialogue Simulation Demo ===\n")
    
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
                    "description": "Identify specific climate risks and differential impacts across vulnerable groups",
                    "interaction_pattern": "spoke_wheel",
                    "required_topics": ["climate_risks", "vulnerable_groups", "current_coping"],
                    "duration_points": 8
                }
                # Limiting to 2 stages for demo purposes
            ],
            "outputs": [
                {
                    "id": "DN_01",
                    "class": "DN",
                    "format": "structured_narrative",
                    "fields": ["climate_impacts", "vulnerable_groups", "differential_impacts"],
                    "quality": {"detail_level": "high", "scientific_accuracy": True},
                    "stage": "S2"
                }
            ],
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
    
    # Create sample personas (simplified versions)
    sample_personas = [
        {
            "id": "persona_elderly_coastal_1",
            "type": "human",
            "name": "Eleanor Martinez",
            "metadata": {
                "demographics": {
                    "gender": "female",
                    "age": 73,
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
                    {"value": "universalism", "weight": 0.6}
                ]
            },
            "climate_impacts": {
                "current": [
                    {"hazard": "coastal_flooding", "description": "Yard floods during king tides", "severity": "moderate"},
                    {"hazard": "extreme_heat", "description": "Health impacts during summer heat waves", "severity": "high"}
                ],
                "projected_2050": [
                    {"hazard": "coastal_flooding", "description": "Home likely inundated during major storms", "severity": "severe"}
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
                    {"option": "community_resilience_hub", "feasibility": "high", "preference": "high"}
                ]
            },
            "biography": "Eleanor Martinez has lived in her coastal home for over 40 years. As a retired teacher, she has deep ties to the community and limited resources on a fixed income. She's observed increasing flooding in her yard over the years.",
            "quotes": [
                {"text": "I've seen the water getting closer every year. The tide comes up into my garden now, which never happened when we first moved here.", "topic": "observed_changes"},
                {"text": "At my age, starting over somewhere new just isn't realistic. This is my home, all my memories are here.", "topic": "retreat_options"}
            ],
            "linguistic_parameters": {
                "formality_level": 0.7,
                "technical_vocabulary": 0.3,
                "regional_markers": 0.4,
                "speech_patterns": ["reflective", "historically_oriented", "direct"]
            }
        },
        {
            "id": "persona_fisher_1",
            "type": "human",
            "name": "Miguel Rodriguez",
            "metadata": {
                "demographics": {
                    "gender": "male",
                    "age": 42,
                    "income_level": "moderate_seasonal",
                    "housing_type": "owned_small_home",
                    "occupation": "commercial_fisher",
                    "location": "harbor_district"
                },
                "vulnerability_factors": {
                    "physical_exposure": 0.7,
                    "sensitivity": 0.6,
                    "adaptive_capacity": 0.5
                },
                "value_priorities": [
                    {"value": "self_determination", "weight": 0.8},
                    {"value": "security", "weight": 0.7},
                    {"value": "achievement", "weight": 0.6}
                ]
            },
            "climate_impacts": {
                "current": [
                    {"hazard": "storm_surge", "description": "Damage to docked boats and equipment", "severity": "severe"},
                    {"hazard": "coastal_erosion", "description": "Harbor access issues during storms", "severity": "moderate"}
                ],
                "projected_2050": [
                    {"hazard": "storm_surge", "description": "Potential destruction of harbor infrastructure", "severity": "severe"},
                    {"hazard": "marine_ecosystem_shifts", "description": "Changing fish populations affecting livelihood", "severity": "high"}
                ]
            },
            "adaptation_context": {
                "enablers": [
                    {"factor": "technical_knowledge", "description": "Extensive knowledge of local waters and weather patterns"},
                    {"factor": "social_networks", "description": "Strong connections within fishing community"}
                ],
                "barriers": [
                    {"factor": "financial_constraints", "description": "Limited capital for major equipment upgrades"},
                    {"factor": "occupation_dependence", "description": "Skills tied to specific location and industry"}
                ],
                "adaptation_options": [
                    {"option": "improved_harbor_protection", "feasibility": "medium", "preference": "high"},
                    {"option": "equipment_upgrades", "feasibility": "medium", "preference": "medium"}
                ]
            },
            "biography": "Miguel Rodriguez is a second-generation commercial fisher who operates a small boat out of Bayshore Harbor. He supports his family through seasonal fishing and faces increasing uncertainty due to weather disruptions and infrastructure damage.",
            "quotes": [
                {"text": "These storms are getting worse every year. Last season I had to repair my boat twice because of surge damage while it was docked.", "topic": "infrastructure_damage"},
                {"text": "We can't just pick up and move. This harbor is our livelihood. We need better protection for our boats and equipment.", "topic": "harbor_protection"}
            ],
            "linguistic_parameters": {
                "formality_level": 0.4,
                "technical_vocabulary": 0.7,
                "regional_markers": 0.6,
                "speech_patterns": ["direct", "practical", "industry_terminology"]
            }
        },
        {
            "id": "persona_renter_1",
            "type": "human",
            "name": "Tamika Johnson",
            "metadata": {
                "demographics": {
                    "gender": "female",
                    "age": 34,
                    "income_level": "low",
                    "housing_type": "rental_apartment",
                    "occupation": "healthcare_aide",
                    "location": "flood_zone_B"
                },
                "vulnerability_factors": {
                    "physical_exposure": 0.8,
                    "sensitivity": 0.7,
                    "adaptive_capacity": 0.3
                },
                "value_priorities": [
                    {"value": "security", "weight": 0.9},
                    {"value": "self_determination", "weight": 0.7},
                    {"value": "universalism", "weight": 0.5}
                ]
            },
            "climate_impacts": {
                "current": [
                    {"hazard": "flooding", "description": "Street flooding blocks access to work during heavy rain", "severity": "high"},
                    {"hazard": "mold", "description": "Persistent moisture problems in apartment building", "severity": "moderate"}
                ],
                "projected_2050": [
                    {"hazard": "flooding", "description": "Potential displacement from current housing", "severity": "severe"},
                    {"hazard": "infrastructure_failure", "description": "Frequent power outages and water service disruptions", "severity": "high"}
                ]
            },
            "adaptation_context": {
                "enablers": [
                    {"factor": "community_connections", "description": "Active in local tenant association"},
                    {"factor": "practical_knowledge", "description": "Experience preparing for and responding to floods"}
                ],
                "barriers": [
                    {"factor": "renter_status", "description": "Limited authority to modify apartment"},
                    {"factor": "financial_constraints", "description": "Cannot afford to relocate to safer housing"},
                    {"factor": "transportation", "description": "Relies on public transportation affected by flooding"}
                ],
                "adaptation_options": [
                    {"option": "tenant_advocacy_coalition", "feasibility": "high", "preference": "high"},
                    {"option": "emergency_shelter_access", "feasibility": "medium", "preference": "medium"}
                ]
            },
            "biography": "Tamika Johnson is a single mother working as a healthcare aide who lives in a ground-floor apartment in a flood-prone area. She relies on public transportation to reach her job and faces increasing challenges with flooding affecting her commute and home.",
            "quotes": [
                {"text": "My landlord won't fix the drainage issues. After every heavy rain, the mold gets worse, and my son's asthma acts up. But I can't afford to move anywhere else.", "topic": "housing_issues"},
                {"text": "When it floods, I've missed shifts because I can't get to work. That means lost income I really can't afford.", "topic": "transportation_disruption"}
            ],
            "linguistic_parameters": {
                "formality_level": 0.5,
                "technical_vocabulary": 0.3,
                "regional_markers": 0.5,
                "speech_patterns": ["pragmatic", "direct", "emotionally_expressive"]
            }
        }
    ]
    
    print(f"Loaded workshop configuration with {len(workshop_config['workshop_parameters']['stages'])} stages")
    print(f"Created {len(sample_personas)} sample personas")
    
    # Initialize and run a simplified simulation
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and configure the simulation
        simulation = WorkshopSimulation(
            llm_provider="openai",
            api_key=api_key,
            output_dir=tmpdir
        )
        
        # Run a very abbreviated simulation for demo purposes
        print("\nRunning an abbreviated workshop simulation...")
        results = simulation.run_workshop(
            workshop_config=workshop_config,
            personas=sample_personas,
            max_turns_per_stage=3,  # Limited turns for demo
            save_outputs=True,
            visualize_interactions=True
        )
        
        # Display a short sample of the transcript
        print("\nSample of generated dialogue:")
        for i, exchange in enumerate(results["transcript"][:5]):
            speaker = exchange.get("display_name", exchange.get("speaker", "Unknown"))
            content = exchange.get("content", "")
            # Truncate long content for display
            if len(content) > 100:
                content = content[:97] + "..."
            print(f"[{speaker}]: {content}")
        
        print(f"\nSimulation complete! Generated {len(results['transcript'])} exchanges")
        print(f"Results saved to {tmpdir}")

# Run the demo if executed directly
if __name__ == "__main__":
    run_demo()
```

## Dependencies

The Multi-Agent Dialogue Simulation module requires the following dependencies:

- **Python 3.8+**: Core programming language
- **OpenAI API or Anthropic API**: For LLM integration
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib**: Visualization of interactions and stance evolution
- **YAML**: Configuration file parsing
- **JSON**: Data serialization and deserialization
- **Tiktoken**: Token counting for context management
- **SciPy**: (Optional) For advanced statistical analysis

All dependencies can be installed via pip:

```bash
pip install openai anthropic pandas numpy matplotlib pyyaml tiktoken scipy
```

## References

1. Jones, R. et al. (2024). "Focus Agent: LLM-Powered Virtual Focus Group." arXiv:2409.01907.
2. Li, M. et al. (2024). "Can Large Language Models Replace Human Participants?" Marketing Science Institute Report 25-101.
3. Cohen, S. et al. (2024). "Virtual Personas for Language Models via an Anthology of Backstories." Berkeley Artificial Intelligence Research Blog.
4. Wang, Y. et al. (2024). "OASIS: Open Agent Social Interaction Simulations with One Million Agents." arXiv:2411.11581v4.
5. Market Research Society (2023). "Synthetic Respondents and other Gen AI applications in Market Research." MRS Delphi Report Series.
6. Hewitt, J. et al. (2024). "LLM Social Simulations Are a Promising Research Method." arXiv:2504.02234.
7. Salminen, J. et al. (2025). "Generative AI for Persona Development: A Systematic Review." arXiv:2504.04927.
8. Brand, J. et al. (2023). "Using LLMs for Market Research." Harvard Business School Working Paper 23-062.
9. Cabrero, D.G. et al. (2022). "A Critique of Personas as representations of 'the other' in Cross-Cultural Technology Design."
10. Microsoft Research (2023). "Autogen: Enabling Next-Generation Large Language Model Applications." Microsoft Research Blog.
11. Mullenbach, L.E. & Wilhelm Stanis, S.A. (2024). "Understanding how justice is considered in climate adaptation approaches." Journal of Environmental Planning and Management, 1-20.
12. Adaptation Scotland (2022). "Climate Change Adaptation Personas — Workshop Outline and Template."
