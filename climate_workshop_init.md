# Climate Adaptation Workshop Initialization Protocol

This notebook implements Module 1 of the LLM-simulated climate adaptation workshop system. The Workshop Initialization Protocol transforms natural language workshop descriptions into structured, machine-executable specifications for driving focus group simulations.

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
- [LLM Client Implementation](#llm-client-implementation)
- [Prompt Templates](#prompt-templates)
- [Workshop Initialization Protocol](#workshop-initialization-protocol)
- [Running a Demo](#running-a-demo)
- [Examining the Results](#examining-the-results)

## Introduction

The Workshop Initialization Protocol is the foundation for our LLM-simulated climate adaptation workshop system. It implements the formalization process described in section 3.2.1 of the project specification, leveraging LLM capabilities to understand and structure natural language inputs.

This module processes:
- Workshop goals expressed in natural language
- Expected deliverables described in plain English
- Additional context about the climate adaptation situation

And produces:
- Structured workshop parameters with formalized purpose directives
- Stage definitions with interaction patterns
- Output specifications with fields and quality metrics
- Constraint sets to guide the LLM simulation

Let's begin by setting up our environment and implementing the necessary components.

## Setup

First, let's install the required packages:

```python
# Install required dependencies
!pip install openai jsonschema pyyaml

# Import necessary libraries
import json
import os
import yaml
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import openai
from jsonschema import validate
```

If you're running this in Google Colab, you'll need to provide your API key. Let's set that up:

```python
# Set up OpenAI API key (for Google Colab)
import os

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

# Set up API key
if IN_COLAB:
    # If API key not in environment, prompt for it
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = input("Enter your OpenAI API key: ")
        
# Verify API key is set
if "OPENAI_API_KEY" not in os.environ:
    print("Please set your OPENAI_API_KEY environment variable")
else:
    print("OpenAI API key is set")
```

## LLM Client Implementation

Now, let's implement the LLM client that will handle interactions with the language model:

```python
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
            self.client = openai.OpenAI(api_key=self.api_key)
        elif provider == "anthropic":
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install the 'anthropic' package to use the Anthropic provider")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 temperature: float = 0.2, max_tokens: int = 1500,
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
                    from anthropic import Anthropic
                    
                    if system_prompt:
                        full_prompt = f"{system_prompt}\n\n{prompt}"
                    else:
                        full_prompt = prompt
                    
                    response = self.client.completions.create(
                        model=self.model,
                        prompt=full_prompt,
                        max_tokens_to_sample=max_tokens,
                        temperature=temperature
                    )
                    return response.completion
                except Exception as e:
                    raise ValueError(f"Error with Anthropic provider: {str(e)}")
                
        except Exception as e:
            logging.error(f"Error generating text with {self.provider}: {str(e)}")
            raise
```

## Prompt Templates

Let's define the prompt templates that will be used to guide the LLM in extracting and formalizing workshop components:

```python
class PromptTemplates:
    """Collection of prompt templates for different tasks in the workshop initialization process."""
    
    # System prompt for all tasks
    SYSTEM_PROMPT = """You are a climate adaptation planning expert helping to formalize workshop design. 
Your task is to transform natural language descriptions into structured formats for machine-executable workshop simulations.
Follow instructions precisely and output only in the requested format."""
    
    # Extract purpose components
    PURPOSE_EXTRACTION = """
You are analyzing a climate adaptation workshop proposal to extract key elements.

WORKSHOP TOPIC: {workshop_topic}
WORKSHOP GOAL: {workshop_goal}
ADDITIONAL CONTEXT: {additional_context}

Extract and organize the following elements from the workshop information:
1. DOMAIN: What specific climate hazard or adaptation domain is being addressed? Include geographic context.
2. PLANNING PHASE: What phase of adaptation planning does this workshop focus on?
3. STAKEHOLDERS: Which specific stakeholder groups are prioritized or especially vulnerable?
4. CLIMATE_SCENARIO: What climate projection or scenario is relevant (if mentioned)?

Format your response as JSON:
{
  "domain": "Specific climate hazard and geographic context",
  "planning_phase": "Identified planning phase",
  "stakeholders": ["Stakeholder group 1", "Stakeholder group 2"],
  "climate_scenario": "Relevant climate scenario if mentioned, otherwise a reasonable assumption"
}
"""

    # Generate purpose directive
    PURPOSE_DIRECTIVE = """
Using the extracted information, create a detailed purpose directive for the climate adaptation workshop.

DOMAIN: {domain}
PLANNING PHASE: {planning_phase}
STAKEHOLDERS: {stakeholders}
CLIMATE SCENARIO: {climate_scenario}
WORKSHOP GOAL: {workshop_goal}
ADDITIONAL CONTEXT: {additional_context}

Create a three-part purpose directive with the following elements:
1. CONTEXT: Provide a detailed environmental and social context paragraph that establishes the climate risk situation.
2. OBJECTIVES: Transform the workshop goal into 2-3 specific, actionable objectives.
3. OUTPUTS_EXPECTED: Summarize the expected outputs in a clear, directive format.

Each element should be detailed, specific to the climate adaptation context, and actionable.

Format your response as JSON:
{
  "context": "Detailed environmental and social context...",
  "objectives": "Specific, actionable objectives...",
  "outputs_expected": "Clear expectations for workshop outputs..."
}
"""

    # Generate constraint sets
    CONSTRAINT_GENERATION = """
Based on the workshop purpose, generate constraint sets to guide the LLM-simulated discussion.

DOMAIN: {domain}
PLANNING PHASE: {planning_phase}
STAKEHOLDERS: {stakeholders}
WORKSHOP GOAL: {workshop_goal}

Create two constraint sets:
1. C_purpose: Core constraints defining the boundaries, perspectives, and goals
2. C_equity: Equity-specific constraints ensuring adaptation justice

Each constraint should be prefixed with a directive tag like #BOUNDARY, #PERSPECTIVE, #GOAL, or #REQUIRE.

Format your response as JSON:
{
  "C_purpose": [
    "#BOUNDARY Content must remain within [specific boundary]",
    "#PERSPECTIVE Special attention to [specific stakeholder] required",
    "#GOAL Discussions must contribute to [specific goal]"
  ],
  "C_equity": [
    "#REQUIRE [specific equity requirement]",
    "#REQUIRE [another equity requirement]"
  ]
}
"""

    # Analyze deliverables
    DELIVERABLE_ANALYSIS = """
Analyze the following expected deliverables for a climate adaptation workshop.

DELIVERABLES:
{deliverables}

WORKSHOP CONTEXT:
{context}

Classify each deliverable into one of these categories:
- DN (Descriptive Narrative): Qualitative descriptions of climate impacts and vulnerabilities
- AI (Analytical Insight): Evidence-based findings or conclusions
- AO (Adaptation Option): Specific action or strategy for addressing climate risk
- IC (Implementation Consideration): Factors affecting option implementation

For each deliverable, determine:
1. Most appropriate output class (DN, AI, AO, IC)
2. Suitable format (structured_list, rich_text, stakeholder_matrix, etc.)
3. Which workshop stage it belongs to (S1-Introduction, S2-Impact Exploration, S3-Adaptation Brainstorm, S4-Strategy Evaluation)

Format your response as a JSON array:
[
  {
    "description": "Original deliverable description",
    "class": "Output class (DN, AI, AO, IC)",
    "format": "Appropriate format",
    "stage": "Workshop stage ID (S1, S2, S3, S4)"
  }
]
"""

    # Extract fields for output
    FIELD_EXTRACTION = """
You are extracting necessary fields for a climate adaptation workshop output.

DELIVERABLE: {deliverable}
OUTPUT CLASS: {output_class}
OUTPUT FORMAT: {output_format}

Based on the deliverable description and output class ({output_class}), identify 4-7 fields that should be included in this structured output.
Each field should be specific, measurable, and relevant to climate adaptation planning.

Format your response as a JSON array of strings, using snake_case for field names:
["field_one", "field_two", "field_three", ...]
"""

    # Generate quality metrics
    QUALITY_METRICS = """
You are defining quality metrics for a climate adaptation workshop output.

DELIVERABLE: {deliverable}
OUTPUT CLASS: {output_class}

Based on the deliverable description and output class ({output_class}), define quality metrics that should be used to evaluate this output.
Consider aspects like:
- Required level of detail
- Scientific accuracy requirements
- Equity considerations
- Minimum counts for items (if applicable)
- Required assessment components

Format your response as a JSON object with metric names as keys and values that are either boolean, string, or numeric:
{
  "metric_one": true,
  "metric_two": "high",
  "metric_three": 0.7
}
"""
```

## Workshop Initialization Protocol

Now, let's implement the main Workshop Initialization Protocol class that will use the LLM client and prompt templates to transform natural language inputs into structured workshop configurations:

```python
class WorkshopInitializationProtocol:
    """
    Implements the Workshop Initialization Protocol that transforms natural language
    workshop inputs into machine-executable specifications using LLMs.
    """
    
    # Output class descriptions
    OUTPUT_CLASSES = {
        "DN": "Descriptive Narrative - Qualitative descriptions of climate impacts and vulnerabilities",
        "AI": "Analytical Insight - Evidence-based findings or conclusions",
        "AO": "Adaptation Option - Specific action or strategy for addressing climate risk",
        "IC": "Implementation Consideration - Factors affecting option implementation"
    }
    
    # Standard interaction patterns for each stage
    STAGE_PATTERNS = {
        "S1": "linear",              # Introduction stage uses linear pattern
        "S2": "spoke_wheel",         # Impact Exploration uses spoke and wheel
        "S3": "mesh_network",        # Adaptation Brainstorm uses mesh network
        "S4": "bipartite_evaluation" # Strategy Evaluation uses bipartite grid
    }
    
    # Standard stage definitions that match the four-stage structure in the paper
    DEFAULT_STAGES = [
        {
            "id": "S1",
            "name": "Introduction",
            "description": "Establish persona identities and initial climate perspectives",
            "required_topics": ["personal_background", "climate_concerns"],
            "duration_points": 5
        },
        {
            "id": "S2",
            "name": "Impact Exploration",
            "description": "Identify specific climate risks and differential impacts across vulnerable groups",
            "required_topics": ["climate_risks", "vulnerable_groups", "current_coping"],
            "duration_points": 8
        },
        {
            "id": "S3",
            "name": "Adaptation Brainstorm",
            "description": "Generate adaptation options and identify implementation enablers/barriers",
            "required_topics": ["adaptation_options", "barriers", "enablers"],
            "duration_points": 12
        },
        {
            "id": "S4",
            "name": "Strategy Evaluation",
            "description": "Assess adaptation options across multiple criteria with emphasis on equity",
            "required_topics": ["feasibility", "equity", "effectiveness"],
            "duration_points": 8
        }
    ]
    
    # JSON Schema for validating the final workshop configuration
    WORKSHOP_SCHEMA = {
        "type": "object",
        "required": ["workshop_parameters", "version", "timestamp"],
        "properties": {
            "workshop_parameters": {
                "type": "object",
                "required": ["purpose", "stages", "outputs", "constraint_sets"],
                "properties": {
                    "purpose": {
                        "type": "object",
                        "required": ["context", "objectives", "outputs_expected"],
                        "properties": {
                            "context": {"type": "string"},
                            "objectives": {"type": "string"},
                            "outputs_expected": {"type": "string"}
                        }
                    },
                    "stages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id", "name", "description", "interaction_pattern", "required_topics", "duration_points"],
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "interaction_pattern": {"type": "string"},
                                "required_topics": {"type": "array", "items": {"type": "string"}},
                                "duration_points": {"type": "number"}
                            }
                        }
                    },
                    "outputs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id", "class", "format", "fields", "quality", "stage"],
                            "properties": {
                                "id": {"type": "string"},
                                "class": {"type": "string", "enum": ["DN", "AI", "AO", "IC"]},
                                "format": {"type": "string"},
                                "fields": {"type": "array", "items": {"type": "string"}},
                                "quality": {"type": "object"},
                                "stage": {"type": "string"}
                            }
                        }
                    },
                    "constraint_sets": {"type": "object"}
                }
            },
            "version": {"type": "string"},
            "timestamp": {"type": "string"}
        }
    }
    
    def __init__(self, llm_provider: str = "openai", api_key: Optional[str] = None, 
                 output_dir: Optional[str] = None, logger=None):
        """
        Initialize the Workshop Initialization Protocol.
        
        Args:
            llm_provider: LLM provider to use ("openai", "anthropic", etc.)
            api_key: API key for the LLM provider
            output_dir: Directory to save the generated configuration
            logger: Optional logger instance
        """
        # Setup LLM client
        self.llm = LLMClient(provider=llm_provider, api_key=api_key)
        
        # Setup output directory (handle Colab paths)
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Default to current directory in Colab
            self.output_dir = Path.cwd() / "config"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logger or self._setup_logger()
        self.logger.info(f"Workshop Initialization Protocol initialized with {llm_provider} provider")
        
        # Store prompt templates
        self.prompts = PromptTemplates()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the initialization protocol."""
        logger = logging.getLogger("workshop_init")
        logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def process_natural_language_input(self, workshop_input: Dict[str, str], 
                                      save_output: bool = True) -> Dict[str, Any]:
        """
        Process natural language workshop input to generate a formalized configuration.
        
        Args:
            workshop_input: Dictionary with natural language workshop parameters
            save_output: Whether to save the configuration to disk
            
        Returns:
            A dictionary containing the formalized workshop configuration
        """
        required_fields = ["workshop_topic", "workshop_goal", "expected_deliverables"]
        for field in required_fields:
            if field not in workshop_input:
                raise ValueError(f"Missing required field: {field}")
        
        # Set default for additional_context if not provided
        if "additional_context" not in workshop_input:
            workshop_input["additional_context"] = ""
            
        self.logger.info(f"Processing workshop input for topic: {workshop_input['workshop_topic']}")
        
        # Step 1: Extract purpose components using LLM
        purpose_components = self._extract_purpose_components(workshop_input)
        self.logger.info(f"Extracted purpose components: domain={purpose_components['domain']}")
        
        # Step 2: Generate constraints using LLM
        constraints = self._generate_constraints(purpose_components, workshop_input)
        self.logger.info(f"Generated {sum(len(v) for v in constraints.values())} constraints")
        
        # Step 3: Create purpose directive using LLM
        purpose = self._create_purpose_directive(purpose_components, workshop_input)
        self.logger.info("Created purpose directive")
        
        # Step 4: Create workshop stages
        stages = self._create_workshop_stages()
        self.logger.info(f"Created {len(stages)} workshop stages")
        
        # Step 5: Analyze deliverables and generate outputs using LLM
        outputs = self._analyze_deliverables(workshop_input["expected_deliverables"], purpose["context"], stages)
        self.logger.info(f"Generated {len(outputs)} formalized outputs")
        
        # Assemble the complete workshop configuration
        workshop_config = {
            "workshop_parameters": {
                "purpose": purpose,
                "stages": stages,
                "outputs": outputs,
                "constraint_sets": constraints
            },
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }
        
        # Validate the configuration
        self._validate_configuration(workshop_config)
        
        # Save the configuration if requested
        if save_output:
            self._save_configuration(workshop_config)
        
        self.logger.info("Workshop configuration created successfully")
        return workshop_config
    
    def _extract_purpose_components(self, workshop_input: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract purpose components from natural language input using LLM.
        
        Args:
            workshop_input: The workshop input
            
        Returns:
            Dictionary with extracted components
        """
        prompt = self.prompts.PURPOSE_EXTRACTION.format(
            workshop_topic=workshop_input["workshop_topic"],
            workshop_goal=workshop_input["workshop_goal"],
            additional_context=workshop_input["additional_context"]
        )
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.prompts.SYSTEM_PROMPT,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            components = json.loads(response)
            
            # Validate required fields
            required_fields = ["domain", "planning_phase", "stakeholders"]
            for field in required_fields:
                if field not in components or not components[field]:
                    self.logger.warning(f"Missing or empty required field in LLM response: {field}")
                    # Set a default value
                    if field == "stakeholders":
                        components[field] = ["General population"]
                    else:
                        components[field] = f"Undefined {field}"
            
            # Ensure stakeholders is a list
            if isinstance(components["stakeholders"], str):
                components["stakeholders"] = [components["stakeholders"]]
                
            # Add default climate scenario if not provided
            if "climate_scenario" not in components or not components["climate_scenario"]:
                components["climate_scenario"] = "Current and projected climate impacts"
                
            return components
            
        except Exception as e:
            self.logger.error(f"Error extracting purpose components: {str(e)}")
            # Fallback to basic extraction
            return {
                "domain": workshop_input["workshop_topic"],
                "planning_phase": "Adaptation planning",
                "stakeholders": ["Community members"],
                "climate_scenario": "Current and projected climate impacts"
            }
    
    def _generate_constraints(self, purpose_components: Dict[str, Any], 
                            workshop_input: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Generate constraint sets using LLM.
        
        Args:
            purpose_components: Extracted purpose components
            workshop_input: Original workshop input
            
        Returns:
            Dictionary of constraint sets
        """
        prompt = self.prompts.CONSTRAINT_GENERATION.format(
            domain=purpose_components["domain"],
            planning_phase=purpose_components["planning_phase"],
            stakeholders=", ".join(purpose_components["stakeholders"]),
            workshop_goal=workshop_input["workshop_goal"]
        )
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.prompts.SYSTEM_PROMPT,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            constraints = json.loads(response)
            
            # Validate constraint sets
            if "C_purpose" not in constraints or not constraints["C_purpose"]:
                self.logger.warning("Missing or empty C_purpose constraints in LLM response")
                constraints["C_purpose"] = [
                    f"#BOUNDARY Content must remain within {purpose_components['domain']} context",
                    f"#PHASE Discussions must address {purpose_components['planning_phase']}",
                    f"#GOAL Discussions must contribute to {workshop_input['workshop_goal']}"
                ]
                
            if "C_equity" not in constraints or not constraints["C_equity"]:
                self.logger.warning("Missing or empty C_equity constraints in LLM response")
                constraints["C_equity"] = [
                    "#REQUIRE Adaptation options must explicitly consider distributional impacts",
                    "#REQUIRE Implementation considerations must address access barriers for vulnerable groups"
                ]
                
            return constraints
            
        except Exception as e:
            self.logger.error(f"Error generating constraints: {str(e)}")
            # Fallback to basic constraints
            return {
                "C_purpose": [
                    f"#BOUNDARY Content must remain within {purpose_components['domain']} context",
                    f"#PHASE Discussions must address {purpose_components['planning_phase']}",
                    f"#GOAL Discussions must contribute to {workshop_input['workshop_goal']}"
                ],
                "C_equity": [
                    "#REQUIRE Adaptation options must explicitly consider distributional impacts",
                    "#REQUIRE Implementation considerations must address access barriers for vulnerable groups"
                ]
            }
    
    def _create_purpose_directive(self, purpose_components: Dict[str, Any], 
                                workshop_input: Dict[str, str]) -> Dict[str, str]:
        """
        Create the purpose directive using LLM.
        
        Args:
            purpose_components: Extracted purpose components
            workshop_input: Original workshop input
            
        Returns:
            Dictionary with context, objectives, and outputs_expected
        """
        prompt = self.prompts.PURPOSE_DIRECTIVE.format(
            domain=purpose_components["domain"],
            planning_phase=purpose_components["planning_phase"],
            stakeholders=", ".join(purpose_components["stakeholders"]),
            climate_scenario=purpose_components.get("climate_scenario", ""),
            workshop_goal=workshop_input["workshop_goal"],
            additional_context=workshop_input["additional_context"]
        )
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.prompts.SYSTEM_PROMPT,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            purpose = json.loads(response)
            
            # Validate required fields
            required_fields = ["context", "objectives", "outputs_expected"]
            for field in required_fields:
                if field not in purpose or not purpose[field]:
                    self.logger.warning(f"Missing or empty required field in purpose directive: {field}")
                    if field == "context":
                        purpose[field] = f"{purpose_components['domain']} facing {purpose_components['climate_scenario']}."
                    elif field == "objectives":
                        purpose[field] = workshop_input["workshop_goal"]
                    else:
                        purpose[field] = "Generate appropriate adaptation outputs."
                        
            return purpose
            
        except Exception as e:
            self.logger.error(f"Error creating purpose directive: {str(e)}")
            # Fallback to basic purpose directive
            return {
                "context": f"{purpose_components['domain']} facing {purpose_components.get('climate_scenario', 'climate impacts')}. Priority stakeholders include {', '.join(purpose_components['stakeholders'])}.",
                "objectives": workshop_input["workshop_goal"],
                "outputs_expected": f"Expected outputs include: {workshop_input['expected_deliverables']}"
            }
    
    def _create_workshop_stages(self) -> List[Dict[str, Any]]:
        """
        Create the workshop stages with interaction patterns.
        
        Returns:
            List of stage definitions with interaction patterns
        """
        # Start with default stages
        stages = self.DEFAULT_STAGES.copy()
        
        # Add interaction patterns
        for stage in stages:
            stage_id = stage["id"]
            stage["interaction_pattern"] = self.STAGE_PATTERNS.get(stage_id, "linear")
        
        return stages
    
    def _analyze_deliverables(self, deliverables: str, context: str, 
                            stages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze deliverables and formalize outputs using LLM.
        
        Args:
            deliverables: Expected deliverables as string
            context: Workshop context for LLM
            stages: Workshop stages
            
        Returns:
            List of formalized outputs
        """
        prompt = self.prompts.DELIVERABLE_ANALYSIS.format(
            deliverables=deliverables,
            context=context
        )
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.prompts.SYSTEM_PROMPT,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            deliverable_analysis = json.loads(response)
            
            # Ensure we have a list
            if not isinstance(deliverable_analysis, list):
                self.logger.warning("Unexpected format in deliverable analysis")
                deliverable_analysis = []
                
            # Process each analyzed deliverable
            outputs = []
            for i, analysis in enumerate(deliverable_analysis):
                output_id = f"{analysis['class']}_{i+1:02d}"
                
                # Extract fields using LLM
                fields = self._extract_fields(analysis['description'], analysis['class'], analysis['format'])
                
                # Generate quality metrics using LLM
                quality = self._generate_quality_metrics(analysis['description'], analysis['class'])
                
                # Create the formalized output
                output = {
                    "id": output_id,
                    "class": analysis['class'],
                    "format": analysis['format'],
                    "fields": fields,
                    "quality": quality,
                    "stage": analysis['stage']
                }
                
                outputs.append(output)
            
            # Ensure each stage has at least one output (except S1)
            stage_ids = {stage["id"] for stage in stages}
            covered_stages = {output["stage"] for output in outputs}
            for stage_id in stage_ids:
                if stage_id not in covered_stages and stage_id != "S1":
                    # Create a default output for this stage
                    self.logger.info(f"Adding default output for uncovered stage {stage_id}")
                    default_output = self._create_default_output_for_stage(stage_id, len(outputs) + 1)
                    outputs.append(default_output)
            
            return outputs
            
        except Exception as e:
            self.logger.error(f"Error analyzing deliverables: {str(e)}")
            # Fallback to basic outputs
            return [
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
                    "fields": ["description", "beneficiaries", "implementation_challenges", "resource_needs"],
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
    
    def _extract_fields(self, deliverable: str, output_class: str, output_format: str) -> List[str]:
        """
        Extract fields for an output using LLM.
        
        Args:
            deliverable: Deliverable description
            output_class: Output class (DN, AI, AO, IC)
            output_format: Output format
            
        Returns:
            List of field names
        """
        prompt = self.prompts.FIELD_EXTRACTION.format(
            deliverable=deliverable,
            output_class=output_class,
            output_format=output_format
        )
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.prompts.SYSTEM_PROMPT,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            fields = json.loads(response)
            
            # Ensure we have a list
            if not isinstance(fields, list):
                self.logger.warning("Unexpected format in field extraction")
                fields = []
                
            # Ensure we have at least some fields
            if not fields:
                # Use default fields based on output class
                default_fields = {
                    "DN": ["climate_impacts", "vulnerable_groups", "differential_impacts", "current_coping_strategies"],
                    "AI": ["key_finding", "supporting_evidence", "implications", "confidence_level"],
                    "AO": ["description", "primary_beneficiaries", "implementation_challenges", "resource_needs", "timeframe"],
                    "IC": ["stakeholder_group", "role", "interests", "potential_barriers", "engagement_strategy"]
                }
                fields = default_fields.get(output_class, ["description", "details", "implications"])
                
            return fields
            
        except Exception as e:
            self.logger.error(f"Error extracting fields: {str(e)}")
            # Fallback to basic fields
            default_fields = {
                "DN": ["climate_impacts", "vulnerable_groups", "differential_impacts", "current_coping_strategies"],
                "AI": ["key_finding", "supporting_evidence", "implications", "confidence_level"],
                "AO": ["description", "primary_beneficiaries", "implementation_challenges", "resource_needs", "timeframe"],
                "IC": ["stakeholder_group", "role", "interests", "potential_barriers", "engagement_strategy"]
            }
            return default_fields.get(output_class, ["description", "details", "implications"])
    
    def _generate_quality_metrics(self, deliverable: str, output_class: str) -> Dict[str, Any]:
        """
        Generate quality metrics for an output using LLM.
        
        Args:
            deliverable: Deliverable description
            output_class: Output class (DN, AI, AO, IC)
            
        Returns:
            Dictionary of quality metrics
        """
        prompt = self.prompts.QUALITY_METRICS.format(
            deliverable=deliverable,
            output_class=output_class
        )
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.prompts.SYSTEM_PROMPT,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            quality_metrics = json.loads(response)
            
            # Ensure we have a dictionary
            if not isinstance(quality_metrics, dict):
                self.logger.warning("Unexpected format in quality metrics generation")
                quality_metrics = {}
                
            # Ensure we have at least some metrics
            if not quality_metrics:
                # Use default metrics based on output class
                default_metrics = {
                    "DN": {
                        "detail_level": "high",
                        "scientific_accuracy": True,
                        "intersectionality_required": True
                    },
                    "AI": {
                        "evidence_based": True,
                        "actionable": True
                    },
                    "AO": {
                        "feasibility_min": 0.70,
                        "equity_required": True,
                        "cobenefit_note": True
                    },
                    "IC": {
                        "comprehensiveness": "medium",
                        "actionability": True,
                        "resource_identification": True
                    }
                }
                quality_metrics = default_metrics.get(output_class, {"quality_required": "high"})
                
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error generating quality metrics: {str(e)}")
            # Fallback to basic metrics
            default_metrics = {
                "DN": {
                    "detail_level": "high",
                    "scientific_accuracy": True,
                    "intersectionality_required": True
                },
                "AI": {
                    "evidence_based": True,
                    "actionable": True
                },
                "AO": {
                    "feasibility_min": 0.70,
                    "equity_required": True,
                    "cobenefit_note": True
                },
                "IC": {
                    "comprehensiveness": "medium",
                    "actionability": True,
                    "resource_identification": True
                }
            }
            return default_metrics.get(output_class, {"quality_required": "high"})
    
    def _create_default_output_for_stage(self, stage_id: str, output_count: int) -> Dict[str, Any]:
        """
        Create a default output for a stage that doesn't have any outputs assigned.
        
        Args:
            stage_id: The ID of the stage
            output_count: Current count of outputs for ID generation
            
        Returns:
            A default output definition
        """
        stage_output_mappings = {
            "S2": {
                "class": "DN",
                "format": "structured_narrative",
                "fields": ["climate_impacts", "vulnerable_groups", "differential_impacts"],
                "quality": {"detail_level": "medium", "scientific_accuracy": True}
            },
            "S3": {
                "class": "AO",
                "format": "structured_list",
                "fields": ["description", "beneficiaries", "challenges"],
                "quality": {"feasibility_min": 0.60, "equity_required": True}
            },
            "S4": {
                "class": "IC",
                "format": "stakeholder_matrix",
                "fields": ["stakeholder", "role", "engagement_strategy"],
                "quality": {"comprehensiveness": "low", "actionability": True}
            }
        }
        
        mapping = stage_output_mappings.get(stage_id, {
            "class": "AI",
            "format": "bullet_points",
            "fields": ["key_finding", "implications"],
            "quality": {"actionable": True}
        })
        
        output = {
            "id": f"{mapping['class']}_{output_count:02d}",
            "class": mapping["class"],
            "format": mapping["format"],
            "fields": mapping["fields"],
            "quality": mapping["quality"],
            "stage": stage_id
        }
        
        return output
    
    def _validate_configuration(self, workshop_config: Dict[str, Any]) -> None:
        """
        Validate the workshop configuration against the schema.
        
        Args:
            workshop_config: The workshop configuration to validate
            
        Raises:
            ValueError: If the configuration is invalid
        """
        try:
            validate(instance=workshop_config, schema=self.WORKSHOP_SCHEMA)
            self.logger.info("Workshop configuration validated successfully")
        except Exception as e:
            self.logger.error(f"Workshop configuration validation failed: {str(e)}")
            raise ValueError(f"Invalid workshop configuration: {str(e)}")
    
    def _save_configuration(self, workshop_config: Dict[str, Any], 
                          filename: Optional[str] = None) -> None:
        """
        Save the workshop configuration to file.
        
        Args:
            workshop_config: The workshop configuration to save
            filename: Optional filename (defaults to workshop_config.json)
        """
        if filename is None:
            filename = "workshop_config.json"
            
        config_path = self.output_dir / filename
        try:
            with open(config_path, "w") as f:
                json.dump(workshop_config, f, indent=2)
            self.logger.info(f"Workshop configuration saved to {config_path}")
        except Exception as e:
            self.logger.error(f"Error saving workshop configuration: {str(e)}")
            raise
        
        # Also save as YAML for human readability
        yaml_path = self.output_dir / filename.replace(".json", ".yaml")
        try:
            with open(yaml_path, "w") as f:
                yaml.dump(workshop_config, f, default_flow_style=False)
            self.logger.info(f"Workshop configuration also saved as YAML to {yaml_path}")
        except Exception as e:
            self.logger.warning(f"Error saving YAML configuration: {str(e)}")
```

## Running a Demo

Let's now run a demo to see the Workshop Initialization Protocol in action:

```python
def run_demo():
    """Run a demonstration of the LLM-powered Workshop Initialization Protocol."""
    print("\n=== LLM-Powered Workshop Initialization Protocol Demo ===\n")
    
    # Get OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Create example workshop input
    workshop_input = {
        "workshop_topic": "Coastal flooding adaptation in Bayshore, a small fishing community facing sea level rise",
        "workshop_goal": "We need to develop strategies to help low-income residents and local fishers cope with increasing flood events while ensuring elderly residents aren't left behind",
        "expected_deliverables": """
        1. A list of at least 10 possible adaptation actions ranked by feasibility
        2. An analysis of how flooding impacts different community groups
        3. Recommendations for implementation that consider local capacity constraints
        """,
        "additional_context": "The community has about 5,000 residents with 40% living in high-risk zones. Recent flooding has damaged fishing infrastructure and low-income housing. Local government has limited budget for adaptation."
    }
    
    print("Natural Language Workshop Input:")
    print(f"Topic: {workshop_input['workshop_topic']}")
    print(f"Goal: {workshop_input['workshop_goal']}")
    print(f"Expected Deliverables: {workshop_input['expected_deliverables']}")
    print(f"Additional Context: {workshop_input['additional_context']}")
    print("\nProcessing with LLM to transform into structured configuration...")
    
    # Initialize the protocol with OpenAI
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        protocol = WorkshopInitializationProtocol(
            llm_provider="openai",
            api_key=api_key,
            output_dir=tmpdir
        )
        
        # Process the natural language input
        workshop_config = protocol.process_natural_language_input(workshop_input)
        
        # Store the configuration for examination
        global demo_config
        demo_config = workshop_config
        
        print("\nWorkshop configuration created successfully!")
        
# Run the demo
run_demo()
```

## Examining the Results

Let's look at the LLM-generated configuration:

```python
# Examine the workshop purpose
if 'demo_config' in globals():
    purpose = demo_config["workshop_parameters"]["purpose"]
    print("\nGenerated Workshop Purpose:")
    print(f"Context: {purpose['context']}")
    print(f"Objectives: {purpose['objectives']}")
    print(f"Expected Outputs: {purpose['outputs_expected']}")
    
    # Print constraint sets
    print("\nLLM-Generated Constraint Sets:")
    for set_name, constraints in demo_config["workshop_parameters"]["constraint_sets"].items():
        print(f"  {set_name}:")
        for constraint in constraints:
            print(f"    - {constraint}")
    
    # Print output summary
    outputs = demo_config["workshop_parameters"]["outputs"]
    stages = demo_config["workshop_parameters"]["stages"]
    print(f"\nDefined {len(outputs)} workshop outputs across {len(stages)} stages:")
    
    for stage in stages:
        print(f"\nStage {stage['id']}: {stage['name']}")
        print(f"  Description: {stage['description']}")
        print(f"  Interaction Pattern: {stage['interaction_pattern']}")
        
        # Print outputs for this stage
        stage_outputs = [o for o in outputs if o['stage'] == stage['id']]
        if stage_outputs:
            print(f"  Outputs ({len(stage_outputs)}):")
            for output in stage_outputs:
                print(f"    - {output['id']} ({output['class']}): {output['format']}")
                print(f"      Fields: {', '.join(output['fields'])}")
                quality_metrics = [f"{k}={v}" for k, v in output['quality'].items()]
                print(f"      Quality metrics: {', '.join(quality_metrics[:3])}...")
        else:
            print("  No outputs defined for this stage")
else:
    print("Please run the demo first to generate a configuration")
```

Let's also visualize the output breakdown by class:

```python
import matplotlib.pyplot as plt
import numpy as np

# Visualize output distribution by class
if 'demo_config' in globals():
    outputs = demo_config["workshop_parameters"]["outputs"]
    
    # Count outputs by class
    output_classes = ["DN", "AI", "AO", "IC"]
    class_counts = {cls: len([o for o in outputs if o["class"] == cls]) for cls in output_classes}
    
    # Create a bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(class_counts.keys(), class_counts.values(), color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    plt.title('Output Distribution by Class')
    plt.xlabel('Output Class')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    # Add labels to each bar
    for i, (cls, count) in enumerate(class_counts.items()):
        plt.text(i, count + 0.1, str(count), ha='center')
    
    # Add a legend explaining each class
    class_descriptions = {
        "DN": "Descriptive Narrative",
        "AI": "Analytical Insight",
        "AO": "Adaptation Option",
        "IC": "Implementation Consideration"
    }
    
    legend_labels = [f"{cls}: {class_descriptions[cls]}" for cls in output_classes]
    plt.legend(legend_labels, loc='upper right')
    
    plt.tight_layout()
    plt.show()
else:
    print("Please run the demo first to generate visualization data")
```

## Conclusion

In this notebook, we've implemented the Workshop Initialization Protocol, the first module of our LLM-simulated climate adaptation workshop system. This module successfully transforms natural language workshop descriptions into structured, machine-executable configurations using the power of large language models.

Key capabilities demonstrated include:
- Processing natural language input to extract domain, planning phase, and stakeholders
- Generating purpose directives and constraint sets
- Classifying deliverables into standardized output types
- Extracting appropriate fields and quality metrics for each output
- Creating a complete workshop configuration ready for the next modules

This approach makes the system more accessible to users while maintaining the technical rigor needed for the subsequent simulation modules.
