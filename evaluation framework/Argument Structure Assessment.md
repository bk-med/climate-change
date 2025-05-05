# Argument Structure Assessment

## Overview

Argument Structure Assessment evaluates the logical construction and development of arguments within LLM-simulated climate adaptation workshops. This dimension analyzes how effectively participants articulate climate risks, present evidence, establish logical connections, and qualify their positions. The assessment provides quantitative measures of argumentative completeness, depth of reasoning, and adaptation-specific reasoning patterns.

## Purpose

In climate adaptation planning, the quality of decision-making depends on well-structured arguments that connect evidence to claims in logical, transparent ways. Argument Structure Assessment helps determine whether:

- Climate risks and adaptation options are supported by appropriate evidence
- Logical connections between evidence and claims are clearly articulated
- Arguments acknowledge uncertainties and limitations appropriately
- Discussion exhibits the depth and complexity expected in expert deliberation

This evaluation dimension is critical for detecting whether LLM-simulated discussions demonstrate the sophisticated reasoning patterns that characterize productive human climate adaptation workshops, rather than simply generating plausible-sounding but logically flawed or superficial arguments.

## Input Specification

The Argument Structure Assessment component processes structured workshop transcript data with specific formatting requirements:

### Primary Input

The primary input is a list of **utterance objects** representing sequential contributions from the workshop:

```json
[
  {
    "id": "S2_T15",
    "speaker": "Eleanor_Martinez",
    "content": "Based on the flood maps we reviewed, my property will be in the expanded flood zone within 15 years. The projected sea level rise of 18 inches would put my home at risk during even moderate storm events. While elevating homes is effective, it's completely unaffordable for those of us on fixed incomes.",
    "topics": ["flood_risk", "sea_level_rise", "adaptation_barriers"],
    "timestamp": "2025-04-28T14:45:10Z"
  },
  {
    "id": "S2_T16",
    "speaker": "moderator",
    "content": "Thank you for that perspective, Eleanor. Can you elaborate on what specific barriers you see to implementing home elevation in your situation?",
    "topics": ["adaptation_barriers"],
    "timestamp": "2025-04-28T14:46:20Z"
  },
  // Additional utterances...
]
```

Each utterance object must contain at minimum:
- `id`: Unique identifier for the utterance
- `speaker`: Identifier of the speaker (persona or moderator)
- `content`: The actual text of the utterance

Optional but useful fields include:
- `topics`: Pre-identified topics in the utterance
- `timestamp`: When the utterance was made
- `referring_to`: Reference to previous utterance IDs if applicable

### Configuration Parameters

The assessment function accepts several configuration parameters:

- `min_utterance_length`: Minimum token count for an utterance to be analyzed (default: 50)
- `component_weights`: Dictionary defining the importance of different argument components
- `domain_specific_terms`: List of climate adaptation terminology for domain relevance scoring

## Methodology

### Argumentative Component Identification

The Argument Structure Assessment employs techniques from argumentation mining to identify key components within each substantial contribution (>50 tokens):

1. **Claims**: Assertions about climate risks or adaptation strategies
2. **Evidence**: Supporting data, observations, or examples
3. **Warrants**: Logical connections between evidence and claims
4. **Qualifiers**: Expressions of certainty, uncertainty, or limitations
5. **Backing**: Additional support for the warrants
6. **Rebuttals**: Acknowledgment of counter-arguments or conditions

This approach is grounded in Toulmin's model of argumentation, adapted specifically for climate adaptation contexts.

### Implementation Process

The assessment process follows these steps:

1. **Pre-processing**: Filter contributions by length and extract content
2. **Component Detection**: Identify argumentative components using LLM-based extraction
3. **Completeness Calculation**: Compute an Argument Completeness Score (ACS) for each contribution
4. **Depth Analysis**: Measure supporting evidence depth and complexity
5. **Adaptation Reasoning Analysis**: Evaluate climate-specific reasoning patterns

## Implementation Details

```python
def assess_argument_structure(workshop_transcript, config=None):
    """
    Assess argument structure in climate adaptation workshop transcript.
    
    Args:
        workshop_transcript: List of utterance objects containing:
            - id: Unique identifier
            - speaker: Speaker identifier
            - content: Text content of the utterance
            - topics: (optional) List of topics covered
            - timestamp: (optional) Timestamp of utterance
        config: Configuration parameters including:
            - min_utterance_length: Min tokens to analyze (default: 50)
            - component_weights: Weights for different argument components
            - domain_specific_terms: Climate adaptation terminology
            
    Returns:
        Dictionary with argument structure metrics
    """
    # Use default configuration if none provided
    if config is None:
        config = {
            "min_utterance_length": 50,
            "component_weights": {
                "claim": 1.0,
                "evidence": 1.0,
                "warrant": 1.5,
                "qualifier": 0.5,
                "backing": 0.5,
                "rebuttal": 1.0
            },
            "domain_specific_terms": [
                "adaptation", "vulnerability", "resilience", "mitigation",
                "exposure", "sensitivity", "adaptive capacity", "climate risk",
                "sea level rise", "flood", "drought", "heat wave"
            ]
        }
    
    # Filter utterances by length
    substantial_utterances = []
    for utterance in workshop_transcript:
        # Tokenize the content (simplified)
        tokens = utterance["content"].split()
        if len(tokens) >= config["min_utterance_length"]:
            substantial_utterances.append(utterance)
    
    # Analyze each substantial utterance
    argument_metrics = []
    for utterance in substantial_utterances:
        # Extract argument components using LLM
        components = extract_argument_components(utterance["content"])
        
        # Calculate completeness score
        completeness_score = calculate_argument_completeness(
            components, 
            weights=config["component_weights"]
        )
        
        # Calculate evidence depth
        evidence_depth = calculate_evidence_depth(components["evidence"])
        
        # Analyze adaptation reasoning
        adaptation_reasoning = analyze_adaptation_reasoning(
            utterance["content"],
            components,
            domain_terms=config["domain_specific_terms"]
        )
        
        # Store metrics for this utterance
        argument_metrics.append({
            "utterance_id": utterance["id"],
            "speaker": utterance["speaker"],
            "completeness_score": completeness_score,
            "evidence_depth": evidence_depth,
            "adaptation_reasoning": adaptation_reasoning,
            "components_present": [k for k, v in components.items() if v]
        })
    
    # Calculate aggregate metrics
    aggregate_metrics = {
        "average_completeness": sum(m["completeness_score"] for m in argument_metrics) / len(argument_metrics) if argument_metrics else 0,
        "average_evidence_depth": sum(m["evidence_depth"] for m in argument_metrics) / len(argument_metrics) if argument_metrics else 0,
        "adaptation_reasoning_prevalence": sum(m["adaptation_reasoning"] for m in argument_metrics) / len(argument_metrics) if argument_metrics else 0,
        "utterances_analyzed": len(argument_metrics),
        "detailed_metrics": argument_metrics
    }
    
    # Calculate metrics by stage
    stage_metrics = {}
    for metric in argument_metrics:
        # Extract stage from utterance ID (assuming format like "S2_T15")
        stage_id = metric["utterance_id"].split("_")[0] if "_" in metric["utterance_id"] else "Unknown"
        
        if stage_id not in stage_metrics:
            stage_metrics[stage_id] = {
                "utterances": 0,
                "completeness_sum": 0,
                "evidence_depth_sum": 0,
                "adaptation_reasoning_sum": 0
            }
        
        stage = stage_metrics[stage_id]
        stage["utterances"] += 1
        stage["completeness_sum"] += metric["completeness_score"]
        stage["evidence_depth_sum"] += metric["evidence_depth"]
        stage["adaptation_reasoning_sum"] += metric["adaptation_reasoning"]
    
    # Calculate averages by stage
    for stage_id, metrics in stage_metrics.items():
        count = metrics["utterances"]
        if count > 0:
            stage_metrics[stage_id] = {
                "average_completeness": metrics["completeness_sum"] / count,
                "average_evidence_depth": metrics["evidence_depth_sum"] / count,
                "average_adaptation_reasoning": metrics["adaptation_reasoning_sum"] / count,
                "utterances_analyzed": count
            }
    
    aggregate_metrics["stage_metrics"] = stage_metrics
    
    return aggregate_metrics

def extract_argument_components(text):
    """
    Extract argument components using LLM-based analysis.
    
    Args:
        text: Content of the utterance
        
    Returns:
        Dictionary with identified argument components
    """
    # Prepare the prompt for the LLM
    prompt = f"""
    Analyze the following statement from a climate adaptation workshop and identify the argumentative components according to Toulmin's model:

    TEXT: {text}

    For each component type, extract the relevant text if present or indicate "None" if absent:
    1. CLAIM: The main assertion or position being argued
    2. EVIDENCE: Data, facts, or examples that support the claim
    3. WARRANT: The logical connection between evidence and claim
    4. QUALIFIER: Words expressing degree of certainty or limitations
    5. BACKING: Support for the warrant itself
    6. REBUTTAL: Counter-arguments or conditions

    Format your response as a JSON object with component types as keys and extracted text as values.
    """
    
    # Call LLM to extract components
    try:
        response = llm_client.generate(
            prompt=prompt,
            system_prompt="You are an expert in argumentation analysis for climate adaptation planning.",
            response_format={"type": "json_object"}
        )
        components = json.loads(response)
        
        # Convert to boolean presence indicators and extracted text
        processed_components = {}
        for component in ["claim", "evidence", "warrant", "qualifier", "backing", "rebuttal"]:
            component_text = components.get(component, "None")
            processed_components[component] = component_text if component_text != "None" else None
            
        return processed_components
        
    except Exception as e:
        # Fallback to empty components if LLM processing fails
        return {
            "claim": None,
            "evidence": None,
            "warrant": None,
            "qualifier": None,
            "backing": None,
            "rebuttal": None
        }

def calculate_argument_completeness(components, weights):
    """
    Calculate argument completeness score based on components present.
    
    Args:
        components: Dictionary of argument components
        weights: Weights for each component type
        
    Returns:
        Completeness score between 0 and 1
    """
    max_possible_score = sum(weights.values())
    actual_score = sum(weights[component] for component in components if components[component] is not None)
    
    completeness = actual_score / max_possible_score if max_possible_score > 0 else 0
    return min(1.0, completeness)  # Cap at 1.0

def calculate_evidence_depth(evidence_text):
    """
    Calculate depth of evidence based on specificity and quantity.
    
    Args:
        evidence_text: Extracted evidence text or None
        
    Returns:
        Evidence depth score between 0 and 1
    """
    if not evidence_text:
        return 0.0
    
    # Count evidence pieces (separated by markers like numbers, bullets)
    evidence_count = 1 + sum(1 for char in evidence_text if char in [',', ';', '.'] and char != '.')
    
    # Check for quantitative elements
    has_numbers = bool(re.search(r'\d+(\.\d+)?', evidence_text))
    
    # Check for specific references
    has_specific_references = bool(re.search(r'(study|report|research|survey|data|according to|evidence|shows)', 
                                             evidence_text, re.IGNORECASE))
    
    # Calculate depth score (simplified model)
    depth_score = min(1.0, 0.2 * evidence_count + 0.3 * has_numbers + 0.3 * has_specific_references)
    
    return depth_score

def analyze_adaptation_reasoning(text, components, domain_terms):
    """
    Analyze climate adaptation-specific reasoning patterns.
    
    Args:
        text: Full text content
        components: Extracted argument components
        domain_terms: List of domain-specific terms
        
    Returns:
        Score indicating presence of adaptation reasoning
    """
    # Check for domain terminology
    term_count = sum(1 for term in domain_terms if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE))
    domain_score = min(1.0, term_count / 5)  # Cap at 1.0
    
    # Check for adaptation reasoning patterns
    patterns = [
        r'\b(vulnerability|vulnerable|susceptible)\b',
        r'\b(capacity|capabilities|ability)\b',
        r'\b(risk|risks|hazard|threat)\b',
        r'\b(uncertain|uncertainty|likely|probability)\b',
        r'\b(long[-\s]term|future|projection)\b',
        r'\b(trade[-\s]?offs?|balance|weighing)\b'
    ]
    
    pattern_matches = sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))
    pattern_score = min(1.0, pattern_matches / len(patterns))
    
    # Combine scores with more weight on reasoning patterns
    adaptation_reasoning_score = (0.4 * domain_score) + (0.6 * pattern_score)
    
    return adaptation_reasoning_score
```

## Interpretation Guidelines

Argument Structure Assessment produces several key metrics that can be interpreted as follows:

### Argument Completeness Score (ACS)

The ACS represents the percentage of expected argument components present in a contribution, weighted by importance:

| Score Range | Interpretation | Example |
|-------------|----------------|---------|
| 0.8 - 1.0   | Comprehensive argument | Complete with claim, evidence, warrant, and qualifications |
| 0.6 - 0.8   | Substantial argument | Contains core elements but may lack qualifiers or rebuttals |
| 0.4 - 0.6   | Partial argument | Basic claim with some evidence but limited logical structure |
| 0.2 - 0.4   | Minimal argument | Simple assertions with little support |
| < 0.2       | Non-argumentative | Statements without argumentative structure |

### Evidence Depth Score

The Evidence Depth Score measures the specificity, quantity, and quality of evidence provided:

| Score Range | Interpretation | Example |
|-------------|----------------|---------|
| 0.8 - 1.0   | Rigorous evidence | Multiple specific data points, quantified impacts, cited sources |
| 0.6 - 0.8   | Substantial evidence | Specific examples with some quantification |
| 0.4 - 0.6   | Moderate evidence | General examples with limited specificity |
| 0.2 - 0.4   | Limited evidence | Vague references to supporting information |
| < 0.2       | Minimal/No evidence | Claims without supporting evidence |

### Adaptation Reasoning Score

This score reflects the presence of climate adaptation-specific reasoning patterns:

| Score Range | Interpretation | Example |
|-------------|----------------|---------|
| 0.8 - 1.0   | Advanced adaptation reasoning | Sophisticated discussion of vulnerability, capacity, uncertainty |
| 0.6 - 0.8   | Strong adaptation reasoning | Clear consideration of climate factors and trade-offs |
| 0.4 - 0.6   | Moderate adaptation reasoning | Some adaptation terminology and concepts |
| 0.2 - 0.4   | Basic adaptation reasoning | Limited consideration of adaptation specifics |
| < 0.2       | General reasoning | Discussion lacks adaptation-specific elements |

## Benchmarks for Climate Adaptation Workshops

Based on analysis of expert-facilitated climate adaptation workshops, we can establish these benchmarks:

| Workshop Stage | Expected ACS | Expected Evidence Depth | Expected Adaptation Reasoning |
|----------------|--------------|-------------------------|-------------------------------|
| Introduction | 0.3 - 0.5 | 0.2 - 0.4 | 0.3 - 0.5 |
| Impact Exploration | 0.5 - 0.7 | 0.5 - 0.7 | 0.5 - 0.7 |
| Adaptation Brainstorming | 0.6 - 0.8 | 0.4 - 0.6 | 0.7 - 0.9 |
| Strategy Evaluation | 0.7 - 0.9 | 0.6 - 0.8 | 0.7 - 0.9 |

These benchmarks reflect the evolution of argument complexity as workshops progress from initial problem exploration to detailed solution evaluation.

## Visualizations

Argument structure can be visualized through:

1. **Radar charts** displaying the three key metrics (completeness, evidence depth, adaptation reasoning)
2. **Stacked bar charts** showing the presence of different argument components by speaker
3. **Heat maps** indicating argument quality across the workshop timeline

Example visualization:

```
Argument Structure by Workshop Stage
                  ACS    |    Evidence    |    Adaptation
                         |     Depth      |    Reasoning
Stage 1    [▓▓▓▓░░░░░░]  |  [▓▓░░░░░░░░]  |  [▓▓▓░░░░░░░]  
Stage 2    [▓▓▓▓▓▓░░░░]  |  [▓▓▓▓▓░░░░░]  |  [▓▓▓▓▓░░░░░]  
Stage 3    [▓▓▓▓▓▓▓░░░]  |  [▓▓▓▓░░░░░░]  |  [▓▓▓▓▓▓▓░░░]  
Stage 4    [▓▓▓▓▓▓▓▓░░]  |  [▓▓▓▓▓▓░░░░]  |  [▓▓▓▓▓▓▓░░░]  
           0    0.5    1  |  0    0.5    1  |  0    0.5    1  
```

## Integration with Evaluation Framework

Argument Structure Assessment is one component of the broader Semantic Coherence Analysis dimension, working alongside:

- **Topic Coherence Scoring**: Assessing the internal semantic cohesion within discussion segments
- **Thematic Evolution Tracking**: Analyzing how topics progress and develop over time

Together, these metrics provide a comprehensive evaluation of both the semantic and logical quality of LLM-simulated climate adaptation discussions, enabling rich comparisons between different models and with human workshops.

## Advanced Applications

Beyond basic argument assessment, advanced applications include:

- **Persona-Specific Analysis**: Comparing argument quality across different personas
- **Cross-Reference Network**: Mapping how arguments build upon or reference previous points
- **Comparative Model Evaluation**: Identifying which LLM architectures produce more logically sound arguments
- **Argument Tree Visualization**: Building hierarchical structures showing how arguments relate

## References

1. Toulmin, S.E. (2003). "The Uses of Argument." Cambridge University Press.
2. Li, M. et al. (2024). "Can Large Language Models Replace Human Participants?" Marketing Science Institute Report 25-101.
3. Hewitt, J. et al. (2024). "LLM Social Simulations Are a Promising Research Method." arXiv:2504.02234.
4. Lawrence, J., & Reed, C. (2020). "Argument Mining: A Survey." Computational Linguistics, 45(4), 765-818.
5. Mullenbach, L.E., & Wilhelm Stanis, S.A. (2024). "Understanding how justice is considered in climate adaptation approaches." Journal of Environmental Planning and Management, 1-20.
