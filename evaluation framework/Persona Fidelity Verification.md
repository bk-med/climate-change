# Persona Fidelity Verification

## Overview

Persona Fidelity Verification is a comprehensive assessment methodology for evaluating how faithfully LLM-simulated stakeholders maintain their defined characteristics throughout climate adaptation workshops. This evaluation dimension quantifies the authenticity and consistency with which synthetic personas represent their assigned demographic profiles, value systems, and linguistic traits—essential factors in producing reliable insights for equitable adaptation planning.

## Purpose

In climate adaptation planning, ensuring diverse stakeholder representation is crucial for developing strategies that address the needs of all community members, particularly those most vulnerable to climate impacts. Persona Fidelity Verification helps determine whether:

- Simulated stakeholders maintain consistent positions aligned with their defined vulnerability profiles
- Synthetic participants authentically express value systems appropriate to their backgrounds
- Personas maintain realistic and consistent linguistic patterns throughout discussions
- Potential bias or stereotyping is identified and quantified

This evaluation dimension is particularly important for assessing whether LLM-simulated workshops can generate reliable insights about differential impacts and adaptation preferences across diverse community segments—critical information for developing equitable adaptation strategies.

## Input Specification

The Persona Fidelity Verification component processes both workshop transcript data and persona definitions with specific formatting requirements:

### Workshop Transcript Input

The primary transcript input is a list of **utterance objects** representing sequential contributions from the workshop:

```json
[
  {
    "id": "S2_T15",
    "speaker": "Eleanor_Martinez",
    "content": "I've lived in my coastal home for over 40 years and have seen the water getting closer every year. The tide comes up into my garden now, which never happened when we first moved here. While I understand the need for planning, fixed incomes like mine make major renovations impossible without assistance.",
    "topics": ["observed_changes", "personal_experience", "financial_constraints"],
    "timestamp": "2025-04-28T14:45:10Z"
  },
  {
    "id": "S2_T16",
    "speaker": "moderator",
    "content": "Thank you for sharing that experience, Eleanor. How have these changes affected your sense of security in your home?",
    "topics": ["moderation", "security_concerns"],
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

### Persona Definition Input

For each persona in the workshop, a detailed definition is required:

```json
{
  "id": "persona_elderly_coastal_1",
  "type": "human",
  "name": "Eleanor_Martinez",
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
      {"option": "community_resilience_hub", "feasibility": "high", "preference": "high"}
    ]
  },
  "linguistic_parameters": {
    "formality_level": 0.7,
    "technical_vocabulary": 0.3,
    "regional_markers": 0.4,
    "speech_patterns": ["reflective", "historically_oriented", "direct"]
  }
}
```

The persona definition must include at minimum:
- `id`: Unique identifier matching the `speaker` field in utterances
- `name`: Display name of the persona
- `metadata`: Core characteristics including demographics and values
- `linguistic_parameters`: Speaking style parameters

### Configuration Parameters

The verification function accepts several configuration parameters:

- `value_classification_model`: Model to use for value classification (default: BART-large-mnli)
- `embedding_model`: Model to use for generating embeddings (default: text-embedding-3-large)
- `stance_dimensions`: List of stance dimensions to evaluate
- `linguistic_features`: List of linguistic features to track

## Methodology

Persona Fidelity Verification comprises three complementary metrics that together provide a comprehensive assessment of simulation quality:

### 1. Stance Consistency Index (SCI)

The Stance Consistency Index quantifies alignment between predefined persona stances and their expressed positions throughout discussions. For each persona, key stance dimensions are established based on their demographic profile, vulnerability factors, and adaptation context. The SCI evaluates whether utterances remain consistent with these predefined stances.

### 2. Value Expression Congruence (VEC)

Value Expression Congruence assesses how consistently personas express their underlying value systems. Based on Schwartz's value theory adapted for climate contexts, each persona is defined with a hierarchical value structure. The VEC measures whether the frequency and emphasis of expressed values in utterances align with the persona's predefined value priorities.

### 3. Linguistic Signature Stability (LSS)

Linguistic Signature Stability evaluates the consistency of persona-specific language patterns throughout the discussion. This includes lexical features, syntactic complexity, pragmatic markers, and sociocultural indicators. LSS helps identify whether a persona's "voice" remains stable and authentic across the simulation.

## Implementation Details

```python
def verify_persona_fidelity(workshop_transcript, persona_definitions, config=None):
    """
    Evaluate persona fidelity across multiple dimensions.
    
    Args:
        workshop_transcript: List of utterance objects containing:
            - id: Unique identifier
            - speaker: Speaker identifier
            - content: Text content of the utterance
            - topics: (optional) List of topics covered
            - timestamp: (optional) Timestamp of utterance
        persona_definitions: Dictionary mapping persona IDs to their definitions
        config: Configuration parameters
            
    Returns:
        Dictionary with persona fidelity metrics
    """
    # Use default configuration if none provided
    if config is None:
        config = {
            "stance_dimensions": [
                "retreat_vs_protect", 
                "individual_vs_collective", 
                "engineered_vs_nature_based",
                "incremental_vs_transformative",
                "present_vs_future_focused"
            ],
            "value_classification_model": "BART-large-mnli",
            "embedding_model": "text-embedding-3-large",
            "linguistic_features": [
                "sentence_length", 
                "vocabulary_richness",
                "formality_markers",
                "hedge_words", 
                "certainty_expressions"
            ]
        }
    
    # Group utterances by persona
    persona_utterances = {}
    for utterance in workshop_transcript:
        speaker = utterance["speaker"]
        if speaker == "moderator":
            continue
            
        if speaker not in persona_utterances:
            persona_utterances[speaker] = []
            
        persona_utterances[speaker].append(utterance)
    
    # Calculate fidelity metrics for each persona
    persona_metrics = {}
    for persona_id, utterances in persona_utterances.items():
        # Skip if no persona definition available
        if persona_id not in persona_definitions:
            continue
            
        persona_def = persona_definitions[persona_id]
        
        # 1. Calculate Stance Consistency Index
        stance_index = calculate_stance_consistency(
            utterances, 
            persona_def, 
            config["stance_dimensions"],
            config["embedding_model"]
        )
        
        # 2. Calculate Value Expression Congruence
        value_congruence = calculate_value_congruence(
            utterances, 
            persona_def,
            config["value_classification_model"]
        )
        
        # 3. Calculate Linguistic Signature Stability
        linguistic_stability = calculate_linguistic_stability(
            utterances, 
            persona_def,
            config["linguistic_features"]
        )
        
        # Store metrics for this persona
        persona_metrics[persona_id] = {
            "name": persona_def.get("name", persona_id),
            "utterance_count": len(utterances),
            "stance_consistency_index": stance_index,
            "value_expression_congruence": value_congruence,
            "linguistic_signature_stability": linguistic_stability,
            "overall_fidelity_score": (stance_index + value_congruence + linguistic_stability) / 3
        }
    
    # Calculate aggregate metrics
    aggregate_metrics = {
        "average_stance_consistency": sum(p["stance_consistency_index"] for p in persona_metrics.values()) / len(persona_metrics) if persona_metrics else 0,
        "average_value_congruence": sum(p["value_expression_congruence"] for p in persona_metrics.values()) / len(persona_metrics) if persona_metrics else 0,
        "average_linguistic_stability": sum(p["linguistic_signature_stability"] for p in persona_metrics.values()) / len(persona_metrics) if persona_metrics else 0,
        "overall_fidelity": sum(p["overall_fidelity_score"] for p in persona_metrics.values()) / len(persona_metrics) if persona_metrics else 0,
        "personas_analyzed": len(persona_metrics),
        "detailed_metrics": persona_metrics
    }
    
    return aggregate_metrics
```

### Stance Consistency Index (SCI)

```python
def calculate_stance_consistency(utterances, persona_definition, stance_dimensions, embedding_model):
    """
    Calculate Stance Consistency Index for a persona.
    
    Args:
        utterances: List of utterances by this persona
        persona_definition: Definition of the persona
        stance_dimensions: List of stance dimensions to evaluate
        embedding_model: Model to use for embeddings
        
    Returns:
        Stance Consistency Index (0-1)
    """
    # Define expected stances based on persona definition
    expected_stances = derive_expected_stances(persona_definition, stance_dimensions)
    
    # If no expected stances could be derived, return 0
    if not expected_stances:
        return 0.0
    
    # Extract stance signals from utterances
    observed_stances = {}
    
    # Analyze each utterance
    for utterance in utterances:
        # Extract stance signals from current utterance
        stance_signals = extract_stance_signals(utterance["content"], stance_dimensions, embedding_model)
        
        # Update observed stances with new signals
        for dimension, signal in stance_signals.items():
            if dimension not in observed_stances:
                observed_stances[dimension] = []
            observed_stances[dimension].append(signal)
    
    # Calculate consistency for each dimension
    dimension_consistency = {}
    
    for dimension in expected_stances:
        # Skip if no observed signals for this dimension
        if dimension not in observed_stances or not observed_stances[dimension]:
            continue
            
        expected = expected_stances[dimension]
        observed = sum(observed_stances[dimension]) / len(observed_stances[dimension])
        
        # Calculate consistency as 1 - normalized distance
        distance = abs(expected - observed)
        max_possible_distance = 1.0  # Max distance on a 0-1 scale
        consistency = 1.0 - (distance / max_possible_distance)
        
        dimension_consistency[dimension] = consistency
    
    # Return average consistency across all dimensions
    if not dimension_consistency:
        return 0.0
        
    return sum(dimension_consistency.values()) / len(dimension_consistency)

def derive_expected_stances(persona_definition, stance_dimensions):
    """
    Derive expected stances for a persona based on their definition.
    
    Args:
        persona_definition: Definition of the persona
        stance_dimensions: List of stance dimensions to evaluate
        
    Returns:
        Dictionary of expected stance positions for each dimension
    """
    expected_stances = {}
    
    # Map demographic and vulnerability factors to stance positions
    demographics = persona_definition.get("metadata", {}).get("demographics", {})
    vulnerability = persona_definition.get("metadata", {}).get("vulnerability_factors", {})
    adaptation = persona_definition.get("adaptation_context", {})
    values = persona_definition.get("metadata", {}).get("value_priorities", [])
    
    # Derive stance on retreat vs. protect
    if "retreat_vs_protect" in stance_dimensions:
        # Start with neutral position
        retreat_vs_protect = 0.5
        
        # Adjust based on demographics
        if demographics.get("age", 0) > 65:
            # Older residents often prefer protection of existing homes
            retreat_vs_protect -= 0.1
            
        if demographics.get("housing_type", "") == "owned_single_family":
            # Homeowners often prefer protection
            retreat_vs_protect -= 0.1
        
        # Adjust based on vulnerability
        if vulnerability.get("physical_exposure", 0) > 0.7:
            # Higher exposure may necessitate retreat
            retreat_vs_protect += 0.2
            
        # Adjust based on adaptation preferences
        adaptation_options = adaptation.get("adaptation_options", [])
        for option in adaptation_options:
            if "retreat" in option.get("option", "").lower() and option.get("preference", "") == "high":
                retreat_vs_protect += 0.3
            elif "protection" in option.get("option", "").lower() and option.get("preference", "") == "high":
                retreat_vs_protect -= 0.3
        
        # Ensure value is between 0-1
        retreat_vs_protect = max(0.0, min(1.0, retreat_vs_protect))
        expected_stances["retreat_vs_protect"] = retreat_vs_protect
    
    # Similar derivations for other stance dimensions...
    
    return expected_stances

def extract_stance_signals(text, stance_dimensions, embedding_model):
    """
    Extract stance signals from text using LLM or embedding-based analysis.
    
    Args:
        text: Content of the utterance
        stance_dimensions: List of stance dimensions to evaluate
        embedding_model: Model to use for embeddings
        
    Returns:
        Dictionary of stance signals for each dimension
    """
    # This could be implemented using various approaches:
    
    # 1. LLM-based classification approach
    prompt = f"""
    Analyze the following statement from a climate adaptation workshop and estimate the speaker's position on these stance dimensions:

    TEXT: {text}

    For each dimension, provide a score from 0.0 to 1.0:
    1. retreat_vs_protect: 0=strongly favors protection, 1=strongly favors retreat
    2. individual_vs_collective: 0=emphasizes individual solutions, 1=emphasizes collective solutions
    3. engineered_vs_nature_based: 0=prefers engineered solutions, 1=prefers nature-based solutions
    4. incremental_vs_transformative: 0=favors incremental change, 1=favors transformative change
    5. present_vs_future_focused: 0=focused on immediate needs, 1=focused on long-term future

    Only include dimensions clearly expressed in the text. Format your response as a JSON object with dimension names as keys and scores as values.
    """
    
    # 2. Embedding similarity approach
    stance_signals = {}
    
    for dimension in stance_dimensions:
        # Create axis endpoints
        if dimension == "retreat_vs_protect":
            low_endpoint = "We should invest in protecting our existing homes and infrastructure from climate impacts."
            high_endpoint = "We should gradually relocate away from the most vulnerable areas to safer locations."
        elif dimension == "individual_vs_collective":
            low_endpoint = "Individual homeowners should take responsibility for adapting their own properties."
            high_endpoint = "We need community-wide solutions and shared responsibility for adaptation."
        # Define other dimension endpoints similarly...
        
        # Get embeddings
        text_embedding = get_embedding(text, embedding_model)
        low_embedding = get_embedding(low_endpoint, embedding_model)
        high_embedding = get_embedding(high_endpoint, embedding_model)
        
        # Calculate similarities
        low_similarity = cosine_similarity(text_embedding, low_embedding)
        high_similarity = cosine_similarity(text_embedding, high_embedding)
        
        # Calculate position on the dimension axis
        total_similarity = low_similarity + high_similarity
        if total_similarity > 0:
            position = high_similarity / total_similarity
            stance_signals[dimension] = position
    
    return stance_signals
```

### Value Expression Congruence (VEC)

```python
def calculate_value_congruence(utterances, persona_definition, classification_model):
    """
    Calculate Value Expression Congruence for a persona.
    
    Args:
        utterances: List of utterances by this persona
        persona_definition: Definition of the persona
        classification_model: Model to use for value classification
        
    Returns:
        Value Expression Congruence score (0-1)
    """
    # Extract expected value priorities from persona definition
    expected_values = {}
    total_weight = 0
    
    for value_item in persona_definition.get("metadata", {}).get("value_priorities", []):
        value = value_item.get("value", "")
        weight = value_item.get("weight", 0)
        if value and weight > 0:
            expected_values[value] = weight
            total_weight += weight
    
    # Normalize weights to sum to 1
    if total_weight > 0:
        expected_values = {k: v/total_weight for k, v in expected_values.items()}
    else:
        return 0.0  # No valid value weights
    
    # Define value categories and example expressions
    value_categories = {
        "security": [
            "safety", "stability", "protection", "risk reduction",
            "security", "sheltering", "defense", "keeping safe"
        ],
        "tradition": [
            "heritage", "history", "customs", "practices", 
            "cultural", "traditional", "way of life", "roots"
        ],
        "universalism": [
            "environment", "sustainability", "equity", "fairness",
            "justice", "shared", "common good", "protecting nature"
        ],
        "self_determination": [
            "autonomy", "freedom", "choice", "control", "independence",
            "self-reliance", "deciding", "my own terms"
        ],
        "achievement": [
            "success", "progress", "growth", "prosperity", "improvement",
            "advancing", "building", "developing", "achieving"
        ]
    }
    
    # Analyze value expressions in utterances
    observed_values = {value: 0 for value in expected_values}
    total_expressions = 0
    
    for utterance in utterances:
        # Classify values expressed in the utterance
        expressed_values = classify_values(utterance["content"], value_categories, classification_model)
        
        # Update counts
        for value, count in expressed_values.items():
            if value in observed_values:
                observed_values[value] += count
                total_expressions += count
    
    # Calculate normalized observed value distribution
    if total_expressions > 0:
        observed_distribution = {k: v/total_expressions for k, v in observed_values.items()}
    else:
        observed_distribution = {k: 0 for k in observed_values}
    
    # Calculate congruence using weighted similarity
    congruence = 0
    for value in expected_values:
        expected_weight = expected_values[value]
        observed_weight = observed_distribution.get(value, 0)
        
        # Higher weight for values that should be more prominent
        value_importance = expected_weight
        value_congruence = 1 - abs(expected_weight - observed_weight)
        
        congruence += value_importance * value_congruence
    
    # Normalize by sum of expected weights (which should be 1)
    return congruence

def classify_values(text, value_categories, model_name):
    """
    Classify values expressed in text using zero-shot classification.
    
    Args:
        text: Text to analyze
        value_categories: Dictionary mapping values to example expressions
        model_name: Model to use for classification
        
    Returns:
        Dictionary of values with their occurrence counts
    """
    # This could be implemented using various approaches:
    
    # 1. Zero-shot classification with BART-large-mnli
    value_counts = {}
    
    for value, expressions in value_categories.items():
        # Combine expressions into a hypothesis statement
        hypothesis = f"This text expresses the value of {value}, which includes concepts like {', '.join(expressions)}"
        
        # Classify using the model
        classification_result = zero_shot_classify(text, hypothesis, model_name)
        
        if classification_result > 0.7:  # Confidence threshold
            value_counts[value] = value_counts.get(value, 0) + 1
    
    # 2. LLM-based approach
    prompt = f"""
    Analyze the following statement from a climate adaptation workshop and identify which values the speaker expresses:

    TEXT: {text}

    For each of these values, indicate if it is expressed in the text (0=not expressed, 1=weakly expressed, 2=strongly expressed):
    1. Security: Safety, stability, protection from threats
    2. Tradition: Respect for customs, cultural heritage, established ways of life
    3. Universalism: Environmental protection, social justice, equity, common good
    4. Self-determination: Freedom of choice, autonomy, independence, personal control
    5. Achievement: Success, progress, improvement, growth, development

    Format your response as a JSON object with value names as keys and expression scores as values.
    """
    
    return value_counts
```

### Linguistic Signature Stability (LSS)

```python
def calculate_linguistic_stability(utterances, persona_definition, linguistic_features):
    """
    Calculate Linguistic Signature Stability for a persona.
    
    Args:
        utterances: List of utterances by this persona
        persona_definition: Definition of the persona
        linguistic_features: List of linguistic features to track
        
    Returns:
        Linguistic Signature Stability score (0-1)
    """
    # Extract expected linguistic parameters
    expected_params = persona_definition.get("linguistic_parameters", {})
    
    # Define expected linguistic signature based on definition
    expected_signature = {
        "formality_level": expected_params.get("formality_level", 0.5),
        "technical_vocabulary": expected_params.get("technical_vocabulary", 0.5),
        "regional_markers": expected_params.get("regional_markers", 0.5),
        "sentence_length": derive_expected_sentence_length(expected_params),
        "vocabulary_richness": derive_expected_vocabulary_richness(expected_params),
        "hedging_frequency": derive_expected_hedging(expected_params)
    }
    
    # Extract linguistic features from each utterance
    utterance_features = []
    
    for utterance in utterances:
        features = extract_linguistic_features(utterance["content"], linguistic_features)
        utterance_features.append(features)
    
    # Calculate variance for each feature
    feature_variances = {}
    for feature in expected_signature:
        if feature not in linguistic_features:
            continue
            
        # Extract feature values across utterances
        values = [features.get(feature, 0) for features in utterance_features if feature in features]
        
        if not values:
            continue
            
        # Calculate variance
        mean_value = sum(values) / len(values)
        variance = sum((x - mean_value) ** 2 for x in values) / len(values)
        
        # Store normalized variance (lower is better)
        feature_variances[feature] = min(1.0, variance / 0.25)  # Normalize with expected variance
    
    # Calculate deviation from expected signature
    signature_deviations = {}
    
    for feature in expected_signature:
        if feature not in linguistic_features:
            continue
            
        # Calculate average observed value
        values = [features.get(feature, 0) for features in utterance_features if feature in features]
        
        if not values:
            continue
            
        observed_avg = sum(values) / len(values)
        expected = expected_signature[feature]
        
        # Calculate normalized deviation
        deviation = abs(observed_avg - expected) / max(expected, 1-expected)
        signature_deviations[feature] = min(1.0, deviation)
    
    # Calculate stability score (higher is better)
    # Combine low variance (consistency) with low deviation (authenticity)
    if not feature_variances or not signature_deviations:
        return 0.0
        
    variance_score = 1 - (sum(feature_variances.values()) / len(feature_variances))
    deviation_score = 1 - (sum(signature_deviations.values()) / len(signature_deviations))
    
    # Overall linguistic stability score
    stability_score = 0.4 * variance_score + 0.6 * deviation_score
    
    return stability_score

def extract_linguistic_features(text, features_to_extract):
    """
    Extract linguistic features from text.
    
    Args:
        text: Text to analyze
        features_to_extract: List of features to extract
        
    Returns:
        Dictionary of linguistic features
    """
    linguistic_features = {}
    
    # Tokenize text
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    words = text.split()
    
    # Extract sentence length
    if "sentence_length" in features_to_extract:
        if sentences:
            avg_sentence_length = len(words) / len(sentences)
            linguistic_features["sentence_length"] = min(1.0, avg_sentence_length / 30)  # Normalize
    
    # Extract vocabulary richness (type-token ratio)
    if "vocabulary_richness" in features_to_extract:
        if words:
            unique_words = len(set(w.lower() for w in words))
            ttr = unique_words / len(words)
            linguistic_features["vocabulary_richness"] = min(1.0, ttr * 2)  # Normalize
    
    # Extract formality markers
    if "formality_markers" in features_to_extract:
        formal_markers = ["therefore", "consequently", "furthermore", "additionally", "nevertheless", 
                         "however", "thus", "hence", "regarding", "concerning"]
        informal_markers = ["anyway", "like", "so", "you know", "kinda", "sorta", "pretty much",
                          "stuff", "thing", "gonna", "wanna"]
        
        formal_count = sum(1 for marker in formal_markers if marker in text.lower())
        informal_count = sum(1 for marker in informal_markers if marker in text.lower())
        
        if formal_count + informal_count > 0:
            formality = formal_count / (formal_count + informal_count)
        else:
            formality = 0.5  # Neutral if no markers detected
            
        linguistic_features["formality_markers"] = formality
    
    # Extract hedge words
    if "hedge_words" in features_to_extract:
        hedges = ["maybe", "perhaps", "possibly", "probably", "kind of", "sort of", "might", 
                 "could be", "I think", "I believe", "in my opinion", "seems like"]
        
        hedge_count = sum(1 for hedge in hedges if hedge in text.lower())
        normalized_count = min(1.0, hedge_count / 3)  # Normalize
        
        linguistic_features["hedge_words"] = normalized_count
    
    # Extract certainty expressions
    if "certainty_expressions" in features_to_extract:
        certainty_phrases = ["definitely", "certainly", "without a doubt", "absolutely", 
                            "undoubtedly", "surely", "clearly", "obviously", "always", "never"]
        
        certainty_count = sum(1 for phrase in certainty_phrases if phrase in text.lower())
        normalized_count = min(1.0, certainty_count / 3)  # Normalize
        
        linguistic_features["certainty_expressions"] = normalized_count
    
    return linguistic_features
```

## Interpretation Guidelines

Each of the three metrics provides distinct insights into the quality of persona simulation:

### Stance Consistency Index (SCI)

The SCI measures how well personas maintain consistent positions on key adaptation issues:

| Score Range | Interpretation | Example |
|-------------|----------------|---------|
| 0.8 - 1.0   | High consistency | Persona maintains positions fully aligned with defined profile |
| 0.6 - 0.8   | Moderate consistency | Mostly consistent with occasional minor deviations |
| 0.4 - 0.6   | Inconsistent | Significant deviations from expected positions |
| < 0.4       | Highly inconsistent | Positions contradict persona definition |

When evaluating climate adaptation workshops, it's important to recognize that some stance evolution may be appropriate as personas respond to new information or engage with other perspectives. The SCI should be interpreted with this context in mind.

### Value Expression Congruence (VEC)

The VEC assesses alignment between expressed values and predefined value priorities:

| Score Range | Interpretation | Example |
|-------------|----------------|---------|
| 0.8 - 1.0   | High congruence | Values expressed with frequencies matching persona definition |
| 0.6 - 0.8   | Moderate congruence | General alignment with some value expression imbalances |
| 0.4 - 0.6   | Low congruence | Significant misalignment in value expressions |
| < 0.4       | Very low congruence | Values expressed contradict defined priorities |

Climate adaptation contexts often involve value trade-offs, such as balancing individual property rights with community protection. Effective persona simulation should reflect these value tensions in ways consistent with the persona's defined priorities.

### Linguistic Signature Stability (LSS)

The LSS evaluates consistency in language patterns:

| Score Range | Interpretation | Example |
|-------------|----------------|---------|
| 0.8 - 1.0   | High stability | Consistent, authentic linguistic patterns throughout |
| 0.6 - 0.8   | Moderate stability | Generally consistent with minor variations |
| 0.4 - 0.6   | Inconsistent | Noticeable fluctuations in speaking style |
| < 0.4       | Highly inconsistent | Erratic language patterns that undermine authenticity |

Different workshop stages may naturally elicit different linguistic patterns (e.g., more technical language during strategy evaluation). This contextual variation should be considered when interpreting LSS scores.

## Benchmarks for Persona Fidelity

Based on analysis of expert-crafted and validated personas, we can establish these benchmarks:

| Persona Type | Expected SCI | Expected VEC | Expected LSS |
|--------------|--------------|--------------|--------------|
| General community members | 0.7 - 0.9 | 0.6 - 0.8 | 0.7 - 0.9 |
| Technical stakeholders | 0.7 - 0.9 | 0.7 - 0.9 | 0.7 - 0.9 |
| Vulnerable groups | 0.8 - 1.0 | 0.7 - 0.9 | 0.8 - 1.0 |
| Non-human entities | 0.8 - 1.0 | 0.8 - 1.0 | 0.9 - 1.0 |

Higher standards are appropriate for personas representing vulnerable groups, as misrepresentation can have serious equity implications for adaptation planning.

## Visualizations

Persona fidelity can be visualized through:

1. **Radar charts** displaying the three key metrics (SCI, VEC, LSS) for each persona
2. **Time-series graphs** showing how consistency measures evolve throughout the workshop
3. **Heat maps** comparing fidelity metrics across different LLM architectures

Example visualization:

```
Persona Fidelity Comparison Across Models
                 SCI   |    VEC    |    LSS
Eleanor    [▓▓▓▓▓▓▓▓░]  [▓▓▓▓▓▓░░░]  [▓▓▓▓▓▓▓░░]   GPT-4.0
(Elderly)  [▓▓▓▓▓▓▓░░]  [▓▓▓▓▓▓░░░]  [▓▓▓▓▓▓▓▓░]   GPT-4.5
           [▓▓▓▓▓▓▓▓░]  [▓▓▓▓▓▓▓░░]  [▓▓▓▓▓▓▓▓▓]   Claude-3.7

Miguel     [▓▓▓▓▓▓░░░]  [▓▓▓▓▓▓▓░░]  [▓▓▓▓▓▓░░░]   GPT-4.0
(Fisher)   [▓▓▓▓▓▓▓░░]  [▓▓▓▓▓▓▓░░]  [▓▓▓▓▓▓▓░░]   GPT-4.5
           [▓▓▓▓▓▓▓▓░]  [▓▓▓▓▓▓▓▓░]  [▓▓▓▓▓▓▓░░]   Claude-3.7

           0    0.5   1  0    0.5   1  0    0.5   1
```

## Integration with Evaluation Framework

Persona Fidelity Verification is one component of the broader evaluation framework, complementing:

- **Semantic Coherence Analysis**: Assessing the quality of discussion content
- **Engagement Metrics**: Measuring participation patterns and interactions
- **Output Quality Assessment**: Evaluating the adaptation solutions generated

Together, these dimensions provide a comprehensive assessment of LLM-simulated climate adaptation workshops, enabling meaningful comparisons between different models and with human-facilitated sessions.

## Advanced Applications

Beyond basic verification, advanced applications include:

- **Cross-Influence Analysis**: Measuring how personas influence each other's positions
- **Identity Consistency Testing**: Evaluating whether personas maintain consistent self-references
- **Stereotype Detection**: Identifying potential stereotyping in persona representations
- **Comparative Model Evaluation**: Determining which LLM architectures produce the most authentic personas

## References

1. Salminen, J. et al. (2025). "Generative AI for Persona Development: A Systematic Review." arXiv:2504.04927.
2. Schwartz, S.H. (2012). "An Overview of the Schwartz Theory of Basic Values." Online Readings in Psychology and Culture, 2(1).
3. Li, M. et al. (2024). "Can Large Language Models Replace Human Participants?" Marketing Science Institute Report 25-101.
4. Hewitt, J. et al. (2024). "LLM Social Simulations Are a Promising Research Method." arXiv:2504.02234.
5. Mullenbach, L.E., & Wilhelm Stanis, S.A. (2024). "Understanding how justice is considered in climate adaptation approaches." Journal of Environmental Planning and Management, 1-20.
6. Cohen, S. et al. (2024). "Virtual Personas for Language Models via an Anthology of Backstories." Berkeley Artificial Intelligence Research Blog.
7. Wang, Y. et al. (2024). "OASIS: Open Agent Social Interaction Simulations with One Million Agents." arXiv:2411.11581v4.
