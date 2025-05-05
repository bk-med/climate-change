# Topic Coherence Scoring

## Overview

Topic Coherence Scoring is a quantitative method for assessing the semantic consistency and logical flow of discussions in LLM-simulated climate adaptation workshops. This metric evaluates how well semantically related terms co-occur within discussion segments, providing an objective measure of topical consistency and focus throughout the simulated dialogue.

## Purpose

In climate adaptation discussions, maintaining coherent discourse around specific themes (e.g., vulnerability factors, adaptation options, implementation challenges) is essential for generating meaningful insights. Topic Coherence Scoring helps evaluate whether:

- Discussions maintain focus on relevant adaptation topics
- Related concepts appropriately cluster together in conversation 
- The dialogue progresses logically without disjointed topic shifts
- Domain-specific terminology is used in semantically appropriate contexts

This scoring mechanism is particularly valuable for identifying instances where LLM-generated dialogue may appear fluent on the surface but lacks deeper semantic consistency that would be expected in expert-facilitated human workshops.

## Input Specification

The Topic Coherence Scoring component processes structured workshop transcript data with specific formatting requirements:

### Primary Input

The primary input is a list of **utterance objects** representing sequential contributions from the workshop:

```json
[
  {
    "id": "S2_T15",
    "speaker": "Eleanor_Martinez",
    "content": "I've noticed that flooding in my neighborhood has gotten worse over the past decade. When I first moved here, it would only flood during major hurricanes, but now we see water in the streets during normal high tides several times a year.",
    "topics": ["observed_changes", "flooding", "local_knowledge"],
    "timestamp": "2025-04-28T14:45:10Z"
  },
  {
    "id": "S2_T16",
    "speaker": "moderator",
    "content": "Thank you for sharing that observation, Eleanor. Has anyone else noticed similar changes in their areas?",
    "topics": ["moderation", "observed_changes"],
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

The scoring function accepts several configuration parameters:

- `segment_size`: Number of utterances to include in each segment (default: 5-10)
- `top_k_keywords`: Number of top keywords to extract per segment (default: 20)
- `window_size`: Context window size for co-occurrence calculation (default: 20)
- `custom_stopwords`: Domain-specific words to exclude from analysis
- `domain_vocabulary`: Climate adaptation terminology for enhanced keyword extraction

## Methodology

### Mathematical Foundation

Topic Coherence Scoring employs Normalized Pointwise Mutual Information (NPMI) to quantify the statistical coherence of topic distributions. NPMI measures the degree to which word pairs co-occur more frequently than would be expected by chance alone.

For a pair of words $(w_i, w_j)$, the NPMI is calculated as:

$$NPMI(w_i, w_j) = \frac{\log\left(\frac{P(w_i, w_j)}{P(w_i)P(w_j)}\right)}{-\log P(w_i, w_j)}$$

Where:
- $P(w_i, w_j)$ is the joint probability of words $w_i$ and $w_j$ co-occurring
- $P(w_i)$ and $P(w_j)$ are the marginal probabilities of each word occurring independently

The NPMI value ranges from -1 (words never occur together) to 1 (words always occur together), with 0 indicating independence.

### Implementation Process

The Topic Coherence Scoring process follows these steps:

1. **Segmentation**: Divide the workshop transcript into meaningful segments (e.g., by stage or topic)
2. **Keyword Extraction**: For each segment, extract the top 20-30 keywords using TF-IDF weighting
3. **NPMI Calculation**: Calculate NPMI scores for all keyword pairs within each segment
4. **Aggregation**: Average the NPMI scores to produce a coherence score for the segment
5. **Threshold Comparison**: Compare scores against established benchmarks for human discussions

## Implementation Details

```python
def calculate_workshop_topic_coherence(workshop_transcript, config=None):
    """
    Calculate topic coherence scores for a full workshop transcript.
    
    Args:
        workshop_transcript: List of utterance objects containing:
            - id: Unique identifier
            - speaker: Speaker identifier
            - content: Text content of the utterance
            - topics: (optional) List of topics covered
            - timestamp: (optional) Timestamp of utterance
        config: Configuration parameters including:
            - segment_size: Number of utterances per segment (default: 8)
            - top_k_keywords: Number of top keywords to extract (default: 20)
            - window_size: Context window for co-occurrence (default: 20)
            - custom_stopwords: Domain-specific words to exclude
            - domain_vocabulary: Climate terminology to prioritize
            
    Returns:
        Dictionary with coherence scores by segment and overall metrics
    """
    # Use default configuration if none provided
    if config is None:
        config = {
            "segment_size": 8,
            "top_k_keywords": 20,
            "window_size": 20,
            "custom_stopwords": ["climate", "change", "adaptation", "workshop"],
            "domain_vocabulary": [
                "vulnerability", "resilience", "mitigation", "exposure", 
                "sensitivity", "adaptive capacity", "sea level rise"
            ]
        }
    
    # Segment the transcript
    segments = []
    segment_ids = []
    for i in range(0, len(workshop_transcript), config["segment_size"]):
        segment = workshop_transcript[i:i + config["segment_size"]]
        if len(segment) >= 3:  # Ensure minimum segment size for meaningful analysis
            segments.append(segment)
            # Create segment identifier (e.g., "S1_1", "S1_2")
            stage_id = segment[0]["id"].split("_")[0] if "_" in segment[0]["id"] else "S?"
            segment_ids.append(f"{stage_id}_{len(segments)}")
    
    # Calculate coherence for each segment
    coherence_results = []
    for i, (segment, segment_id) in enumerate(zip(segments, segment_ids)):
        coherence_score = calculate_segment_topic_coherence(
            segment, 
            top_k=config["top_k_keywords"],
            window_size=config["window_size"],
            stopwords=config["custom_stopwords"],
            domain_vocabulary=config["domain_vocabulary"]
        )
        
        # Store results with metadata
        coherence_results.append({
            "segment_id": segment_id,
            "segment_index": i,
            "utterance_range": (segment[0]["id"], segment[-1]["id"]),
            "coherence_score": coherence_score,
            "speakers": list(set(u["speaker"] for u in segment)),
            "topics": list(set(topic for u in segment if "topics" in u for topic in u["topics"]))
        })
    
    # Calculate aggregate metrics
    stage_scores = {}
    for result in coherence_results:
        stage_id = result["segment_id"].split("_")[0]
        if stage_id not in stage_scores:
            stage_scores[stage_id] = []
        stage_scores[stage_id].append(result["coherence_score"])
    
    stage_averages = {
        stage: sum(scores) / len(scores) if scores else 0
        for stage, scores in stage_scores.items()
    }
    
    overall_average = sum(r["coherence_score"] for r in coherence_results) / len(coherence_results) if coherence_results else 0
    
    # Return complete results
    return {
        "overall_coherence": overall_average,
        "stage_coherence": stage_averages,
        "segment_results": coherence_results,
        "config_used": config
    }

def calculate_segment_topic_coherence(segment, top_k=20, window_size=20, 
                                    stopwords=None, domain_vocabulary=None):
    """
    Calculate topic coherence for a segment of workshop transcript.
    
    Args:
        segment: List of utterance objects in the segment
        top_k: Number of top keywords to extract
        window_size: Context window size for co-occurrence
        stopwords: List of words to exclude
        domain_vocabulary: List of domain terms to prioritize
        
    Returns:
        Average NPMI score for the segment
    """
    # Combine all utterances into a single text
    full_text = " ".join([utterance["content"] for utterance in segment])
    
    # Extract top keywords using TF-IDF
    keywords = extract_top_keywords(
        full_text, 
        k=top_k, 
        stopwords=stopwords,
        domain_vocabulary=domain_vocabulary
    )
    
    # Calculate NPMI for all keyword pairs
    npmi_scores = []
    for i in range(len(keywords)):
        for j in range(i+1, len(keywords)):
            word_i = keywords[i]
            word_j = keywords[j]
            npmi = calculate_npmi(word_i, word_j, full_text, window_size=window_size)
            npmi_scores.append(npmi)
    
    # Return average NPMI score
    return sum(npmi_scores) / len(npmi_scores) if npmi_scores else 0.0

def extract_top_keywords(text, k=20, stopwords=None, domain_vocabulary=None):
    """
    Extract top keywords from text using TF-IDF and domain knowledge.
    
    Args:
        text: Text to analyze
        k: Number of keywords to extract
        stopwords: List of words to exclude
        domain_vocabulary: Domain-specific terms to prioritize
        
    Returns:
        List of top keywords
    """
    # Implementation would use a library like scikit-learn or custom TF-IDF
    # For brevity, this is a simplified placeholder
    
    # In a real implementation, you would:
    # 1. Tokenize and preprocess the text
    # 2. Remove stopwords (both standard and custom)
    # 3. Calculate TF-IDF scores
    # 4. Boost scores for domain vocabulary terms
    # 5. Return the top k terms
    
    # This is where you could also use an LLM to extract key terms:
    # prompt = f"Extract the {k} most important climate adaptation terms from this text: {text}"
    # keywords = llm_client.generate(prompt)
    
    # For demo purposes, return a placeholder
    return ["flooding", "sea_level", "adaptation", "vulnerability", 
            "community", "infrastructure", "planning", "coastal",
            "resilience", "property", "risk", "policy"]

def calculate_npmi(word_i, word_j, text, window_size=20):
    """
    Calculate NPMI for a pair of words.
    
    Args:
        word_i: First word
        word_j: Second word
        text: Text corpus to analyze
        window_size: Context window size for co-occurrence
        
    Returns:
        NPMI score for the word pair
    """
    # Tokenize text
    tokens = text.split()
    total_windows = max(1, len(tokens) - window_size + 1)
    
    # Count occurrences
    count_i = sum(1 for idx in range(len(tokens)) if tokens[idx] == word_i)
    count_j = sum(1 for idx in range(len(tokens)) if tokens[idx] == word_j)
    
    # Count co-occurrences within window
    co_occur = 0
    for idx in range(total_windows):
        window = tokens[idx:idx+window_size]
        if word_i in window and word_j in window:
            co_occur += 1
    
    # Calculate probabilities
    p_i = count_i / len(tokens)
    p_j = count_j / len(tokens)
    p_ij = co_occur / total_windows
    
    # Avoid division by zero
    if p_ij == 0 or p_i == 0 or p_j == 0:
        return 0.0
    
    # Calculate NPMI
    pmi = math.log(p_ij / (p_i * p_j))
    npmi = pmi / (-math.log(p_ij))
    
    return npmi
```

## Interpretation Guidelines

Topic Coherence Scores can be interpreted according to these general guidelines:

| Score Range | Interpretation | Example Scenario |
|-------------|----------------|------------------|
| 0.5 - 1.0   | High coherence | Expert-facilitated discussion with strong thematic focus |
| 0.3 - 0.5   | Moderate coherence | Typical productive workshop discussion with occasional topic shifts |
| 0.1 - 0.3   | Low coherence | Fragmented discussion with weak semantic relationships |
| < 0.1       | Very low coherence | Disjointed, potentially hallucinated or off-topic content |

When evaluating climate adaptation discussions specifically, domain experts have established the following benchmarks based on analysis of human focus groups:

- **Impact Exploration Stage**: Expect scores of 0.35-0.55 due to broader exploration of diverse impacts
- **Adaptation Brainstorming Stage**: Expect scores of 0.40-0.60 as solutions cluster around specific approaches
- **Strategy Evaluation Stage**: Expect scores of 0.45-0.65 as criteria-based evaluation creates stronger semantic links

## Visualizations

Topic coherence can be visualized in several ways:

1. **Time-series graphs** showing coherence scores as the workshop progresses
2. **Heat maps** displaying NPMI relationships between key terms
3. **Network graphs** showing semantic clusters within the discussion

Example visualization:

```
Stage 1 Coherence: 0.37 [████████████████░░░░░░░░] Moderate
Stage 2 Coherence: 0.52 [█████████████████████░░░] High
Stage 3 Coherence: 0.48 [████████████████████░░░░] Moderate-High
Stage 4 Coherence: 0.61 [████████████████████████] High
```

## Integration with Evaluation Framework

Topic Coherence Scoring is one component of the broader Semantic Coherence Analysis dimension. It works alongside:

- **Thematic Evolution Tracking**: Assessing how topics progress and develop over time
- **Argument Structure Assessment**: Evaluating the logical construction of arguments

These metrics collectively provide a comprehensive evaluation of the semantic quality of LLM-simulated climate adaptation discussions, enabling comparisons between different LLM models and with human-facilitated workshops.

## References

1. Li, M. et al. (2024). "Can Large Language Models Replace Human Participants?" Marketing Science Institute Report 25-101.
2. Hewitt, J. et al. (2024). "LLM Social Simulations Are a Promising Research Method." arXiv:2504.02234.
3. Newman, D., et al. (2010). "Automatic Evaluation of Topic Coherence." Human Language Technologies: The 2010 Annual Conference of the North American Chapter of the ACL, 100-108.
4. Röder, M., et al. (2015). "Exploring the Space of Topic Coherence Measures." Proceedings of the Eighth ACM International Conference on Web Search and Data Mining, 399-408.
