# Climate Adaptation Workshop

## Overview

This project simulates climate adaptation workshops using language models (LLMs). It provides a complete solution for simulating stakeholder engagement in climate adaptation planning, with particular emphasis on representing vulnerable communities.

## Key Features

- **Persona Generation**: Creation of 4 to 20 personas representing different community profiles
- **Dialogue Simulation**: Facilitation of realistic interactions between personas
- **Comprehensive Evaluation**: Analysis of discussion quality and results
- **Automatic Saving**: Storage of results in the `workshop_autpout` folder

## Project Structure

```
climate-adaptation-workshop/
├── src/
│   ├── workshop_initialization/   # Workshop initialization
│   ├── persona_development/       # Persona generation
│   ├── dialogue_simulation/       # Dialogue simulation
│   └── evaluation/               # Results evaluation
├── workshop_autpout/            # Output directory
│   ├── personas/                # Generated personas
│   │   ├── index.json          # Personas index with statistics
│   │   └── *.json              # Individual persona files
│   ├── dialogue/                # Dialogue history
│   └── evaluation/              # Evaluation reports
└── requirements.txt             # Dependencies
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/climate-adaptation-workshop.git
cd climate-adaptation-workshop

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your Azure OpenAI API keys
```

## Usage

### Basic Example

```python
# Launch the workshop
python src/run_workshop.py
```

The script will:
1. Ask for the number of personas to generate (4-20)
2. Generate personas with varied profiles
3. Launch dialogue between personas
4. Evaluate results
5. Save everything in `workshop_autpout`

### Persona Structure

Each persona is generated with the following information:
```json
{
    "name": "string",
    "age": number,
    "profession": "string",
    "family_situation": "string",
    "income": "string",
    "housing": "string",
    "vulnerabilities": ["string"],
    "current_impacts": ["string"],
    "projected_impacts": ["string"],
    "adaptation_capacity": "string",
    "obstacles": ["string"],
    "adaptation_options": ["string"],
    "quotes": ["string"],
    "metadata": {
        "creation_date": "ISO datetime",
        "demographic": "string",
        "location": "string",
        "version": "1.0"
    }
}
```

### Index File

The `workshop_autpout/personas/index.json` file contains:
```json
{
    "creation_date": "ISO datetime",
    "persona_count": number,
    "statistics": {
        "by_demographic": {
            "demographic1": count,
            "demographic2": count
        },
        "by_location": {
            "location1": count,
            "location2": count
        },
        "by_age": {
            "young": count,
            "adult": count,
            "senior": count
        }
    },
    "personas": [/* List of personas */]
}
```

### Available Profiles

Personas are generated from 20 profiles distributed across 5 zones:

1. **Coastal Area**
   - Elderly person
   - Fisherman
   - Coastal tourism worker
   - Coastal property owner

2. **Flood Zone**
   - Low-income family
   - Single parent
   - Renting family
   - Flood zone resident

3. **Urban Area**
   - Young adult
   - Student
   - Urban professional
   - Urban service worker

4. **Downtown**
   - Business owner
   - Restaurant owner
   - Shop owner
   - Office worker

5. **Rural Area**
   - Farmer
   - Rural resident
   - Agricultural worker
   - Rural business owner

## Main Modules

### 1. Workshop Initialization (`workshop_initialization/`)

The initialization module manages workshop configuration and defines its structure.

#### Workshop Configuration
```python
workshop_input = {
    "context": "Workshop context description",
    "objectives": "List of objectives",
    "outputs": "Expected results"
}
```

#### Workshop Structure
The workshop is divided into 4 stages:

1. **Introduction (S1)**
   - Establish persona identities
   - Present initial perspectives
   - Duration: 5 points

2. **Impact Exploration (S2)**
   - Identify climate risks
   - Analyze differential impacts
   - Duration: 8 points

3. **Adaptation Brainstorming (S3)**
   - Generate adaptation options
   - Identify facilitators/obstacles
   - Duration: 12 points

4. **Strategy Evaluation (S4)**
   - Evaluate options against multiple criteria
   - Duration: 8 points

#### Configuration Files
- `workshop_config.json`: Complete configuration
- `workshop_config.yaml`: Readable configuration version

### 2. Dialogue Management (`dialogue_manager.py`)

The dialogue manager orchestrates interactions between personas.

#### Features
- Realistic dialogue generation
- Workshop stage management
- Persona perspective integration
- Dialogue history saving

#### Dialogue Format
```
[Persona Name]: Message
```

#### Dialogue Example
```
[Marie Dubois]: As an elderly resident, I'm particularly concerned about flooding...
[Jean Martin]: As a fisherman, I'm already seeing the impact on our fishing areas...
```

#### Dialogue History
Dialogues are saved in `workshop_autpout/dialogue/` with:
- Workshop stage
- Involved participants
- Exchange content
- Topics covered

### 3. Evaluation Module (`evaluation/`)

The evaluation module analyzes workshop quality and relevance across several dimensions.

#### 1. Main Evaluator (`workshop_evaluator.py`)
```python
class WorkshopEvaluator:
    """Evaluates overall workshop quality."""
    
    def evaluate_dialogue(self, dialogue_history: List[Dict]) -> Dict:
        """Evaluates dialogue history."""
        
    def generate_report(self, evaluation: Dict) -> None:
        """Generates complete evaluation report."""
```

#### 2. Topic Coherence (`topic_coherence.py`)
```python
class TopicCoherenceAnalyzer:
    """Analyzes topic coherence."""
    
    def analyze_coherence(self, dialogue: str) -> float:
        """Calculates coherence score (0-1)."""
        
    def identify_key_topics(self, dialogue: str) -> List[str]:
        """Identifies main topics."""
```

#### 3. Thematic Evolution (`thematic_evolution.py`)
```python
class ThematicEvolutionTracker:
    """Tracks theme evolution in dialogue."""
    
    def track_evolution(self, dialogue_history: List[Dict]) -> Dict:
        """Analyzes theme evolution."""
        
    def identify_theme_shifts(self) -> List[Dict]:
        """Identifies theme changes."""
```

#### 4. Argument Structure (`argument_structure.py`)
```python
class ArgumentStructureAnalyzer:
    """Analyzes argument structure."""
    
    def analyze_arguments(self, dialogue: str) -> Dict:
        """Evaluates argument quality."""
        
    def identify_argument_patterns(self) -> List[str]:
        """Identifies argument patterns."""
```

#### 5. Persona Fidelity (`persona_fidelity.py`)
```python
class PersonaFidelityEvaluator:
    """Evaluates persona consistency."""
    
    def evaluate_fidelity(self, persona: Dict, dialogue: str) -> float:
        """Calculates fidelity score (0-1)."""
        
    def check_consistency(self) -> Dict:
        """Checks behavior consistency."""
```

## Evaluation Metrics

### 1. Semantic Coherence
- **Coherence Score**: 0-1
  - Topic coherence
  - Logical progression
  - Discussion relevance

### 2. Persona Fidelity
- **Fidelity Score**: 0-1
  - Perspective consistency
  - Character maintenance
  - Reaction authenticity

### 3. Argument Structure
- **Structure Score**: 0-1
  - Argument clarity
  - Position support
  - Logical progression

### 4. Thematic Evolution
- **Evolution Score**: 0-1
  - Theme progression
  - Perspective integration
  - Idea development

## Evaluation Report Format

```json
{
    "evaluation_summary": {
        "overall_score": number,
        "coherence_score": number,
        "fidelity_score": number,
        "structure_score": number,
        "evolution_score": number
    },
    "detailed_analysis": {
        "topic_coherence": {
            "score": number,
            "key_topics": ["string"],
            "coherence_patterns": ["string"]
        },
        "persona_fidelity": {
            "score": number,
            "consistency_issues": ["string"],
            "strengths": ["string"]
        },
        "argument_structure": {
            "score": number,
            "patterns": ["string"],
            "improvements": ["string"]
        },
        "thematic_evolution": {
            "score": number,
            "theme_progression": ["string"],
            "integration_points": ["string"]
        }
    },
    "recommendations": ["string"],
    "metadata": {
        "evaluation_date": "ISO datetime",
        "workshop_id": "string",
        "version": "string"
    }
}
```

## Using the Evaluation Module

```python
from evaluation.workshop_evaluator import WorkshopEvaluator

# Create evaluator
evaluator = WorkshopEvaluator()

# Evaluate workshop
evaluation = evaluator.evaluate_dialogue(dialogue_history)

# Generate report
evaluator.generate_report(evaluation)
```

## Dependencies

- Python 3.8 or higher
- Azure OpenAI API
- Required environment variables:
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_API_VERSION`
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_DEPLOYMENT_NAME`

## Ethical Considerations

This project follows these ethical principles:
- Transparency in persona creation
- Equitable representation of vulnerable groups
- Privacy protection
- Complementarity with real community engagement

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
