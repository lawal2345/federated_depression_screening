# Methodology: Federated Learning for Depression Screening

## Overview

This project builds a privacy-preserving depression screening system that allows multiple hospitals to collaborate without sharing sensitive patient data. The system uses federated learning with differential privacy to train a machine learning model that can identify depression symptoms from clinical text.

The primary goal of this project was to demonstrate differential privacy implementation and prove the absence of data leakage in a federated learning system for healthcare applications. While model performance was secondary, the focus was on showing that sensitive patient data could not be reconstructed from the shared model updates.

## Phase 1: Data Analysis and Feature Extraction

### What Was Done
In this implementation, the Hugging Face dataset moremilk/CoT_Reasoning_Clinical_Diagnosis_Mental_Health is used as a proxy dataset to simulate clinical reasoning for mental health diagnoses, including depression, rather than real hospital or healthcare institution records which are private and confidential.

### Technical Implementation
Bio_ClinicalBERT, a specialized version of BERT trained on medical texts, was used to understand the semantic meaning of clinical cases. This model converts text into high-dimensional vectors that capture medical concepts better than general-purpose language models.

11 depression criteria templates were created based on PHQ-9 and DSM-5 standards:
- Low mood and sadness
- Loss of interest (anhedonia)  
- Sleep disturbances
- Fatigue and energy loss
- Appetite and weight changes
- Guilt and worthlessness feelings
- Concentration problems
- Psychomotor changes
- Suicidal thoughts
- Functional impairment
- Duration of symptoms

For each clinical case, semantic similarity scores were calculated between the case text and each depression criterion. This process provides numerical measures of how strongly each depression symptom appears in the clinical presentation.

### Results
The feature extraction process created a dataset where each case has:
- 11 similarity scores (one per depression criterion)
- Binary indicators for strong symptom presence (similarity > 0.7)
- An overall depression semantic score (average of all similarities)
- A count of how many criteria are strongly present

## Phase 2: Hospital Partitioning and Privacy Framework

### What Was Done
The dataset was partitioned to simulate three different hospital environments based on clinical characteristics, then a privacy framework was built to protect patient data during federated learning.

### Hospital Specialization Strategy
**Hospital A (Geriatric Focus)**: Cases with higher difficulty levels, somatic symptoms, and complex medical presentations. This represents older patients with more complicated depression presentations.

**Hospital B (Young Adult Focus)**: Cases involving bipolar disorder, anxiety, substance use, and acute presentations. This represents younger patients with different symptom patterns.

**Hospital C (Mixed Urban Population)**: Cases involving trauma, personality disorders, and social factors. This represents a diverse urban hospital serving various demographics.

### Why This Partitioning Matters
Real hospitals serve different patient populations with varying depression presentations. The partitioning creates realistic data distributions that test whether the federated model can learn from diverse sources and generalize across different patient types.

### Privacy Framework Design
Differential privacy was implemented to ensure individual patient information cannot be extracted from model updates. Each hospital has different privacy budgets based on their dataset size and sensitivity requirements:

**Hospital A**: Very conservative privacy (ε=0.2 per round, 15 total rounds)
**Hospital B**: Moderate privacy (ε=0.3 per round, 20 total rounds) 
**Hospital C**: Balanced privacy (ε=0.25 per round, 16 total rounds)

The privacy mechanism adds calibrated noise to model updates before sharing, ensuring that no single patient's data can be reverse-engineered from the shared information.

### Data Sharing Rules
Strict protocols were established for what can and cannot be shared:
- **Allowed**: Noisy model weights, aggregated metrics, hospital identifiers for routing
- **Forbidden**: Raw patient data, individual predictions, case IDs, embeddings

## Phase 3: Federated Model Architecture and Training

### Model Design Philosophy
A two-head architecture was built that both classifies depression severity and explains its reasoning. This addresses the critical need for interpretable AI in healthcare settings.

### Architecture Components

**Backbone Network**: Bio_ClinicalBERT with LoRA (Low-Rank Adaptation) adapters. The pre-trained weights are frozen and only small adapter layers are trained. This approach:
- Reduces computational requirements
- Prevents catastrophic forgetting of medical knowledge
- Makes federated learning more efficient by sharing only adapter weights

**Head A - Depression Screening**: A neural network that classifies depression into four severity levels:
- No Depression (score < 0.6)
- Mild Depression (score 0.6-0.65) 
- Moderate Depression (score 0.65-0.7)
- Severe Depression (score > 0.7)

**Head B - Rationale Generation**: Uses Flan-T5-Small with LoRA adapters to generate clinical explanations. This head identifies which depression criteria are present and creates human-readable rationales for the classification decision.

**Uncertainty Quantification**: Monte Carlo dropout samples multiple predictions to estimate model confidence. High uncertainty cases are flagged for human review.

**Out-of-Distribution Detection**: Identifies unusual cases that differ significantly from training data, indicating potential need for specialist consultation.

### Federated Learning Process

**Round 1: Initialization**
- Each hospital starts with the same pre-trained model
- Local datasets are prepared with proper train/test splits
- Privacy budgets are allocated

**Local Training Phase**
Each hospital trains their model for 2 epochs on local data:
- Uses AdamW optimizer with learning rate 1e-4
- Applies gradient clipping (max norm = 1.0) for privacy protection
- Combines classification loss with rationale generation loss

**Privacy-Preserving Aggregation**
- Extract LoRA adapter weights from each hospital's model
- Add differential privacy noise scaled to each hospital's privacy budget
- Share noisy weights with central aggregator
- Weighted averaging based on dataset sizes
- Distribute updated global weights back to hospitals

**Evaluation and Monitoring**
- Test on hospital-specific test sets
- Monitor uncertainty levels and out-of-distribution cases
- Track privacy budget consumption

### Training Optimization
Several techniques were used to make training efficient:
- Mixed precision training when available
- Memory management for GPU constraints  
- Batch processing to handle varying case lengths
- Temperature scaling for probability calibration

### Validation Strategy
The system evaluates performance on multiple metrics:
- **Accuracy**: Correct depression severity classification
- **AUC**: Area under ROC curve for each severity class
- **Uncertainty Calibration**: Alignment between confidence and accuracy
- **Privacy Budget Tracking**: Ensuring differential privacy guarantees are maintained

## Technical Considerations

### Hardware Requirements
The system was trained on the GPU of a Mac M1 Pro

### Limitations and Future Work
This implementation uses simplified privacy accounting suitable for educational purposes. Production systems would require:
- Renyi Differential Privacy accounting
- Cryptographically secure aggregation
- Formal security audits
- Regulatory compliance validation

## Clinical Relevance

This proof of concept addresses some healthcare challenges:
- **Data Silos**: Hospitals can collaborate without data sharing agreements
- **Privacy Compliance**: Meets requirements for patient data protection
- **Scalability**: Can incorporate new hospitals without retraining from scratch

The methodology demonstrates how advanced machine learning can be deployed in healthcare while maintaining strict privacy protections