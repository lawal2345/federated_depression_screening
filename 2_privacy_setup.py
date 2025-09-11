import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import hashlib
import json

print("Current dataset analysis:")
print(f"Total cases: {len(clinicalbert_features_df)}")
print(f"Depression cases (likely_depression=True): {clinicalbert_features_df['likely_depression'].sum()}")
print(f"Mean depression score: {clinicalbert_features_df['depression_semantic_score'].mean():.3f}")
print(f"Depression criteria count distribution:")
print(clinicalbert_features_df['depression_criteria_count'].value_counts().sort_index())

random.seed(42)

def assign_hospital(row):
    topic = str(row['topic']).lower()
    difficulty = row['difficulty']
    depression_score = row['depression_semantic_score']
    criteria_count = row['depression_criteria_count']

    hospital_a_indicators = [
        'cognitive' in topic,
        'dementia' in topic,
        'geriatric' in topic,
        'medical' in topic and 'comorbid' in topic,
        difficulty >= 4,
        'somatic' in topic,
        criteria_count >= 3 and depression_score > 0.67
    ]

    hospital_b_indicators = [
        'anxiety' in topic,
        'bipolar' in topic,
        'substance' in topic,
        'stress' in topic,
        'acute' in topic,
        'mania' in topic or 'manic' in topic,
        depression_score > 0.68 and criteria_count <= 2,
        difficulty <= 3
    ]

    hospital_c_indicators = [
        'trauma' in topic,
        'personality' in topic,
        'social' in topic,
        'borderline' in topic,
        'ptsd' in topic,
        'dissociat' in topic,
        difficulty == 3 or difficulty == 4,
        0.63 <= depression_score <= 0.67
    ]

    scores = {
        'A': sum(hospital_a_indicators),
        'B': sum(hospital_b_indicators),
        'C': sum(hospital_c_indicators)
    }

    if max(scores.values()) == 0:
        if depression_score > 0.68:
            return random.choice(['A', 'B'])
        elif depression_score < 0.62:
            return 'C'
        else:
            return random.choice(['A', 'B', 'C'])
    
    return max(scores, key=scores.get)

df_merged['hospital'] = df_merged.apply(assign_hospital, axis=1)

print("Hospital assignment complete!")
print("\nHospital distribution:")
hospital_counts = df_merged['hospital'].value_counts()
print(hospital_counts)
print(f"\nPercentages:")
for hospital in ['A', 'B', 'C']:
    pct = (hospital_counts[hospital] / len(df_merged)) * 100
    print(f"Hospital {hospital}: {pct:.1f}%")

print("\n" + "="*60)
print("HOSPITAL CHARACTERISTICS ANALYSIS")
print("="*60)

for hospital in ['A', 'B', 'C']:
    hospital_data = df_merged[df_merged['hospital'] == hospital]
    
    print(f"\nHOSPITAL {hospital} ({len(hospital_data)} cases)")
    print("-" * 40)
    
    print(f"Depression prevalence: {hospital_data['likely_depression'].mean():.3f}")
    print(f"Mean depression score: {hospital_data['depression_semantic_score'].mean():.3f}")
    print(f"Mean criteria count: {hospital_data['depression_criteria_count'].mean():.2f}")
    
    print(f"Difficulty distribution:")
    diff_dist = hospital_data['difficulty'].value_counts().sort_index()
    for diff, count in diff_dist.items():
        pct = (count / len(hospital_data)) * 100
        print(f"  Level {diff}: {count} ({pct:.1f}%)")
    
    print(f"Top 5 topics:")
    top_topics = hospital_data['topic'].value_counts().head(5)
    for topic, count in top_topics.items():
        pct = (count / len(hospital_data)) * 100
        print(f"  {topic[:50]}...: {count} ({pct:.1f}%)")

hospital_datasets = {}

for hospital in ['A', 'B', 'C']:
    hospital_data = df_merged[df_merged['hospital'] == hospital].copy()
    hospital_data = hospital_data.reset_index(drop=True)
    hospital_datasets[hospital] = hospital_data
    print(f"Hospital {hospital} dataset created: {len(hospital_data)} cases")

print(f"\nHospital datasets created successfully!")
print("Available datasets:", list(hospital_datasets.keys()))

print("Setting up Privacy Framework for Federated Depression Screening")
print("=" * 70)
print("NOTE: Simplified implementation of privacy network")
print("Not suitable for production without full cryptographic security")

@dataclass
class PrivacyConfig:
    hospital_id: str
    dataset_size: int
    epsilon_per_round: float
    total_rounds: int
    delta: float
    max_grad_norm: float

    def __post_init__(self):
        self.sensitivity = self.max_grad_norm
        self.sigma = self.sensitivity * np.sqrt(2 * np.log(1.25/self.delta)) / self.epsilon_per_round

print("Configuring differential privacy parameters for each hospital...")

privacy_configs = {}

privacy_configs['A'] = PrivacyConfig(
    hospital_id='A',
    dataset_size=1046,
    epsilon_per_round=0.2,
    total_rounds=1,
    delta=1e-6,
    max_grad_norm=1.0
)

privacy_configs['B'] = PrivacyConfig(
    hospital_id='B',
    dataset_size=126,
    epsilon_per_round=0.3,
    total_rounds=20,
    delta=1e-3,
    max_grad_norm=1.0
)

privacy_configs['C'] = PrivacyConfig(
    hospital_id='C',
    dataset_size=1834,
    epsilon_per_round=0.25,
    total_rounds=16,
    delta=1e-6,
    max_grad_norm=1.0
)

for hospital_id, config in privacy_configs.items():
    print(f"\nHospital {hospital_id} Privacy Configuration:")
    print(f"  Dataset size: {config.dataset_size}")
    print(f"  Epsilon per round: {config.epsilon_per_round}")
    print(f"  Total rounds allowed: {config.total_rounds}")
    print(f"  Total epsilon budget: {config.epsilon_per_round * config.total_rounds}")
    print(f"  Delta (δ): {config.delta}")
    print(f"  Max gradient norm: {config.max_grad_norm}")
    print(f"  Calculated noise σ: {config.sigma:.4f}")

class FederatedSecurityProtocol:
    def __init__(self):
        self.allowed_sharing = {
            'model_adapter_weights': True,
            'gradient_updates': True,
            'loss_metrics': True,
            'patient_data': False,
            'raw_embeddings': False,
            'individual_predictions': False,
            'case_ids': False,
            'hospital_identifiers': True
        }

        self.required_protections = {
            'adapter_weights': ['differential_privacy', 'secure_aggregation'],
            'gradients': ['gradient_clipping', 'differential_privacy'],
            'metrics': ['aggregation_threshold_min_3']
        }
    
    def validate_sharing(self, data_type: str) -> bool:
        return self.allowed_sharing(data_type, False)
    
    def get_protections(self, data_type: str) -> List[str]:
        return self.required_protections.get(data_type, [])

class DifferentialPrivacyManager:
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.rounds_used = 0
    
    def add_noise_to_gradients(self, gradients: np.ndarray) -> np.ndarray:
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > self.config.max_grad_norm:
            gradients = gradients * (self.config.max_grad_norm / grad_norm)

        noise = np.random.normal(
            loc=0.0,
            scale=self.config.sigma,
            size=gradients.shape
        )

        noisy_gradients = gradients + noise
        return noisy_gradients
    
    def update_privacy_accounting(self):
        self.rounds_used += 1

        if self.rounds_used > self.config.total_rounds:
            raise ValueError(f"Privacy budget exceeded! Used: {self.rounds_used}, Max: {self.config.total_rounds}")
    
    def get_remaining_rounds(self) -> int:
        return max(0, self.config.total_rounds - self.rounds_used)
    
    def can_participate(self) -> int:
        return self.rounds_used < self.config.total_rounds
    
    def get_privacy_spent(self) -> float:
        return self.rounds_used * self.config.epsilon_per_round

class SecureAggregator:
    def __init__(self, min_participants=2):
        self.min_participants = min_participants
        self.aggregation_history = []
        print(f"Using trusted aggregator, and not cryptographic aggregator")

    def aggregate_weights(self, hospital_updates: Dict[str, np.ndarray]) -> np.ndarray:
        if len(hospital_updates) < self.min_participants:
            raise ValueError(f"Need at least {self.min_participants} participants for aggregation")

        total_samples = sum(privacy_configs[h].dataset_size for h in hospital_updates.keys())

        aggregated_weights = None
        for hospital_id, weights in hospital_updates.items():
            weight_factor = privacy_configs[hospital_id].dataset_size / total_samples

            if aggregated_weights is None:
                aggregated_weights = weights * weight_factor
            else:
                aggregated_weights += weights * weight_factor
        
        self.aggregation_history.append({
            'round': len(self.aggregation_history) + 1,
            'participant_count': len(hospital_updates),
            'total_samples': total_samples,
            'timestamp': np.datetime64('now')
        })
        return aggregated_weights
    
    def get_aggregation_stats(self) -> Dict:
        if not self.aggregation_history:
            return {'rounds': 0, 'avg_participants': 0}
        
        return {
            'rounds': len(self.aggregation_history),
            'avg_participants': np.mean([r['participant_count'] for r in self.aggregation_history]),
            'total_samples_last_round': self.aggregation_history[-1]['total_samples']
        }

class FederatedLearningCoordinator:
    def __init__(self, privacy_configs: Dict, security_protocol: FederatedSecurityProtocol, secure_aggregator: SecureAggregator):
        self.privacy_configs = privacy_configs
        self.dp_managers = {h: DifferentialPrivacyManager(config)
                            for h, config in privacy_configs.items()}
        self.security_protocol = security_protocol
        self.secure_aggregator = secure_aggregator
        self.global_round = 0

    def plan_communication_round(self) -> Dict:
        participating_hospitals = []
        for hospital_id, manager in self.dp_managers.items():
            if manager.can_participate():
                participating_hospitals.append(hospital_id)
        
        round_plan = {
            'round_number': self.global_round + 1,
            'participating_hospitals': participating_hospitals,
            'total_participants': len(participating_hospitals),
            'can_proceed': len(participating_hospitals) >= self.secure_aggregator.min_participants
        }
        return round_plan
    
    def simulate_communication_round(self, round_plan: Dict):
        if not round_plan['can_proceed']:
            print(f"Round {round_plan['round_number']}: Insufficient participants")
            return False
        
        print(f"\nCommunication Round {round_plan['round_number']}")
        print(f" Participants: {round_plan['participating_hospitals']}")

        hospital_updates = {}

        for hospital_id in round_plan['participating_hospitals']:
            simulated_weights = np.random.normal(0, 0.1, size=100)

            noisy_weights = self.dp_managers[hospital_id].add_noise_to_weights(simulated_weights)
            hospital_updates[hospital_id] = noisy_weights

            self.dp_managers[hospital_id].update_privacy_accounting()

            remaining = self.dp_managers[hospital_id].get_remaining_rounds()
            spent = self.dp_managers[hospital_id].get_privacy_spent()
            print(f" Hospital {hospital_id}: ε spent = {spent:.2f}, rounds remaining = {remaining}")
        
        try:
            global_weights = self.secure_aggregator.aggregate_weights(hospital_updates)
            print(f" Global model updated successfully")
            self.global_round += 1
            return True
        except Exception as e:
            print(f" Aggregation failed: {e}")
            return False
        
    def get_privacy_summary(self) -> Dict:
        summary = {}
        for hospital_id, manager in self.dp_managers.items():
            config = manager.config
            summary['hospital_id'] = {
                'total_epsilon_budget': config.epsilon_per_round * config.total_rounds,
                'epsilon_spent': manager.get_privacy_spent(),
                'rounds_used': manager.rounds_used,
                'rounds_remaining': manager.get_remaining_rounds(),
                'can_participate': manager.can_participate()
            }
        return summary

security_protocol = FederatedSecurityProtocol()
secure_aggregator = SecureAggregator(min_participants=2)
fed_coordinator = FederatedLearningCoordinator(
    privacy_configs=privacy_configs,
    security_protocol=security_protocol,
    secure_aggregator=secure_aggregator
)

print("\nData Sharing Security Protocol:")
print("-" * 40)
for data_type, allowed in security_protocol.allowed_sharing.items():
    status = "✓ ALLOWED" if allowed else "✗ FORBIDDEN"
    print(f"{data_type:25}: {status}")

print("\nFederated Learning Coordinator initialized")
print("Educational privacy framework ready for Hours 2-3!")
print("\nLimitations acknowledged:")
print("- Simplified DP accounting (not RDP/moments accountant)")
print("- Trusted aggregator (not cryptographically secure)")
print("- Educational demonstration only")