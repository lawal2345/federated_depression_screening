import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import roc_auc_score, accuracy_score
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ModelConfig:
    backbone_model = "emilyalsentzer/Bio_ClinicalBERT"
    max_length: int = 512
    hidden_size: int = 768

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    depression_classes: int = 4
    rationale_max_lenght: int = 200

    mc_dropout_samples: int = 10
    dropout_rate: float = 0.1

    temperature_scaling: bool = True
    ood_threshold: float = 0.8

    t5_max_input_length: int = 128
    t5_max_output_length: int = 100
    t5_generation_temperature: float = 0.7
    t5_lora_r: int = 8

@dataclass
class ClinicalKnowledgeBase:
    def __init__(self):
        self.depression_criteria = {
            'low_mood': {
                'description': "depressed mood most of the day",
                'examples': ["feeling sad", "empty mood", "tearfulness", "hopelessness"]
            },
            'anhedonia': {
                'description': "loss of interest or pleasure in activities",
                'examples': ["no longer enjoys hobbies", "loss of motivation", "decreased pleasure"]
            },
            'sleep_problems': {
                'description': "sleep disturbances",
                'examples': ["insomnia", "early awakening", "hypersomnia", "restless sleep"]
            },
            'fatigue': {
                'description': "fatigue or loss of energy",
                'examples': ["tired all the time", "no energy", "exhaustion", "weakness"]
            },
            'appetite_changes': {
                'description': "appetite or weight changes",
                'examples': ["loss of appetite", "weight loss", "overeating", "weight gain"]
            },
            'guilt_worthlessness': {
                'description': "feelings of worthlessness or guilt",
                'examples': ["feeling worthless", "excessive guilt", "self-blame", "inadequacy"]
            },
            'concentration_problems': {
                'description': "difficulty concentrating or making decisions",
                'examples': ["trouble focusing", "indecisiveness", "memory problems", "distractibility"]
            },
            'psychomotor_changes': {
                'description': "psychomotor agitation or retardation",
                'examples': ["restlessness", "pacing", "slowed movements", "sluggishness"]
            },
            'suicidal_thoughts': {
                'description': "thoughts of death or suicide",
                'examples': ["death wishes", "suicidal ideation", "wanting to die", "suicide plans"]
            }
        }
        
        self.severity_criteria = {
            0: {"label": "No Depression", "threshold": 0.2, "description": "Minimal or no symptoms"},
            1: {"label": "Mild Depression", "threshold": 0.4, "description": "2-3 symptoms with mild impairment"},
            2: {"label": "Moderate Depression", "threshold": 0.7, "description": "4-6 symptoms with moderate impairment"},
            3: {"label": "Severe Depression", "threshold": 1.0, "description": "7+ symptoms with severe impairment"}
        }

class DepressionScreeningBackbone(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config.backbone_model)
        self.backbone = AutoModel.from_pretrained(config.backbone_model)

        for param in self.backbone.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["query", "value", "key", "dense"]
        )

        self.backbone = get_peft_model(self.backbone, lora_config)
        print(f"✓ Added LoRA adapters - trainable parameters: {self.backbone.print_trainable_parameters()}")

    def forward(self, input_ids, attention_mask, structured_features=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        avg_embedding = outputs.last_hidden_state.mean(dim=1)

        combined_embedding = torch.cat([cls_embedding, avg_embedding], dim=-1)
        
        if structured_features is not None:
            combined_embedding = torch.cat([combined_embedding, structured_features], dim=-1)
        
        return combined_embedding

class DepressionScreeningHead(nn.Module):
    def __init__(self, input_size: int, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, config.depression_classes)
        )

        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, features, return_uncertainty=False):
        if return_uncertainty and self.training == False:
            self.train()
            predictions = []
            for _ in range(self.config.mc_dropout_samples):
                logits = self.classifier(features)
                predictions.append(torch.softmax(logits / self.temperature, dim=-1))
            self.eval()

            predictions = torch.stack(predictions)
            mean_pred = predictions.mean(dim=0)
            uncertainty = predictions.var(dim=0).sum(dim=-1)
            
            return {
                'predictions': mean_pred,
                'uncertainty': uncertainty,
                'raw_logits': self.classifier(features)
            }
        else:
            logits = self.classifier(features)
            if self.config.temperature_scaling:
                probs = torch.softmax(logits / self.temperature, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
            
            return {
                'predictions': probs,
                'raw_logits': logits
            }

class RationaleGenerationHead(nn.Module):
    def __init__(self, input_size: int, config: ModelConfig, knowledge_base: ClinicalKnowledgeBase):
        super().__init__()
        self.config = config
        self.knowledge_base = knowledge_base

        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
        self.t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

        for param in self.t5_model.parameters():
            param.requires_grad = False
        
        from peft import LoraConfig, get_peft_model, TaskType
        t5_lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"]
        )
        self.t5_model = get_peft_model(self.t5_model, t5_lora_config)

        self.feature_projection = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256), 
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 512)
        )

        self.criteria_detector = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, len(knowledge_base.depression_criteria))
        )

        self.severity_prompts = {
            0: "Generate a clinical explanation for why this patient shows minimal depression symptoms based on the detected indicators: ",
            1: "Generate a clinical explanation for why this patient shows mild depression symptoms based on the detected indicators: ",
            2: "Generate a clinical explanation for why this patient shows moderate depression symptoms based on the detected indicators: ",
            3: "Generate a clinical explanation for why this patient shows severe depression symptoms based on the detected indicators: "
        }
        
        print("✓ T5-Small loaded for rationale generation with LoRA adapters")
    
    def forward(self, features, predicted_class, structured_symptom_scores=None):
        batch_size = features.size(0)
        
        criteria_logits = self.criteria_detector(features)
        criteria_probs = torch.sigmoid(criteria_logits)
        
        explanations = []
        
        for i in range(batch_size):
            explanation = self._generate_t5_explanation(
                features[i:i+1],
                predicted_class[i].item() if torch.is_tensor(predicted_class) else predicted_class,
                criteria_probs[i].cpu().numpy()
            )
            explanations.append(explanation)
        
        return {
            'criteria_predictions': criteria_probs,
            'explanations': explanations
        }
    
    def _generate_t5_explanation(self, single_feature, predicted_class, criteria_probs, threshold=0.5):
        detected_criteria = []
        criteria_names = list(self.knowledge_base.depression_criteria.keys())
        
        for i, prob in enumerate(criteria_probs):
            if prob > threshold:
                criterion_name = criteria_names[i]
                detected_criteria.append({
                    'name': criterion_name,
                    'confidence': prob,
                    'description': self.knowledge_base.depression_criteria[criterion_name]['description']
                })
        
        severity_prompt = self.severity_prompts[min(predicted_class, 3)]
        
        if detected_criteria:
            detected_symptoms = ", ".join([c['description'] for c in detected_criteria[:4]])
            input_text = f"{severity_prompt}{detected_symptoms}. Provide clinical rationale:"
        else:
            input_text = f"{severity_prompt}no significant symptoms detected. Provide clinical rationale:"
        
        inputs = self.t5_tokenizer(
            input_text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True
        )
        
        inputs = {k: v.to(next(self.t5_model.parameters()).device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.t5_model.generate(
                **inputs,
                max_length=100,
                min_length=20,
                num_beams=3,
                early_stopping=True,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.t5_tokenizer.pad_token_id,
                eos_token_id=self.t5_tokenizer.eos_token_id
            )
        
        generated_text = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'explanation': generated_text,
            'detected_criteria': detected_criteria,
            'confidence_score': np.mean([c['confidence'] for c in detected_criteria]) if detected_criteria else 0.0,
            'input_prompt': input_text
        }
    
    def get_t5_parameters(self):
        return {name: param for name, param in self.t5_model.named_parameters() if param.requires_grad}

class OODDetector(nn.Module):
    def __init__(self, input_size: int, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.density_estimator = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128), 
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.register_buffer('training_mean', torch.zeros(input_size))
        self.register_buffer('training_std', torch.ones(input_size))
        self.fitted = False
    
    def fit_statistics(self, training_features):
        self.training_mean = training_features.mean(dim=0)
        self.training_std = training_features.std(dim=0) + 1e-8
        self.fitted = True
    
    def forward(self, features):
        density_score = self.density_estimator(features).squeeze(-1)
        
        if self.fitted:
            normalized_features = (features - self.training_mean) / self.training_std
            distance_score = torch.exp(-0.5 * (normalized_features ** 2).sum(dim=-1))
            
            ood_score = 0.7 * density_score + 0.3 * distance_score
        else:
            ood_score = density_score
        
        return ood_score

class FederatedDepressionScreener(nn.Module):
    def __init__(self, config: ModelConfig, structured_feature_size: int = 11):
        super().__init__()
        self.config = config
        self.knowledge_base = ClinicalKnowledgeBase()
        
        self.backbone = DepressionScreeningBackbone(config)
        
        backbone_output_size = 2 * config.hidden_size + structured_feature_size
        
        self.screening_head = DepressionScreeningHead(backbone_output_size, config)
        self.rationale_head = RationaleGenerationHead(backbone_output_size, config, self.knowledge_base)
        
        self.ood_detector = OODDetector(backbone_output_size, config)
        
    def forward(self, input_ids, attention_mask, structured_features, return_uncertainty=False, return_rationale=True):
        features = self.backbone(input_ids, attention_mask, structured_features)
        
        screening_output = self.screening_head(features, return_uncertainty=return_uncertainty)
        
        predicted_class = torch.argmax(screening_output['predictions'], dim=-1)
        
        outputs = {
            'screening': screening_output,
            'predicted_class': predicted_class,
            'features': features
        }
        
        if return_rationale:
            rationale_output = self.rationale_head(features, predicted_class, structured_features)
            outputs['rationale'] = rationale_output
        
        ood_score = self.ood_detector(features)
        outputs['ood_score'] = ood_score
        
        return outputs

class ClinicalInputProcessor:
    def __init__(self, tokenizer, config: ModelConfig):
        self.tokenizer = tokenizer
        self.config = config
        
    def process_batch(self, cases_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        combined_texts = []
        for _, case in cases_df.iterrows():
            text = f"Question: {case['question']} Answer: {case['answer']}"
            if 'reasoning' in case and pd.notna(case['reasoning']):
                text += f" Clinical Reasoning: {case['reasoning']}"
            combined_texts.append(text)
        
        tokenized = self.tokenizer(
            combined_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        structured_features = self._extract_structured_features(cases_df)
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'structured_features': torch.FloatTensor(structured_features)
        }
    
    def _extract_structured_features(self, cases_df: pd.DataFrame) -> np.ndarray:
        feature_columns = [
            'low_mood_similarity', 'anhedonia_similarity', 'sleep_problems_similarity',
            'fatigue_similarity', 'appetite_changes_similarity', 'guilt_worthlessness_similarity',
            'concentration_problems_similarity', 'psychomotor_changes_similarity', 
            'suicidal_thoughts_similarity', 'functional_impairment_similarity',
            'depression_semantic_score'
        ]
        
        features = []
        for _, case in cases_df.iterrows():
            case_features = []
            for col in feature_columns:
                if col in case:
                    case_features.append(case[col])
                else:
                    case_features.append(0.0)
            features.append(case_features)
        
        return np.array(features)

class ModelCalibrator:
    def __init__(self, model):
        self.model = model
        
    def calibrate(self, val_loader, device):
        all_logits = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                outputs = self.model(**inputs, return_uncertainty=False, return_rationale=False)
                
                all_logits.append(outputs['screening']['raw_logits'])
                all_labels.append(batch['labels'])
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        
        optimizer = torch.optim.LBFGS([self.model.screening_head.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(all_logits / self.model.screening_head.temperature, all_labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        print(f"✓ Model calibrated - Temperature: {self.model.screening_head.temperature.item():.3f}")

class FederatedModelManager:
    def __init__(self, model: FederatedDepressionScreener):
        self.model = model

    def get_adapter_weights(self) -> Dict[str, torch.Tensor]:
        adapter_weights = {}
        
        for name, param in self.model.backbone.backbone.named_parameters():
            if param.requires_grad:
                adapter_weights[f"backbone_{name}"] = param.data.clone()
        
        t5_params = self.model.rationale_head.get_t5_parameters()
        for name, param in t5_params.items():
            adapter_weights[f"t5_{name}"] = param.data.clone()
    
        print(f"✓ Extracted {len(adapter_weights)} adapter weight tensors (backbone + T5)")
        return adapter_weights

    def update_adapter_weights(self, new_weights: Dict[str, torch.Tensor]):
        for name, param in self.model.backbone.backbone.named_parameters():
            backbone_key = f"backbone_{name}"
            if backbone_key in new_weights and param.requires_grad:
                param.data.copy_(new_weights[backbone_key])
        
        t5_params = self.model.rationale_head.get_t5_parameters()
        for name, param in t5_params.items():
            t5_key = f"t5_{name}"
            if t5_key in new_weights:
                param.data.copy_(new_weights[t5_key])
        
        print(f"✓ Updated model with {len(new_weights)} adapter weights (backbone + T5)")

    def add_differential_privacy_noise(self, weights: Dict[str, torch.Tensor], noise_scale: float) -> Dict[str, torch.Tensor]:
        noisy_weights = {}
        for name, weight in weights.items():
            noise = torch.normal(0, noise_scale, size=weight.shape)
            noisy_weights[name] = weight + noise
        
        return noisy_weights

def create_depression_screening_model(config: ModelConfig, structured_feature_size: int = 11) -> FederatedDepressionScreener:
    model = FederatedDepressionScreener(config, structured_feature_size)
    print("✓ Created federated depression screening model")
    print(f"  - Backbone: {config.backbone_model}")
    print(f"  - LoRA rank: {config.lora_r}")
    print(f"  - Depression classes: {config.depression_classes}")
    print(f"  - MC dropout samples: {config.mc_dropout_samples}")
    return model

class HospitalDepressionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor: ClinicalInputProcessor, include_labels=True):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.include_labels = include_labels
        
        self.processed_data = self.processor.process_batch(df)
        
        if include_labels and 'depression_semantic_score' in df.columns:
            self.labels = self._create_labels(df)
        else:
            self.labels = None
    
    def _create_labels(self, df):
        labels = []
        for _, row in df.iterrows():
            score = row['depression_semantic_score']
            criteria_count = row.get('depression_criteria_count', 0)
            
            if score < 0.6 or criteria_count == 0:
                label = 0
            elif score < 0.65 or criteria_count <= 2:
                label = 1
            elif score < 0.7 or criteria_count <= 4:
                label = 2
            else:
                label = 3
            
            labels.append(label)
        
        return torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.processed_data['input_ids'][idx],
            'attention_mask': self.processed_data['attention_mask'][idx],
            'structured_features': self.processed_data['structured_features'][idx]
        }
        
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        
        return item

def prepare_hospital_datasets(hospital_datasets: Dict[str, pd.DataFrame], processor: ClinicalInputProcessor, test_size=0.2):
    prepared_datasets = {}
    
    for hospital_id, df in hospital_datasets.items():
        print(f"\nPreparing datasets for Hospital {hospital_id}...")
        
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['likely_depression'])
        
        train_dataset = HospitalDepressionDataset(train_df, processor, include_labels=True)
        test_dataset = HospitalDepressionDataset(test_df, processor, include_labels=True)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        prepared_datasets[hospital_id] = {
            'train_df': train_df,
            'test_df': test_df,
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'train_loader': train_loader,
            'test_loader': test_loader
        }
        
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        
        train_labels = train_dataset.labels.numpy()
        label_counts = np.bincount(train_labels)
        print(f"  Label distribution: {dict(enumerate(label_counts))}")
    
    return prepared_datasets

class HospitalTrainer:
    def __init__(self, model: FederatedDepressionScreener, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.classification_loss = nn.CrossEntropyLoss()
        self.rationale_loss = nn.BCELoss()
        
    def train_epoch(self, train_loader, optimizer, epoch):
        self.model.train()
        total_loss = 0
        total_acc = 0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(**inputs, return_uncertainty=False, return_rationale=True)
            
            class_loss = self.classification_loss(outputs['screening']['raw_logits'], labels)
            
            rationale_loss = 0
            if 'rationale' in outputs:
                criteria_targets = self._create_criteria_targets(labels, len(outputs['rationale']['criteria_predictions'][0]))
                rationale_loss = self.rationale_loss(outputs['rationale']['criteria_predictions'], criteria_targets)
            
            total_batch_loss = class_loss + 0.1 * rationale_loss
            
            optimizer.zero_grad()
            total_batch_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            predictions = torch.argmax(outputs['screening']['predictions'], dim=-1)
            total_acc += (predictions == labels).float().mean().item()
            
            if batch_idx % 10 == 0:
                print(f'    Batch {batch_idx}/{len(train_loader)}, Loss: {total_batch_loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        
        return avg_loss, avg_acc
    
    def _create_criteria_targets(self, labels, num_criteria):
        batch_size = labels.size(0)
        targets = torch.zeros(batch_size, num_criteria, device=self.device)
        
        for i, label in enumerate(labels):
            if label >= 1:
                targets[i, :3] = 0.7
            if label >= 2:
                targets[i, 3:6] = 0.6
            if label >= 3:
                targets[i, 6:] = 0.8
        
        return targets
    
    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        all_ood_scores = []
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(**inputs, return_uncertainty=True, return_rationale=False)
                
                loss = self.classification_loss(outputs['screening']['raw_logits'], labels)
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs['screening']['predictions'], dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_uncertainties.extend(outputs['screening']['uncertainty'].cpu().numpy())
                all_ood_scores.extend(outputs['ood_score'].cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        try:
            auc_scores = []
            for class_idx in range(4):
                class_labels = (np.array(all_labels) == class_idx).astype(int)
                if len(np.unique(class_labels)) > 1:
                    class_probs = [outputs['screening']['predictions'][i][class_idx].item() 
                                 for i in range(len(all_labels))]
                    auc = roc_auc_score(class_labels, class_probs)
                    auc_scores.append(auc)
            avg_auc = np.mean(auc_scores) if auc_scores else 0.0
        except:
            avg_auc = 0.0
        
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': avg_auc,
            'predictions': all_predictions,
            'labels': all_labels,
            'uncertainties': all_uncertainties,
            'ood_scores': all_ood_scores
        }
        
        return results

class FederatedLearningSimulator:
    def __init__(self, hospitals_data: Dict, model_config: ModelConfig, privacy_configs: Dict):
        self.hospitals_data = hospitals_data
        self.model_config = model_config
        self.privacy_configs = privacy_configs
        
        self.global_model = create_depression_screening_model(model_config)

        self.secure_aggregator = SecureAggregator(min_participants=2)
        self.global_round = 0
        self.global_fed_manager = FederatedModelManager(self.global_model)
        
        self.hospital_models = {}
        self.hospital_trainers = {}
        
        for hospital_id in hospitals_data.keys():
            model = create_depression_screening_model(model_config)
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            trainer = HospitalTrainer(model, device=device)
            self.hospital_models[hospital_id] = model
            self.hospital_trainers[hospital_id] = trainer
        
        self.fed_managers = {
            hospital_id: FederatedModelManager(model) 
            for hospital_id, model in self.hospital_models.items()
        }
        
        for hospital_id in hospitals_data.keys():
            setattr(self, f'rounds_used_{hospital_id}', 0)
    
    def simulate_federated_round(self, round_num: int, hospitals_data: Dict):
        print(f"\nCommunication Round {round_num}")
        
        participating_hospitals = []
        for hospital_id in hospitals_data.keys():
            rounds_used = getattr(self, f'rounds_used_{hospital_id}', 0)
            if rounds_used < self.privacy_configs[hospital_id].total_rounds:
                participating_hospitals.append(hospital_id)
        
        if len(participating_hospitals) < 2:
            print("Insufficient participants for aggregation")
            return False
            
        print(f" Participants: {participating_hospitals}")
        
        hospital_updates = {}
        
        for hospital_id in participating_hospitals:
            print(f"\nHospital {hospital_id} local training...")
            
            model = self.hospital_models[hospital_id]
            train_loader = hospitals_data[hospital_id]['train_loader']

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model.to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
            model.train()
            
            for epoch in range(2):
                epoch_loss = 0
                num_batches = 0
                
                for batch_idx, batch in enumerate(train_loader):
                    inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                    labels = batch['labels'].to(device)
                    
                    outputs = model(**inputs, return_uncertainty=False, return_rationale=False)
                    loss = torch.nn.CrossEntropyLoss()(outputs['screening']['raw_logits'], labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    if batch_idx % 10 == 0:
                        print(f"    Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                print(f"  Epoch {epoch+1} completed, Avg Loss: {avg_loss:.4f}")
            
            import gc
            model.to("cpu")
            torch.mps.empty_cache()
            gc.collect()

            fed_manager = self.fed_managers[hospital_id]
            adapter_weights = fed_manager.get_adapter_weights()
            
            privacy_config = self.privacy_configs[hospital_id]
            noise_scale = privacy_config.sigma
            noisy_weights = fed_manager.add_differential_privacy_noise(adapter_weights, noise_scale)
            
            hospital_updates[hospital_id] = noisy_weights
            
            rounds_used = getattr(self, f'rounds_used_{hospital_id}', 0)
            setattr(self, f'rounds_used_{hospital_id}', rounds_used + 1)
            
            remaining = privacy_config.total_rounds - (rounds_used + 1)
            spent = (rounds_used + 1) * privacy_config.epsilon_per_round
            print(f" Hospital {hospital_id}: ε spent = {spent:.2f}, rounds remaining = {remaining}")
        
        if len(hospital_updates) >= 2:
            print(f"\nAggregating updates from {len(hospital_updates)} hospitals...")
            global_weights = self._aggregate_adapter_weights(hospital_updates)
            
            self.global_fed_manager.update_adapter_weights(global_weights)
            for hospital_id in hospital_updates.keys():
                self.fed_managers[hospital_id].update_adapter_weights(global_weights)
            
            print(f" Global model updated successfully")
            self.global_round += 1
            return True
        else:
            print(f" Aggregation failed: insufficient participants")
            return False
        
    def _aggregate_adapter_weights(self, hospital_updates: Dict) -> Dict[str, torch.Tensor]:
        aggregated = {}
            
        total_samples = sum(self.privacy_configs[h].dataset_size for h in hospital_updates.keys())
            
        for weight_name in list(hospital_updates.values())[0].keys():
            weighted_sum = None
                
            for hospital_id, weights in hospital_updates.items():
                weight_factor = self.privacy_configs[hospital_id].dataset_size / total_samples
                weighted_weight = weights[weight_name] * weight_factor
                    
                if weighted_sum is None:
                    weighted_sum = weighted_weight
                else:
                    weighted_sum += weighted_weight
                
            aggregated[weight_name] = weighted_sum
            
        return aggregated
    
    def evaluate_global_model(self, test_data: Dict):
        print(f"\n{'='*60}")
        print("GLOBAL MODEL EVALUATION")
        print(f"{'='*60}")
        
        overall_results = {}
        
        for hospital_id, data in test_data.items():
            print(f"\nEvaluating on Hospital {hospital_id} test set...")
            
            trainer = HospitalTrainer(self.global_model)
            results = trainer.evaluate(data['test_loader'])
            
            print(f"  Accuracy: {results['accuracy']:.3f}")
            print(f"  AUC: {results['auc']:.3f}")
            print(f"  Loss: {results['loss']:.3f}")
            print(f"  Avg Uncertainty: {np.mean(results['uncertainties']):.3f}")
            print(f"  Avg OOD Score: {np.mean(results['ood_scores']):.3f}")
            
            overall_results[hospital_id] = results
        
        return overall_results

def run_federated_depression_screening(hospital_datasets: Dict, privacy_configs: Dict, num_rounds: int = 3):
    print("="*80)
    print("FEDERATED DEPRESSION SCREENING SYSTEM")
    print("="*80)
    
    config = ModelConfig()
    model = create_depression_screening_model(config)
    processor = ClinicalInputProcessor(model.backbone.tokenizer, config)
    
    print(f"\nStep 1: Preparing hospital datasets...")
    prepared_datasets = prepare_hospital_datasets(hospital_datasets, processor)
    
    print(f"\nStep 2: Initializing federated learning...")
    fed_simulator = FederatedLearningSimulator(prepared_datasets, config, privacy_configs)
    
    print(f"\nStep 3: Running {num_rounds} federated learning rounds...")
    successful_rounds = 0
    
    for round_num in range(1, num_rounds + 1):
        success = fed_simulator.simulate_federated_round(round_num, prepared_datasets)
        if success:
            successful_rounds += 1
    
    print(f"\n✓ Completed {successful_rounds}/{num_rounds} successful federated rounds")
    
    print(f"\nStep 4: Final evaluation...")
    final_results = fed_simulator.evaluate_global_model(prepared_datasets)
    
    print(f"\nStep 5: Generating summary report...")
    generate_summary_report(final_results, successful_rounds)
    
    return {
        'global_model': fed_simulator.global_model,
        'hospital_models': fed_simulator.hospital_models,
        'results': final_results,
        'processor': processor
    }

def generate_summary_report(results: Dict, successful_rounds: int):
    print(f"\n{'='*80}")
    print("FEDERATED LEARNING SUMMARY REPORT")
    print(f"{'='*80}")
    
    print(f"Successful Federated Rounds: {successful_rounds}")
    print(f"Participating Hospitals: {len(results)}")
    
    all_accuracies = [r['accuracy'] for r in results.values()]
    all_aucs = [r['auc'] for r in results.values()]
    
    print(f"\nGlobal Model Performance:")
    print(f"  Average Accuracy: {np.mean(all_accuracies):.3f} ± {np.std(all_accuracies):.3f}")
    print(f"  Average AUC: {np.mean(all_aucs):.3f} ± {np.std(all_aucs):.3f}")
    
    print(f"\nHospital-Specific Results:")
    for hospital_id, result in results.items():
        print(f"  Hospital {hospital_id}:")
        print(f"    Accuracy: {result['accuracy']:.3f}")
        print(f"    AUC: {result['auc']:.3f}")
        print(f"    Avg Uncertainty: {np.mean(result['uncertainties']):.3f}")
    
    all_uncertainties = np.concatenate([r['uncertainties'] for r in results.values()])
    print(f"\nUncertainty Analysis:")
    print(f"  Mean Uncertainty: {np.mean(all_uncertainties):.3f}")
    print(f"  High Uncertainty Cases (>0.5): {np.mean(all_uncertainties > 0.5)*100:.1f}%")
    
    print(f"\n✓ Federated depression screening system ready for deployment!")

def demo_model_usage(model, hospital_data):
    print("\n" + "="*60)
    print("DEMO: Model Usage with Hospital Data")
    print("="*60)
    
    processor = ClinicalInputProcessor(model.backbone.tokenizer, model.config)
    
    sample_data = hospital_data.head(3)
    inputs = processor.process_batch(sample_data)
    
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            structured_features=inputs['structured_features'],
            return_uncertainty=True,
            return_rationale=True
        )
    
    for i in range(len(sample_data)):
        print(f"\nCase {i+1}:")
        print(f"  Prediction: {outputs['screening']['predictions'][i].numpy()}")
        print(f"  Predicted Class: {outputs['predicted_class'][i].item()}")
        print(f"  Uncertainty: {outputs['screening']['uncertainty'][i].item():.3f}")
        print(f"  OOD Score: {outputs['ood_score'][i].item():.3f}")
        print(f"  Explanation: {outputs['rationale']['explanations'][i]['explanation'][:100]}...")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ Using Apple M1/M2 GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓ Using NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("✓ Using CPU")

def main_execution():
    try:
        print("Step 0: Clearing initial memory...")
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        import gc
        gc.collect()
        
        print("\nStep 1: Starting system...")
        print("Available hospital datasets:", list(hospital_datasets.keys()))
        print("Available privacy configs:", list(privacy_configs.keys()))
        
        results = run_federated_depression_screening(
            hospital_datasets=hospital_datasets,
            privacy_configs=privacy_configs,
            num_rounds=1
        )
        
        if results:
            print("\n✅ SYSTEM EXECUTION COMPLETED SUCCESSFULLY!")
            return results
        else:
            print("\n❌ System returned None")
            return None
        
    except Exception as e:
        print(f"\n❌ ERROR during execution: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    config = ModelConfig()
    
    model = create_depression_screening_model(config)
    
    fed_manager = FederatedModelManager(model)
    
    print("\n✓ Federated depression screening system ready!")
    print("✓ Ready for integration with your hospital datasets")
    print("✓ Ready for streamlit deployment")
    
    print("Starting debug execution...")
    results = main_execution()
    print(f"Final result: {type(results)}")