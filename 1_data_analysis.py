from datasets import load_dataset
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using M1 GPU (MPS)")

dataset = load_dataset(
    "moremilk/CoT_Reasoning_Clinical_Diagnosis_Mental_Health",
    token="")

print(f"Dataset splits available: {list(dataset.keys())}")

if 'train' in dataset:
    data = dataset['train']

print(f"Dataset size: {len(data)} entries")
print(f"Dataset features/columns: {data.features}")
print("\n" + "="*50)

df = data.to_pandas()
print("Dataset shape:", df.shape)
print("\n Column names:")
print(df.columns.tolist())
print("\n First few column names and types:")
print(df.dtypes)

print("SAMPLE ENTRY #1:")
print("="*50)
first_entry = df.iloc[0]

for column in df.columns:
    print(f"\n{column.upper()}:")
    print("-" * len(column))
    print(first_entry[column])
    print()

model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model = model.to(device)
model.eval()

print(f"ClinicalBert loaded")

depression_criteria_templates = {
    'low_mood': "patient feels sad, depressed, down, or has low mood",
    'anhedonia': "patient has loss of interest or pleasure in activities", 
    'sleep_problems': "patient has sleep disturbances, insomnia, or hypersomnia",
    'fatigue': "patient feels tired, fatigued, or has low energy",
    'appetite_changes': "patient has appetite changes, weight loss, or weight gain",
    'guilt_worthlessness': "patient feels guilty, worthless, or like a failure",
    'concentration_problems': "patient has trouble concentrating or making decisions",
    'psychomotor_changes': "patient moves slowly or is restless and agitated", 
    'suicidal_thoughts': "patient has thoughts of death, suicide, or self-harm",
    'functional_impairment': "patient has difficulty with work, relationships, or daily functioning",
    'duration_criteria': "symptoms have lasted for weeks or months"
}

print(f"Created {len(depression_criteria_templates)} semantic templates")

def get_text_embedding(text):
    if pd.isna(text) or text == '':
        text = "no information provided"

    encoded = tokenizer(
        str(text),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return embedding[0]

print("\n Generating embeddings for decision criteria templates")

criteria_embeddings = {}
for criterion, template in depression_criteria_templates.items():
    embedding = get_text_embedding(template)
    criteria_embeddings[criterion] = embedding
    print(f"\n Processing clinical cases")

df_expanded = df.copy()
df_expanded['difficulty'] = df_expanded['metadata'].apply(lambda x: x['difficulty'])
df_expanded['reasoning'] = df_expanded['metadata'].apply(lambda x: x['reasoning'])
df_expanded['topic'] = df_expanded['metadata'].apply(lambda x: x['topic'])

def extract_clinicalbert_features(case_data):
    combined_text = f"{case_data['question']} {case_data['answer']} {case_data['reasoning']} {case_data['topic']}"
    case_embedding = get_text_embedding(combined_text)

    features = {'case_id': case_data['id']}

    similarity_scores = {}
    for criterion, criterion_embedding in criteria_embeddings.items():
        similarity = cosine_similarity(
            case_embedding.reshape(1, -1),
            criterion_embedding.reshape(1, -1)
        )[0][0]
        
        features[f'mentions_{criterion}'] = similarity > 0.7
        features[f'{criterion}_similarity'] = similarity
        similarity_scores[criterion] = similarity
    
    all_similarities = [similarity_scores[criterion] for criterion in depression_criteria_templates.keys()]
    
    features['depression_semantic_score'] = np.mean(all_similarities)
    features['likely_depression'] = features['depression_semantic_score'] > 0.65
    
    features['depression_criteria_count'] = sum(
        features[f'mentions_{criterion}'] for criterion in depression_criteria_templates.keys()
    )
    
    return features

print("Extracting ClinicalBERT features for all cases")

all_features = []
for idx, case in df_expanded.iterrows():
    if idx % 100 == 0:
        print(f"  Processed {idx}/3000 cases...")
    
    features = extract_clinicalbert_features(case)
    all_features.append(features)

clinicalbert_features_df = pd.DataFrame(all_features)

print(f"\nFeature extraction complete!")
print(f"Dataset shape: {clinicalbert_features_df.shape}")

print("\nDepression Criteria Detection (semantic similarity > 0.7):")
for criterion in depression_criteria_templates.keys():
    count = clinicalbert_features_df[f'mentions_{criterion}'].sum()
    pct = (count / len(clinicalbert_features_df)) * 100
    print(f"  {criterion}: {count} cases ({pct:.1f}%)")

print(f"\nMax depression score: {clinicalbert_features_df['depression_semantic_score'].max():.3f}")
print(f"Min depression score: {clinicalbert_features_df['depression_semantic_score'].min():.3f}")
print(f"Mean depression score: {clinicalbert_features_df['depression_semantic_score'].mean():.3f}")

for threshold in [0.7, 0.6, 0.5]:
    count = (clinicalbert_features_df['depression_semantic_score'] > threshold).sum()
    print(f"Cases > {threshold}: {count}")

df_merged = df_expanded.merge(clinicalbert_features_df, left_on='id', right_on='case_id', how='inner')
print(f"Merged dataset shape: {df_merged.shape}")

print("Topic distribution in dataset:")
topic_counts = df_merged['topic'].value_counts()
print(topic_counts.head(10))