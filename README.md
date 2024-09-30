# Semantic Text Similarity (STS) Project

**Team Members:**
- Heejun Kwak
- Minjun Kim
- Jeongeun Kim
- DongHyeok Park
- Donghun Han

---

## 1. Project Overview

The project focuses on **Semantic Text Similarity (STS)**, an NLP task to evaluate how semantically similar two sentences are, with a similarity score ranging from **0.0 to 5.0**. Our goal was to implement a model that predicts the human-labeled similarity score between pairs of sentences.
  
### **Development Environment:**
- **Hardware**: Tesla V100 GPU (4 servers)
- **Software**: Linux, Git, Pandas, Hugging Face Transformers, Pytorch Lightning, WandB
- **Collaboration Tools**: 
  - **GitHub**: code sharing
  - **Slack**: server coordination
  - **Notion**: brainstorming and planning
  - **Google Docs**: experiment sharing
  - **Zoom**: real-time meetings
  
### **Project Structure:**

```BOOSTCAMP_PROJECT1-SEMANTIC_TEXT_SIMILARITY
├───data             # Raw, preprocessed, and augmented data
├───STS              # Main project directory
│   ├───baselines    # Model Hyperparameters (config.yaml)
│   ├───EDA          # EDA notebooks
│   ├───src          # Main model code (model.py, data_pipeline.py)
│   ├───train.py
│   ├───inference.py # Inference validation set
│   ├───ensemble.py  # conda
│   └───wandb        # Wandb sweep results and logs
```



---
## 2. Team Collaboration and Roles

Our team adopted an **end-to-end** project approach, where each member worked on all aspects of the project, from data preprocessing to model tuning. This helped ensure everyone gained broad insights into the entire process.

### **Team Roles:**
- **Minjun Kim**: EDA, Data Augmentation (swap sentence, copy sentence, under-sampling), Ensemble (soft-voting, weight voting).
- **Jeongeun Kim**: EDA(statistical and sentence data analysis, Preprocessing (special character and duplicates removal), Augmentation(synonym replacement, swap sentence, under-sampling, copy sentence), Modeling (hyperparameter tuning, large/small model comparison), Ensemble (bagging, boosting, voting).
- **Heejun Kwak**: Environment setup (environment setting using: server, git, conda), EDA (token and preprocessing analysis), Preprocessing (spelling correction, duplicate removal), Augmentation(copy, swap sentence) Modeling (pretrained model comparisons, layer customizing, freezing, hyperparameters tuning), Postprocessing (tuning prediction output labels).
- **DongHyeok Park**: EDA (label and source analysis, search for and removal of [UNK] token), Model Tuning (KrELECTA-discriminator, sBERT), Modeling(custom: weightedMSELoss Function), Ensemble (Connecting the soft voting code with the ensemble preprocessing functions).
- **Donghun Han**: EDA, Model tuning (head layer modification using CLS and SEP tokens), Data augmentation, Ensemble (soft voting).

---
## 3. Project Schedule and Workflow
### **Collaboration**:
- Server: 5 team members worked by sharing 4 servers.
- Git Collaboration: Each team member worked independently on their own branch.
- Remote Repository: Managed the code and moved freely between servers.
- Miniconda: Ensured environment independence and portability (using environment.yaml)

#### **Week 1**:
- Individually experience the project end-to-end
  
#### **Week 2**:
- Collaboration to improve model performance by combining insights and experimentation outcomes.


---

## 4. Project Results

### 4.1 Exploratory Data Analysis (EDA)

The dataset used in this project consists of **4 key files**: `train.csv` (9324 rows), `dev.csv` (550 rows), `test.csv` (1100 rows), and `sample_submission.csv` (1100 rows). Each pair of sentences is assigned a label between **0.0 to 5.0**.

#### **Label Distribution**:
- There was a heavy skew in the dataset towards lower similarity scores (0.0–1.0).
- A significant proportion (~50%) of the data fell within the 0.0–1.0 range, with only ~1.6% of the data belonging to the 4.5–5.0 range, indicating **data imbalance**.

#### **Data Source Analysis**:
- The dataset was crawled from sources like NSMC, Slack, and petitions, resulting in a predominance of informal language.
- **Preprocessing focus**: Correct informal and incorrect text by removing duplicate characters, fixing spacing errors, and using a spelling correction tool.

### 4.2 Data Preprocessing

We observed that the model performed poorly with raw data. Thus, we designed preprocessing steps to ensure the model better captured the sentence semantics:

#### **Preprocessing Steps:**
1. **Character filtering**: Limit repeated characters (e.g., ‘ㅋㅋㅋㅋㅋ’ → ‘ㅋㅋ’).
2. **Spelling correction**: Fix grammar and typographical errors using **Naver's spelling correction tool**.

- **Improved Performance**: By feeding preprocessed data into the model, we saw a significant Pearson correlation improvement (from **0.7806 to 0.8129**).

### 4.3 Data Augmentation

To address the data imbalance, we implemented several augmentation techniques:

- **Under-Sampling**: Reduced the over-represented labels.
- **Over-Sampling**:
  - **Synonym Replacement**: Used **Korean WordNet(KWN)** to replace common nouns with synonyms.
  - **Sentence Swap**: Reversed sentence pairs in the dataset for more variety.
  - **Copy Sentence**: Modified sentences labeled 0.0 to 5.0 by copying one sentence to the other, making the sentence pairs identical.
  - **Translation (ko2en)**: Translated sentences from **Korean to English** to create more data points for under-represented labels.
  
#### **Augmentation Results**:
- By using a combination of augmentation techniques, we improved Pearson correlation by **+0.0397**, reaching a score of **0.8526**.

### 4.4 Modeling

#### **RoBERTa and KoELECTRA Models**:
- **RoBERTa** and **KoELECTRA-discriminator** were selected for their performance on informal datasets.
- We experimented with **custom layers** (LSTM and GRU) to enhance the models' understanding of sentence context, but found that base transformers already captured the necessary context effectively.

#### **Layer Freezing**:
- Freezing the top **12 layers** of RoBERTa-large provided the best balance between preserving general language understanding and allowing task-specific fine-tuning.

#### **Key Findings**:
- **Pretrained Models**: RoBERTa-large consistently outperformed other models, reaching a Pearson correlation of **0.9342**.

### 4.5 Ensemble

We implemented multiple ensemble techniques to further boost performance:

1. **Soft Voting**: Averaged the predictions from multiple models.
2. **Weighted Voting**: Applied weights based on model performance to each model's predictions.
  
- The **soft voting** approach yielded better results, and **weighted voting** did not outperform it. Thus, we finalized our ensemble using soft voting.

### 4.6 Post-processing

Through exploratory data analysis (EDA), we discovered that the labels in the entire dataset are composed in increments of 0.2 (with a small number of data points labeled 0.5, 1.5, 3.5, and 4.5). However, the model made predictions covering from 0.0 to 5.0 in increments of 0.1
Therefore, post-processing was applied to adjust the predicted labels to the nearest actual label based on the label distribution of the validation set when the model's predicted label did not exist among the true labels.

The results indicated that post-processing helped increase the model’s Pearson correlation, especially in under-represented label ranges.

---

## 5. Leaderboard Results

- **Public Score**: **0.9296** (Rank 12)
- **Private Score**: **0.9387** (Rank 7, +5 positions)


