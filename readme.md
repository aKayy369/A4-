# A4: Do you AGREE?
# AT82.05 Artificial Intelligence: Natural Language Understanding (NLU)
# Student Name: Anubhav Kharel , ID - st125999


#  Overview

This project implements BERT from scratch and adapts it into a Sentence-BERT (SBERT) model to capture semantic similarity between sentences for the Natural Language Inference (NLI) task.

The assignment demonstrates:

* Training contextual language models

* Learning sentence embeddings

* Modeling semantic relationships


# Project Structure
A4/
│
├── notebook.ipynb          # Main implementation
├── README.md               # Documentation
├── app/                    # Web application
│   ├── app.py
│   ├── bert_task1_weights.pth
│   ├── sbert_finetuned_model.pth
│   └── word2id.pth

#  Task 1 — Training BERT from Scratch

A BERT encoder was implemented from scratch using PyTorch, including:

* Token embeddings

* Positional embeddings

* Segment embeddings

* Multi-Head Self-Attention (custom implementation)

* Feed-Forward Networks

* Layer Normalization

* Encoder stacking

* Masked Language Modeling (MLM)

* Next Sentence Prediction (NSP)

* The model follows the original BERT architecture concepts.

## Dataset

Training was performed on a 100K subset of the BookCorpus dataset:

📚 BookCorpus Dataset
https://huggingface.co/datasets/bookcorpus/bookcorpus

Only a subset was used due to computational constraints.

## Training Objectives

Masked Language Modeling (MLM)

Next Sentence Prediction (NSP)

## Loss function:

Total Loss = MLM Loss + NSP Loss

## Output

The trained encoder weights were saved for reuse:

bert_task1_weights.pth

# Task 2 — Sentence-BERT for Sentence Embedding
Siamese Architecture

The trained BERT encoder was used inside a Siamese network to generate sentence embeddings.

## Dataset

📘 SNLI Dataset
https://huggingface.co/datasets/snli

Used for Natural Language Inference classification:

* Entailment

* Neutral

* Contradiction


📄 SBERT Paper
https://aclanthology.org/D19-1410/

## Training Strategy

Fine-tuning was performed end-to-end:

* BERT encoder

* SBERT classifier

* Masked mean pooling was used to generate sentence embeddings.

* Similarity Demonstration

Cosine similarity between sentence embeddings:

* Cosine similarity ≈ 0.93



# Task 3 — Evaluation and Analysis

Classification Report (SNLI)
Accuracy: 49%

Class Performance:

Entailment      F1 ≈ 0.48
Neutral         F1 ≈ 0.52
Contradiction   F1 ≈ 0.46



##  Limitations

1. Limited Training Data

**  Only a small subset of BookCorpus and SNLI was used.

2. Small Model Size

**  A reduced BERT configuration was used due to hardware constraints.

3. Word-Level Tokenization

** A simple tokenizer was implemented instead of WordPiece.

4. Computational Constraints

**  Full-scale training was not feasible.

## Proposed Improvements

* Use full BookCorpus + Wikipedia

* Implement WordPiece tokenization

* Increase model depth and embedding size

* Train longer

* Apply data augmentation

# Task 4 — Web Application

A simple web application was developed using Dash.

* Features -  Two input boxes: Premise, Hypothesis

* Predicts NLI label: Entailment, Neutral, Contradiction




# App Interface

<p align="center"> <img src="images/Sample1.png"> </p>
<p align="center"> <img src="images/Sample2.png"> </p>
<p align="center"> <img src="images/Sample3.png"> </p>
<p align="center"> <img src="images/Sample4.png"> </p>

![Demo](images/Sample1.png)
