My Zoom: A Transformer-Based Model for Contextual Feedback Validation

Project Overview :
            This project develops a machine learning model to validate user feedback for EdTech platforms like Zoom, ensuring alignment between feedback text and selected reasons. Using a fine-tuned BERT model, it classifies feedback-reason pairs as "Aligned" or "Not Aligned." The solution includes a Gradio interface for real-time predictions and is deployed on Hugging Face Spaces for public access. The project achieves over 85% accuracy (depending on dataset quality) and enhances feedback quality for platform improvements.
Problem Statement
EdTech platforms like Zoom collect user feedback with dropdown reasons, but misalignment (e.g., positive feedback with negative reasons) degrades data quality. Manual validation is unscalable, necessitating an automated solution. This project uses a transformer-based model to classify feedback-reason pairs, ensuring only relevant feedback is recorded, thus improving user experience and platform development.
Features

Text Preprocessing: 
          Cleans feedback and reason text by lowercasing, removing special characters, fixing typos (e.g., "can not" to "cannot"), and removing stopwords.
Model Training: Fine-tunes bert-base-uncased for binary classification using train.xlsx and evaluation.xlsx.

Evaluation: Computes accuracy, precision, recall, F1-score, and generates a confusion matrix (confusion_matrix.png).

Inference: Provides a Gradio interface for real-time feedback validation.

Deployment: Hosts the application on Hugging Face Spaces.

Technology Stack :

Programming Language: Python 3.11
Machine Learning:
Transformers (4.40.0): BERT model and tokenization
PyTorch (2.3.0): Training backend
Accelerate (0.26.0): GPU/CPU optimization


Data Processing:
Pandas (2.2.2): Excel handling
NLTK (3.8.1): Stopwords removal
Openpyxl (3.1.2): Excel reading


Evaluation/Visualization:
Scikit-learn (1.5.0): Metrics
Matplotlib (3.9.0), Seaborn (0.13.2): Confusion matrix


Deployment:
Gradio (4.31.0): Web interface
Hugging Face Spaces: Hosting


Environment: Google Colab (GPU) for training

Installation

Clone the Repository:
git clone https://github.com/your-username/my-zoom-feedback-validation.git
cd my-zoom-feedback-validation


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Download NLTK Data:
python -c "import nltk; nltk.download('stopwords')"



Dataset

Format: Excel files (train.xlsx, evaluation.xlsx)
Columns:
text: Feedback text (e.g., "Amazing app for classes")
reason: Dropdown reason (e.g., "Good app for online classes")
label: Binary label (1 for Aligned, 0 for Not Aligned)


Source: Provide your own dataset or use sample data with balanced classes.
Note: Ensure sufficient data and balanced classes for optimal performance.

Usage
Training the Model (Google Colab)

Set Up Colab Environment:

Open Google Colab and set the runtime to GPU (Runtime > Change runtime type > GPU).
Upload train.xlsx and evaluation.xlsx to the Colab file system.


Install Dependencies: Run the following in a Colab cell:
!pip cache purge
!pip uninstall -y transformers accelerate
!pip install transformers==4.40.0 torch==2.3.0 gradio==4.31.0 pandas==2.2.2 numpy==1.26.4 scikit-learn==1.5.0 matplotlib==3.9.0 seaborn==0.13.2 nltk==3.8.1 openpyxl==3.1.2 accelerate==0.26.0
!pip show transformers accelerate
!pip list


Run Training:

Copy train_model.py into a Colab cell and execute.
The script:
Preprocesses and tokenizes data.
Fine-tunes BERT for 3 epochs.
Evaluates performance and saves confusion_matrix.png.
Saves the model to ./feedback_model.




Download Outputs:

Download ./feedback_model and confusion_matrix.png for deployment.



Local Testing (Gradio Interface)

Ensure Model is Trained:

Place the ./feedback_model directory in the project root.


Run Gradio Interface:
python app.py


Access the interface at http://localhost:7860.
Input feedback text and reason to get predictions ("Aligned" or "Not Aligned").



Deployment (Hugging Face Spaces)

Create a Space:

Go to Hugging Face Spaces and create a new Space (Python, Gradio).


Upload Files:

Upload app.py, requirements.txt, and the ./feedback_model directory.


Deploy:

Follow the Spaces instructions to build and deploy.
Access the public URL for the Gradio interface.



Project Structure
my-zoom-feedback-validation/
├── train_model.py         # Script for training and evaluating the model
├── app.py                # Gradio interface for inference
├── requirements.txt      # Project dependencies
├── feedback_model/       # Trained model and tokenizer (post-training)
├── confusion_matrix.png  # Evaluation visualization
├── train.xlsx           # Training dataset (not included)
├── evaluation.xlsx      # Evaluation dataset (not included)
├── README.md            # This file

Results

Performance (example, based on dataset):
Accuracy: >85%
Precision: ~0.87
Recall: ~0.85
F1-Score: ~0.86


Confusion Matrix: Visualizes true vs. predicted labels (confusion_matrix.png).
Gradio Interface: Predicts "Aligned" or "Not Aligned" (e.g., "Amazing app for classes" with "Good app for online classes" → "Aligned").
Impact: Enhances feedback quality for Zoom, scalable for EdTech platforms.

Challenges and Solutions

Dependency Conflicts: Resolved by clearing pip cache and installing specific versions (transformers==4.40.0, accelerate==0.26.0).
Dataset Quality: Ensured balanced classes; augmentation can be added if needed.
Deployment: Simplified using Hugging Face Spaces for hosting.

Future Work

Extend to multi-class classification for nuanced feedback.
Integrate with Zoom’s API for real-time validation.
Explore lightweight models (e.g., DistilBERT) for faster inference.

Acknowledgments

Hugging Face for transformers and Spaces.
Google Colab for GPU training.
Zoom for inspiring the use case.

