# 📝 Text Classification using Multiple ML Models

## 📚 Project Overview

This project implements and compares different Machine Learning models for text classification tasks, using a Twitter dataset sourced from Kaggle. The main goal is to predict the sentiment of tweet responses and analyze the performance of various ML techniques.

---

## 🚀 Objectives

- Explore and preprocess real-world text data.
- Train, compare, and visualize the performance of several classification models.
- Generate useful insights for future applications in sentiment analysis and NLP.

---

## 🏗️ Project Structure
```
Text-Classification/
│
├── .env/ # Environment variables (optional)
├── utils/ # Utility functions and helpers
│
├── config/ # Configuration and global resources
│ ├── pycache/ # Python cache files
│ ├── constants.py # Global constants and paths
│ └── kaggle.json # Kaggle API authentication token
│
├── data/ # Raw and processed data
│ ├── twitter_training.csv # Training dataset
│ ├── twitter_validation.csv # Validation dataset
│ └── twitter-entity-sentiment-analysis.zip # Original source (optional)
│
├── images/ # Generated visualizations
│ ├── Distribution_Probabilities_Incorrect.png
│ ├── Distribution_Probabilities_Predicted_Class.png
│ ├── image_1.png
│ ├── image_2.png
│ ├── image_3.png
│ ├── image_4.png
│ ├── image_5.png
│ ├── Model_metrics.png
│ └── sentiment_by_entity.png
│
├── .gitignore # Files/folders ignored by git
├── README.md # Main project documentation
├── requirements.txt # Python libraries and dependencies
└── Text Classification.ipynb # Main project notebook
```


## 🧩 Folder and File Descriptions

- **config/**  
  Project configuration files, credentials, and global constants.

- **data/**  
  Contains the datasets used for training and validating the models.

- **images/**  
  Stores all the charts and visualizations generated during model analysis and evaluation.

- **utils/**  
  Auxiliary functions for preprocessing and analysis, imported in the main notebook.

- **requirements.txt**  
  List of required dependencies to run the project (`scikit-learn`, `pandas`, `matplotlib`, etc.).

- **Text Classification.ipynb**  
  The main notebook containing the full workflow: analysis, training, and visualization.

---

## 🛠️ Technologies and Libraries Used

- **Python 3.x**
- **Pandas** and **NumPy**: Data manipulation and analysis
- **scikit-learn**: Machine Learning models and evaluation
- **Matplotlib** and **Seaborn**: Data visualization
- **Jupyter Notebook**

---

## 📈 Project Workflow

1. **Data loading and exploration:**  

The project begins by loading a dataset of product reviews, where each review is labeled as either positive or negative. An initial exploratory analysis is conducted to understand the class distribution and the typical length of the reviews.

- **Class Distribution Visualization:**
A bar chart is used to display the number of positive,negative, neutral and irrelevant reviews. 
  ![Exploratory Data Visualization for Sentiment and Class Distributions](utils/images/sentiment_by_entity.png)

- **Review Length Analysis:**
A graph is generated to show the number of tweets per type of review. This helps assess whether the dataset is balanced or if one class is overrepresented, which could impact model performance and training dynamics.

  ![Exploratory Data Visualization for Sentiment and Class Distributions](utils/images/image_1.png)


2. **Data Preprocessing**
Before feeding the data into the model, several preprocessing steps are applied to prepare the text for analysis:

- Text Cleaning:
Special characters are removed, all text is converted to lowercase, and stopwords (commonly used words with little semantic value) are eliminated to reduce noise.

- Tokenization:
Each review is split into individual words or "tokens," enabling the model to process text at the word level.

- Vectorization:
The cleaned and tokenized text is transformed into a numerical representation using techniques such as TF-IDF (Term Frequency–Inverse Document Frequency). This method assigns weights to words based on their frequency in a specific review relative to their frequency across the entire dataset, helping the model focus on the most informative terms.
  Clean data, normalize text, tokenize, and vectorize (CountVectorizer, TF-IDF, etc.).

![Visualization of Raw Data Prior to Cleaning](utils/images/image_3.png)
![Raw Data Visualization After Cleaning](utils/images/image_4.png)

This analysis is particularly useful for determining an appropriate maximum input length for the model, ensuring it can effectively process the textual data without truncating valuable information.

![UMBRAL DE N_GRMS PRIOR TO CLEANING](utils/images/image_5.png)
![UMBRAL OF THE N GRAMS AFTER CLEANING](utils/images/image_6.png)
3. **Model training:**  
  Multiple classic ML models are trained and compared for text classification, including:
  - Logistic Regression
  - Naive Bayes
  - Random Forest



4. **Model Evaluation**
After training, the model is evaluated using the test dataset. Key performance metrics such as accuracy, the confusion matrix, and the classification report are calculated to assess its effectiveness.

 **Confusion Matrix:**
Displays the number of correct and incorrect predictions made by the model for each class, offering a clear view of how well the model distinguishes between positive and negative reviews.

 **Classification Report:**
Provides detailed performance metrics, including precision, recall, and F1-score for each class. These metrics offer deeper insights into the model's strengths and areas for improvement.

  ![Model Evaluation and Metrics Overview](utils/images/Model_metrics.png)

5. **Prediction on New Reviews**
Finally, the model is used to predict the sentiment of previously unseen reviews. This step demonstrates the model's ability to generalize beyond the training data and showcases its potential usefulness in real-world applications.
---

## 🏆 Key Results

- Clear performance comparison among classic text classification models.
- Intuitive visualizations to understand each model's strengths and weaknesses.
- Recommendations for future use and improvements of NLP models in both Spanish and English.

 The following visualizations provide a comprehensive overview of the performance of the text classification model. These plots offer valuable insights into how the model assigns prediction probabilities and how they correlate with correct and incorrect classifications.

### 📊 **Probability Distribution of Incorrect Predictions**
This plot illustrates how the model assigns different probability scores to its incorrect predictions.

- X-axis (Prediction Probability): Represents the confidence score assigned by the model to each prediction.

 - Y-axis (Frequency): Indicates how frequently each range of probability values occurs among incorrect predictions.

The chart reveals that most misclassifications are associated with lower probability values, typically around 0.3, suggesting the model tends to be less confident when it makes a mistake. However, there are also instances of incorrect predictions with high confidence scores, indicating that the model can occasionally be overconfident in its wrong decisions — a potential area for improvement.

![Model Evaluation and Metrics Overview](utils/images/Distribution_Probabilities_Incorrect.png)

### 📈 Probability Distribution by Prediction Outcome (Correct vs. Incorrect)

This second plot compares the probability distributions for correct and incorrect predictions.

Orange Curve (Correct): Represents the distribution of probabilities for predictions the model classified correctly.

Blue Curve (Incorrect): Represents the distribution for misclassified predictions.

- X-axis (Prediction Probability): Indicates the model's assigned confidence.

- Y-axis (Frequency): Represents how often each probability range appears.

From the graph, we observe that correct predictions tend to cluster around higher probability values (close to 1.0), demonstrating that the model is generally confident when making accurate predictions. In contrast, incorrect predictions are more spread out across lower probability ranges, though some still occur with high confidence.



![Probability Distribution of Incorrect Predictions](utils/images/Distribution_Probabilities_Predicted_Class.png)

---

## 📂 How to Use This Project

1. **Clone the repository**
  ```bash
  git clone https://github.com/your_username/Text-Classification.git
  cd Text-Classification
  Install the dependencies
  Copiar
  Editar
  pip install -r requirements.txt
  Run the notebook
  ```

Open Text Classification.ipynb using Jupyter Notebook or VSCode.

Replace/update the datasets if you wish to use your own data

---

## 🤝 Credits & Acknowledgments
Dataset from Kaggle Twitter Sentiment Analysis.

Inspiration and resources from the Data Science community.

---

## 📬 Contact

Kevin Astudillo
📧 astudillo.kevim@gmail.com

Ricardo Jaramillo
📧 r.jaramillohernandez@outlook.com

Questions, suggestions, or want to collaborate?
Open an issue or reach out!
