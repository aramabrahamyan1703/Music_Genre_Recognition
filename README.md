# 🎵 Music Genre Recognition

This project implements a machine learning pipeline to classify music tracks into genres using audio features extracted from .wav files. It leverages both traditional classifiers and ensemble methods to achieve high accuracy on the GTZAN dataset.

## 📁 Project Structure
- `main.ipynb` – Core notebook containing data preprocessing, feature extraction, model training, evaluation, and visualization.
- `features_30_sec.csv` – Pre-extracted features for full-length (30-second) audio clips.
- `features_3_sec.csv` – Pre-extracted features for segmented (3-second) audio clips.

## 🧠 Models Used
The following classifiers were trained and evaluated:
- SVM
- k-NN
- Gaussian Naive Bayes
- Random Forest
- QDA
- LDA
- Logistic Regression
- HistGradientBoostingClassifier
- VotingClassifier(Ensemble of the above)

## 📊 Evaluation & Error Analysis
A detailed confusion matrix analysis revealed:
- High accuracy in genres like **blues**, **hip-hop**, **metal**, and **pop**.
- Notable misclassifications between acoustically similar genres:
  - **Jazz ↔ Classical**
  - **Rock ↔ Blues**
  - **Pop ↔ Disco**
These confusions likely stem from overlapping audio features. To mitigate this, future work could involve:
- Expanding the training dataset.
- Utilizing deep learning architectures on spectrograms.

## Prerequisites
`numpy`
`pandas`
`scikit-learn`
`matplotlib`

## Results
|Rank|Model|F1 Score|
|---|---|---|
|1|**VotingClassifier**|0.935|
|2|**SVM (Tuned)**|0.932|
|3|**HistGradientBoostingClassifier (Tuned)**|0.921|
|4|**k-NN (Tuned)**|0.891|
|5|**Random Forest (Reduced Features)**|0.865|
|6|**QDA (Tuned)**|0.768|
|7|**Logistic Regression**|0.714|
|8|**LDA (Tuned)**|0.661|
|9|**Gaussian Naive Bayes**|0.481|

## Acknowledgments
- The creators of the GTZAN dataset.
- Open-source contributors and libraries used in this project.
- The machine learning community for their valuable resources and inspiration.