#  🎭 Facial Emotion Recognition  
<p align="left">
  <img src="https://img.shields.io/badge/Made%20With-Colab-blue?logo=googlecolab&logoColor=white&label=Made%20With" alt="Made with Colab">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/github/repo-size/ShaikhBorhanUddin/Facial-Emotion-Recognition" alt="Repo Size">
  <img src="https://img.shields.io/github/last-commit/ShaikhBorhanUddin/Facial-Emotion-Recognition" alt="Last Commit">
  <img src="https://img.shields.io/github/issues/ShaikhBorhanUddin/Facial-Emotion-Recognition" alt="Issues">
  <img src="https://img.shields.io/badge/Framework-TensorFlow-orange?logo=tensorflow" alt="Framework: TensorFlow">
  <img src="https://img.shields.io/badge/Result%20Visualization-GradCAM%20|%20GradCAM++-red?style=flat&logo=visualstudiocode&logoColor=white" alt="Result Visualization: GradCAM, GradCAM++">
  <img src="https://img.shields.io/badge/Emotion%20Classes-7-critical" alt="Emotion Classes">
  <img src="https://img.shields.io/github/forks/ShaikhBorhanUddin/Facial-Emotion-Recognition?style=social" alt="Forks">
  <img src="https://img.shields.io/badge/Version%20Control-Git-orange?logo=git&logoColor=white" alt="Version Control: Git">
  <img src="https://img.shields.io/badge/Host-GitHub-black?logo=github&logoColor=white" alt="Host: GitHub">
  <img src="https://img.shields.io/badge/Project-Completed-brightgreen" alt="Project Status">

</p>

![Dashboard](https://github.com/ShaikhBorhanUddin/Facial-Emotion-Recognition-Project/blob/main/Image/fer_title_2.png?raw=true)

## 🧾 Overview  

## 🗂️ Project Structure  

```bash
Facial-Emotion-Recognition/
│
├── assets/                    # 📊 Saved model weights, plots, and outputs
├── data/                      # 📁 Raw and preprocessed data
│
├── src/
│   ├── model_training.ipynb   # 🏋️ Training notebook
│   ├── evaluation.ipynb       # 📈 Evaluation metrics
│   └── emotion_predictor.py   # 🤖 Inference script
│
├── requirements.txt           # 📃 Python dependencies
└── README.md
```

## 🔄 Workflow  

## 🗃️ Dataset  

![Dashboard](https://github.com/ShaikhBorhanUddin/Facial-Emotion-Recognition-Project/blob/main/Image/image_distribution.png?raw=true)  

![Dashboard](https://github.com/ShaikhBorhanUddin/Facial-Emotion-Recognition/blob/main/Image/fer_sample_mod.png?raw=true)  

## 🔬 Model Architecture & Experiments  

## 📊 Results  

| Model                   | Accuracy | F1 Score | Loss   | Precision | Recall  | Training       |
|-------------------------|----------|----------|--------|-----------|---------|----------------|
| EfficientNetB5          | 0.8386   | 0.7556   | 0.6814 | 0.8473    | 0.8360  | Successful     |
| EfficientNetB4          | 0.8185   | 0.7384   | 0.6205 | 0.8246    | 0.8097  | Successful     |
| EfficientNetB3          | 0.7913   | 0.7209   | 0.6796 | 0.8034    | 0.7900  | Successful     |
| EfficientNetB2          | 0.8324   | 0.7100   | 0.6005 | 0.8434    | 0.8256  | Successful     |
| EfficientNetB3 Modified | 0.8342   | 0.8535   | 0.7443 | 0.8387    | 0.8284  | Successful     |
| EfficientNetB2 Modified | 0.8083   | 0.6756   | 0.6470 | 0.8129    | 0.7978  | Successful     |
| ConvNeXtBase            | 0.8342   | 0.8535   | 0.7443 | 0.8387    | 0.8284  | Successful     |
| VGG16                   | 0.7444   | 0.3558   | 0.6762 | 0.8164    | 0.6580  | Successful     |
| VGG19                   | 0.8112   | 0.6560   | 0.5309 | 0.8558    | 0.7876  | Successful     |
| ResNet152V2             | 0.6464   | 0.4481   | 1.7839 | 0.6563    | 0.6456  | Overfitted     |
| ResNet101V2             | 0.7362   | 0.4074   | 1.2637 | 0.7373    | 0.7301  | Successful     |
| ResNet50V2              | 0.6985   | 0.3915   | 1.5528 | 0.7040    | 0.6969  | Successful     |

## 📈 ROC Curves  

ROC curve analysis of all tested models are included in this section. ResNet152V2 was excluded due to overfitting in training phase. Also, customized version of EfficientNetB2 and B3 are not included in the analysis.

![Dashboard](https://github.com/ShaikhBorhanUddin/Facial-Emotion-Recognition/blob/main/Image/roc_efficientnet.png?raw=true)  

In the first four image above, the ROC curves for the EfficientNet models B2, B3, B4, and B5 are shown. All four models demonstrate excellent class discrimination, with AUC scores mostly ranging from 0.97 to 1.00 across all seven emotion classes (Anger, Disgust, Fear, Happiness, Neutral, Sadness, and Surprise). Notably, EfficientNetB5 and B4 show near-perfect AUCs of 1.00 for "Happiness" and "Neutral," indicating extremely strong classification performance for those classes. There is minimal deviation among classes, and all models exhibit steep rises near the Y-axis, reflecting high true positive rates and low false positives—hallmarks of effective multi-class classification models.  

![Dashboard](https://github.com/ShaikhBorhanUddin/Facial-Emotion-Recognition/blob/main/Image/roc_vgg.png?raw=true)  

For VGG class, the ROC curves for the VGG16 (left image above) and VGG19 (right image) models are displayed. Both models still achieve reasonably high AUCs (mostly in the 0.93–0.99 range), though slightly lower compared to EfficientNet models. A noticeable decline in performance is seen for "Anger" (VGG19: 0.95) and "Sadness" (VGG19: 0.93), suggesting these emotions were harder to classify accurately. Additionally, the ROC curves exhibit more fluctuation, and the curves are less smooth, which aligns with the models' noted overfitting during training. This performance inconsistency suggests that while VGG models can still capture emotional distinctions, their generalization ability is weaker.

![Dashboard](https://github.com/ShaikhBorhanUddin/Facial-Emotion-Recognition/blob/main/Image/roc_resnet.png?raw=true)  

In the last three image, ConvNeXtBase, ResNet50V2, and ResNet101V2 are analyzed. These models deliver robust classification results, comparable to EfficientNetB2–B5, with AUCs consistently between 0.95 and 1.00. ConvNeXtBase (left image) stands out with a perfect AUC of 1.00 for both "Happiness" and "Neutral," while maintaining strong performance across the other classes. The ROC curves of ResNet101V2 (right image) and ResNet50V2 (middle image) are closely packed with high slopes, indicating minimal class confusion and strong predictive power. Unlike VGG models, these architectures exhibit better generalization, as supported by both their AUC scores and the smoothness and steepness of the ROC curves.

## 📉 Confusion Matrix  


## 🖼️ Visualizations  

## 🌍 Practical Applications  

Facial Emotion Recognition (FER) technology is essential for advancing human-centered AI across various fields, including healthcare, education, retail, automotive, entertainment, and security. In **healthcare**, FER supports therapists by continuously monitoring a patient's emotional state during virtual sessions for timely interventions. In **education**, it enhances e-learning by identifying student confusion or disengagement in real time, allowing for adaptive content delivery. In **retail and customer service**, emotion recognition helps analyze customer sentiment during interactions, improving service and product recommendations. The **automotive** industry benefits from FER through driver monitoring systems that detect drowsiness, frustration, or distraction, enhancing road safety. Furthermore, in **entertainment**, FER creates responsive gaming and virtual reality environments based on player emotions. Lastly, it improves **security and surveillance** by detecting unusual or stress-induced facial expressions in public or high-risk areas, enabling proactive behavioral analysis.  

## 🔧 Tools & Technology  

## 🚧 Future Improvements  

## ⚠️ Limitations  

## 🔗 References  

## 📄 License

## 🙋 Contact  


