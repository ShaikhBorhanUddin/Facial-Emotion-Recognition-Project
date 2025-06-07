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

[Dataset](https://drive.google.com/drive/folders/14GxPnsSpW-rJrU4eOo7fZWZQV-zcxCi9?usp=sharing)  

![Dashboard](https://github.com/ShaikhBorhanUddin/Facial-Emotion-Recognition-Project/blob/main/Image/image_distribution.png?raw=true)  

![Dashboard](https://github.com/ShaikhBorhanUddin/Facial-Emotion-Recognition/blob/main/Image/fer_sample_mod.png?raw=true)  

## 🔬 Model Architecture

All models followed a structured and consistent architecture pipeline involving a pretrained CNN backbone, global average pooling, dense layers with ReLU activation, and a softmax output for multi-class emotion classification. A `Dropout(0.5)` layer was uniformly applied before the classification head to mitigate overfitting.

- **EfficientNetB2 to B5**: These models varied in depth and parameter size (B2: 8.4M to B5: 29.5M) and mostly used a single dense layer of 512 units.

- **Modified EfficientNetB2** included **two dense layers (1024 → 512)**, increasing capacity with 10.7M parameters.

- **Modified EfficientNetB3** further extended this to **three dense layers (1024 → 512 → 256)**, totaling 13M parameters.

- **VGG16** and **VGG19** used their classic convolutional stacks followed by a dense layer of 512 units, with respective parameters of 14.9M and 20.2M.

- **ResNet50V2**, **101V2**, and **152V2** increased in complexity and depth, topping out at **59.3M parameters** for ResNet152V2.

- **ConvNeXtBase**, the most computationally intensive model with **88M parameters**, leveraged modern attention-inspired architecture, ending in a dense layer of 512 units.

Despite architectural differences, the consistent training logic and layered designs allowed fair comparison across models. The Modified EfficientNet and ConvNeXtBase models benefitted from deeper, flexible classification heads and TPU acceleration, leading to superior generalization.  

## 🧪 Experiments  

A series of deep learning models were trained and evaluated to recognize facial emotions across seven categories. All models shared a unified training protocol with 30 epochs, a dropout rate of 0.5, ReLU activation in dense layers, softmax output activation, Adam optimizer, and categorical crossentropy loss. The base learning rate was set to 0.001, with a fine-tuning learning rate of 1e-4 (except VGG16 and VGG19, which used 1e-5). Training was conducted on an **NVIDIA A100 GPU**, except for the **Modified EfficientNetB2** and **Modified EfficientNetB3**, which were trained using **v5e Trillium TPUs** for enhanced scalability.

Batch sizes were tailored to each model's capacity. For example, EfficientNetB5 and VGG19 used a smaller batch size of 128 due to their larger parameter count (29.5M and 20.2M, respectively), while ResNet50V2 and ResNet101V2 utilized batch sizes of 512. These batch size variations were chosen to balance memory usage and convergence efficiency.

All models were evaluated on accuracy, F1 score, precision, recall, and categorical loss. Results clearly favored **ConvNeXtBase** and **Modified EfficientNetB3**, which achieved top-tier performance, while models such as **ResNet152V2 showed signs of overfitting despite their complexity**. This consistent and controlled experimental setup enables reliable performance comparison across architectures and training environments.

## 📊 Results  

Performance matrix for all models are included in this section.  

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

The ConvNeXtBase and EfficientNetB3 Modified models emerged as top performers, both achieving the highest F1 score (0.8535) and excellent accuracy (~83.4%), indicating strong consistency in classification. EfficientNetB5 also performed well with an accuracy of 83.86%, though its F1 score (0.7556) was noticeably lower, suggesting it may not generalize as effectively across all emotion classes. Among traditional architectures, VGG16 performed the worst in F1 score (0.3558), pointing to major misclassifications despite decent precision. VGG19 improved significantly over VGG16 after correction but still lagged behind the top models. Notably, ResNet152V2 was the only model flagged as overfitted, with both accuracy and F1 score dropping below acceptable levels. The results suggest that modern architectures like ConvNeXt and modified EfficientNets handle complex, multi-class emotional data more robustly than deeper or older CNN backbones.  

## 📈 ROC Curves  

ROC curve analysis of all tested models are included in this section. ResNet152V2 was excluded due to overfitting in training phase. Also, customized version of EfficientNetB2 and B3 are not included in the analysis.

![Dashboard](https://github.com/ShaikhBorhanUddin/Facial-Emotion-Recognition/blob/main/Image/roc_efficientnet.png?raw=true)  

In the four image above, the ROC curves for the EfficientNet models B2, B3, B4, and B5 are shown. All four models demonstrate excellent class discrimination, with AUC scores mostly ranging from 0.97 to 1.00 across all seven emotion classes (Anger, Disgust, Fear, Happiness, Neutral, Sadness, and Surprise). Notably, EfficientNetB5 and B4 show near-perfect AUCs of 1.00 for "Happiness" and "Neutral," indicating extremely strong classification performance for those classes. There is minimal deviation among classes, and all models exhibit steep rises near the Y-axis, reflecting high true positive rates and low false positives—hallmarks of effective multi-class classification models.  

![Dashboard](https://github.com/ShaikhBorhanUddin/Facial-Emotion-Recognition/blob/main/Image/roc_vgg.png?raw=true)  

For VGG class, the ROC curves for the VGG16 (left image above) and VGG19 (right image) models are displayed. Both models still achieve reasonably high AUCs (mostly in the 0.93–0.99 range), though slightly lower compared to EfficientNet models. A noticeable decline in performance is seen for "Anger" (VGG19: 0.95) and "Sadness" (VGG19: 0.93), suggesting these emotions were harder to classify accurately. Additionally, the ROC curves exhibit more fluctuation, and the curves are less smooth, which aligns with the models' noted overfitting during training. This performance inconsistency suggests that while VGG models can still capture emotional distinctions, their generalization ability is weaker.

![Dashboard](https://github.com/ShaikhBorhanUddin/Facial-Emotion-Recognition/blob/main/Image/roc_resnet.png?raw=true)  

In the last three image, ConvNeXtBase, ResNet50V2, and ResNet101V2 are analyzed. These models deliver robust classification results, comparable to EfficientNetB2–B5, with AUCs consistently between 0.95 and 1.00. ConvNeXtBase (left image) stands out with a perfect AUC of 1.00 for both "Happiness" and "Neutral," while maintaining strong performance across the other classes. The ROC curves of ResNet101V2 (right image) and ResNet50V2 (middle image) are closely packed with high slopes, indicating minimal class confusion and strong predictive power. Unlike VGG models, these architectures exhibit better generalization, as supported by both their AUC scores and the smoothness and steepness of the ROC curves.

## 📉 Confusion Matrix  

Confusion Matrix of all models discussed in ROC tab are discussed in this section.  

![Dashboard](https://github.com/ShaikhBorhanUddin/Facial-Emotion-Recognition/blob/main/Image/cm_efficient.png?raw=true)  

The images displayed above show confusion matrices for EfficientNet B2, B3, B4, and B5 models (left to right). Across all models, Fear, Happiness, and Neutral consistently show strong classification performance, with high true positive counts—Fear especially stands out with values ranging from 99 to 104. However, Surprise and Sadness frequently get misclassified, particularly into Fear and Neutral, which suggests that the visual features of these emotions may overlap and pose challenges even for deeper networks. A trend of performance improvement is visible as we move from B2 to B5: the number of true positives increases slightly (e.g., Anger improves from 71 to 79, Disgust from 75 to 80), and off-diagonal confusion for classes like Sadness and Surprise becomes more controlled. This indicates that deeper EfficientNet variants (B4 and B5) are better at capturing fine-grained emotional differences.  

![Dashboard](https://github.com/ShaikhBorhanUddin/Facial-Emotion-Recognition/blob/main/Image/cm_vgg.png?raw=true)  

The second images contain confusion matrices for VGG16 (left) and VGG19 (right). The differences in performance are stark—VGG19 outperforms VGG16, particularly in classifying Fear, where true positives jump from 72 to 105. VGG16 struggles notably with Sadness and Surprise, misclassifying Sadness as Neutral (25 times) and Surprise as Fear (8 times). These high misclassification rates point to its limited depth compared to VGG19. VGG19 demonstrates improved balance, especially for difficult classes like Sadness and Disgust, though confusion with overlapping expressions such as Fear–Surprise and Neutral–Sadness still exists. Overall, while both VGG models lag behind EfficientNet B4/B5 in overall robustness, VGG19 shows significant gains over VGG16, benefiting from its deeper convolutional layers and enhanced feature extraction.  

![Dashboard](https://github.com/ShaikhBorhanUddin/Facial-Emotion-Recognition/blob/main/Image/cm_resnet_mod.png?raw=true)  

The confusion matrices show that the ConvNeXtBase model (left) achieves the highest accuracy, with strong predictions across most classes—e.g., 103 correct for Fear, 93 for Happiness, and 86 for Neutral, with minimal confusion (only 8 Neutral misclassified as Sadness). In contrast, the ResNet50V2 model (middle) struggles with overlapping emotions: only 25 Neutral samples are correctly predicted, while 31 and 27 are wrongly classified as Disgust and Sadness, respectively; Surprise also suffers, with only 75 correct and 16 misclassified as Fear. The ResNet101V2 model (right) improves on this with 46 correct Neutral predictions and stronger results for Fear (111 correct), but still misclassifies 36 Neutral as Sadness and 27 Surprise as Fear. Overall, ConvNeXtBase clearly outperforms both ResNet models in both precision and consistency across emotion categories.

## 🖼️ Visualizations  

## 🌍 Practical Applications  

Facial Emotion Recognition (FER) technology is essential for advancing human-centered AI across various fields, including healthcare, education, retail, automotive, entertainment, and security. In **healthcare**, FER supports therapists by continuously monitoring a patient's emotional state during virtual sessions for timely interventions. In **education**, it enhances e-learning by identifying student confusion or disengagement in real time, allowing for adaptive content delivery. In **retail and customer service**, emotion recognition helps analyze customer sentiment during interactions, improving service and product recommendations. The **automotive** industry benefits from FER through driver monitoring systems that detect drowsiness, frustration, or distraction, enhancing road safety. Furthermore, in **entertainment**, FER creates responsive gaming and virtual reality environments based on player emotions. Lastly, it improves **security and surveillance** by detecting unusual or stress-induced facial expressions in public or high-risk areas, enabling proactive behavioral analysis.  

## 🔧 Tools & Technology  

`Python` `tensorFlow` `Keras` `EfficientNet` `VGG` `ResNet` `ConvNeXt` `A100` `Trillium`  
This Facial Emotion Recognition project was implemented primarily using Python, with TensorFlow and Keras serving as the core deep learning frameworks. These libraries provided the foundation for building, training, fine-tuning, and evaluating various state-of-the-art convolutional neural network architectures including EfficientNet, VGG, ResNet, and ConvNeXt. For training efficiency and scalability, most models were trained using Google Colab Pro with NVIDIA A100 GPUs, while the modified EfficientNetB2 and B3 models were trained on v5e Trillium TPUs, ensuring faster computations and reduced training time.

`Matplotlib` `seaborn` `GradCAM`  
In addition to model development, the project employed NumPy and pandas for data manipulation and preprocessing. Visualization and interpretability were emphasized through the use of Matplotlib, Seaborn, and Grad-CAM techniques (including Grad-CAM++) to highlight the key regions influencing model predictions. These techniques aided in gaining insights into model decision-making, particularly for misclassified emotion classes.

`Git` `GitHub`  
Version control was managed using Git, with all development hosted on GitHub, ensuring reproducibility and collaborative workflow. The Jupyter Notebook format was used throughout for experiment tracking, result logging, and visual analysis. Overall, the technology stack was designed to balance ease of development, training speed, and interpretability of results.  

## 🚧 Future Improvements  

[FERV39k](https://github.com/wangyanckxx/FERV39k)  

[AffectNet](https://www.kaggle.com/datasets/mstjebashazida/affectnet)  

## ⚠️ Limitations  

Emotion is influenced by more than just facial expressions. Sound, movement, body language, the environment, objects, and psychological factors all significantly contribute to how we interpret emotions.   

![Dashboard](https://github.com/ShaikhBorhanUddin/Facial-Emotion-Recognition/blob/main/Image/body_language.png?raw=true)
**Figure**: Example of hand gesture, surrounding environment or body language not considered in the dataset  

For example, If we focus only on the facial expression in the image above (woman with salad bowl) ignoring the contextual cues like the salad bowl and fork, the emotion displayed can easily be misinterpreted as pain or frustration rather than disgust for salad. However, the models experimented in this study focuses solely on front-facing facial features within a controlled setting, which may limit the overall understanding of emotional expressions.

![Dashboard](https://github.com/ShaikhBorhanUddin/Facial-Emotion-Recognition/blob/main/Image/sideway_view.png?raw=true)
**Figure**: Sideway view images also not included in the model training  

The dataset does not fully align with [Plutchik's wheel of emotions](https://github.com/ShaikhBorhanUddin/Facial-Emotion-Recognition/blob/main/Image/wheel%20of%20emoion.png) , making it impossible to train models for subtle or blended emotions such as sarcasm, enthusiasm, or embarrassment.

![Dashboard](https://github.com/ShaikhBorhanUddin/Facial-Emotion-Recognition/blob/main/Image/blended_emotions_mod.png?raw=true)
**Figure**: Example of blended or ambiguous emotions  

Due to hardware limitations, experimentation with larger datasets or more complex models (like EfficientNetB7, ViT-base, or DenseNet201) was not feasible.  

![Dashboard](https://github.com/ShaikhBorhanUddin/Facial-Emotion-Recognition/blob/main/Image/resource_usage.png?raw=true)  

## 🔗 References  

## 📄 License

## 🙋 Contact  


