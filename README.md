# AI-Framework
Cardiovascular diseases (CVD) require accurate early prediction. This study proposes an XAI-based model using clinical factors like age, blood pressure, and cholesterol. With preprocessing, SMOTE-Tomek, and ML models, it achieves strong ROC-AUC and F1-score, while SHAP and LIME improve interpretability.
1 Introduction
The incidence of cardiovascular diseases (CVDs) is still high globally, neces-
sitating the development of a reliable risk prediction tool. Although ML models
perform better in terms of accuracy and prediction, issues such as interpretability
and data imbalance hinder their implementation. Statistical modeling is easier
to understand but cannot detect nonlinear patterns among the risk factors [3,4].
AI/ML advancements have improved prediction capability, but most of the mod-
els are opaque to users due to insufficient explainability [10,14,18].
Current solutions usually emphasize one aspect out of two – either achieving
higher predictive accuracy or greater model interpretability. Problems like class
imbalance, calibration problems, and conflicting explanations are still underex-
plored, indicating the necessity of an all-in-one approach. This paper introduces
an end-to-end solution for explainable AI for predicting cardiovascular diseases
from clinical and lifestyle factors. Contributions are:
1. Framework integrating preprocessing, feature selection, SMOTE Tomek bal-
ancing, ensemble modeling, and explainability methods.
2. Ensemble modeling using stacking, weighted, and cost sensitive methods for
better performance in data imbalance scenarios.
3. Proposes a new Explainability Stability Index (ESI) that compares the sta-
bility of SHAP and LIME explainers.
4. Threshold optimization based on F1 score and Isotonic calibration for higher
reliability.
5. Evaluation with multiple metrics, ablation study, and explainability analysis.
Structure of the remaining paper: Section 2 covers related works, Section 3
covers data set details, Section 4 discusses the methodology, and Sections 5 to 7
discuss experiments, results, and analysis.

2 Related Work
The subsequent sections provide brief reviews of past research conducted on
heart disease prediction based on machine learning with regard to three key
issues, which are the methodologies employed to create predictive models, en-
semble modeling, and the use of interpretable machine learning.

2.1 Machine Learning for CVD Prediction
AI models have been extensively applied to CVD prediction because of the
ability to identify intricate associations between various clinical and lifestyle vari-
ables [14]. Traditional models like logistic regression, which are easy to interpret
but do not account for nonlinear relationships, have been the starting point in
previous researches [3,4]. With the development of computers, new algorithms
can be developed and utilized to improve the prediction accuracy, such as de-
cision tree, random forest, and SVM [5,6]. The combination of these models,
known as ensemble learning, can lead to further improvements [9,10]. Neverthe-
less, most models only consider accuracy and ignore other problems.
Title Suppressed Due to Excessive Length 3

2.2 Ensemble and Hybrid Models
The advantage of ensemble learning lies in its ability to make use of more
than one base learner in order to minimize bias and variance [10]. Methods such
as bagging, boosting, and stacking have been utilized in the field of healthcare,
where models like random forests and gradient boosting have proven to be ef-
ficient when it comes to discovering complex relationships within CVDs [5,7].
Nonetheless, the problem with current methods is that they use one learning
approach only and may be inadequate for dealing with heterogeneous data.

2.3 Explainable AI in Healthcare
The increasing applications of machine learning in the healthcare field have
created the need for transparency and explainability [13]. XAI techniques provide
insight into the decisions made by the models and make them more transparent
[18]. Some of the most popular explainability tools include SHAP and LIME.
SHAP uses the concept of feature importance to provide global and local ex-
planations, while LIME focuses only on the local explanation of individual data
points [16].

2.4 Research Gap
Although there have been some improvements in machine learning approaches
for predicting cardiovascular diseases, several important issues have not yet
been addressed. In most researches, predictive performance is favored over in-
terpretability, although interpretability is essential for clinical applications [18].
Moreover, there is inadequate tuning and calibration within hybrid models [9,10,13].
The use of interpretability tools like SHAP and LIME is usually analyzed in iso-
lation, and their stability has not been adequately considered [16,17].

3 Dataset Description
This section discusses the dataset used, including details on its source, fea-
tures, and pre-processing methods. The dataset was collected using the National
Health and Nutrition Examination Survey (NHANES). It is an open-source
dataset from the CDC [2] available at: NHANES Dataset.
This data set comprises demographics, clinical indicators (such as blood pres-
sure and cholesterol levels), and behavior attributes including smoking, alcohol
intake, and physical exercise habits. In preprocessing, missing values are imputed
using the median approach, categorical data are coded, and all invalid observa-
tions are discarded from the analysis. The resulting data set includes clinical and
behavioral variables [14], with feature engineering done to represent nonlinear
relationships among features. The target is binary, where hypertension occurs
if systolic blood pressure exceeds 140 mmHg or diastolic blood pressure exceeds
90 mmHg.

4 Proposed Methodology
This section presents the proposed explainable AI framework for cardiovascu-
lar disease prediction. The approach integrates preprocessing, feature engineer-
ing, hybrid modeling, optimization, and explainability into a unified pipeline.
4 Authors Suppressed Due to Excessive Length.

4.1 Framework Architecture
Multi-source data is brought together into one comprehensive database that
encompasses both clinical and lifestyle factors. This database is then divided
into two parts – the training set and the testing set – using stratified sampling
to ensure that the distribution of the classes is maintained. The model uses a
pipeline approach that comprises several stages. The overall architecture of the
proposed framework is shown in Figure 1. The pipeline integrates preprocessing,
feature engineering, hybrid ensemble modeling, and explainability mechanisms
to ensure accurate and interpretable cardiovascular risk prediction.
Fig. 1. Overview of the proposed explainable AI framework for cardiovascular disease
risk prediction, illustrating the complete pipeline from data preprocessing to hybrid
modeling, optimization, and explainability analysis.

4.2 Feature Engineering
Feature engineering is done in order to account for the non-linear associations
between clinical and behavioral factors. This entails interaction terms, ratios,
polynomials, and aggregations of lifestyle factors.

4.3 Machine Learning Models
Several algorithms are used to model various data distributions. Logistic
Regression acts as a baseline algorithm [4], whereas Random Forest and SVM
learn non-linear dependencies [5,6]. LightGBM is used due to its computational
efficiency and high accuracy on tabular data. Parameters are tuned using Grid-
SearchCV, and imbalanced classes are dealt with using class weights and sam-
pling methods.

4.4 Hybrid Ensemble Modeling
For enhancing accuracy and robustness, hybrid ensembling is employed. In
weighted models, predictions are made using weights generated by performance
metrics [10], while in stacking, predictions are improved using a meta-learner
[9]. Cost-sensitive learning can be used for improving classification of minority
classes.
Title Suppressed Due to Excessive Length 5

4.5 Optimization and Explainability
Optimization of threshold is done through choosing the cut-off point for
probability based on maximum F1-score [14]. Calibration of probability using
isotonic regression enhances prediction accuracy. Explainability is achieved using
SHAP and LIME, which offer global and local explanations [16,17]. To measure
consistency between these techniques, Explainability Stability Index (ESI) can
be used, calculated as:

ESI= αS+ βJ+ γ(1− Rd)

where S denotes Spearman rank correlation, J represents Jaccard similar-
ity, and Rd is the normalized rank difference. This metric ensures reliability of
explanations in clinical decision-making.

4.6 Experimental Setup
The dataset is partitioned into training and testing datasets using stratified
sampling to ensure equal distribution of classes between them. Data prepro-
cessing is performed only on the training set to avoid data leakage. Accuracy,
precision, recall, F1-score, and ROC-AUC scores are used as evaluation metrics
[14]. Taking into account the class imbalance problem, F1-score is given a special
emphasis, and the cutoff point at which it is maximized is chosen. Calibration
of probabilities is estimated via calibration curves. The model is developed us-
ing Python programming language with the use of Scikit-learn, LightGBM, and
Imbalanced-learn libraries and Pandas and NumPy packages for data prepro-
cessing. Model explainability is carried out with SHAP and LIME [16]
5 Results and Performance Analysis
This section presents the performance of individual and hybrid models, along
with ROC and calibration analysis to evaluate classification capability and prob-
ability reliability.

5.1 Model Performance
As shown in Table 3, each model performs differently. The Logistic Regression
is easy to interpret but does not detect any non-linearities. The Random Forest
improves its performance through ensemble learning, whereas the SVM model
performs moderately well. The LightGBM gives the best performance, providing
the highest accuracy and ROC-AUC values, respectively, at 0.9776 and 0.9951.
Model Acc. Sens. Spec. AUC Prec. 
F1
LightGBM 0.9776 0.9513 0.9847 0.9951 0.9430 0.9471
RandomForest 0.9338 0.9115 0.9398 0.9701 0.8016 0.8530
SVM 0.8192 0.6814 0.8560 0.8563 0.5580 0.6135
LogisticReg. 0.8267 0.6018 0.8867 0.8424 0.5862 0.5939
Table 1. Evaluation of Individual Models
Hybrid approaches further enhance efficiency through the integration of com-
plimentary capabilities. According to Table 4, the best accuracy (0.9804) and
6 Authors Suppressed Due to Excessive Length
F1-score (0.9532) were obtained from Weighted Hybrid and Cost-Sensitive ap-
proaches, whereas the best ROC-AUC (0.9957) was obtained from Stacking.
Model Accuracy Sensitivity Specificity ROC-AUC Precision F1-score
Stacking 0.974837 0.964602 0.977568 0.995659 0.919831 0.941685
Weighted Hybrid 0.980429 0.946903 0.989374 0.991673 0.959641 0.953229
Cost-Sensitive 0.980429 0.946903 0.989374 0.991594 0.959641 0.953229
Table 2. Evaluation of Hybrid Models

5.2 ROC and Calibration Analysis
It can be seen from the graphs below (Figures 1) that all the hybrids have a
good performance in terms of classification and their ROC curves are consider-
ably high from the random line.
Fig. 2. ROC Curve for Hybrid Models
The results of the calibration plot (Figure 2) show that the proposed frame-
work offers reliable probability estimation. All of the models (stacking, weighted,
and cost-sensitive) follow closely the ideal line of calibration, although the latter
one demonstrates some deviations.
Fig. 3. Calibration Curve for Hybrid Models
Title Suppressed Due to Excessive Length 7
6 Ablation and Explainability Analysis
In this part, the contributions made by critical parts of the proposed frame-
work are evaluated, and its interpretability is analyzed. From the ablation study,
it can be observed that hybrid model without excluding any critical factors gives
the best results in terms of ROC-AUC score (ROC-AUC = 0.9956). This indi-
cates that the framework used for hybrid model formulation works very well.
Without feature engineering, the performance degrades significantly (ROC-AUC
= 0.7027). It can also be seen that hybrid models perform better than the best
standalone model. In terms of interpretability, SHAP provides a global expla-
nation while LIME provides a local explanation to the model and identifies
important clinical features such as blood pressure and age.

Comparison Spearman (S) Kendall (τ ) Jaccard (J) ESI
SHAP vs LIME 0.82 0.67 0.82 0.77
Top-10 Features 0.85 0.73 0.82 0.80
Table 3. Evaluation of Explainability Consistency
The high ESI values indicate strong consistency between explanation meth-
ods, reinforcing the reliability and trustworthiness of the proposed framework.
7 Discussion and Conclusion
The findings show that hybrid ensembles consistently outperform individ-
ual algorithms in terms of all performance measures. Feature generation and
addressing imbalanced classes improve the effectiveness of the models, whereas
stacking and cost-sensitive learning increase robustness. SHAP and LIME high-
light important variables like blood pressure and age, providing interpretability
of the model. Nevertheless, there are some limitations associated with this ap-
proach. First, the machine learning algorithm is trained using only one data set,
which might affect the ability of the model to generalize well. Second, the use
of explainability approaches adds computational costs, and the settings of ESI
might impact the outcomes.
Future Work: Future research will focus on validation using larger and
diverse datasets, as well as integrating deep learning and temporal health data
to further improve predictive performance.

References
1. Talukder, M. A., Talaat, A. S., Kazi, M., & Khraisat, A. (2025). XAI-HD: an
explainable artificial intelligence framework for heart disease detection. Artificial
Intelligence Review, 58(12), 1-78.
2. Talaat, F. M., Elnaggar, A. R., Shaban, W. M., Shehata, M., & Elhosseini, M.
(2024). CardioRiskNet: a hybrid AI-based model for explainable risk prediction
and prognosis in cardiovascular disease. Bioengineering, 11(8), 822.
3. Kiran, I., Shahzad, A., Ur, S., Alhussein, M., Aslam, S., & Aurangzeb, K. (2025).
An AI-enabled framework for transparency and interpretability in cardiovascular
disease risk prediction. Computers, Materials, & Continua, 82(3), 5057.
4. Shah, P., Shukla, M., Dholakia, N. H., & Gupta, H. (2025). Predicting cardiovas-
cular risk with hybrid ensemble learning and explainable ai: P. shah et al. Scientific
8 Authors Suppressed Due to Excessive Length
Reports, 15(1), 17927.
5. Bilal, A., Alzahrani, A., Almohammadi, K., Saleem, M., Farooq, M. S., & Sarwar,
R. (2025). Explainable AI-driven intelligent system for precision forecasting in
cardiovascular disease. Frontiers in Medicine, 12, 1596335.
6. Sushmitha, GLND, & Utukuru, S. (2025). Age-based disease prediction and health
monitoring: integrating explainable AI and deep learning techniques. Iran Journal
of Computer Science , 8 (2), 393-402.
7. Adelusi, BS, Osamika, D., Kelvin-Agwu, MC, Mustapha, AY, Forkuo, AY, &
Ikhalea, N. (2025). A machine learning-driven predictive framework for early de-
tection and prevention of cardiovascular diseases in US healthcare systems. Engi-
neering and Technology Journal , 10 (5), 4727-4751.
8. Singh, M., Kumar, A., Khanna, NN, Laird, JR, Nicolaides, A., Faa, G., ... &
Suri, JS (2024). Artificial intelligence for cardiovascular disease risk assessment in
personalized framework: a scoping review. EClinicalMedicine , 73 .
9. Wajid, I., Dan, L., & Wang, Q. (2025). Hybrid Ensemble Approaches for Car-
diovascular Disease Prediction: Leveraging Interpretable AI for Clinical Insight.
Intelligence-Based Medicine , 100297.
10. Tawfeek, M.A., Alrashdi, I., Alruwaili, M., & Allahem, H. (2025). Cardiovascu-
lar disease detection: A hybrid machine learning-AI framework for personalized
diagnosis and risk assessment. Plos one , 20 (10), e0335421.
11. Ganie, SM, Pramanik, PKD, & Zhao, Z. (2025). Ensemble learning with explain-
able AI for improved heart disease prediction based on multiple datasets. Scientific
reports , 15 (1), 13912.
12. Lane, H., Valko, M., Rath, S., Walker, M. D., Olson, M. L., & Kramer, S. (2025).
Harder to mend than to break?—counterfactual explainable artificial intelligence
for lifestyle medicine and heart disease prediction. Journal of Medical Artificial
Intelligence, 8, 3.
13. Waqar, M., Shahnawaz, M. B., Saleem, S., Dawood, H., Muhammad, U., & Da-
wood, H. (2025). Enhancing heart attack prediction: Feature identification from
multiparametric cardiac data using explainable AI. Algorithms, 18(6), 333.
14. Teshale, A. B., Htun, H. L., Vered, M., Owen, A. J., & Freak-Poli, R. (2024). A
systematic review of artificial intelligence models for time-to-event outcome applied
in cardiovascular disease risk prediction. Journal of medical systems, 48(1), 68.
15. Srinivasan, S. M., & Sharma, V. (2025). Applications of AI in cardiovascular dis-
ease detection—A review of the specific ways in which AI is being used to detect
and diagnose cardiovascular diseases. AI in Disease Detection: Advancements and
Applications, 123-146.
16. Xu, H., Zhu, T., Zhang, L., Zhou, W., & Yu, P. S. (2024). Update selective param-
eters: Federated machine unlearning based on model explanation. IEEE Transac-
tions on Big Data, 11(2), 524-539.
17. Donmez, T. B., Kutlu, M., Mansour, M., & Yildiz, M. Z. (2025). Explainable AI in
action: a comparative analysis of hypertension risk factors using SHAP and LIME.
Neural Computing and Applications, 37(5), 4053-4074.
18. Arrieta, A.B., et al.: Explainable Artificial Intelligence (XAI): Concepts, tax-
onomies, opportunities and challenges toward responsible AI. Information Fusion
(2023)
