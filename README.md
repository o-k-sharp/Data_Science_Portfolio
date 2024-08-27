# Successful Game Features Problem

## Research Question: 
### Which game features most significantly influence success on Steam and how can developers leverage these features to enhance a game’s performance?

## 1: Executive Summary

The project used Kaggle data to identify factors influencing Steam game success, applying a Random Forest algorithm after thorough data preprocessing. 
Initial results showed high accuracy but revealed issues with low recall and F1 scores, indicating that the model struggled to identify successful 
games effectively.

Hyperparameter tuning improved model performance, resulting in a final model with significantly enhanced recall and F1 scores. The final model 
demonstrated superior performance in predicting game success, with increased capability to correctly identify successful games compared to the 
initial model. The Random Forest model identified key predictors of video game success, such as genre and user ratings, which can directly inform 
game development and marketing strategies. These insights suggest that developers should prioritise optimising user experience in specific genres to 
maximise game success

## 2: Introduction & Project Background
With the exponential growth of the video game industry, as highlighted by Bond and Beale (2009, p.418), it is becoming increasingly challenging for 
individual games to distinguish themselves in a crowded market. This project aims to employ a Random Forest algorithm to analyse data from Steam, a 
leading digital distribution platform for video games, to identify the key features that contribute to a game’s success. Understanding these attributes is 
crucial for stakeholders, including game developers, artists, and marketers, as it provides actionable insights that can inform the development and promotion 
of future games. A ”great quality of the random forest algorithm is that it is very easy to measure the relative importance of each feature on the 
prediction” (Donges and Whitfield, 2024). The Random Forest algorithm was chosen for its robustness and ability to model complex feature interactions. 
This method suits the gaming industry’s non-linear factor interactions.

## 3: Data Collection & Preparation
In this project, Python was utilised for all stages of the Extract, Transform, Load (ETL) pipeline and subsequent analysis. Python was selected over R due to 
its more comprehensible code structure, as highlighted by Sudhakar (2018), although R has the advantage in visualisations due to ”many packages that enhance 
the various techniques of doing things” (Sudhakar,2018). Data for this study was obtained from two comprehensive datasets available on 
Kaggle (Bathwal, 2024; Whim, 2023), which included key variables such as app ID (used for dataset integration), developer, publisher, discounted price, rating, 
and genre. As illustrated in Figure 1, all data transformation, analysis, and visualisation tasks were executed within the Python environment, employing Pandas 
for data manipulation, Scikit-learn for data analysis and Matplotlib and Seaborn for visualisation purposes.


#### Steam Data ETL Pipeline image
![ScreenShot: ETL Pipeline](assets/Game Analysis Images/Steam Data ETL Pipline.jpg)
* Figure 1: Steam Data ELT Pipeline *

Challenges included handling non-numeric variables, currency conversion, missing values, and special characters. These issues were resolved during the data 
transformation stage, as depicted in Figure 2.

#### Data Transformation steps image

An example of the code addressing one of these challenges is presented in Figure 3. This code demonstrates the conversion of non-numeric categorical variables 
into a format compatible with Scikit-learn in Python. For categorical variables with high cardinality, frequency encoding was employed to maintain model efficiency 
by ”considering the frequency of a specific category’s appearance, rather than solely relying on the categorical value of the attribute” (Hosni, 2023, p.463.). 
Alternative methods like target encoding were considered but deemed less appropriate due to the potential leakage of information from the target variable as 
outlined by Trevisan (2022).

#### Pyhton Categorical encoding image

## 4: Data Analysis
Before conducting any formal analysis, exploratory data analysis (EDA) was performed. This process involved examining the summary statistics and basic information 
of the dataset, as well as generating scatter plots, heat maps, box plots, and histograms to explore relationships between features. An example of this process is
illustrated in Figure 4.

#### Box Plot Image

#### Histogram Image

The rating distribution histogram in Figure 5 established the success metric for the model. Given the left-skewed nature of this distribution, it was determined 
that a game would be classified as successful if it achieved a rating of 85 or higher. The metric was constructed using the code shown in Figure 6, with all 
multicollinear variables being excluded. The removal of multicollinear variables is a well-regarded method for addressing multicollinearity, as discussed by 
Paul (2006).

#### success metric code Image

A random forest model was chosen to identify the key determinants of success due to its robust performance and high accuracy, as well as its capacity to capture 
non-linear relationships. Ali et al. (2021,p.272) assert that ”it can be concluded that the Random Forest achieves increased classification performance and yields 
results that are accurate and precise in the cases of a large number of instances.” Alternative models like Gradient Boosting Machines (GBM) were dismissed due 
to high memory consumption as discussed by Natekin and Knoll (2013 p19) The model underwent extensive evaluation and cross-validation, with the results summarised
in Table 1.

#### initial Scores Table 1

The data presented in Table 1, along with the charts in Figure 7, indicate that the initial model may have been overfitting. Although the model’s accuracy is 
high—suggesting strong discriminatory power and an ability to distinguish between classes with a high degree of precision—the recall and F1 score are relatively 
low, indicating that the model struggles to accurately identify successful games. This issue is further highlighted by the confusion matrix, which shows that 
only 19% of successful games were correctly predicted as successes.

#### initial ROC and learning curves

Bonnet (2023) notes that ”the choice between precision and recall often hinges on the specific application and the associated costs of errors.” Given our 
objective to identify the key features that influence a game’s success, it is imperative that the model achieves higher recall and F1 scores than those observed 
in the initial model. Hyperparameter tuning, as shown in Figure 8, reduced false negatives.

#### feature tuning image

### 5:  Results

When comparing the performance of the initial model, as shown in Table 1 and Figure 7, with the performance of the final model, as depicted in Table 2 and 
Figure 9, it is evident that both models exhibit high accuracy (above 90%), with the initial model being 1% higher than the final model. However, due to the 
class imbalance in the dataset, where only approximately 10% of the games fall into the successful category, accuracy alone is an insufficient evaluation
metric, as argued by Goyal (2021).

#### final model performance table & curve charts

The final model demonstrates a significantly higher recall and F1 score, which is evident in the confusion matrix presented in Figure 9. In this matrix, 
49% of the successful games are correctly predicted as successes, representing a 30% improvement over the original model. In the context of the research 
question addressed in this project, the final model performs better. Although its precision is lower, the significantly higher recall and F1 score indicate
that it is more effective at identifying successful games. For the purpose of predicting which features influence game success on Steam, the final model 
is superior, as it is more adept at identifying a broader range of successful games (higher recall) while maintaining a balanced trade-off between precision 
and recall (higher F1 score). Key success features include player peaks, current player count, store genres, discount pricing, and release day. This suggests 
that developers should focus on improving user engagement within popular genres. These findings are illustrated in Figure 10.

#### feature importance chart

### 6:  Recommendations

With 90% accuracy and a strong true positive rate (Figure 11), the model reliably identifies influential features in a game’s success. However, further 
analysis is warranted to explore the specific values of these features that contribute most significantly to success. For instance, investigating whether 
releasing a game at the beginning of the month rather than in the middle correlates with higher success rates could provide valuable insights.

#### Final ROC

Despite the improvements in recall and F1 scores in the final Random Forest Classification model compared to the initial model, these metrics remain 
relatively low, just below 0.5. This indicates that the model may benefit from additional tuning and feature engineering to enhance the accuracy of 
identifying successful games.

Expanding the dataset to include additional variables, such as developer team size, levels of independence, and development challenges, could offer a 
more comprehensive understanding of the factors driving game success. Additionally, incorporating social media sentiment analysis could provide real-time 
insights into player perceptions, enabling developers to respond more quickly to market trends than the current data allows. This broader analysis would provide
game developers and stakeholders with deeper insights into the critical features influencing a game’s performance.

Zhao et al .(2018, p407) stated that ‘The ensemble nature of the model helps random forests outperform any individual decision tree. However, it also leads 
to a poor model interpretability, which significantly hinders the model from being used in fields that require transparent and explainable predictions’. 
Therefore exploring alternative models, such as Linear Discriminant Analysis or Decision trees could be advantageous due to the increased transparency offered, 
which would help build greater trust among stakeholders by providing clearer explanations of how predictions are made, addressing the inherent opacity of Random Forest
models.


# Images
![give your image a name](image path e.g. /assets/image_name.png)

# URl Links
[Mark Down Sheet](https://www.markdownguide.org/cheat-sheet/)
