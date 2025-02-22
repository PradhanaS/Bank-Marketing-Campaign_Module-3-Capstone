# Bank-Marketing-Campaign
Capstone Module 3
"""
**Dirancang Oleh:** Pradhana Satria - JCDS 2502

---

## A. Pemahaman Permasalahan Usaha

### Konteks

Sebuah perusahaan yang bergerak di perbankan ingin meluncurkan kampanye pemasaran berjudul "Bank Marketing Campaign" yang bertujuan untuk meningkatkan pemahaman tentang perilaku nasabah dalam melakukan deposit. Kampanye ini bertujuan untuk mengklasifikasikan nasabah yang melakukan deposit dan yang tidak melakukan deposit. Dengan informasi ini, bank berharap dapat meningkatkan strategi pemasarannya untuk menarik lebih banyak nasabah untuk melakukan deposit dan memperkuat hubungan dengan nasabah yang sudah ada.

### Pernyataan Masalah

Bank tersebut bertujuan untuk mengoptimalkan kampanye pemasaran "Bank Marketing Campaign" dengan mengklasifikasikan calon deposan dan non-deposan secara akurat. Tantangan utama adalah meminimalkan dua jenis kesalahan kritis:

- **False Positive (FP):** Di mana kampanye ditargetkan pada nasabah yang pada akhirnya tidak melakukan deposit. Hal ini mengakibatkan pemborosan sumber daya dan upaya pemasaran.
    - FP: Kampanye diberikan, tetapi tidak melakukan deposit

- **False Negative (FN):** Di mana kampanye gagal menargetkan nasabah yang sebenarnya akan melakukan deposit jika mereka didekati. Hal ini mengakibatkan hilangnya peluang bagi bank untuk memperluas basis deposannya.
    - FN: Kampanye tidak diberikan, tetapi melakukan deposit

Dengan mengidentifikasi kesalahan-kesalahan ini secara akurat, bank dapat memperbaiki strategi pemasarannya untuk memastikan bahwa sumber daya digunakan secara efektif, memaksimalkan return on investment (ROI), dan meningkatkan keberhasilan kampanye secara keseluruhan.

### Tujuan

Berdasarkan permasalahan tersebut, bank ingin memiliki kemampuan untuk memprediksi kemungkinan seorang nasabah akan melakukan deposit atau tidak, sehingga kampanye pemasaran dapat difokuskan pada nasabah yang berpotensi melakukan deposit. Bank juga ingin mengidentifikasi faktor-faktor yang mempengaruhi keputusan nasabah untuk melakukan deposit, sehingga dapat merancang strategi pemasaran yang lebih efektif dan personal dalam mendekati nasabah potensial, serta meningkatkan jumlah deposit secara keseluruhan.

### Pendekatan Analisis

Tujuan utama kita adalah menganalisis data untuk mengidentifikasi pola yang membedakan nasabah yang berpotensi melakukan deposit dari yang tidak. Proses ini melibatkan pengumpulan dan eksplorasi data, pra-pemrosesan, serta teknik rekayasa fitur untuk memastikan kualitas data yang optimal.

Selanjutnya, kita akan membangun model klasifikasi yang mampu memprediksi probabilitas seorang nasabah akan melakukan deposit atau tidak. Model ini akan menggunakan teknik resampling dan penyesuaian hyperparameter untuk mengatasi ketidakseimbangan data dan meningkatkan akurasi prediksi.

Dengan memanfaatkan hasil analisis ini, bank dapat merancang strategi pemasaran yang lebih efektif dan personal, yang tidak hanya meningkatkan jumlah deposit tetapi juga memperkuat hubungan dengan nasabah. Model prediksi yang dihasilkan akan membantu bank dalam mengoptimalkan alokasi sumber daya pemasaran dan memaksimalkan return on investment (ROI) dengan meminimalkan kesalahan False Positive (FP) dan False Negative (FN).

## B. Data Understanding

### Informasi pada Kolom Data

| Atribut | Tipe Data | Deskripsi |
| --- | --- | --- |
| age | Integer | Usia nasabah |
| job | Text | Jenis pekerjaan nasabah |
| balance | Integer | Saldo rekening nasabah |
| housing | Text | Pinjaman uang untuk rumah |
| loan | Text | Pinjaman uang |
| contact | Text | Media komunikasi yang digunakan untuk menghubungi nasabah |
| month | Text | Bulan terakhir ketika kontak dilakukan dengan nasabah |
| campaign | Integer | Jumlah total upaya kontak yang dilakukan selama kampanye |
| pdays | Integer | Jumlah hari sejak nasabah terakhir kali dihubungi dalam kampanye sebelumnya |
| poutcome | Text | Hasil dari kampanye pemasaran sebelumnya |
| deposit | Text | Nasabah melakukan deposit atau tidak |

### Proporsi Target

Proporsi deposit atau target terbilang stabil karena perbedaan antara kedua target klasifikasi hanya berbeda 5 persen, dan proses ini akan berkurang efeknya pada saat penggunaan resampling (None, NearMiss, SMOTE, Random Under Sampling, Random Over Sampling).

## C. Data Cleaning, Feature Selection, & Feature Engineering

### Pembersihan Data

- **Handling Missing Values**:
  - Checked for missing values and found none.
  - Identified and handled duplicate entries by removing 8 duplicate rows.

- **Handling Outliers**:
  - Analyzed numerical columns for outliers using box plots.
  - Decided to retain outliers as they were not extreme and could provide valuable information.

- **Feature `poutcome`**:
  - Analyzed the `poutcome` feature and identified 'unknown' values.
  - Replaced 'unknown' values with 'success' where `pdays` > 0 and `deposit` = 1.
  - Replaced 'other' with `NaN` to be imputed later.

- **Feature `job`**:
  - Replaced 'unknown' values in the `job` feature with `NaN` for imputation.

### Feature Selection

- **Initial Features**:
  - Selected initial features based on their relevance to the business problem.
  - Features included: `age`, `job`, `balance`, `housing`, `loan`, `contact`, `month`, `campaign`, `pdays`, `poutcome`, and `deposit`.

- **Target Variable**:
  - Converted the target variable `deposit` from 'yes'/'no' to 1/0 for binary classification.

### Feature Engineering

- **New Features**:
  - **`last_campaign`**: Created a feature indicating whether the customer received the last campaign based on `pdays`.
    ```python
    df['last_campaign'] = df['pdays'].apply(lambda x: 'yes' if x < 0 else 'no')
    ```

  - **`avg_balance_age_segment`**: Created a feature representing the average balance for each age segment.
    ```python
    df['age_segment'] = pd.qcut(df['age'], q=4, labels=['Segment_1', 'Segment_2', 'Segment_3', 'Segment_4'])
    average_balance_per_segment = df.groupby('age_segment')['balance'].mean().reset_index()
    average_balance_per_segment.columns = ['age_segment', 'avg_balance_age_segment']
    df = df.merge(average_balance_per_segment, on='age_segment', how='left')
    df.drop(columns=['age_segment'], inplace=True)
    ```

### Data Preparation

- **Defining X and y**:
  - Defined `X` as the feature set and `y` as the target variable.
    ```python
    X = df.drop(columns='deposit')
    y = df['deposit']
    ```

- **Train-Test Split**:
  - Split the data into training (85%) and testing (15%) sets using stratified sampling to maintain the target distribution.
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y)
    ```

### Preprocessing Pipeline

- **Imputation, Encoding, and Scaling**:
  - Created a preprocessing pipeline to handle missing values, encode categorical variables, and scale numerical features.
    ```python
    transformer = ColumnTransformer([
        ('impute_onehot', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), ['contact', 'poutcome', 'job', 'month', 'last_campaign']),
        ('impute_binary', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('binary', OrdinalEncoder())
        ]), ['housing', 'loan']),
        ('robust', RobustScaler(), ['age', 'balance', 'campaign', 'pdays', 'avg_balance_age_segment']),
        ('iterative', IterativeImputer(max_iter=10, random_state=RANDOM_SEED), ['age', 'balance'])
    ], remainder='passthrough')
    ```

## D. Modeling & Evaluation

### Model Selection

- **Models Considered**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors
  - Voting Classifier
  - Stacking Classifier
  - AdaBoost
  - Gradient Boosting
  - XGBoost

### Resampling Techniques

- **Techniques Used**:
  - SMOTE
  - NearMiss
  - Random Under Sampling
  - Random Over Sampling

### Hyperparameter Tuning

- **Optuna**:
  - Used Optuna for hyperparameter tuning to find the best combination of hyperparameters for the models.

### Evaluation Metrics

- **Metrics Used**:
  - Recall
  - Precision
  - F2-score
  - F3-score

### Model Performance

- **Without Resampling**:
  - Best Model: Logistic Regression
  - F2-score: 65.55%

- **With Resampling**:
  - Best Model: Stacking Classifier with NearMiss
  - Recall: 71.61%

## E. Conclusion & Recommendations

### Conclusion

- The model achieved an overall accuracy of 66%.
- There is an imbalance in precision and recall for class 0 and class 1 due to the focus on recall and threshold optimization.
- After threshold optimization, recall increased to around 83%, but precision and accuracy decreased, indicating a trade-off between recall and precision.

### Recommendations

- **Business**:
  - Use the Stacking Classifier with NearMiss and threshold optimization for better performance.
  - Regularly evaluate costs to ensure the model provides optimal financial benefits.
  - Continuously monitor and update the model to maintain accuracy and effectiveness.

- **Features**:
  - Include more features such as `Avg Order Value`, `Purchase Frequency`, `Time Spent in Platform`, `Pages Visited`, `Customer Loyalty Program`, and `Avg Order Value in Competitor Price`.

- **Model Optimization**:
  - Use advanced hyperparameter tuning with `GridSearchCV`.
  - Experiment with different models like SVM and ANN.
  - Explore other scoring methods beyond F2-score.

By following these recommendations, the bank can optimize its marketing campaigns, increase the number of customers making deposits, and maximize return on investment (ROI) while minimizing false positives and false negatives.
"""



# Bank Marketing Campaign

**Tujuan:** Capstone Modul 3

**Dirancang Oleh:** Pradhana Satria - JCDS 2502

---

## A. Pemahaman Permasalahan Usaha

### Konteks

Sebuah perusahaan yang bergerak di perbankan ingin meluncurkan kampanye pemasaran berjudul "Bank Marketing Campaign" yang bertujuan untuk meningkatkan pemahaman tentang perilaku nasabah dalam melakukan deposit. Kampanye ini bertujuan untuk mengklasifikasikan nasabah yang melakukan deposit dan yang tidak melakukan deposit. Dengan informasi ini, bank berharap dapat meningkatkan strategi pemasarannya untuk menarik lebih banyak nasabah untuk melakukan deposit dan memperkuat hubungan dengan nasabah yang sudah ada.

### Pernyataan Masalah

Bank tersebut bertujuan untuk mengoptimalkan kampanye pemasaran "Bank Marketing Campaign" dengan mengklasifikasikan calon deposan dan non-deposan secara akurat. Tantangan utama adalah meminimalkan dua jenis kesalahan kritis:

- **False Positive (FP):** Di mana kampanye ditargetkan pada nasabah yang pada akhirnya tidak melakukan deposit. Hal ini mengakibatkan pemborosan sumber daya dan upaya pemasaran.
    - FP: Kampanye diberikan, tetapi tidak melakukan deposit

- **False Negative (FN):** Di mana kampanye gagal menargetkan nasabah yang sebenarnya akan melakukan deposit jika mereka didekati. Hal ini mengakibatkan hilangnya peluang bagi bank untuk memperluas basis deposannya.
    - FN: Kampanye tidak diberikan, tetapi melakukan deposit

Dengan mengidentifikasi kesalahan-kesalahan ini secara akurat, bank dapat memperbaiki strategi pemasarannya untuk memastikan bahwa sumber daya digunakan secara efektif, memaksimalkan return on investment (ROI), dan meningkatkan keberhasilan kampanye secara keseluruhan.

### Tujuan

Berdasarkan permasalahan tersebut, bank ingin memiliki kemampuan untuk memprediksi kemungkinan seorang nasabah akan melakukan deposit atau tidak, sehingga kampanye pemasaran dapat difokuskan pada nasabah yang berpotensi melakukan deposit. Bank juga ingin mengidentifikasi faktor-faktor yang mempengaruhi keputusan nasabah untuk melakukan deposit, sehingga dapat merancang strategi pemasaran yang lebih efektif dan personal dalam mendekati nasabah potensial, serta meningkatkan jumlah deposit secara keseluruhan.

### Pendekatan Analisis

Tujuan utama kita adalah menganalisis data untuk mengidentifikasi pola yang membedakan nasabah yang berpotensi melakukan deposit dari yang tidak. Proses ini melibatkan pengumpulan dan eksplorasi data, pra-pemrosesan, serta teknik rekayasa fitur untuk memastikan kualitas data yang optimal.

Selanjutnya, kita akan membangun model klasifikasi yang mampu memprediksi probabilitas seorang nasabah akan melakukan deposit atau tidak. Model ini akan menggunakan teknik resampling dan penyesuaian hyperparameter untuk mengatasi ketidakseimbangan data dan meningkatkan akurasi prediksi.

Dengan memanfaatkan hasil analisis ini, bank dapat merancang strategi pemasaran yang lebih efektif dan personal, yang tidak hanya meningkatkan jumlah deposit tetapi juga memperkuat hubungan dengan nasabah. Model prediksi yang dihasilkan akan membantu bank dalam mengoptimalkan alokasi sumber daya pemasaran dan memaksimalkan return on investment (ROI) dengan meminimalkan kesalahan False Positive (FP) dan False Negative (FN).

## B. Data Understanding

### Informasi pada Kolom Data

| Atribut | Tipe Data | Deskripsi |
| --- | --- | --- |
| age | Integer | Usia nasabah |
| job | Text | Jenis pekerjaan nasabah |
| balance | Integer | Saldo rekening nasabah |
| housing | Text | Pinjaman uang untuk rumah |
| loan | Text | Pinjaman uang |
| contact | Text | Media komunikasi yang digunakan untuk menghubungi nasabah |
| month | Text | Bulan terakhir ketika kontak dilakukan dengan nasabah |
| campaign | Integer | Jumlah total upaya kontak yang dilakukan selama kampanye |
| pdays | Integer | Jumlah hari sejak nasabah terakhir kali dihubungi dalam kampanye sebelumnya |
| poutcome | Text | Hasil dari kampanye pemasaran sebelumnya |
| deposit | Text | Nasabah melakukan deposit atau tidak |

### Proporsi Target

Proporsi deposit atau target terbilang stabil karena perbedaan antara kedua target klasifikasi hanya berbeda 5 persen, dan proses ini akan berkurang efeknya pada saat penggunaan resampling (None, NearMiss, SMOTE, Random Under Sampling, Random Over Sampling).

## C. Data Cleaning, Feature Selection, & Feature Engineering

### Pembersihan Data

- **Handling Missing Values**:
  - Checked for missing values and found none.
  - Identified and handled duplicate entries by removing 8 duplicate rows.

- **Handling Outliers**:
  - Analyzed numerical columns for outliers using box plots.
  - Decided to retain outliers as they were not extreme and could provide valuable information.

- **Feature `poutcome`**:
  - Analyzed the `poutcome` feature and identified 'unknown' values.
  - Replaced 'unknown' values with 'success' where `pdays` > 0 and `deposit` = 1.
  - Replaced 'other' with `NaN` to be imputed later.

- **Feature `job`**:
  - Replaced 'unknown' values in the `job` feature with `NaN` for imputation.

### Feature Selection

- **Initial Features**:
  - Selected initial features based on their relevance to the business problem.
  - Features included: `age`, `job`, `balance`, `housing`, `loan`, `contact`, `month`, `campaign`, `pdays`, `poutcome`, and `deposit`.

- **Target Variable**:
  - Converted the target variable `deposit` from 'yes'/'no' to 1/0 for binary classification.

### Feature Engineering

- **New Features**:
  - **`last_campaign`**: Created a feature indicating whether the customer received the last campaign based on `pdays`.
    ```python
    df['last_campaign'] = df['pdays'].apply(lambda x: 'yes' if x < 0 else 'no')
    ```

  - **`avg_balance_age_segment`**: Created a feature representing the average balance for each age segment.
    ```python
    df['age_segment'] = pd.qcut(df['age'], q=4, labels=['Segment_1', 'Segment_2', 'Segment_3', 'Segment_4'])
    average_balance_per_segment = df.groupby('age_segment')['balance'].mean().reset_index()
    average_balance_per_segment.columns = ['age_segment', 'avg_balance_age_segment']
    df = df.merge(average_balance_per_segment, on='age_segment', how='left')
    df.drop(columns=['age_segment'], inplace=True)
    ```

### Data Preparation

- **Defining X and y**:
  - Defined `X` as the feature set and `y` as the target variable.
    ```python
    X = df.drop(columns='deposit')
    y = df['deposit']
    ```

- **Train-Test Split**:
  - Split the data into training (85%) and testing (15%) sets using stratified sampling to maintain the target distribution.
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y)
    ```

### Preprocessing Pipeline

- **Imputation, Encoding, and Scaling**:
  - Created a preprocessing pipeline to handle missing values, encode categorical variables, and scale numerical features.
    ```python
    transformer = ColumnTransformer([
        ('impute_onehot', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), ['contact', 'poutcome', 'job', 'month', 'last_campaign']),
        ('impute_binary', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('binary', OrdinalEncoder())
        ]), ['housing', 'loan']),
        ('robust', RobustScaler(), ['age', 'balance', 'campaign', 'pdays', 'avg_balance_age_segment']),
        ('iterative', IterativeImputer(max_iter=10, random_state=RANDOM_SEED), ['age', 'balance'])
    ], remainder='passthrough')
    ```

## D. Modeling & Evaluation

### Model Selection

- **Models Considered**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors
  - Voting Classifier
  - Stacking Classifier
  - AdaBoost
  - Gradient Boosting
  - XGBoost

### Resampling Techniques

- **Techniques Used**:
  - SMOTE
  - NearMiss
  - Random Under Sampling
  - Random Over Sampling

### Hyperparameter Tuning

- **Optuna**:
  - Used Optuna for hyperparameter tuning to find the best combination of hyperparameters for the models.

### Evaluation Metrics

- **Metrics Used**:
  - Recall
  - Precision
  - F2-score
  - F3-score

### Model Performance

- **Without Resampling**:
  - Best Model: Logistic Regression
  - F2-score: 65.55%

- **With Resampling**:
  - Best Model: Stacking Classifier with NearMiss
  - Recall: 71.61%

## E. Conclusion & Recommendations

### Conclusion

- The model achieved an overall accuracy of 66%.
- There is an imbalance in precision and recall for class 0 and class 1 due to the focus on recall and threshold optimization.
- After threshold optimization, recall increased to around 83%, but precision and accuracy decreased, indicating a trade-off between recall and precision.

### Recommendations

- **Business**:
  - Use the Stacking Classifier with NearMiss and threshold optimization for better performance.
  - Regularly evaluate costs to ensure the model provides optimal financial benefits.
  - Continuously monitor and update the model to maintain accuracy and effectiveness.

- **Features**:
  - Include more features such as `Avg Order Value`, `Purchase Frequency`, `Time Spent in Platform`, `Pages Visited`, `Customer Loyalty Program`, and `Avg Order Value in Competitor Price`.

- **Model Optimization**:
  - Use advanced hyperparameter tuning with `GridSearchCV`.
  - Experiment with different models like SVM and ANN.
  - Explore other scoring methods beyond F2-score.

By following these recommendations, the bank can optimize its marketing campaigns, increase the number of customers making deposits, and maximize return on investment (ROI) while minimizing false positives and false negatives.
