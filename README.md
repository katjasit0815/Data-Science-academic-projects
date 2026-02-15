# Data Science academic projects
My works/projects from the ML course

## Технологический стек моих проектов
*   **Языки:** Python 
*   **Базовые библиотеки** Pandas, NumPy, Matplotlib, Seaborn
*   **Machine Learning:** Scikit-learn, Gradient Boosting (XGBoost/CatBoost/LGBM)
*   **Deep Learning:** PyTorch, TensorFlow, Keras 
*   **NLP & CV:** BERT, Hugging Face, Fine-tuning
*   **Оптимизация моделей:** Distillation, Quantization

---

##  Проекты

### 01. Анализ риска сердечных приступов (Heart Risk Attack)
*   **Задача:** Разведочный анализ данных (EDA) и построение моделей для предсказания риска сердечного приступа.
*   **Стек:** Python, Pandas, Matplotlib, KNN, Linear Regression.

### 02. Анализ тональности текста (Sentiment Analysis)
*   **Задача:** Классификация эмоциональной окраски текстов с использованием вероятностных подходов.
*   **Стек:** Scikit-learn, Naive Bayes, NLP preprocessing.

### 03. Определение мошеннических транзакций (Credit Card Fraud Detection)
*   **Задача:** Обнаружение аномалий и классификация транзакций в условиях сильно несбалансированных данных.
*   **Стек:** Scikit-learn, имбаланс-техники (SMOTE/Random Undersampling), метрики Precision-Recall.

### 04. Прогнозирование оттока клиентов банка (Bank Churners)
*   **Задача:** Предсказание вероятности ухода клиента из банка.
*   **Стек:** Gradient Boosting, Feature Engineering, классификация.

### 05. Классификация 3 классов животных (Animal Faces Classification)
*   **Задача:** Многоклассовая классификация изображений с применением стратегии Fine-tuning. 
*   **Стек:** Python, TensorFlow, Keras, EfficientNet, Image Processing (PIL), Scikit-learn (Label Encoding).
*   **Особенности**: Реализация конвейера обработки данных с использованием коллбэков (EarlyStopping, ReduceLROnPlateau) для оптимизации процесса обучения.

### 06. Распознавание эмоций (Face Expression Recognition)
*   **Задача:** Создание компактной и быстрой модели для распознавания эмоций с использованием методов сжатия.
*   **Стек:** TensorFlow, Keras (EfficientNet), Scikit-learn, PIL.
*   **Особенности**: Обработка несбалансированных выборок (Class Weights), дистилляция и квантизация модели ученика EfficientNetB0

### 07. Регрессия рейтинга отзывов на лекарства (Drug Review Rating Regression)
*   **Задача:** Глубокий анализ текстов и предсказание рейтинга (регрессия) на основе отзывов о препаратах.
*   **Стек:** PyTorch, Hugging Face Transformers (PUBMedBERT, DistilBERT).
*   **Особенности**: Оптимизация обучения: Mixed Precision (FP16/autocast) для экономии памяти, кастомные пайплайны данных (DataLoaders), Linear Schedule с прогревом (Warmup). Сложная предобработка текста (RegEx, HTML-очистка).
