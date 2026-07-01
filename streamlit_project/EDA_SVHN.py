import os
import urllib.request
import scipy.io 
import numpy as np 
import streamlit as st 
import matplotlib.pyplot as plt 
import pandas as pd 
import altair as alt 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split, KFold, cross_val_score 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix 
import seaborn as sns 
import plotly.figure_factory as ff 
st.set_page_config(layout="wide")


@st.cache_data 
def download_and_load_svhn():
    urls = {
        "train": "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
        "test": "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
    }
    
    results = {}
    for name, url in urls.items():
        filename = f"{name}_32x32.mat"
        
        if not os.path.exists(filename):
            with st.spinner(f"Скачиваю {filename}..."):
                urllib.request.urlretrieve(url, filename)
        
        data = scipy.io.loadmat(filename)
        X = data['X']
        y = data['y'].flatten()
        y[y == 10] = 0
        X = X.transpose((3, 0, 1, 2))
        results[name] = (X, y)
        
    return results["train"][0], results["train"][1], results["test"][0], results["test"][1]

train_X, train_y, test_X, test_y = download_and_load_svhn()

st.markdown("<h1 style='text-align: center;'>Исследовательский анализ данных датасета SVHN</h1>", unsafe_allow_html=True)
st.write('')
st.write('')
st.write('')
col1, col2 = st.columns(2, gap='large')

# Экземпляры изображений разных классов
with col1:
    with st.container(border=True):
        st.markdown("<h3 style='text-align: center;'>Экземпляры изображений разных классов</h3>", unsafe_allow_html=True)
        cols = st.columns(10)
        samples_per_class = 5
        for class_index in range(10):
            with cols[class_index]:
                st.write(f"Class {class_index}")
        
                class_indices = np.where(train_y == class_index)[0][:samples_per_class]
        
                for example_index in class_indices:
                    image = train_X[example_index]
                    st.image(image.astype(np.uint8), width=65)              
                    
unique_classes, counts = np.unique(train_y, return_counts=True)
sorted_class_counts_df = pd.DataFrame({
    'Class': unique_classes.astype(str),
    'Count': counts
}).sort_values(by='Count',ascending=False)

#Гистограмма экземпляров разных классов меток
with col2:
    with st.container(border=True):
        st.markdown("<h3 style='text-align: center;'>Гистограмма экземпляров разных классов меток</h3>", unsafe_allow_html=True)
        st.write('')
        chart = alt.Chart(sorted_class_counts_df).mark_bar().encode(
            x=alt.X('Class:O', title='Класс', sort=sorted_class_counts_df['Count'].tolist()),
            y=alt.Y('Count:Q', title='Количество экземпляров'),
        ).properties(
            height=410,
            width=700,
        )
        st.altair_chart(chart)
    st.write('По данным гистограммы можно сделать вывод, что распределение экземпляров классов имеет экспоненциальный характер. Это, в свою очередь, говорит о **дисбалансе** классов в датасете.')

# Обучение KNN модели
st.write('')
st.write('')
st.markdown("<h2 style='text-align: center;'>Обучение KNN модели. Кросс-валидация</h2>", unsafe_allow_html=True)
st.write('')

train_X = train_X[:1000] 
train_y = train_y[:1000]
test_X = test_X[:100] 
test_y = test_y[:100] 

X_train_flattened = train_X.reshape(train_X.shape[0], -1) 
X_test_flattened = test_X.reshape(test_X.shape[0], -1)  

knn = KNeighborsClassifier(n_neighbors=2)
knn_model = knn.fit(X_train_flattened, train_y)
knn_predictions = knn.predict(X_test_flattened)

col1,col2,col3 = st.columns([1,2,1], gap='large')
with col2:
    selected_value = st.select_slider(
            "Выберите количество соседей k",
            options=[1, 2, 3, 5, 8, 10, 15, 20, 25, 50],
            value=2,
            key="select_k"
        )
    selected_value = st.session_state.select_k

col1, col2, col3 = st.columns([0.9, 2, 1.5], gap='large')

num_folds = 5
k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]

k_to_accuracy = {}
k_to_f1_ = {}

for k in k_choices:
    clf = KNeighborsClassifier(n_neighbors=k)

    accuracy = cross_val_score(clf, train_X.reshape(len(train_X), -1), train_y, cv=num_folds, scoring='accuracy')
    k_to_accuracy[k] = accuracy.mean()

    f1 = cross_val_score(clf, train_X.reshape(len(train_X), -1), train_y, cv=num_folds, scoring='f1_macro')
    k_to_f1_[k] = f1.mean()

results_df = pd.DataFrame({
    'k': list(k_to_accuracy.keys()),
    'accuracy mean': list(k_to_accuracy.values()),
    'f1-score mean': list(k_to_f1_.values())
})
with col1:
    st.write('')
    st.write('')
    st.write('**Результаты кросс-валидации**')
    st.table(results_df)

    best_k = max(k_to_accuracy, key=k_to_accuracy.get)
    st.write("Лучшее значение k:", best_k)

with col2:

    knn = KNeighborsClassifier(n_neighbors=selected_value)
    knn.fit(X_train_flattened, train_y) 

    predicted_labels = knn.predict(X_test_flattened)
    cm = confusion_matrix(test_y, predicted_labels)

    fig = ff.create_annotated_heatmap(
        z=cm,
        x=[f'Predicted {i}' for i in range(cm.shape[1])],
        y=[f'True {i}' for i in range(cm.shape[0])],
        colorscale='Blues'
    )
       
    fig.update_layout(
        title=f'Матрица путаницы (k={selected_value})',
        xaxis_title='Предсказанные метки',
        yaxis_title='Истинные метки',
        margin=dict(t=100) 
    )
    st.plotly_chart(fig)

with col3: 
    st.write('')
    st.write('')   
    st.write('')     
    predictions_df = pd.DataFrame(predicted_labels, columns=['Predicted Class'])

    histogram = alt.Chart(predictions_df).mark_bar().encode(
    x=alt.X('Predicted Class:O', title='Предсказанный класс'),
    y=alt.Y('count():Q', title='Частота'),
    ).properties(
    title='Гистограмма предсказаний',
    height=410,
    width=700,
    )
    st.altair_chart(histogram, use_container_width=True)


    # accuracy = accuracy_score(test_y, predicted_labels)
    # st.write(f'Accuracy: {accuracy:.2f}')
