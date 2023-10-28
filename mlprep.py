import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from sklearn import preprocessing
st.set_page_config(
    page_title="Data Preprocessing & ML",
    page_icon=":pencil2:",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.title("Simplified Data: Automate Data Preprocessing and Machine Learning Modeling")

uploaded_file = st.file_uploader("Upload CSV file:")


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.header("Exploratory Data Analysis")

    # Show the DataFrame
    st.subheader("Raw Data")
    st.dataframe(df)


    st.subheader("Number of Unique Values for Object Columns")
    nunique = df.select_dtypes(include='object').nunique()
    st.write(nunique)


    st.subheader("Percentage of Missing Values")
    check_missing = df.isnull().sum() * 100 / df.shape[0]
    missing = check_missing[check_missing > 0].sort_values(ascending=False)
    st.write(missing)

    st.subheader("Dropping Columns with >20% Missing Values")
    columns_to_remove = missing[missing > 20].index
    df = df.drop(columns=columns_to_remove)
    st.dataframe(df)


    st.subheader("Dropping Rows with Null Values in Object Columns")
    df = df.dropna(subset=df.select_dtypes(include=['object']).columns)
    st.dataframe(df)


    st.subheader("Remove Selected Columns")
    selected_columns = st.multiselect("Select columns to remove", df.columns)
    df.drop(columns=selected_columns, inplace=True)
    st.dataframe(df)


    st.subheader("Fill Null Values in Numeric Columns")
    select_method = st.selectbox('Fill null values with:', ("Mean", "Median"))
    numeric_columns = df.select_dtypes(include=['float', 'int'])
    unique_value_counts = df[numeric_columns.columns].nunique()
    columns_to_fill = unique_value_counts[unique_value_counts > 10].index

    if select_method == "Mean":
        df[columns_to_fill] = df[columns_to_fill].fillna(df[columns_to_fill].mean())
    elif select_method == "Median":
        df[columns_to_fill] = df[columns_to_fill].fillna(df[columns_to_fill].median())

    st.dataframe(df)


    st.subheader("Drop Rows with Null Values in Numeric Columns with <10 Unique Values")
    columns_to_drop_rows = unique_value_counts[unique_value_counts < 10].index
    df = df.dropna(subset=columns_to_drop_rows)
    st.dataframe(df)

    st.header("Label Encoding for Object Columns")
    for col in df.select_dtypes(include=['object']).columns:
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(df[col].unique())
        df[col] = label_encoder.transform(df[col])
        st.write(f"{col}: {df[col].unique()}")

    st.header("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.header("Machine Learning Modelling")


    select_method2 = st.selectbox('Select your Machine Learning Prediction Method', ("Classification", "Regressor"))

    if select_method2 == "Classification":
        select_method2a = st.selectbox('Select your Machine Learning Classification Algorithm', ("Decision Tree Classifier", "Random Forest Classifier"))
        if select_method2a == "Decision Tree Classifier":
            rs1 = st.selectbox('Random State', (0, 42))
            md1 = st.selectbox('Max Depth', (3, 4, 5, 6, 7, 8))
            msl1 = st.selectbox('Min Sample Leaf', (1, 2, 3, 4))
            mss1 = st.selectbox('Min Sample Split', (2, 3, 4))
            from sklearn.tree import DecisionTreeClassifier
            dtree = DecisionTreeClassifier(random_state=rs1, max_depth=md1, min_samples_leaf=msl1, min_samples_split=mss1, class_weight='balanced')
            dtree.fit(X_train, y_train)

            from sklearn.metrics import accuracy_score
            y_pred = dtree.predict(X_test)
            st.write("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score, log_loss
            st.write('F-1 Score : ',(f1_score(y_test, y_pred, average='micro')))
            st.write('Precision Score : ',(precision_score(y_test, y_pred, average='micro')))
            st.write('Recall Score : ',(recall_score(y_test, y_pred, average='micro')))
            st.write('Jaccard Score : ',(jaccard_score(y_test, y_pred, average='micro')))

            imp_df = pd.DataFrame({
                "Feature Name": X_train.columns,
                "Importance": dtree.feature_importances_
            })
            fi = imp_df.sort_values(by="Importance", ascending=False)

            fi2 = fi.head(10)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=fi2, x='Importance', y='Feature Name', ax=ax)
            ax.set_title('Top 10 Feature Importance Each Attributes (Deicision Tree Classifier)', fontsize=18)
            ax.set_xlabel('Importance', fontsize=16)
            ax.set_ylabel('Feature Name', fontsize=16)


            st.pyplot(fig)

            import shap
            explainer = shap.TreeExplainer(dtree)
            shap_values = explainer.shap_values(X_test)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_test, show=False)
            st.pyplot(fig)

            # compute SHAP values
            explainer = shap.TreeExplainer(dtree)
            shap_values = explainer.shap_values(X_test)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values[1], X_test.values, feature_names = X_test.columns)
            st.pyplot(fig)

            from sklearn.metrics import confusion_matrix            
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 5))
            sns.heatmap(data=cm,linewidths=.5, annot=True,  cmap = 'Blues')
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
            all_sample_title = 'Accuracy Score for Decision Tree: {0}'.format(dtree.score(X_test, y_test))
            plt.title(all_sample_title, size = 15)
            st.pyplot(fig)
        
        if select_method2a == "Random Forest Classifier":
            rs2 = st.selectbox('Random State', (0, 42))
            md2 = st.selectbox('Max Depth', (3, 4, 5, 6, 7, 8))
            mf2 = st.selectbox('Max Features', ('sqrt', 'log2', None))
            ne2 = st.selectbox('N Estimator', (100, 200))
            from sklearn.ensemble import RandomForestClassifier
            rfc = RandomForestClassifier(random_state=rs2, max_depth=md2, max_features=mf2, n_estimators=ne2, class_weight='balanced')
            rfc.fit(X_train, y_train)

            from sklearn.metrics import accuracy_score
            y_pred = rfc.predict(X_test)
            st.write("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score, log_loss
            st.write('F-1 Score : ',(f1_score(y_test, y_pred, average='micro')))
            st.write('Precision Score : ',(precision_score(y_test, y_pred, average='micro')))
            st.write('Recall Score : ',(recall_score(y_test, y_pred, average='micro')))
            st.write('Jaccard Score : ',(jaccard_score(y_test, y_pred, average='micro')))

            imp_df = pd.DataFrame({
                "Feature Name": X_train.columns,
                "Importance": rfc.feature_importances_
            })
            fi = imp_df.sort_values(by="Importance", ascending=False)

            fi2 = fi.head(10)
            # Create the barplot
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=fi2, x='Importance', y='Feature Name', ax=ax)
            ax.set_title('Top 10 Feature Importance Each Attributes (Random Forest Classifier)', fontsize=18)
            ax.set_xlabel('Importance', fontsize=16)
            ax.set_ylabel('Feature Name', fontsize=16)


            st.pyplot(fig)

            import shap
            explainer = shap.TreeExplainer(rfc)
            shap_values = explainer.shap_values(X_test)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_test, show=False)
            st.pyplot(fig)


            explainer = shap.TreeExplainer(rfc)
            shap_values = explainer.shap_values(X_test)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values[1], X_test.values, feature_names = X_test.columns)
            st.pyplot(fig)

            from sklearn.metrics import confusion_matrix            
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 5))
            sns.heatmap(data=cm,linewidths=.5, annot=True,  cmap = 'Blues')
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
            all_sample_title = 'Accuracy Score for Decision Tree: {0}'.format(rfc.score(X_test, y_test))
            plt.title(all_sample_title, size = 15)
            st.pyplot(fig)
        
    if select_method2 == "Regressor":
        select_method2a = st.selectbox('Select your Machine Learning Regressor Algorithm', ("Decision Tree Regressor", "Random Forest Regressor"))
        if select_method2a == "Decision Tree Regressor":
            rs1 = st.selectbox('Random State', (0, 42))
            md1 = st.selectbox('Max Depth', (3, 4, 5, 6, 7, 8))
            mf1 = st.selectbox('Max Features', ('sqrt', 'log2'))
            mss1 = st.selectbox('Min Sample Split', (2, 4, 6, 8))
            msl1 = st.selectbox('Min Sample Leaf', (1, 2, 3, 4))
            from sklearn.tree import DecisionTreeRegressor
            dtree = DecisionTreeRegressor(random_state=rs1, max_depth=md1, max_features=mf1, min_samples_leaf=msl1, min_samples_split=mss1)
            dtree.fit(X_train, y_train)

            from sklearn import metrics
            from sklearn.metrics import mean_absolute_percentage_error
            import math
            y_pred = dtree.predict(X_test)
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            r2 = metrics.r2_score(y_test, y_pred)
            rmse = math.sqrt(mse)

            st.write('MAE is {}'.format(mae))
            st.write('MAPE is {}'.format(mape))
            st.write('MSE is {}'.format(mse))
            st.write('R2 score is {}'.format(r2))
            st.write('RMSE score is {}'.format(rmse))

            imp_df = pd.DataFrame({
                "Feature Name": X_train.columns,
                "Importance": dtree.feature_importances_
            })
            fi = imp_df.sort_values(by="Importance", ascending=False)

            fi2 = fi.head(10)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=fi2, x='Importance', y='Feature Name', ax=ax)
            ax.set_title('Top 10 Feature Importance Each Attributes (Decision Tree Regressor)', fontsize=18)
            ax.set_xlabel('Importance', fontsize=16)
            ax.set_ylabel('Feature Name', fontsize=16)

            # Display the plot in Streamlit
            st.pyplot(fig)

            import shap
            explainer = shap.TreeExplainer(dtree)
            shap_values = explainer.shap_values(X_test)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_test)
            st.pyplot(fig)

            explainer = shap.Explainer(dtree, X_test)
            shap_values = explainer(X_test)
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0])
            st.pyplot(fig)
        
        if select_method2a == "Random Forest Regressor":
            rs1 = st.selectbox('Random State', (0, 42))
            md1 = st.selectbox('Max Depth', (3, 5, 7, 9))
            mf1 = st.selectbox('Max Features', ('log2', 'sqrt'))
            mss1 = st.selectbox('Min Sample Split', (2, 5, 10))
            msl1 = st.selectbox('Min Sample Leaf', (1, 2, 4))
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(random_state=rs1, max_depth=md1, min_samples_split=mss1, min_samples_leaf=msl1, max_features=mf1)
            rf.fit(X_train, y_train)

            from sklearn import metrics
            from sklearn.metrics import mean_absolute_percentage_error
            import math
            y_pred = rf.predict(X_test)
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            r2 = metrics.r2_score(y_test, y_pred)
            rmse = math.sqrt(mse)

            st.write('MAE is {}'.format(mae))
            st.write('MAPE is {}'.format(mape))
            st.write('MSE is {}'.format(mse))
            st.write('R2 score is {}'.format(r2))
            st.write('RMSE score is {}'.format(rmse))

            imp_df = pd.DataFrame({
                "Feature Name": X_train.columns,
                "Importance": rf.feature_importances_
            })
            fi = imp_df.sort_values(by="Importance", ascending=False)

            fi2 = fi.head(10)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=fi2, x='Importance', y='Feature Name', ax=ax)
            ax.set_title('Top 10 Feature Importance Each Attributes (Random Forest Regressor)', fontsize=18)
            ax.set_xlabel('Importance', fontsize=16)
            ax.set_ylabel('Feature Name', fontsize=16)

            # Display the plot in Streamlit
            st.pyplot(fig)

            import shap
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_test)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_test)
            st.pyplot(fig)

            explainer = shap.Explainer(rf, X_test)
            shap_values = explainer(X_test)
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0])
            st.pyplot(fig)

