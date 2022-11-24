import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
import pickle
import os
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import hstack
from scipy.sparse import coo_matrix, hstack
data=pd.read_csv(r"C:\Users\KAPURIYA\Downloads\preprocessed_data (1).csv",nrows=50000)
Y=data["project_is_approved"].values#Y is target column
X=data.drop(["project_is_approved"],axis=1)
#splited the datafor train and test
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=10,stratify=Y)
#split the data for cross validation
x_train,x_cv,y_train,y_cv=train_test_split(X_train,Y_train,random_state=10,stratify=Y_train)

vec_school=CountVectorizer()
X_school_state_train=vec_school.fit(x_train["school_state"].values)
X_school_state_train=vec_school.transform(x_train["school_state"].values)
pickle.dump(vec_school,open("vec_school.pkl","wb"))
#f=pickle.load(open("vec_school.pkl","rb"))
#f.transform(x_train["school_state"].values)

vec_techer_prefix=CountVectorizer()
X_teacher_prefix_train=vec_techer_prefix.fit(x_train["teacher_prefix"].values)
X_teacher_prefix_train=vec_techer_prefix.transform(x_train["teacher_prefix"].values)
pickle.dump(vec_techer_prefix,open("vec_techer_prefix.pkl","wb"))
#f=pickle.load(open("vec_techer_prefix.pkl","rb"))
#f.transform(x_train["teacher_prefix"].values)

vec_project_grade_category=CountVectorizer()
X_project_grade_category_train=vec_project_grade_category.fit(x_train["project_grade_category"].values)
X_project_grade_category_train=vec_project_grade_category.transform(x_train["project_grade_category"].values)
pickle.dump(vec_project_grade_category,open("vec_project_grade_category.pkl","wb"))
#f=pickle.load(open("vec_project_grade_category.pkl","rb"))
#f.transform(x_train["teacher_prefix"].values)

norm_teacher_number_of_previously_posted_projects=Normalizer()
X_teacher_number_of_previously_posted_projects_train=norm_teacher_number_of_previously_posted_projects.fit_transform(x_train["teacher_number_of_previously_posted_projects"].values.reshape(-1,1))
X_teacher_number_of_previously_posted_projects_train=norm_teacher_number_of_previously_posted_projects.transform(x_train["teacher_number_of_previously_posted_projects"].values.reshape(-1,1))
pickle.dump(norm_teacher_number_of_previously_posted_projects,open("norm_teacher_number_of_previously_posted_projects.pkl","wb"))
#f=pickle.load(open("norm_teacher_number_of_previously_posted_projects.pkl","rb"))
#f.transform(x_train["teacher_number_of_previously_posted_projects"].values)

vec_clean_categories=CountVectorizer()
X_clean_categories_train=vec_clean_categories.fit(x_train["clean_categories"].values)
X_clean_categories_train=vec_clean_categories.transform(x_train["clean_categories"].values)
pickle.dump(vec_clean_categories,open("vec_clean_categories.pkl","wb"))
#f=pickle.load(open("vec_clean_categories.pkl","rb"))
#f.transform(x_train["clean_categories"].values)

vec_clean_subcategories=CountVectorizer()
X_clean_subcategories_train=vec_clean_subcategories.fit(x_train["clean_subcategories"].values)
X_clean_subcategories_train=vec_clean_subcategories.transform(x_train["clean_subcategories"].values)
pickle.dump(vec_clean_subcategories,open("vec_clean_subcategories.pkl","wb"))
#f=pickle.load(open("vec_clean_subcategories.pkl","rb"))
#f.transform(x_train["clean_subcategories"].values)


vec_essay=CountVectorizer(min_df=10,max_features=5000)
X_essay_bow_train=vec_essay.fit(x_train["essay"].values)
X_essay_bow_train=vec_essay.transform(x_train["essay"].values)
pickle.dump(vec_essay,open("vec_essay.pkl","wb"))
#f=pickle.load(open("vec_essay.pkl","rb"))
#f.transform(x_train["essay"].values)


norm_price=Normalizer()
X_price_train=norm_price.fit_transform(x_train["price"].values.reshape(-1,1))
X_price_train=norm_price.transform(x_train["price"].values.reshape(-1,1))
pickle.dump(norm_price,open("norm_price.pkl","wb"))
#f=pickle.load(open("norm_price.pkl","rb"))-
#f.transform(x_train["price"].values)


from scipy.sparse import hstack
df_train=hstack((X_school_state_train,X_teacher_prefix_train,X_project_grade_category_train,X_teacher_number_of_previously_posted_projects_train,X_clean_categories_train,X_clean_subcategories_train,X_essay_bow_train,X_price_train))
clf = MultinomialNB(alpha=5,class_prior=[0.5,0.5])
clf.fit(df_train,y_train)
#print(clf.predict(df_train))
pickle.dump(clf,open("clf.pkl","wb"))

#classifier=pickle.load(open("clf.pkl","rb"))
#f.predict(x_train["essay"].values)
#print(classifier.predict(df_train.toarray()[0]))


