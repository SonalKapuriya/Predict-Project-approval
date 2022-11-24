import streamlit as st
import numpy as np
import pickle
from scipy.sparse import coo_matrix, hstack

vec_school=pickle.load(open("vec_school.pkl","rb"))

vec_techer_prefix=pickle.load(open("vec_techer_prefix.pkl","rb"))

vec_project_grade_category=pickle.load(open("vec_project_grade_category.pkl","rb"))

norm_teacher_number_of_previously_posted_projects=pickle.load(open("norm_teacher_number_of_previously_posted_projects.pkl","rb"))

vec_clean_categories=pickle.load(open("vec_clean_categories.pkl","rb"))

vec_clean_subcategories=pickle.load(open("vec_clean_subcategories.pkl","rb"))


vec_essay=pickle.load(open("vec_essay.pkl","rb"))

norm_price=pickle.load(open("norm_price.pkl","rb"))
naive_bays_clf=pickle.load(open("clf.pkl","rb"))


def project_approved(school_state,teacher_prefix,grade,no_of_previous_projects,clean_category,sub_clean_category,essay,price):
    
    school_state=vec_school.transform(np.array([str(school_state)]))
    teacher_prefix=vec_techer_prefix.transform(np.array([str(teacher_prefix)]))
    grade=vec_project_grade_category.transform(np.array([str(grade)]))
    no_of_previous_projects=norm_teacher_number_of_previously_posted_projects.transform((np.array([int(no_of_previous_projects)])).reshape(-1,1))
    clean_category=vec_clean_categories.transform(np.array([str(clean_category)]))
    sub_clean_category=vec_clean_subcategories.transform(np.array([str(sub_clean_category)]))
    essay=vec_essay.transform(np.array([essay]))
    price=norm_price.transform(np.array([int(price)]).reshape(-1,1))
    final_vector=hstack((school_state,teacher_prefix,grade,no_of_previous_projects,clean_category,sub_clean_category,essay,price)).toarray()
    prob=naive_bays_clf.predict_proba(final_vector)
    result=naive_bays_clf.predict(final_vector)
    return prob


def main():
    st.title("Project Approval Prediction at DonorChoose.org")
    
    if st.button("About"):

        st.text("This model predicts chances of your project being funded by DonorChoose.org, DonorChoose is an organization which provide funding for educational and research purposes.")
        
    
    school_state = st.selectbox("Select State of your School",('tx', 'mn', 'dc', 'wi', 'la', 'hi', 'va', 'ct', 'oh', 'nh', 'wa', 'ok', 'ga', 'ny', 'nd', 'ms', 'ma', 'az', 'vt', 'tn', 'or', 'sd', 'wv', 'co', 'mo', 'md', 'id', 'fl', 'ak', 'sc', 'al', 'mi', 'ca', 'ri', 'ar', 'in', 'me', 'ut', 'ky', 'pa', 'de', 'nv', 'nc', 'ks', 'nm', 'il', 'mt', 'ne', 'nj', 'wy', 'ia'))
    teacher_prefix= st.selectbox("Select Teacher prefix",('teacher', 'mr', 'dr', 'mrs', 'ms'))
    grade= st.selectbox("Select Project Grade Category",('grades_prek_2', 'grades_9_12', 'grades_3_5', 'grades_6_8'))
    no_of_previous_projects=st.slider('Select no. of previously posted projects by your teacher',min_value=0,max_value=400)
    clean_category = st.selectbox("Select Category",('health_sports math_science', 'math_science literacy_language', 'math_science music_arts', 'history_civics appliedlearning', 'history_civics music_arts', 'music_arts specialneeds', 'specialneeds health_sports', 'specialneeds', 'appliedlearning music_arts', 'math_science specialneeds', 'health_sports warmth care_hunger', 'music_arts history_civics', 'history_civics warmth care_hunger', 'health_sports specialneeds', 'appliedlearning history_civics', 'literacy_language', 'music_arts appliedlearning', 'health_sports music_arts', 'math_science health_sports', 'appliedlearning health_sports', 'math_science appliedlearning', 'health_sports history_civics', 'appliedlearning literacy_language', 'health_sports literacy_language', 'music_arts', 'math_science history_civics', 'literacy_language history_civics', 'literacy_language health_sports', 'specialneeds music_arts', 'history_civics specialneeds', 'literacy_language music_arts', 'appliedlearning', 'math_science', 'history_civics math_science', 'history_civics health_sports', 'literacy_language specialneeds', 'literacy_language math_science', 'history_civics', 'music_arts health_sports', 'health_sports', 'health_sports appliedlearning', 'appliedlearning specialneeds', 'appliedlearning math_science', 'literacy_language appliedlearning', 'history_civics literacy_language'))

    sub_clean_category= st.selectbox("Select Subcategory",('communityservice', 'environmentalscience health_lifescience', 'environmentalscience performingarts', 'civics_government mathematics', 'esl literature_writing', 'communityservice visualarts', 'college_careerprep foreignlanguages', 'gym_fitness', 'health_lifescience parentinvolvement', 'other visualarts', 'foreignlanguages health_wellness', 'college_careerprep economics', 'appliedsciences civics_government', 'extracurricular performingarts', 'appliedsciences literacy', 'appliedsciences charactereducation', 'socialsciences specialneeds', 'mathematics nutritioneducation', 'civics_government economics', 'health_lifescience teamsports', 'appliedsciences health_wellness', 'communityservice mathematics', 'literature_writing nutritioneducation', 'environmentalscience socialsciences', 'college_careerprep mathematics', 'environmentalscience history_geography', 'other teamsports', 'communityservice specialneeds', 'health_wellness specialneeds', 'appliedsciences socialsciences', 'parentinvolvement', 'foreignlanguages visualarts', 'environmentalscience specialneeds', 'esl history_geography', 'extracurricular socialsciences', 'college_careerprep earlydevelopment', 'civics_government socialsciences', 'nutritioneducation specialneeds', 'financialliteracy health_wellness', 'literature_writing performingarts', 'earlydevelopment teamsports', 'other', 'music socialsciences', 'college_careerprep extracurricular', 'communityservice literacy', 'foreignlanguages music', 'health_wellness performingarts', 'esl environmentalscience', 'appliedsciences environmentalscience', 'appliedsciences earlydevelopment', 'environmentalscience parentinvolvement', 'civics_government health_wellness', 'charactereducation environmentalscience', 'literacy', 'college_careerprep literacy', 'communityservice esl', 'esl health_wellness', 'civics_government performingarts', 'nutritioneducation socialsciences', 'mathematics socialsciences', 'charactereducation health_lifescience', 'mathematics specialneeds', 'environmentalscience music', 'health_lifescience performingarts', 'literature_writing visualarts', 'extracurricular', 'health_wellness socialsciences', 'nutritioneducation teamsports', 'health_wellness other', 'earlydevelopment', 'performingarts specialneeds', 'literature_writing teamsports', 'mathematics teamsports', 'health_wellness warmth care_hunger', 'economics environmentalscience', 'college_careerprep parentinvolvement', 'literacy literature_writing', 'charactereducation extracurricular', 'civics_government literature_writing', 'health_lifescience mathematics', 'college_careerprep health_lifescience', 'economics', 'earlydevelopment extracurricular', 'socialsciences', 'literacy music', 'college_careerprep nutritioneducation', 'earlydevelopment health_lifescience', 'college_careerprep communityservice', 'esl music', 'extracurricular health_lifescience', 'environmentalscience foreignlanguages', 'parentinvolvement socialsciences', 'appliedsciences nutritioneducation', 'charactereducation mathematics', 'esl visualarts', 'gym_fitness music', 'college_careerprep literature_writing', 'communityservice health_wellness', 'appliedsciences gym_fitness', 'extracurricular literature_writing', 'economics music', 'communityservice literature_writing', 'extracurricular nutritioneducation', 'literacy other', 'appliedsciences visualarts', 'extracurricular specialneeds', 'charactereducation earlydevelopment', 'specialneeds visualarts', 'appliedsciences other', 'appliedsciences foreignlanguages', 'literacy performingarts', 'health_lifescience history_geography', 'charactereducation esl', 'college_careerprep socialsciences', 'esl parentinvolvement', 'charactereducation gym_fitness', 'appliedsciences health_lifescience', 'earlydevelopment economics', 'gym_fitness literature_writing', 'civics_government esl', 'gym_fitness specialneeds', 'health_wellness teamsports', 'economics foreignlanguages', 'economics mathematics', 'gym_fitness health_wellness', 'mathematics music', 'health_wellness history_geography', 'appliedsciences specialneeds', 'appliedsciences mathematics', 'appliedsciences communityservice', 'music specialneeds', 'health_lifescience visualarts', 'health_lifescience music', 'charactereducation civics_government', 'extracurricular teamsports', 'literature_writing specialneeds', 'health_wellness literature_writing', 'esl mathematics', 'earlydevelopment parentinvolvement', 'literacy teamsports', 'earlydevelopment performingarts', 'literacy parentinvolvement', 'mathematics performingarts', 'gym_fitness visualarts', 'environmentalscience gym_fitness', 'music other', 'mathematics', 'economics literature_writing', 'foreignlanguages mathematics', 'performingarts socialsciences', 'charactereducation nutritioneducation', 'environmentalscience financialliteracy', 'college_careerprep performingarts', 'history_geography performingarts', 'financialliteracy literacy', 'communityservice environmentalscience', 'gym_fitness history_geography', 'civics_government health_lifescience', 'communityservice parentinvolvement', 'extracurricular health_wellness', 'esl earlydevelopment', 'teamsports visualarts', 'communityservice socialsciences', 'civics_government history_geography', 'literature_writing mathematics', 'charactereducation visualarts', 'charactereducation college_careerprep', 'foreignlanguages performingarts', 'appliedsciences extracurricular', 'college_careerprep visualarts', 'charactereducation communityservice', 'environmentalscience visualarts', 'teamsports', 'music visualarts', 'financialliteracy history_geography', 'extracurricular other', 'foreignlanguages health_lifescience', 'foreignlanguages', 'history_geography literature_writing', 'charactereducation teamsports', 'esl extracurricular', 'health_lifescience literature_writing', 'civics_government financialliteracy', 'history_geography other', 'civics_government environmentalscience', 'history_geography', 'college_careerprep', 'economics specialneeds', 'esl socialsciences', 'esl specialneeds', 'earlydevelopment gym_fitness', 'other socialsciences', 'appliedsciences history_geography', 'college_careerprep specialneeds', 'earlydevelopment mathematics', 'earlydevelopment nutritioneducation', 'appliedsciences', 'extracurricular gym_fitness', 'charactereducation specialneeds', 'earlydevelopment foreignlanguages', 'economics nutritioneducation', 'gym_fitness literacy', 'health_lifescience literacy', 'foreignlanguages specialneeds', 'history_geography mathematics', 'extracurricular literacy', 'civics_government extracurricular', 'college_careerprep financialliteracy', 'extracurricular parentinvolvement', 'financialliteracy', 'history_geography specialneeds', 'visualarts', 'other parentinvolvement', 'environmentalscience health_wellness', 'earlydevelopment visualarts', 'performingarts visualarts', 'communityservice earlydevelopment', 'civics_government college_careerprep', 'performingarts teamsports', 'history_geography socialsciences', 'health_lifescience nutritioneducation', 'appliedsciences college_careerprep', 'earlydevelopment literature_writing', 'earlydevelopment history_geography', 'foreignlanguages history_geography', 'parentinvolvement visualarts', 'communityservice financialliteracy', 'college_careerprep teamsports', 'history_geography warmth care_hunger', 'earlydevelopment literacy', 'charactereducation financialliteracy', 'gym_fitness teamsports', 'extracurricular history_geography', 'communityservice health_lifescience', 'charactereducation music', 'charactereducation other', 'charactereducation foreignlanguages', 'health_lifescience other', 'appliedsciences teamsports', 'gym_fitness mathematics', 'appliedsciences performingarts', 'other specialneeds', 'college_careerprep health_wellness', 'charactereducation parentinvolvement', 'socialsciences visualarts', 'civics_government literacy', 'appliedsciences economics', 'college_careerprep environmentalscience', 'communityservice history_geography', 'economics socialsciences', 'gym_fitness nutritioneducation', 'earlydevelopment environmentalscience', 'environmentalscience other', 'health_wellness nutritioneducation', 'charactereducation literacy', 'history_geography music', 'environmentalscience extracurricular', 'literature_writing socialsciences', 'earlydevelopment health_wellness', 'civics_government foreignlanguages', 'extracurricular mathematics', 'music performingarts', 'college_careerprep gym_fitness', 'financialliteracy specialneeds', 'college_careerprep music', 'literacy nutritioneducation', 'literacy socialsciences', 'economics literacy', 'civics_government communityservice', 'mathematics visualarts', 'foreignlanguages socialsciences', 'college_careerprep other', 'health_lifescience health_wellness', 'charactereducation health_wellness', 'performingarts', 'communityservice nutritioneducation', 'esl literacy', 'esl performingarts', 'literacy visualarts', 'health_wellness music', 'health_lifescience specialneeds', 'extracurricular music', 'gym_fitness other', 'foreignlanguages gym_fitness', 'nutritioneducation', 'financialliteracy visualarts', 'nutritioneducation visualarts', 'charactereducation performingarts', 'communityservice extracurricular', 'literature_writing other', 'parentinvolvement specialneeds', 'charactereducation', 'foreignlanguages literacy', 'literacy specialneeds', 'esl', 'health_wellness visualarts', 'economics financialliteracy', 'charactereducation history_geography', 'appliedsciences parentinvolvement', 'college_careerprep history_geography', 'earlydevelopment specialneeds', 'health_wellness', 'communityservice other', 'appliedsciences esl', 'esl other', 'civics_government specialneeds', 'economics health_lifescience', 'civics_government visualarts', 'charactereducation socialsciences', 'literature_writing', 'music', 'environmentalscience mathematics', 'college_careerprep esl', 'earlydevelopment other', 'economics other', 'earlydevelopment music', 'esl financialliteracy', 'esl health_lifescience','health_wellness parentinvolvement', 'mathematics parentinvolvement', 'esl foreignlanguages', 'charactereducation literature_writing', 'economics history_geography', 'environmentalscience literacy', 'earlydevelopment socialsciences', 'history_geography teamsports', 'esl gym_fitness', 'health_wellness literacy', 'communityservice performingarts', 'communityservice music', 'environmentalscience literature_writing', 'history_geography literacy', 'gym_fitness health_lifescience', 'extracurricular visualarts', 'mathematics other', 'literature_writing music', 'specialneeds teamsports', 'history_geography visualarts', 'financialliteracy mathematics', 'music teamsports', 'appliedsciences music', 'gym_fitness performingarts', 'esl nutritioneducation', 'nutritioneducation other', 'specialneeds', 'health_lifescience', 'gym_fitness parentinvolvement', 'health_wellness mathematics', 'appliedsciences literature_writing', 'literature_writing parentinvolvement', 'environmentalscience', 'environmentalscience nutritioneducation', 'civics_government', 'foreignlanguages other', 'health_lifescience socialsciences', 'foreignlanguages literature_writing', 'literacy mathematics'))
    essay=st.text_area("Write few lines describing applications & objective of your project", value="plz enter text in small letters")
    price=st.number_input('Enter the fund in $ needed for project',help='Do not write $ symbol')
    
    
    
    if st.button("Predict"):
        
        probability=project_approved(school_state,teacher_prefix,grade,no_of_previous_projects,clean_category,sub_clean_category,essay,price)
        st.success('The chances of your project being funded is {}%'.format(np.round((probability[0][1])*100,2)))

    if st.button("Suggestion on your chances"):
        st.text("If chances are greater than 50% then it's highly likely that your project will be funded")
    
if __name__=='__main__':
    main()                           

