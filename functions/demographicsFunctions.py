'''
Read a subset of data from file

@relFilePath : file path of district data, relative to python notebook
@state   : state 
@year    : year

@return:  dataframe with a subset of data
'''
def read_cols_light(relFilePath, state, year):
    
    #read from file
    data_df = pd.read_csv(relFilePath)
                              
    
    #People Topic Gender and Age
    people_sexage_data = data_df[(data_df['Topic']=='People') & (data_df['Subject']=="Sex and Age")]
    people_sexage_data = people_sexage_data.set_index('Title').transpose()
    #people_cols = people_sexage_data.columns (if we want to add age)
    people_sexage_cols = ['Male','Female']

    
    #People Race
    people_race_data = data_df[(data_df['Topic']=='People') & (data_df['Subject']=="Race")]
    people_race_data = people_race_data.set_index('Title').transpose()
    people_race_cols = people_race_data.columns.difference(['Total population']).tolist()

        
    #People Hipanic or Latino
    people_hispanics_data = data_df[(data_df['Topic']=='People') & (data_df['Subject']=="Hispanic or Latino and Race")]
    people_hispanics_data = people_hispanics_data.set_index('Title').transpose()
    #people_hispanics_cols = people_hispanics_data.columns.difference(['Total population']).tolist()
    people_hispanics_cols = ['Hispanic or Latino (of any race)','Not Hispanic or Latino']

        

    #Education Topic
    education_data = data_df[(data_df['Topic']=='Education') & (data_df['Subject']=="Educational Attainment")]
    education_data = education_data.set_index('Title').transpose()
    #education_cols = education_data.columns.difference(['Total population']).tolist()
    education_cols = ["Percent high school graduate or higher","Percent bachelor's degree or higher"]

    #Socioeconomic Topic
    socioeconomic_data = data_df[(data_df['Topic']=='Socioeconomic') & (data_df['Subject']=="Income and Benefits (In 2017 inflation-adjusted dollars)")]
    socioeconomic_data = socioeconomic_data.set_index('Title').transpose()
    #socioeconomic_cols = socioeconomic_data.columns.difference(['Total population']).tolist()
    socioeconomic_cols = ["Median household income (dollars)","Mean household income (dollars)"]

    
    #Workers Topic
    workers_data = data_df[(data_df['Topic']=='workers_data') & (data_df['Subject']=="Employment Status")]
    workers_data = workers_data.set_index('Title').transpose()
    #workers_cols = workers_data.columns.difference(['Total population']).tolist()
    workers_cols = ["Unemployment Rate"]

    
    #COLUMNS TO READ
    list_cols = people_sexage_cols + people_race_cols+\
                people_hispanics_cols+education_cols +\
                socioeconomic_cols + workers_cols
    
                          
    #create the dataFrame
    df = pd.DataFrame()
    for col in list_cols:
        df[col]=data_df.set_index('Title').transpose()[col]
    
    #Add state and Year column
    df['state']=state
    df['year'] = year
    
    #Remove the first 2 entries and MOE
    df=df.transpose()
    df=df.drop(['Topic','Subject'],axis=1)
    df = df[df.columns.drop(list(df.filter(regex='MOE')))]
    df=df.transpose()
    
    #return the dataframe
    return df  
