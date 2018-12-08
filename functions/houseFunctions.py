import pandas as pd
import numpy as np
import pickle
'''
Parse 1976-2016 house data from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2

@relFilePath : file path of house results data, relative to python notebook

@return:  dataframe indexed by (year, state, district)
'''

def load_data(relFilePath, minYear=2010):
    ''' Keep only the winner and 2nd place candidates within each state's district for every year.
    arguments:
        relFilePath -- path to the data file (csv)
        minYear -- only records for all years from and after the min year will be kept (int)
    returns:
        dataframe with only the winners (pandas.dataframe)
        dataframe with only the 2nd place candidates (pandas.dataframe)
    '''

    data_df = pd.read_csv(relFilePath)

    winners_df = pd.DataFrame()
    winners2_df = pd.DataFrame()
    for key, shard in data_df.groupby(['year', 'state_po', 'district']):        
        if int(key[0]) >= minYear:
            #convention: 2nd winner = 1st winner if only 1 player
            winners_df = winners_df.append(shard.loc[shard['candidatevotes'].idxmax()])
            sortedIndices = (shard['candidatevotes'].values.argsort()[::-1])
            if len(sortedIndices) > 1:
                winners2_df = winners2_df.append(shard.iloc[sortedIndices[1]])
            else:
                winners2_df = winners2_df.append(shard.iloc[sortedIndices[0]])
    return winners_df, winners2_df

def clean_index(df, clean_before_build=True):
    '''Performs general clean up tasks on the key columns. Generates the master key.
    arguments:
        df -- dataframe to clean up, should contain the columns 'district', 'state_po' and 'year' (pandas.dataframe)
    returns:
        dataframe with cleaned key columns and index (pandas.dataframe)
    '''
    if clean_before_build:
        # drop default index
        df = df.reset_index().drop(['index','state'], axis=1)
        # rename state code
        df = df.rename(columns={'state_po' : 'state'})
    #format year and district columns as ints
    df = df.astype({'year': int, 'district': int})
    #make sure all districts start with 1
    df.loc[df['district']==0, 'district'] = 1

    # glue together the columns to get a more descriptive index    
    df.index = ['{0}_{1:02d}_{2}'.format(row['state'],row['district'],row['year']) for _,row in df.iterrows()]

    return df

def fetch_index(df, df2, save=False, load=False):
    '''Helper function for generating/loading master index for syncing between data sources.
    arguments:
        df -- dataframe to parse index from, MUST CONTAIN FULL COPIES OF THE 'district', 'state_po', 'year' COLUMNS (pandas.dataframe)
    returns:
        dataframe with master index for syncing between data sources.
    '''

    if not load:
        # Make a dummy dataframe so everyone else can make complete dataframes
        tmp1 = df[['district', 'state', 'year']]
        tmp2 = df2[['district', 'state', 'year']]
        master_index = pd.concat([tmp1, tmp2])

        if save:
            pickle.dump(master_index, open('Datasets/master_index.p', 'wb'))
        return master_index

    else:
        master_index = pickle.load(open('Datasets/master_index.p', 'rb'))
        return master_index    

def fetch_trimmed_data(df1, df2, minYear=2012):
    '''Compile training data. Additional cleaning and processing to generate additional features.
    arguments:
        df1 -- dataframe to compile training data from, should be loaded through load_data() and cleaned with clean_index()
        df2 -- dataframe with 2nd place candidates for each race
        minYear -- only records for all years from and after the min year will be kept (int)
    returns:
        dataframe containing training data.
    '''

    df1 = df1[['district', 'state', 'year', 'party', 'candidatevotes', 'totalvotes', 'candidate']]
    df2 = df2[['district', 'state', 'year', 'party', 'candidatevotes', 'totalvotes', 'candidate']]

    ########################################## ADDITIONAL CLEANING RELATED TO PARTY ##########################################

    #democratic-farmer-labor -> democratic party (one entry in 2012)
    df1.loc[df1['party'] == 'democratic-farmer-labor', 'party'] = 'democrat'
    #tax revolt -> republican party (one entry in 2012)
    df1.loc[df1['party'] == 'tax revolt', 'party'] = 'republican'

    #no clear indication which was to cast it, go by candidates closest affiliation between democrat/republican  
    #independent -> democrat (one entry in 2004 -- bernard sanders)
    df1.loc[df1['party'] == 'independent', 'party'] = 'democrat' 
    #reform -> republican (two entires in 2002, 2004 -- henry e. brown jr., 2002 -- barbara dale washer)
    df1.loc[df1['party'] == 'reform', 'party'] = 'republican'
    #republican/democrat -> republican (one entry in 2002) -- Don Sherwood
    df1.loc[df1['party'] == 'republican/democrat', 'party'] = 'republican'
    
    #KS 1.0: republican (tea party) -- might be nan because he ran under republican party ticket but he's actually from tea party?
    #KS 2.0: republican (tea party)
    #KS 3.0: republican (?)
    #KS 4.0: republican (tea party)
    #LA 1.0: republican (it's complicated)
    #LA 2.0: democrat (?)
    #LA 3.0: republican
    #if there is a run off election, we don't include it in the data. so vote counts could be iffy (e.g. see issues with verifying LA 3.0 vote counts)
    #winner may be in correct then if the votes are not from the run-off election, like they should be! in this case..
    #LA 4.0: republican (tea party? maybe...)
    #LA 5.0: republican (tea party caucus)
    #LA 6.0: republican (tea party)
    #MS 1.0: republican
    #MS 2.0: democrat (??)
    #MS 3.0: republican
    #MS 4.0: republican
    #ND 0.0: republican
    #WY 0.0: republican

    df1.loc[(pd.isnull(df1['party'])) & (df1['state'] == 'KS') & (df1['district'] == 1.0), 'party'] = 'republican'
    df1.loc[(pd.isnull(df1['party'])) & (df1['state'] == 'KS') & (df1['district'] == 2.0), 'party'] = 'republican'
    df1.loc[(pd.isnull(df1['party'])) & (df1['state'] == 'KS') & (df1['district'] == 3.0), 'party'] = 'republican'
    df1.loc[(pd.isnull(df1['party'])) & (df1['state'] == 'KS') & (df1['district'] == 4.0), 'party'] = 'republican'
    
    df1.loc[(pd.isnull(df1['party'])) & (df1['state'] == 'LA') & (df1['district'] == 1.0), 'party'] = 'republican'
    df1.loc[(pd.isnull(df1['party'])) & (df1['state'] == 'LA') & (df1['district'] == 2.0), 'party'] = 'democrat'
    df1.loc[(pd.isnull(df1['party'])) & (df1['state'] == 'LA') & (df1['district'] == 3.0), 'party'] = 'republican'
    df1.loc[(pd.isnull(df1['party'])) & (df1['state'] == 'LA') & (df1['district'] == 4.0), 'party'] = 'republican'
    df1.loc[(pd.isnull(df1['party'])) & (df1['state'] == 'LA') & (df1['district'] == 5.0), 'party'] = 'republican'
    df1.loc[(pd.isnull(df1['party'])) & (df1['state'] == 'LA') & (df1['district'] == 6.0), 'party'] = 'republican'
    
    df1.loc[(pd.isnull(df1['party'])) & (df1['state'] == 'MS') & (df1['district'] == 1.0), 'party'] = 'republican'
    df1.loc[(pd.isnull(df1['party'])) & (df1['state'] == 'MS') & (df1['district'] == 2.0), 'party'] = 'democrat'
    df1.loc[(pd.isnull(df1['party'])) & (df1['state'] == 'MS') & (df1['district'] == 3.0), 'party'] = 'republican'
    df1.loc[(pd.isnull(df1['party'])) & (df1['state'] == 'MS') & (df1['district'] == 4.0), 'party'] = 'republican'
    
    df1.loc[(pd.isnull(df1['party'])) & (df1['state'] == 'ND') & (df1['district'] == 1.0), 'party'] = 'republican'
    df1.loc[(pd.isnull(df1['party'])) & (df1['state'] == 'WY') & (df1['district'] == 1.0), 'party'] = 'republican'

    df1.loc[(pd.isnull(df1['party'])) & (df1['state'] == 'CO') & (df1['district'] == 6.0), 'party'] = 'republican'

    #democratic-farmer-labor -> democratic party (one entry in 2012)
    df2.loc[df2['party'] == 'democratic-farmer-labor', 'party'] = 'democrat'
    #tax revolt -> republican party (one entry in 2012)
    df2.loc[df2['party'] == 'tax revolt', 'party'] = 'republican'   
    
    #no clear indication which was to cast it, go by candidates closest affiliation between democrat/republican  
    #independent -> democrat (one entry in 2004 -- bernard sanders)
    df2.loc[df2['party'] == 'independent', 'party'] = 'democrat' 
    #reform -> republican (two entires in 2002, 2004 -- henry e. brown jr., 2002 -- barbara dale washer)
    df2.loc[df2['party'] == 'reform', 'party'] = 'republican'
    #republican/democrat -> republican (one entry in 2002) -- Don Sherwood
    df2.loc[df2['party'] == 'republican/democrat', 'party'] = 'republican'
    
    #KS 1.0: republican (tea party) -- might be nan because he ran under republican party ticket but he's actually from tea party?
    #KS 2.0: republican (tea party)
    #KS 3.0: republican (?)
    #KS 4.0: republican (tea party)
    #LA 1.0: republican (it's complicated)
    #LA 2.0: democrat (?)
    #LA 3.0: republican
    #if there is a run off election, we don't include it in the data. so vote counts could be iffy (e.g. see issues with verifying LA 3.0 vote counts)
    #winner may be in correct then if the votes are not from the run-off election, like they should be! in this case..
    #LA 4.0: republican (tea party? maybe...)
    #LA 5.0: republican (tea party caucus)
    #LA 6.0: republican (tea party)
    #MS 1.0: republican
    #MS 2.0: democrat (??)
    #MS 3.0: republican
    #MS 4.0: republican
    #ND 0.0: republican
    #WY 0.0: republican

    df2.loc[(pd.isnull(df2['party'])) & (df2['state'] == 'KS') & (df2['district'] == 1.0), 'party'] = 'republican'
    df2.loc[(pd.isnull(df2['party'])) & (df2['state'] == 'KS') & (df2['district'] == 2.0), 'party'] = 'republican'
    df2.loc[(pd.isnull(df2['party'])) & (df2['state'] == 'KS') & (df2['district'] == 3.0), 'party'] = 'republican'
    df2.loc[(pd.isnull(df2['party'])) & (df2['state'] == 'KS') & (df2['district'] == 4.0), 'party'] = 'republican'
    
    df2.loc[(pd.isnull(df2['party'])) & (df2['state'] == 'LA') & (df2['district'] == 1.0), 'party'] = 'republican'
    df2.loc[(pd.isnull(df2['party'])) & (df2['state'] == 'LA') & (df2['district'] == 2.0), 'party'] = 'democrat'
    df2.loc[(pd.isnull(df2['party'])) & (df2['state'] == 'LA') & (df2['district'] == 3.0), 'party'] = 'republican'
    df2.loc[(pd.isnull(df2['party'])) & (df2['state'] == 'LA') & (df2['district'] == 4.0), 'party'] = 'republican'
    df2.loc[(pd.isnull(df2['party'])) & (df2['state'] == 'LA') & (df2['district'] == 5.0), 'party'] = 'republican'
    df2.loc[(pd.isnull(df2['party'])) & (df2['state'] == 'LA') & (df2['district'] == 6.0), 'party'] = 'republican'
    
    df2.loc[(pd.isnull(df2['party'])) & (df2['state'] == 'MS') & (df2['district'] == 1.0), 'party'] = 'republican'
    df2.loc[(pd.isnull(df2['party'])) & (df2['state'] == 'MS') & (df2['district'] == 2.0), 'party'] = 'democrat'
    df2.loc[(pd.isnull(df2['party'])) & (df2['state'] == 'MS') & (df2['district'] == 3.0), 'party'] = 'republican'
    df2.loc[(pd.isnull(df2['party'])) & (df2['state'] == 'MS') & (df2['district'] == 4.0), 'party'] = 'republican'
    
    df2.loc[(pd.isnull(df2['party'])) & (df2['state'] == 'ND') & (df2['district'] == 1.0), 'party'] = 'republican'
    df2.loc[(pd.isnull(df2['party'])) & (df2['state'] == 'WY') & (df2['district'] == 1.0), 'party'] = 'republican'

    df2.loc[(pd.isnull(df2['party'])) & (df2['state'] == 'CO') & (df2['district'] == 6.0), 'party'] = 'republican'

        
    ########################################## ADDITIONAL PROCESSING W. ASSUMPTIONS ##########################################
    

    poll = pickle.load(open('Datasets/national_poll.p', 'rb'))
    for year in range(minYear, int(max(df1['year'].values))+1, 2):
        #convention: t-> current election, t-2 (tm2) -> previous election
        for index_t, row in df1.iterrows():
            if row['year'] == year:
                index_tm2 = index_t.replace(str(year), str(year-2))
                if index_tm2 in df1.index:
                    #a district is dropped if it does not exist in all years being processed (implictly assuming districts are the same shape across all years)

                    #################### POLLING FEATURES ####################
                    poll_t = poll.loc[poll.index == index_t, 'national_poll'].values[0]
                    poll_tm2 = poll.loc[poll.index == index_tm2, 'national_poll'].values[0]
                    df1.loc[df1.index == index_t, 'national_poll'] = poll_t
                    df1.loc[df1.index == index_t, 'national_poll_prev'] = poll_tm2
                    df1.loc[df1.index == index_t, 'national_poll_delta_subtract'] = poll_t - poll_tm2
                    df1.loc[df1.index == index_t, 'national_poll_delta_divide'] = poll_t/poll_tm2
                    #################### POLLING FEATURES ####################

                    #################### PREVIOUS WINNERS ####################
                    df1.loc[df1.index == index_t, 'previous_party'] = df1.loc[df1.index == index_tm2, 'party'].values[0]
                    #################### PREVIOUS WINNERS ####################

                    
                    #################### MARGIN FEATURES ####################
                    #convention: when signed, always defined as dem +ve and rep -ve
                    winner_totalvotes = df1.loc[df1.index == index_tm2, 'totalvotes'].values[0]
                    loser_totalvotes = df2.loc[df2.index == index_tm2, 'totalvotes'].values[0]
                    if winner_totalvotes == 0:
                        winner_margin = 1
                    else:
                        winner_margin = (df1.loc[df1.index == index_tm2, 'candidatevotes'].values[0])/(winner_totalvotes)
                    if loser_totalvotes == 0:
                        loser_margin = 1
                    else:
                        loser_margin  = (df2.loc[df2.index == index_tm2, 'candidatevotes'].values[0])/(loser_totalvotes)

                    if winner_margin == loser_margin:
                        #only 1 player
                        loser_margin = 1e-10
                    else:
                        loser_margin = (df2.loc[df2.index == index_tm2, 'candidatevotes'].values[0])/(df2.loc[df2.index == index_tm2, 'totalvotes'].values[0])                    
                    ### see convention for 2nd winner when only 1 player ###

                    label_dem = 'dem_win_margin_prev'
                    label_rep = 'rep_win_margin_prev'
                    label_sm = 'margin_signed_minus_prev'
                    label_um = 'margin_unsigned_minus_prev'
                    label_sd = 'margin_signed_divide_prev'
                    label_ud = 'margin_unsigned_divide_prev'

                    if df1.loc[df1.index == index_tm2, 'party'].values[0] == 'democrat':

                        df1.loc[df1.index == index_t, label_dem] = winner_margin
                        df1.loc[df1.index == index_t, label_rep] = loser_margin

                        df1.loc[df1.index == index_t, label_sm] = winner_margin - loser_margin
                        if loser_margin != 0:
                            df1.loc[df1.index == index_t, label_sd] = winner_margin/loser_margin
                        else:
                            df1.loc[df1.index == index_t, label_sd] = winner_margin/(1e-10)

                    elif df1.loc[df1.index == index_tm2, 'party'].values[0] == 'republican':

                        df1.loc[df1.index == index_t, label_dem] = loser_margin
                        df1.loc[df1.index == index_t, label_rep] = winner_margin

                        df1.loc[df1.index == index_t, label_sm] = loser_margin - winner_margin
                        if winner_margin != 0:
                            df1.loc[df1.index == index_t, label_sd] = loser_margin/winner_margin
                        else:
                            df1.loc[df1.index == index_t, label_sd] = loser_margin/(1e-10)

                    df1.loc[df1.index == index_t, label_um] = winner_margin - loser_margin
                    if loser_margin != 0:
                        df1.loc[df1.index == index_t, label_ud] = winner_margin/loser_margin
                    else:
                        df1.loc[df1.index == index_t, label_ud] = winner_margin/(1e-10)
                        
                        #if previous winner was democrat

                    #################### MARGIN FEATURES ####################

                    # to-do features
                    # incumbent? name check
                    # summary statistics for winning margins changing over time

                else:
                    df1 = df1[df1.index != index_t]

    #trim df1 down to only 1 election before minyear
    df1 = df1[df1['year'] != minYear - 2]

    #################### PREVIOUS WINNER FEATURES ####################
    df1.loc[df1['previous_party'] == 'democrat', 'dem_win_prev'] = 1
    df1.loc[df1['previous_party'] != 'democrat', 'dem_win_prev'] = 0

    df1.loc[df1['previous_party'] == 'republican', 'rep_win_prev'] = 1
    df1.loc[df1['previous_party'] != 'republican', 'rep_win_prev'] = 0
    #################### PREVIOUS WINNER FEATURES ####################


    #################### OBSERVED WINNER ####################
    df1.loc[df1['party'] == 'democrat', 'dem_win'] = 1
    df1.loc[df1['party'] != 'democrat', 'dem_win'] = 0

    df1.loc[df1['party'] == 'republican', 'rep_win'] = 1
    df1.loc[df1['party'] != 'republican', 'rep_win'] = 0
    #################### OBSERVED WINNER ####################
    return df1            