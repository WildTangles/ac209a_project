import pandas as pd
import numpy as np
import pickle
'''
Parse 1976-2016 house data from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2

@relFilePath : file path of house results data, relative to python notebook

@return:  dataframe indexed by (year, state, district)
'''

def parse_data(relFilePath, minYear=2010):

    data_df = pd.read_csv(relFilePath)

    winners_df = pd.DataFrame()
    for key, shard in data_df.groupby(['year', 'state_po', 'district']):
        if int(key[0]) >= minYear:
            winners_df = winners_df.append(shard.loc[shard['candidatevotes'].idxmax()])
    return winners_df

def parse_index(full_interest, save=False,load=False):

    if not load:
        # Make a dummy dataframe so everyone else can make complete dataframes
        master_index = full_interest[['district','state_po','year']].reset_index().drop('index', axis=1)
        master_index = master_index.rename(columns={'state_po' : 'state'}) # rename state code
        master_index = master_index.astype({'year' : int, 'district' : int}) # 
        master_index.loc[master_index['district']==0, 'district'] = 1 # make sure all districts start with 1

        # glue together the columns to get a more descriptive index
        master_index.index = ['{0}_{1:02d}_{2}'.format(row['state'],row['district'],row['year']) for _,row in master_index.iterrows()]

        if save:
            pickle.dump(master_index, open('Datasets/master_index.p', 'wb'))
        return master_index

    else:
        # Load the file
        master_index = pickle.load(open('Datasets/master_index.p', 'rb'))
        return master_index
    
def fetch_training_set(full_interest, minYear=2012):
    
    sub_interest = full_interest[['district', 'state_po', 'year', 'party', 'candidatevotes', 'totalvotes', 'candidate']]    
    
    #if democratic-farmer-labor it's made to be democratic party (one entry in 2012)
    sub_interest.loc[sub_interest['party'] == 'democratic-farmer-labor', 'party'] = 'democrat'
    #if tax revolt it's made to be republican party (one entry in 2012)
    sub_interest.loc[sub_interest['party'] == 'tax revolt', 'party'] = 'republican'
    
    #missing data due to issues with party?        
    
    #KS 1.0: republican (tea party) -- might be nan because he ran under republican party ticket but he's actually from tea party?
    #KS 2.0: republican (tea party)
    #KS 3.0: republican (?)
    #KS 4.0: republican (tea party)
    #LA 1.0: republican (it's complicated)
    #LA 2.0: democrat (?)
    #LA 3.0: republican
    #if there is a run off election, we don't include it in the data. so vote counts could be iffy (e.g. see issues with verifying LA 3.0 vote counts)
    #winner may be in correct then if the votes are not from the run-off election, like they should be! in this case
    #TO DO
    #LA 4.0: republican (tea party? maybe...)
    #LA 5.0: republican (tea party caucus)
    #LA 6.0: republican (tea party)
    #MS 1.0: republican
    #MS 2.0: democrat (??)
    #MS 3.0: republican
    #MS 4.0: republican
    #ND 0.0: republican
    #WY 0.0: republican
    
    sub_interest.loc[(pd.isnull(sub_interest['party'])) & (sub_interest['state_po'] == 'KS') & (sub_interest['district'] == 1.0), 'party'] = 'republican'
    sub_interest.loc[(pd.isnull(sub_interest['party'])) & (sub_interest['state_po'] == 'KS') & (sub_interest['district'] == 2.0), 'party'] = 'republican'
    sub_interest.loc[(pd.isnull(sub_interest['party'])) & (sub_interest['state_po'] == 'KS') & (sub_interest['district'] == 3.0), 'party'] = 'republican'
    sub_interest.loc[(pd.isnull(sub_interest['party'])) & (sub_interest['state_po'] == 'KS') & (sub_interest['district'] == 4.0), 'party'] = 'republican'
    
    sub_interest.loc[(pd.isnull(sub_interest['party'])) & (sub_interest['state_po'] == 'LA') & (sub_interest['district'] == 1.0), 'party'] = 'republican'
    sub_interest.loc[(pd.isnull(sub_interest['party'])) & (sub_interest['state_po'] == 'LA') & (sub_interest['district'] == 2.0), 'party'] = 'democrat'
    sub_interest.loc[(pd.isnull(sub_interest['party'])) & (sub_interest['state_po'] == 'LA') & (sub_interest['district'] == 3.0), 'party'] = 'republican'
    sub_interest.loc[(pd.isnull(sub_interest['party'])) & (sub_interest['state_po'] == 'LA') & (sub_interest['district'] == 4.0), 'party'] = 'republican'
    sub_interest.loc[(pd.isnull(sub_interest['party'])) & (sub_interest['state_po'] == 'LA') & (sub_interest['district'] == 5.0), 'party'] = 'republican'
    sub_interest.loc[(pd.isnull(sub_interest['party'])) & (sub_interest['state_po'] == 'LA') & (sub_interest['district'] == 6.0), 'party'] = 'republican'
    
    sub_interest.loc[(pd.isnull(sub_interest['party'])) & (sub_interest['state_po'] == 'MS') & (sub_interest['district'] == 1.0), 'party'] = 'republican'
    sub_interest.loc[(pd.isnull(sub_interest['party'])) & (sub_interest['state_po'] == 'MS') & (sub_interest['district'] == 2.0), 'party'] = 'democrat'
    sub_interest.loc[(pd.isnull(sub_interest['party'])) & (sub_interest['state_po'] == 'MS') & (sub_interest['district'] == 3.0), 'party'] = 'republican'
    sub_interest.loc[(pd.isnull(sub_interest['party'])) & (sub_interest['state_po'] == 'MS') & (sub_interest['district'] == 4.0), 'party'] = 'republican'
    
    sub_interest.loc[(pd.isnull(sub_interest['party'])) & (sub_interest['state_po'] == 'ND') & (sub_interest['district'] == 0.0), 'party'] = 'republican'
    sub_interest.loc[(pd.isnull(sub_interest['party'])) & (sub_interest['state_po'] == 'WY') & (sub_interest['district'] == 0.0), 'party'] = 'republican'
    
    
    #general cleanup
    sub_interest = sub_interest.reset_index().drop('index', axis=1)
    sub_interest = sub_interest.rename(columns={'state_po': 'state'})
    sub_interest = sub_interest.astype({'year': int, 'district': int})
    sub_interest.loc[sub_interest['district']==0, 'district'] = 1
    
    sub_interest.index = ['{0}_{1:02d}_{2}'.format(row['state'], row['district'], row['year']) for _,row in sub_interest.iterrows()]
    #general cleanup
    
    #previous winner
    #drop row if district does not exist in previous
    #+ we're assuming that district 1 is the same shape as distict 1 from other years
    
    poll_data = pickle.load(open('Datasets/national_poll.p','rb'))
    
    for year in range(minYear, max(sub_interest['year'].values)+1, 2):
        for idx, row in sub_interest.iterrows():
            if row['year'] == year:                
                alt_idx = idx.replace(str(year), str(year-2))
                if alt_idx in sub_interest.index:
                    
                    nPoll = poll_data.loc[poll_data.index == idx, 'national_poll'].values[0]
                    nPollPrev = poll_data.loc[poll_data.index == alt_idx, 'national_poll'].values[0]
                    
                    sub_interest.loc[sub_interest.index == idx, 'national_poll'] = nPoll
                    sub_interest.loc[sub_interest.index == idx, 'national_poll_prior'] = nPollPrev
                    sub_interest.loc[sub_interest.index == idx, 'national_poll_delta_sub'] = nPoll - nPollPrev
                    sub_interest.loc[sub_interest.index == idx, 'national_poll_delta_div'] = nPoll/nPollPrev
                    
                    sub_interest.loc[sub_interest.index == idx, 'previous_party'] = sub_interest.loc[sub_interest.index == alt_idx, 'party'].values[0]
                    
                    if sub_interest.loc[sub_interest.index == alt_idx, 'party'].values[0] == 'democrat':
                        percent_winner = (sub_interest.loc[sub_interest.index == alt_idx, 'candidatevotes'].values[0])/(sub_interest.loc[sub_interest.index == alt_idx, 'totalvotes'].values[0])
                        percent_loser = 1 - percent_winner
                        
                        sub_interest.loc[sub_interest.index == idx, 'prior_win_dem_percent'] = percent_winner
                        sub_interest.loc[sub_interest.index == idx, 'prior_win_rep_percent'] = percent_loser
                        if percent_loser == 0:
                            percent_loser = 0.0000001
                        #%winner/%loser
                        sub_interest.loc[sub_interest.index == idx, 'prior_margin_unsigned_divide'] = percent_winner/percent_loser
                        #%winner/%loser
                        sub_interest.loc[sub_interest.index == idx, 'prior_margin_unsigned_minus'] = percent_winner - percent_loser
                        #%dem/%rep
                        sub_interest.loc[sub_interest.index == idx, 'prior_margin_signed_divide'] = percent_winner/percent_loser
                        #%dem - %rep
                        sub_interest.loc[sub_interest.index == idx, 'prior_margin_signed_minus'] = percent_winner - percent_loser
                        
                    elif sub_interest.loc[sub_interest.index == alt_idx, 'party'].values[0] == 'republican':
                        percent_winner = (sub_interest.loc[sub_interest.index == alt_idx, 'candidatevotes'].values[0])/(sub_interest.loc[sub_interest.index == alt_idx, 'totalvotes'].values[0])
                        percent_loser = 1 - percent_winner
                        if percent_loser == 0:
                            percent_loser = 0.0000001
                        
                        sub_interest.loc[sub_interest.index == idx, 'prior_win_dem_percent'] = percent_loser
                        sub_interest.loc[sub_interest.index == idx, 'prior_win_rep_percent'] = percent_winner
                        #%winner/%loser
                        sub_interest.loc[sub_interest.index == idx, 'prior_margin_unsigned_divide'] = percent_winner/percent_loser
                        #%winner/%loser
                        sub_interest.loc[sub_interest.index == idx, 'prior_margin_unsigned_minus'] = percent_winner - percent_loser
                        #%dem/%rep
                        sub_interest.loc[sub_interest.index == idx, 'prior_margin_signed_divide'] = percent_loser/percent_winner
                        #%dem - %rep
                        sub_interest.loc[sub_interest.index == idx, 'prior_margin_signed_minus'] = percent_loser - percent_winner
                else:
                    #if the district does not exist in the previous election, drop that district (that row) from this year's records as well
                    sub_interest = sub_interest[sub_interest.index != idx]
        
    sub_interest = sub_interest[sub_interest['year'] != minYear-2]
    #previous winner
    
    sub_interest.loc[sub_interest['previous_party'] == 'democrat', 'prior_win_dem'] = 1
    sub_interest.loc[sub_interest['previous_party'] != 'democrat', 'prior_win_dem'] = 0
    
    sub_interest.loc[sub_interest['previous_party'] == 'republican', 'prior_win_rep'] = 1
    sub_interest.loc[sub_interest['previous_party'] != 'republican', 'prior_win_rep'] = 0
    
    return sub_interest