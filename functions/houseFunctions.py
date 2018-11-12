'''
Parse 1976-2016 house data from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2

@relFilePath : file path of house results data, relative to python notebook

@return:  dataframe indexed by (year, state, district)
'''

def parse_data(relFilePath):

    data_df = pd.read_csv(relFilePath)

    winners_df = pd.DataFrame()
    for key, shard in data_df.groupby(['year', 'state_po', 'district']):
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
    