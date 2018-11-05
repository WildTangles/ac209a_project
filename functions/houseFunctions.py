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
    