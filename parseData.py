'''
This script is used to parse the
selected data categories from the
Predict 'Em All Pokemon GO dataset.
This will greatly cut down the initialization time of the training.
'''

import pandas as pd

# Retrieve the data from the csv file.
fdata = pd.read_csv('../predictemall/300k.csv', low_memory=False)


# Need only these fields: 'pokemonId', 'latitude', 'longitude', 'appearedHour', 'appearedMinute',
# 'weather', 'temperature', 'terrainType','population_density'
parsedata = fdata[['pokemonId', 'latitude', 'longitude', 'appearedHour', 'appearedMinute',
                   'temperature', 'terrainType']]

# Write the newly transformed data to a csv
parsedata.to_csv(path_or_buf='../predictemall/parsedData.csv', index=False)

filename = ['../predictemall/parsedData.csv']
