import pandas as pd

Z_25 = pd.read_csv(f'data/annotations/Z25.csv', sep=';')

Z_25['ID'][1] = 'A0013'

# save
Z_25.to_csv(f'data/annotations/Z25.csv', sep=';', index=False)


Z_39 = pd.read_csv(f'data/annotations/Z39.csv', sep=';')

Z_39['ID'][4] = 'A0019'

# save
Z_39.to_csv(f'data/annotations/Z39.csv', sep=';', index=False)