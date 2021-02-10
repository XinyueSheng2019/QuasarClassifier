import pandas as pd 

import pandas as pd
data = pd.read_csv('plasticc_test_lightcurves_01.csv.gz', compression='gzip',
                   error_bad_lines=False)
data.to_csv('plasticc_test_lightcurves_01.csv', index=False)