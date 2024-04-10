# Code Execution

The following is the line of code that executes the data extraction:

'''
data, df_pours, pourIDs = p.get_all_temperatures_for_site(
            p.get_api_token(), site_info, site_id=SITE_ID
        )
'''

The 'data' object is the dataframe which contains the relevant fields.
