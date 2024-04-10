# Code Execution

The following is the line of code that executes the data extraction:

```
data, df_pours, pourIDs = p.get_all_temperatures_for_site(
            p.get_api_token(), site_info, site_id=SITE_ID
        )
```

The `data` object is the dataframe which contains the relevant fields.

The `p.get_api_token()` variable can be replaced with a DigitalBuild API token. The `site_info` is a dataframe that contains the output of the `"https://api.converge.io/sites"` API endpoint. The `site_id` should be the `id` value from the correct site in the `site_info` dataframe.
