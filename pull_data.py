#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:32:46 2024

@author: djelal
"""

import glob
from io import StringIO
import json
import os
import numpy as np
import pandas as pd

pd.set_option("display.precision", 8)

import requests
from scipy.interpolate import CubicSpline
import time
import warnings

from Encryption.token_manager import (
    decrypt_token,
    load_encrypted_token_from_file,
)

from Libraries import progress as pg


def get_api_token():
    """
    Decrypt the DigitalBuild API token and return the token as a string.

    Returns
    -------
    api_token : str
        The DigitalBuild API token.

    """
    # Load the encryption key from the environment variable
    encryption_key = os.environ["TOKEN_ENCRYPTION_KEY"]

    # Load the encrypted token from the file

    script_dir = os.path.dirname(os.path.realpath(__file__))
    encryption_path = os.path.join(script_dir, "Encryption")
    token_file_path = encryption_path + "/encrypted_digital_build_token.txt"
    encrypted_token = load_encrypted_token_from_file(token_file_path)

    # Decrypt the API token
    api_token = decrypt_token(encrypted_token, encryption_key)

    return api_token


def find_id_from_name(
    df: pd.DataFrame, name: str, name_col=None, id_col=None
) -> int:
    """
    Given a DataFrame, df, that contains a 'name' and an 'id' column, find the
    'id' that matches a given name.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame that contains a 'name' and 'id' coloumn.
    name : str
        The name to be matched to an ID.
    name_col : str, OPTIONAL
        Optionally define a different header for the 'name' column in which to
        search a match for the name input variable.
    id_col : str, OPTIONAL
        Optionally define a different header for the 'id' column in which to
        extract the ID int that matches the found name.

    Returns
    -------
    id_num : int
        The ID that matches the given name.

    """
    # Remove case and padded spaces from name
    name = name.lower().strip()

    # Default name column is 'name'
    if name_col is None:
        name_col = "name"

    # Default ID column is 'id'
    if id_col is None:
        id_col = "id"

    # Check that the name and ID columns exist in the DataFrame
    assert name_col in df.columns, "The header name_col is not in DataFrame"
    assert id_col in df.columns, "The header id_col is not in DataFrame"

    # Search for name and get corresponding ID
    try:
        id_num = df.loc[
            df[name_col].str.lower().str.strip() == name, id_col
        ].iloc[0]
    except:
        raise Exception(' No match was found for the given "name" entry.')

    return id_num


def find_name_from_id(
    df: pd.DataFrame, id_num: int, id_col=None, name_col=None
) -> str:
    """
    Given a DataFrame, df, that contains a 'name' and an 'id' column, find the
    'name' that matches a given 'id'.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame that contains a 'name' and 'id' coloumn.
    id_num : int
        The ID to be matched to a name.
    id_col : str, OPTIONAL
        Optionally define a different header for the 'id' column in which to
        extract the ID int that matches the found name.
    name_col : str, OPTIONAL
        Optionally define a different header for the 'name' column in which to
        search a match for the name input variable.

    Returns
    -------
    name: str
        The name that matches the given ID.

    """
    # Default name column is 'name'
    if name_col is None:
        name_col = "name"

    # Default ID column is 'id'
    if id_col is None:
        id_col = "id"

    # Check that the name and ID columns exist in the DataFrame
    assert name_col in df.columns, "The header name_col is not in DataFrame"
    assert id_col in df.columns, "The header id_col is not in DataFrame"

    # Search for name and get corresponding ID
    try:
        name = df.loc[df[id_col] == id_num, name_col].iloc[0]
    except:
        raise Exception(' No match was found for the given "id" entry.')

    return name


def get_site_info(auth_token: str) -> tuple[list, pd.DataFrame]:
    """
    Extract the top-level site information for all registered sites to-date and
    return that information as both a JSON object and a Pandas DataFrame.

    Each site has the following information associated with it:

        1. features [dictionary] - contains information of whether the site
                has bluetooth and whether live data is being collected.

        2. id [int] - Site ID

        3. name [str] - Registered site name

        4. createdAt [str] - Datetime that site was registered

        5. updatedAt [str] - Datetime that site information was updated

        6. addressLine1 [str]

        7. addressLine2 [str]

        8. town [str]

        9. country [str]

        10. postCode [str]

        11. latitude [float]

        12. longitude [float]

        13. measurementLocale [str] - metric / imperial

        14. branding [str] - Branding of devices ('converge' or other brand)

        15. organisationID [int]

        16. productsConfigurations [dictionary] - contains information of the
                product type, subscription type, whether the site is using a
                legacy product, whether it is a precast site and other tidbits.

        17. contact [str] - May contain contact information of connect on site

    Parameters
    ----------
    auth_token : str
        DigitalBuild API token.

    Returns
    -------
    json_response : list
        JSON object parsed as a list of dictionaries containing all site info.
    df : pd.DataFrame
        DataFrame of all site info.

    """
    # api call with variables
    request_string = "https://api.converge.io/sites"
    response = requests.get(
        request_string, headers={"PRIVATE-TOKEN": auth_token}
    )
    json_response = response.json()
    json_str = json.dumps(json_response)
    json_data = StringIO(json_str)

    # Convert the JSON response to a DataFrame
    df = pd.read_json(json_data)

    return json_response, df

def get_mix_design(site_id: int, auth_token: str) -> pd.DataFrame:
    
    # API request string
    request = (
        "https://api.converge.io/concrete_mix_designs?"
        + "&siteId="
        + str(site_id)
        + "&includeDrafts=true"
    )

    response = requests.get(
        request, headers={"PRIVATE-TOKEN": auth_token}
    )

    # List of dictionaries, where each list element is a different device
    # stream
    json_response = response.json()

    # Convert list of dictionaries to a DataFrame
    df_designs = pd.DataFrame(json_response)
    
    return df_designs

def get_sensors_for_site(
    site_id: int, auth_token: str
) -> tuple[list, pd.DataFrame]:
    """
    Pull the list of all sensors and the name of the pour to which they are
    associated at a given site. Fields returned are:

        1. id [int] - Sensor ID
        2. name [str] - Name of pour to which sensor belongs
        3. url [str] - app.converge.io URL containing customer-facing pour info

    Parameters
    ----------
    site_id : int
        Site ID.
    auth_token : str
        DigitalBuild API token.

    Returns
    -------
    json_response : list
        JSON object parsed as a list of dictionaries containing sensor IDs and
        associated pour names.
    df : pd.DataFrame
        DataFrame of all sensor IDs and associated pour names.

    """
    # api call with variables
    request_string = (
        "https://api.converge.io/integration/digitalbuild/sensors?siteId="
        + str(site_id)
    )  # the metric needs to be 2 for temperature profile#
    response = requests.get(
        request_string, headers={"PRIVATE-TOKEN": auth_token}
    )

    # response.json() will be a dictionary of sensor_id as Key and then a list
    # of dictionaries of data points in the form {streamId, value, time}
    json_response = response.json()

    # Convert the JSON response to a DataFrame
    df = pd.read_json(json.dumps(json_response))

    return json_response, df


def get_sensor_status(site_id: int, sensor_id: int, auth_token: str) -> dict:
    """
    Pull status of a specific sensor from a chosen site. The returned dictionary
    contains the following key-value pairs:

        1. id [int] - Sensor ID
        2. name [str] - Name of pour to which sensor belongs
        3. url [str] - converge.io URL with customer-facing pour data
        4. siteID
        5. temperature [float] - Last recorded temperature
        6. strength [float] - Last recorded strength
        7. startAt [str] - Time at which temperature stream began
        8. lastDatumAt [str] - Time at which last temperature was measured
        9. milestones [list] - Registered strength milestones
        10. pourStatus [str] - String stating if the pour has finished

    Parameters
    ----------
    site_id : int
        Site ID.
    sensor_id : int
        Sensor ID.
    auth_token : str
        DigitalBuild API.

    Returns
    -------
    sensor_status : dict
        Dictionary containing all sensor status attributes listed above.

    """
    request_string = (
        "https://api.converge.io/integration/digitalbuild/sensors/"
        + str(sensor_id)
        + "/status?siteId="
        + str(site_id)
    )
    response = requests.get(
        request_string, headers={"PRIVATE-TOKEN": auth_token}
    )

    sensor_status = response.json()

    return sensor_status


def get_devices_for_site(site_id, auth_token):
    """
    Pull the list of all the signal-live devices for a given site. Fields
    returned are:

        1. id [int] - Device ID
        2. name [str] - Name of Device (allocated by registering user)
        3. url [str] - converge.io URL with customer-facing pour data
        4. probes [dict] - Dictionary containing 'name' and 'url' of probe and
                probe webapp data stream, respectively.

    """
    # api call with variables
    request_string = (
        "https://api.converge.io/integration/digitalbuild/devices?siteId="
        + str(site_id)
    )  # the metric needs to be 2 for temperature profile#
    response = requests.get(
        request_string, headers={"PRIVATE-TOKEN": auth_token}
    )

    # response.json() will be a dictionary of sensor_id as Key and then a list
    # of dictionaries of data points in the form {streamId, value, time}
    json_response = response.json()

    # Convert the JSON response to a DataFrame
    df = pd.read_json(json.dumps(json_response))

    return json_response, df

def add_mix_columns(df_pours: pd.DataFrame) -> pd.DataFrame:
    """
    Given the DataFrame of all pours for a given site (which is the output of
    the get_pours_for_site() function), add two new columns titled mixId and
    mixName that are the concrete mix ID and name for a pour, respectively.

    Parameters
    ----------
    df_pours : pd.DataFrame
        The output of the get_pour_from_site() function.

    Returns
    -------
    df_pours : pd.DataFrame
        The DataFrame with the concrete mix IDs and names added.

    """
    df_pours["mixId"] = df_pours.concreteMix.apply(
        lambda x: x["id"] if x == x else -1
    )

    df_pours["mixName"] = df_pours.concreteMix.apply(
        lambda x: x["name"] if x == x else "Undefined"
    )

    return df_pours


def get_pours_for_site(
    site_id: int, auth_token: str
) -> tuple[list, pd.DataFrame]:
    """
    Pull the list of all pours for a given site. Fields returned are:

        1. id [int] - Pour ID
        2. name [str] - Name of pour
        3. elementType [str] - Element type
        4. concreteMix [dict] - Contains the numerical 'id' and 'name' of the
                concrete mix used in the pour
        5. sensors [dict] - Contains the sensor 'id' and 'name' of the sensors
                that are in the given pour
        6. webUrl [str] - converge.io URL of the pour
        7. dataUrl [str] - DigitalBuild API call that returns the stream of
                strength measurements
        8. differentialsUrl [str] - converge.io URL of temperature
                differentials (if available)
        9. mixId [int] - The ID for the concrete mix used in the pour (-1 if
                no mix is defined)
        10. mixName [str] - The name of the concrete mix used in the pour
                ('Undefined' if no mix is defined)

    Parameters
    ----------
    site_id : int
        Site ID.
    auth_token : str
        DigitalBuild API token.

    Returns
    -------
    json_response : list
        JSON object parsed as a list of dictionaries containing the above list
        of information per pour.
    df : pd.DataFrame
        DataFrame object of the above listed information per pour.

    """
    # api call with variables
    request_string = (
        "https://api.converge.io/integration/digitalbuild/pours?siteId="
        + str(site_id)
    )  # the metric needs to be 2 for temperature profile#

    response = requests.get(
        request_string, headers={"PRIVATE-TOKEN": auth_token}
    )

    # response.json() will be a dictionary of sensor_id as Key and then a list
    # of dictionaries of data points in the form {streamId, value, time}
    json_response = response.json()
    json_str = json.dumps(json_response)
    json_data = StringIO(json_str)
    
    # Convert the JSON response to a DataFrame
    df_pours = pd.read_json(json_data)

    try:
        # Add the concrete mix IDs and names for each pour as separate columns in
        # the pandas DataFrame
        df_pours = add_mix_columns(df_pours)
    except:
        df_pours["mixId"] = -1
        df_pours["mixName"] = "Undefined"

    return json_response, df_pours


def filter_pours_for_specific_mix(
    df_pours: pd.DataFrame, mixId: int = None, mixName: str = None
) -> tuple[pd.DataFrame, list]:
    """
    Filter out the DataFrame of all pours for a given site, choosing only the
    rows that contain pours from a user-defined concrete mix. The mix can be
    defined by either its name (string) or its ID (int).

    Parameters
    ----------
    df_pours : pd.DataFrame
        The output of the get_pours_for_site() function.
    mixId : int, optional
        The mix ID can be defined explicitly. The default is None.
    mixName : str, optional
        Optionally, the mix can be defined by its name. The default is None.

    Raises
    ------
    ValueError
        If both mixName and mixId are undefined, an exception is raised.

    Returns
    -------
    df_pours : pd.DataFrame
        The filtered DataFrame containing only pours that use the desired
        concrete mix.
    pourIds : list
        The list of all pour IDs that match the desired concrete mix

    """
    # If the mixName is defined, then find the mixId
    if mixName:
        mixId = find_id_from_name(df_pours, mixName)
    elif mixId is None:
        raise ValueError(" Either mixName or mixId must be defined.")

    # Filter out only the pours with the specified concrete mix
    df_pours = df_pours[df_pours["mixId"] == mixId]

    # List of all matching pour IDs
    pourIDs = list(df_pours["id"].values)

    return df_pours, pourIDs


def get_data_from_pour(pour_id, site_id, auth_token, dtype="temperature"):
    """
    Pull the measured data from all sensors/devices associated with a given
    Pour. The return fields are:

        1. streamId [str] - Stream ID for raw temperature stream
        2. value [float] - Recorded temperature value
        3. time [int] - Unix time as number of ns from universal reference

    If the dtype variable is specified as 'temperature' of 'strength' then the
    temperature or strength are extracted, respectively.

    Parameters
    ----------
    pour_id : int
        Pour ID.
    site_id : int
        Site ID.
    auth_token : str
        DigitalBuild API token.
    dtype : str, optional
        Variable denoting which data to extract (temperature or strength).


    Returns
    -------
    json_data_from_pour : dict
        A dictionary of dictionaries. The top-level dictionary has key-value
        pairs of (deviceID, list). The list allocated to each device is a list
        of dictionaries, where each dictionary contains the streamId, value
        and timestamp of the recorded temperature value.

    """
    if dtype == "temperature":
        metricId = 2
    elif dtype == "strength":
        metricId = 3
    else:
        raise ValueError('dtype must be either "temperature" or "strength"')

    # api call with variables
    request_string = (
        "https://api.converge.io/integration/digitalbuild/pours/"
        + str(pour_id)
        + "/data?siteId="
        + str(site_id)
        + "&metricId="
        + str(metricId)
    )
    response = requests.get(
        request_string, headers={"PRIVATE-TOKEN": auth_token}
    )

    # response.json() will be a dictionary of sensor_id as Key and then a list
    # of dictionaries of data points in the form {streamId, value, time}
    data_from_pour = response.json()

    return data_from_pour


def convert_data_from_pour_to_dataframe(
        data_from_pour: dict,
        dtype: str = 'temperature'
    ) -> pd.DataFrame:
    """
    Convert the output from the get_data_from_pour() function into a pandas
    DataFrame that contains all devices and all measurements from the pour ID
    that was used to extract the data_from_pour object. The output DataFrame
    contains the following columns:

        1. device_name [str] - Device ID in the form of a string
        2. time [pd.Timestamp] - The time of the corresponding measurement in
                form of a Pandas timestamp
        3. stream_id [str] - The ID of the temperature stream
        4. temperature [float] - The measured temperature at the timestamp for
                the given device
        5. timedelta [pd.Timedelta] - The relative difference between adjacent
                timestamps in the 'time' column. The first entry (for which
                there is no timedelta) contains a NaT (Not a Time) data type
                entry.

    Parameters
    ----------
    data_from_pour : dict
        The output of the get_data_from_pour() function.
    dtype : str
        Define if the data in data_from_pour is temperature or strength.

    Returns
    -------
    pour_data : pd.DataFrame
        The dataframe containing the above fields.

    """
    # Initialise a list that will contain the flattened data recorded from all
    # devices in the pour
    flat_data = []
    for device, records in data_from_pour.items():
        # For a given device, we sort the recorded temperatures in order of
        # ascending time
        sorted_records = sorted(records, key=lambda x: x["time"])

        # For a given set of sorted measurements, we iterate through all
        # records (where each record is a dictionary) and unpack the dictionary
        # values within a tuple, beginning with the device ID.
        for record in sorted_records:
            flat_data.append((device, *record.values()))
            
    assert dtype in ['temperature', 'strength'], (
        "Data type, dtype, must be 'temperature' or 'strength'."
    )

    # We define the columns of the MultiIndex DataFrame in the correct order
    # relative to the above iteratively unpacked values
    columns = pd.MultiIndex.from_tuples(
        [
            ("device_name", ""),
            ("stream_id", ""),
            (dtype, ""),
            ("time", ""),
        ]
    )

    # Create the DataFrame with the MultiIndex columns
    df = pd.DataFrame(flat_data, columns=columns)

    # Set the MultiIndex values of the DataFrame to be given by the device_name
    # and time of measurement
    df = df.set_index([("device_name", ""), ("time", "")])

    # We ensure the final DataFrame stores the time values in Datetime format
    pour_data = df.reset_index()
    pour_data[("time", "")] = pd.to_datetime(pour_data[("time", "")], unit="s")

    # Create a copy of the DataFrame with single-level columns
    pour_data_copy = pour_data.copy()
    pour_data_copy.columns = pour_data_copy.columns.get_level_values(0)

    # Calculate timedelta values using the 'diff' method after grouping by
    # 'device_name'
    pour_data_copy["timedelta"] = pour_data_copy.groupby("device_name")[
        "time"
    ].diff()

    # Add the calculated timedelta values back to the original DataFrame
    pour_data[("timedelta", "")] = pour_data_copy["timedelta"]
    
    pour_data.columns = pour_data.columns.droplevel(1)
    
    return pour_data


def get_transform_info(auth_token: str, siteId: int) -> pd.DataFrame:
    """
    Extract information about all transform streams attached to the given site.
    This function calls the /transforms API path. The output of this path
    contains many variables, but we are only concerned with the 'startAt'
    element, which contains the timestamp at which the strength measurements
    began (if strength measurements exist).

    Parameters
    ----------
    auth_token : str
        DigitalBuild API token.
    siteId : int
        The site ID.

    Returns
    -------
    df_transforms : pd.DataFrame
        The DataFrame of all transform information, where each row encodes
        information for a different device stream. Only the startAt column of
        the DataFrame is of importance.

    """
    # API request string
    request = "https://api.converge.io/transforms?" + "&siteId=" + str(siteId)

    response = requests.get(
        request, headers={"PRIVATE-TOKEN": get_api_token()}
    )

    # List of dictionaries, where each list element is a different device
    # stream
    json_response = response.json()

    # Convert list of dictionaries to a DataFrame
    df_transforms = pd.DataFrame(json_response)

    return df_transforms


def get_transform_start_times_for_streamIds(
    auth_token, data: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """
    Given a DataFrame, data, that contains a column of stream IDs and a column
    of site IDs from which those stream IDs derive, output a dictionary that
    maps each stream ID to its corresponding transform start time. This
    dictionary is then used by the Pandas map() function to add a new column
    to the data DataFrame, which contains the startAt time for the stream ID
    in every row.

    Parameters
    ----------
    auth_token : str
        DigitalBuild API token.
    data : pd.DataFrame
        A DataFrame contain one columns of stream IDs and another column of
        site IDs, such that the site from which a given stream ID derives is
        defined.

    Returns
    -------
    data: pd.DataFrame
        The input DataFrame with the added 'startAt' column, containing the
        corresponding transform start time for each stream ID.
    start_times: dict
        The dictionary containing the (stream ID, transform start time)
        key-value pairs.

    """
    # The list of all unique site IDs in the DataFrame
    uniqueSiteIDs = data["siteId"].unique()

    # Initialise the final dictionary of all start times
    start_times = {}

    # Iterate over all sites and add the (stream ID, start time) pairs to the
    # start_times dictionary
    for siteId in uniqueSiteIDs:
        df_transforms = get_transform_info(auth_token, siteId)
        uniqueStreamIDs = data["stream_id"].unique()
        for streamId in uniqueStreamIDs:
            try:
                current_time = pd.to_datetime(
                    df_transforms[
                        df_transforms["streamId"] == streamId
                    ].startAt.iloc[0]
                ).tz_convert("UTC")

                current_time = current_time.tz_localize(None)
            except:
                current_time = pd.NaT

            start_times[streamId] = current_time

    data["startAt"] = data["stream_id"].map(start_times)

    return data, start_times


def get_true_startAt(data: pd.DataFrame) -> pd.DataFrame:
    # The minimum absolute temperature difference between two data points for
    # a new start time to be allocated
    TEMP_DIFF_THRESHOLD = 1.0

    # We will search for the true startAt time within a half day forward or
    # backwards from the registered start time
    pourTime_range = {"start": -0.5, "end": 0.5}

    # Find all row indices that fall within this pourTime range
    pourTime_rows = (data["pourTime"] >= pourTime_range["start"]) & (
        data["pourTime"] <= pourTime_range["end"]
    )

    data["startAtTrue"] = pd.NaT

    devices_with_altered_starts = []
    devices = data["device_name"].unique()
    for device in devices:
        # Row indices of the current device
        device_rows = data["device_name"] == device

        # Find the rows that correspond to the correct time window for the
        # current device
        rows = pourTime_rows & device_rows

        # Take the absolut diff of the temperatures within that range
        temp_diff = data.loc[rows, "temperature"].diff().abs()

        # Are there any startAt candidates?
        candidates = temp_diff >= TEMP_DIFF_THRESHOLD
        if candidates.any():
            # If there are, find the maximum temperature diff above the minimum
            # threshold, and allocate the 'time' value as the true startAt time
            max_diff = temp_diff.max()

            # Find the rows with a temperature difference that matches the max
            max_diff_rows = temp_diff == max_diff

            # Get the index value of those rows and choose that last index
            # (i.e. the latest one in time)
            max_diff_ind = temp_diff.loc[max_diff_rows].index[-1]

            # Get the time at which the max diff occurred
            max_diff_time = data["time"].iloc[max_diff_ind]

            # Allocate the new startAt time to all startAt_true rows of this
            # device
            data.loc[device_rows, "startAtTrue"] = max_diff_time

            # Add the current device to the list of devices with altered
            # start times
            devices_with_altered_starts.append(device)

    return data, devices_with_altered_starts


def get_temperatures_from_specific_pour(
    auth_token: str, site_name=None, site_id=None, pour_name=None, pour_id=None
) -> tuple[pd.DataFrame, int, int]:
    """
    Extract all temperature measurements from a given pour. The output of this
    function is a DataFrame, pour_data, containing all measurements and
    relevant device names and times, as well as other labels and derived
    attributes.

    The fields returned in the DataFrame are:

        1. device_name [str] - Device ID in the form of a string
        2. time [pd.Timestamp] - The time of the corresponding measurement in
                form of a Pandas timestamp
        3. stream_id [str] - The ID of the temperature stream
        4. temperature [float] - The measured temperature at the timestamp for
                the given device
        5. timedelta [pd.Timedelta] - The relative difference between adjacent
                timestamps in the 'time' column. The first entry (for which
                there is no timedelta) contains a NaT (Not a Time) data type
                entry.
        6. strength [float] - Strength measurement (if present)
        7. pourId [int] - The pour ID
        8. pourName [str] - The pour name
        9. siteId [int] - The site ID
        10. siteName [str] - The site name
        11. elementType [str] - The type of the element being poured
        12. startAt [pd.Timestamp] - Time at which pour was started
        13. pourTime [float] - Time since startAt [in days]
        14. startAtTrue [pd.Timestamp] - True start time (if different)
        15. mixId [ind] - The concrete mix ID
        16. mixName [str] - The concrete mix Name

    This function requires that both a site and pour be specified. However, it
    permits the user to define a site by either its registered name (a string)
    or its registered site ID value (an int). Likewise, a pour can be specified
    by either its pour_name or its pour_id.

    It is not permitted for neither a name nor an ID to be provided for a site
    or pour, and doing so will raise an Exception.

    Parameters
    ----------
    auth_token : str
        DigitalBuild API token.
    site_name : str, optional
        The registered site name. The default is None.
    site_id : str, optional
        The registered site ID. The default is None.
    pour_name : str, optional
        The name of the pour from within the defined site. The default is None.
    pour_id : str, optional
        The ID of the pour from within the defined site. The default is None.

    Raises
    ------
    ValueError
        If neither the name or the ID value are provided for a site or pour the
        function will raise an Exception.

    Returns
    -------
    pour_data : pd.DataFrame
        A DataFrame containing the fields defined above.
    site_id : int
        The site ID from which the pour is chosen.
    pour_id : int
        The pour ID from which the data is derived.

    """
    # Read in all site names, IDs and other site info as a DataFrame
    _, df_sites = get_site_info(auth_token)

    if site_name:
        site_id = find_id_from_name(df_sites, site_name)
    elif site_id is None:
        raise ValueError(" Please provide either the site_id or site_name.")
    else:
        site_name = find_name_from_id(df_sites, site_id)

    _, df_pours = get_pours_for_site(site_id, auth_token)

    if pour_name:
        pour_id = find_id_from_name(df_pours, pour_name)
    elif pour_id is None:
        raise ValueError(" Please provide either the pour_id or pour_name.")
    else:
        pour_name = find_name_from_id(df_pours, pour_id)

    # Extract temperature data from current pour and site
    data_from_pour = get_data_from_pour(pour_id, site_id, auth_token)
    pour_data = convert_data_from_pour_to_dataframe(data_from_pour)   

    strength_from_pour = get_data_from_pour(
        pour_id, site_id, auth_token, dtype='strength'
    )
    if len(strength_from_pour):
        strength_data = convert_data_from_pour_to_dataframe(
            strength_from_pour, dtype='strength'
        )
        
        pour_data = pd.merge(
            left=pour_data,
            right=strength_data[["device_name", "time", "strength"]],
            on=["device_name","time"],
            how="inner",
        )
    else:
        pour_data['strength'] = np.NaN

    # Add the pour and site IDs and names to the DataFrame
    pour_data["pourId"] = pour_id
    pour_data["pourName"] = pour_name
    pour_data["siteId"] = site_id
    pour_data["siteName"] = site_name

    # Add the elementType to the pour_data DataFrame
    elementType = df_pours[df_pours["id"] == pour_id]["elementType"].iloc[0]
    pour_data["elementType"] = elementType

    # Add the transform start times to a column in the DataFrame called
    # 'startAt' (returns NaT value if no startAt time exists)
    pour_data, _ = get_transform_start_times_for_streamIds(
        auth_token, pour_data
    )

    # Calculate the time since pour time in units of days (if startAt exists)
    try:
        pour_data["pourTime"] = pour_data["time"] - pour_data["startAt"]
        pour_data["pourTime"] = pour_data["pourTime"].dt.total_seconds() / (
            24 * 60 * 60
        )
    except:
        pour_data["pourTime"] = np.NaN

    # Check to see if the pour should occur at a different time. If so,
    # populate the startAtTrue column with the appropriate time for that device
    pour_data, devices_with_altered_starts = get_true_startAt(pour_data)

    # Overwrite pour times with new start times, where applicable
    if devices_with_altered_starts:
        for device in devices_with_altered_starts:
            inds = pour_data["device_name"] == device
            pourTime = (
                pour_data.loc[inds, "time"]
                - pour_data.loc[inds, "startAtTrue"]
            ).dt.total_seconds() / (24 * 60 * 60)

            pour_data.loc[inds, "pourTime"] = pourTime.values

    # Create dictionary mappings from pourId to mixId and mixName
    map2mixId = dict(zip(df_pours["id"], df_pours["mixId"]))
    map2mixName = dict(zip(df_pours["id"], df_pours["mixName"]))

    # Add mixId and mixName attributes as columns in the DataFrame
    pour_data["mixId"] = pour_data["pourId"].map(map2mixId)
    pour_data["mixName"] = pour_data["pourId"].map(map2mixName)

    return pour_data


def get_temperatures_from_set_of_pours(
    auth_token: str, pourIDs: list, site_name: str = None, site_id: int = None
) -> pd.DataFrame:
    """
    Get all data from a set of user-defined pours belonging to a user-defined
    site. The set of pours is defined by a list of pour IDs. The site is
    defined by either its name or its site ID.

    The output of this function is a pandas DataFrame that contains all
    temperature data points extracted from every device in every pour. The
    output DataFrame contains the same columns as the output from
    get_temperatures_from_specific_pour().

    Parameters
    ----------
    auth_token : str
        The DigitalBuild API token.
    pourIDs : list
        A list of all pour IDs to be extracted.
    site_name : str, optional
        The name of the site. The default is None.
    site_id : int, optional
        The ID of the site. The default is None.

    Returns
    -------
    data : pd.DataFrame
        DataFrame containing all temperature measurements from all devices
        in the set of pours defined by the pourIDs list.

    """
    num_pours = len(pourIDs)
    assert num_pours > 0, " pourIDs must not be an empty list"

    # Begin timing the function
    init_time = time.time()

    # Extract the first pour
    data = get_temperatures_from_specific_pour(
        auth_token, site_name=site_name, site_id=site_id, pour_id=pourIDs[0]
    )

    # Begin tracking progress of data extraction
    pg.loop_progress(0, num_pours, init_time)

    # If the number of pours is greater than 1, iterate over all additional
    # pours and append to the existing DataFrame
    if num_pours > 1:
        for ind in range(1, num_pours):
            next_pour_data = get_temperatures_from_specific_pour(
                auth_token,
                site_name=site_name,
                site_id=site_id,
                pour_id=pourIDs[ind],
            )

            pg.loop_progress(ind, num_pours, init_time)

            warnings.filterwarnings(
                "ignore",
                message=".*The behavior of DataFrame concatenation.*"
            )
            
            data = pd.concat((data, next_pour_data), axis=0)

    return data


def get_temperatures_for_site_and_mix(
    auth_token: str,
    site_info: pd.DataFrame,
    site_id: int = None,
    site_name: str = None,
    mix_id: int = None,
    mix_name: str = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Extract all temperature values for only a specified concrete mix from the
    given site. The output DataFrame contains the same columns as the output
    from get_temperatures_from_specific_pour().

    Parameters
    ----------
    auth_token : str
        DigitalBuild API token.
    site_info : pd.DataFrame
        DataFrame output from the get_site_info() function.
    site_id : int, optional
        A site ID. The default is None.
    site_name : str, optional
        A site name. The default is None.
    mix_id : int, optional
        A concrete mix ID. The default is None.
    mix_name : str, optional
        A concrete mix name. The default is None.

    Raises
    ------
    Exception
        Raised if either the no site or no mix is specified.

    Returns
    -------
    data : pd.DataFrame
        All temperatures for all devices in pours of the specified mix at the
        specified sit.
    df_mix : pd.DataFrame
        DataFrame of information about all pours of the desired mix type.
    pourIDs : list
        list of all pour IDs for the chosen pours.

    """
    if site_name:
        site_id = find_id_from_name(site_info, site_name)
    elif site_id is None:
        raise Exception(" Please specify a valid site_id or site_name")

    # Get DataFrame of all pours on site
    _, df_pours = get_pours_for_site(site_id, auth_token)

    if mix_name:
        mix_id = find_id_from_name(
            df_pours, mix_name, name_col="mixName", id_col="mixId"
        )
    elif mix_id is None:
        raise Exception(" Please specify a valid mix_id or mix_name")

    # Fiter df_pours for only the rows that contain the specified concrete mix
    df_mix, pourIDs = filter_pours_for_specific_mix(df_pours, mixId=mix_id)

    if site_name is None:
        site_name = find_name_from_id(site_info, site_id)
    if mix_name is None:
        mix_name = find_name_from_id(
            df_pours, mix_id, id_col="mixId", name_col="mixName"
        )

    print("\n")
    print(
        ' EXTRACTING TEMPERATURES FOR MIX "{0}" AT SITE "{1}"'.format(
            mix_name, site_name
        )
    )
    dash_str = "-"
    print(110 * dash_str, "\n")

    # Extract all temperatures from pours with specified mix
    data = get_temperatures_from_set_of_pours(
        auth_token, pourIDs, site_id=site_id
    )

    return data, df_mix, pourIDs


def get_all_temperatures_for_site(
    auth_token: str,
    site_info: pd.DataFrame,
    site_id: int = None,
    site_name: str = None,
) -> pd.DataFrame:
    """
    Extract all temperature values from all pours on the given site. The output
    DataFrame contains the same columns as the output from
    get_temperatures_from_specific_pour().

    Parameters
    ----------
    auth_token : str
        DigitalBuild API token.
    site_info : pd.DataFrame
        DataFrame output from the get_site_info() function.
    site_id : int, optional
        A site ID. The default is None.
    site_name : str, optional
        A site name. The default is None.

    Raises
    ------
    Exception
        Raised if either the no site is specified.

    Returns
    -------
    data : pd.DataFrame
        All temperatures for all devices at the specified sit.
    df_mix : pd.DataFrame
        DataFrame of information about all pours.
    pourIDs : list
        list of all pour IDs at the site.

    """
    if site_name:
        site_id = find_id_from_name(site_info, site_name)
    elif site_id is None:
        raise Exception(" Please specify a valid site_id or site_name")

    # Get DataFrame of all pours on site
    _, df_pours = get_pours_for_site(site_id, auth_token)

    assert len(df_pours) > 0, "ERROR: No pours registered for this site."

    if site_name is None:
        site_name = find_name_from_id(site_info, site_id)

    print(' EXTRACTING ALL TEMPERATURES FROM SITE "{0}"'.format(site_name))
    dash_str = "-"
    print(110 * dash_str, "\n")

    # Get all pour IDs
    pourIDs = list(df_pours.id.values)

    # Extract all temperatures from pours with specified mix
    data = get_temperatures_from_set_of_pours(
        auth_token, pourIDs, site_id=site_id
    )

    return data, df_pours, pourIDs
