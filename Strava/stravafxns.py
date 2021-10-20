import pandas as pd
import requests
import urllib3
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pandas.api.types import CategoricalDtype

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

auth_url = "https://www.strava.com/oauth/token"
activites_url = "https://www.strava.com/api/v3/athlete/activities"
single_act_url = "https://www.strava.com/api/v3/activities"

clientID = "73092"
client_secret = "7ce9c1a9ef12c146f5e3e25bc5761e2d7c6e7009"
code = "52ec5853b048a139d88b4754b2783c7753c9c1e8"
refresh_tok = "24507a9c490400cc0f281775ba0ee7b1db1d2677"
access_tok = "ccb217b4c599b9a15b6550cbd1d2bca2986c937a"

payload = {
    'client_id': "73092",
    'client_secret': '7ce9c1a9ef12c146f5e3e25bc5761e2d7c6e7009',
    'refresh_token': '24507a9c490400cc0f281775ba0ee7b1db1d2677',
    'grant_type': "refresh_token",
    'f': 'json'
}

def getAccess():
    print("Requesting Token...\n")
    res = requests.post(auth_url, data=payload, verify=False)
    access_token = res.json()['access_token']
    print("Access Token = {}\n".format(access_token))
    return access_token

def getActivities(access_token):
    header = {'Authorization': 'Bearer ' + access_token}
    param = {'per_page': 200, 'page': 5}
    my_dataset = requests.get(activites_url, headers=header, params=param).json()
    output = pd.json_normalize(my_dataset)
    return output

def getActivity(actID,access_token):
    header = {'Authorization': 'Bearer ' + access_token}
    param = {'per_page': 200, 'page': 1}
    activity = requests.get(single_act_url+'/'+str(actID)+'/streams?keys=time,distance,heartrate,cadence,watts,temp&key_by_type=true', headers=header).json()
    #print(activity)
    activityPD = pd.json_normalize(activity)
    return activityPD


# test = getActivities(getAccess())
# print(test.info)