import numpy as np
import pandas as pd
import json
import re
import time
import math
import csv
from sklearn.utils import shuffle
import pandas as pd
import requests
from darkskylib.darksky import forecast
from datetime import datetime as dt
from datetime import timedelta
from dateutil import parser as dateparse

with open('secret_key1.txt', 'r') as myfile:
    WEATHER_KEY = myfile.read().replace('\n', '')

# with open('secret_key2.txt', 'r') as myfile:
#     AGRO_KEY = myfile.read().replace('\n', '')

############################################################################

class Data(object):
    """
    Class to handle loading and processing of raw datasets.
    """

    def __init__(self, data_source, num_of_classes):
        """
        Initialization of a Data object.
        Args:
            data_source (str): Raw data file path
            num_of_classes (int): Number of classes in data
        """
        self.no_of_classes = num_of_classes
        self.data_source = data_source

    ######################################
    def load_data(self):
        """
        Load raw data from the source file into data variable.
        Returns: None
        """
        data = []
        with open(self.data_source, 'r', encoding='utf-8') as f:
            rdr = csv.reader(f, delimiter='\t')
            for row in rdr:
                # fill up data array
                data.append(row)

        self.data = np.array(data, dtype='float64')
        # do counts of each
        df = pd.DataFrame(self.data)
        print(self.data_source, 'total class counts:')
        print(df[0].value_counts())
        return(self.data)

    ######################################
    def get_all_data(self):
        """
        Return all loaded data from data variable.
        Returns:
            (np.ndarray) Data transformed from raw to indexed form with associated one-hot label.
        """
        numericaldata = []
        classes = []
        one_hot = np.eye(self.no_of_classes, dtype='int64')
        for d in self.data:
            c = int(float(d[0]))
            classes.append(one_hot[c-1])
            numericaldata.append(d[1:])
        # return arrays [365, -122.2, -82, 75, 12, ..], [1,0,0,0, ..]
        return np.asarray(numericaldata, dtype='float64'), np.asarray(classes)

############################################################################

class APIdata(object):
    """
    Class to handle downloading of dataset from APIs
    """

    def __init__(self):
        self.data = []
        # The more genuses will help it learn about danger in general .. and help classify the one genus I want?
        self.genuses = [
            "Ursus",
            "Puma",
            "Heloderma",
            "Crotalus"
        ]


    ######################################
    def unix_time_millis(dt):
        epoch = dt.utcfromtimestamp(0)
        return (dt - epoch).total_seconds() * 1000.0

    ######################################
    def inaturalistAPI(self):
        """
        Load observations from iNaturalist
        60/min rate-limit
        """
        MAX_OBSERVATIONS = 1000
        print('MAX_OBSERVATIONS: ' + str(MAX_OBSERVATIONS))
        datapoints = {}
        # collect all datapoints and return it
        for genus in self.genuses:
            observations = []
            url = "http://api.inaturalist.org/v1/taxa?q=" + genus + "&rank=genus"
            with requests.Session() as s:
                time.sleep(1)
                content = s.get(url)
                if content.status_code == 200:
                    content = content.json()
                    taxon_id = content["results"][0]["id"]
                    finished = False
                    ct = 0
                    page = 1
                    skip = 0

                    while not finished:
                        obs_url = ''.join([
                            'http://api.inaturalist.org/v1/observations?',
                            '&taxon_id=' + str(taxon_id),
                            '&page=' + str(page),
                            '&quality_grade=research',
                            '&per_page=30&order=desc&order_by=observed_on'
                        ])
                        with requests.Session() as s2:
                            time.sleep(1)
                            content2 = s2.get(obs_url)
                            if content2.status_code == 200:
                                content2 = content2.json()
                                # tag with class number
                                for res in content2['results']:
                                    res['genus'] = genus
                                    res['classnumber'] = self.genuses.index(genus) + 1
                                observations = observations + content2['results']
                                page += 1
                                skip += 30
                                if skip >= content2['total_results'] or skip >= MAX_OBSERVATIONS:
                                    finished = True

                                # small amount for testing
                                # if page > 1:
                                #     finished = True
                            else:
                                print('status_code 2:', content2.status_code, obs_url)
                else:
                    print('status_code 1:', content.status_code, url)

            # have all genus observations
            print(len(observations), 'observations:', genus)

            for obs in observations:
                if obs['location'] is not None and obs['time_observed_at'] is not None:
                    datapoint = {
                        'time_observed_at': obs["time_observed_at"],
                        'lat': obs['location'].split(',')[0],
                        'lon': obs['location'].split(',')[1],
                        'classnumber': obs['classnumber'],
                        'genus': obs['genus'],
                        'name': obs['taxon']['name']
                    }
                    datapoints[obs["id"]] = datapoint

                    # For NDVI
                    # need minimum of 1 hectare for bounding box
                    # halfside_km = 0.051
                    # latmin, lonmin, latmax, lonmax = boundingBox(
                    #     float(datapoint['lat']), float(datapoint['lon']), halfside_km)
                    # datapoint['latmin'] = latmin
                    # datapoint['lonmin'] = lonmin
                    # datapoint['latmax'] = latmax
                    # datapoint['lonmax'] = lonmax

        # return data for weather API function to use
        return(datapoints)

    ######################################

    def weatherAPI(self, datapoints, predict=False):
        """
        Load observations from weather api
        10,000 calls/$
        """
        print('getting darksky ...')
        print('total observations:', len(datapoints))
        print('will cost $', (len(datapoints) / 10000) * 2, 'dollars')
        count = 0
        for id in datapoints:
            # for testing small amounts, break soon
            count += 1
            # if count == 20:
            #     break
            if count % 50 == 0:
                print(count, 'weather api done of', len(datapoints))

            du = dateparse.parse(datapoints[id]['time_observed_at'])
            # default to noon of the day
            date_o = dt(du.year, du.month, du.day, 12)
            new_years_day = dt(year=date_o.year, month=1, day=1)
            day_of_year = (date_o - new_years_day).days + 1
            datapoints[id]['day_of_year'] = day_of_year

            date_o_minus1 = date_o - timedelta(1) # subtract 1 day
            date_o_minus2 = date_o - timedelta(2) # subtract 2 days
            date_o_minus7 = date_o - timedelta(7) # subtract a week

            t = date_o.isoformat()
            loc = WEATHER_KEY, datapoints[id]['lat'], datapoints[id]['lon']

            # get day's and past forcasts
            print('danger at', loc[1], loc[2])
            weather = forecast(*loc, time=t)
            w_previous = forecast(*loc, time=date_o_minus1.isoformat())
            # uncomment these to go farther back in time
            # w_previous2 = forecast(*loc, time=date_o_minus2.isoformat())
            # w_previous7 = forecast(*loc, time=date_o_minus7.isoformat())

            # darksky example saved in data folder (has other points like uvIndex, dewPoint, etc.)
            if (hasattr(weather, 'daily') and len(weather.daily) > 0):
                d0 = weather.daily[0]

                # barometric pressure
                if hasattr(d0, 'pressure'):
                    datapoints[id]['pressure'] = d0.pressure
                else:
                    datapoints[id]['pressure'] = 1015.0  # sea level avg good idea?

                # temps
                if hasattr(d0, 'temperatureMax') and hasattr(d0, 'temperatureMin'):
                    datapoints[id]['temperatureMax'] = d0.temperatureMax
                    datapoints[id]['temperatureMin'] = d0.temperatureMin
                    # precipitation biased towards days people go out w/ their cameras .. soil moisture is more important
                    if hasattr(d0, 'precipProbability'):
                        datapoints[id]['precipProbability'] = d0.precipProbability
                    else:
                        datapoints[id]['precipProbability'] = 0.0

                    # previous day datapoints
                    if (hasattr(w_previous, 'daily') and len(w_previous.daily) > 0):
                        d1 = w_previous.daily[0]
                        if hasattr(d1, 'temperatureMax') and hasattr(d1, 'temperatureMin'):
                            datapoints[id]['temperatureMaxPrevDay1'] = d1.temperatureMax
                            datapoints[id]['temperatureMinPrevDay1'] = d1.temperatureMin
                            if hasattr(d1, 'precipProbability'):
                                datapoints[id]['precipProbabilityPreviousDay'] = d1.precipProbability
                            else:
                                datapoints[id]['precipProbabilityPreviousDay'] = 0.0

                    # uncomment these to go farther back in time
                    # if (hasattr(w_previous2, 'daily') and len(w_previous2.daily) > 0):
                    #     d2 = w_previous2.daily[0]
                    #     if hasattr(d2, 'temperatureMax') and hasattr(d2, 'temperatureMin'):
                    #         datapoints[id]['temperatureMaxPrevDay2'] = d2.temperatureMax
                    #         datapoints[id]['temperatureMinPrevDay2'] = d2.temperatureMin
                    #         if hasattr(d2, 'precipProbability'):
                    #             datapoints[id]['precipProbabilityPreviousDay2'] = d2.precipProbability
                    #         else:
                    #             datapoints[id]['precipProbabilityPreviousDay2'] = 0.0
                    #
                    # if (hasattr(w_previous7, 'daily') and len(w_previous7.daily) > 0):
                    #     d7 = w_previous7.daily[0]
                    #     if hasattr(d7, 'temperatureMax') and hasattr(d7, 'temperatureMin'):
                    #         datapoints[id]['temperatureMaxPrevDay7'] = d7.temperatureMax
                    #         datapoints[id]['temperatureMinPrevDay7'] = d7.temperatureMin
                    #         if hasattr(d7, 'precipProbability'):
                    #             datapoints[id]['precipProbabilityPreviousDay7'] = d7.precipProbability
                    #         else:
                    #             datapoints[id]['precipProbabilityPreviousDay7'] = 0.0

        # print(json.dumps(datapoints))

        if predict:
            # if doing predictions, just return the datapoints, else save the training data
            return datapoints
        else:
            # when done, make a dataframe
            df = pd.read_json(json.dumps(datapoints), orient='index')
            #df.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
            df.to_csv('data/total_points.tsv', sep='\t')

            # pare down to just what is needed for ml
            df2 = df.drop(columns=['name', 'genus', 'time_observed_at'])
            df2= df2.dropna() # drop rows w/ any missing data

            # put class first
            classnumber = df2['classnumber']
            df2 = df2.drop(columns=['classnumber'])
            df2.insert(0, 'classnumber', classnumber)

            print('headers: ', list(df2))
            data = df2.as_matrix()
            tsv = shuffle(data)
            TRAINING_SPLIT = 0.80
            Ntrain = int(TRAINING_SPLIT * len(tsv))
            tsvTrain, tsvTest = tsv[:Ntrain], tsv[Ntrain:]

            np.savetxt('data/classes.txt', self.genuses, delimiter='\n', fmt="%s")
            np.savetxt('data/headers.txt', list(df2), delimiter='\n', fmt="%s")
            np.savetxt('data/train.tsv', tsvTrain, delimiter='\t', fmt="%s")
            np.savetxt('data/test.tsv', tsvTest, delimiter='\t', fmt="%s")

            return None

############################################################################

# NDVI stuff below, putting off for now.
# soil moisture history starting feb 1st 2018
# https://samples.openweathermap.org/agro/1.0/soil/history?polyid=5aaa8052cbbbb5000b73ff66&start=1517443200&end=1519776000&appid=bb0664ed43c153aa072c760594d775a7
# For NDVI
# From: https://stackoverflow.com/questions/238260/how-to-calculate-the-bounding-box-for-a-given-lat-lng-location
# degrees to radians
# def deg2rad(degrees):
#     return math.pi * degrees / 180.0
# # radians to degrees
#
# def rad2deg(radians):
#     return 180.0 * radians / math.pi
#
# # Semi-axes of WGS-84 geoidal reference
# WGS84_a = 6378137.0  # Major semiaxis [m]
# WGS84_b = 6356752.3  # Minor semiaxis [m]
#
# # Earth radius at a given latitude, according to the WGS-84 ellipsoid [m]
# def WGS84EarthRadius(lat):
#     # http://en.wikipedia.org/wiki/Earth_radius
#     An = WGS84_a * WGS84_a * math.cos(lat)
#     Bn = WGS84_b * WGS84_b * math.sin(lat)
#     Ad = WGS84_a * math.cos(lat)
#     Bd = WGS84_b * math.sin(lat)
#     return math.sqrt((An * An + Bn * Bn) / (Ad * Ad + Bd * Bd))
#
# # Bounding box surrounding the point at given coordinates,
# # assuming local approximation of Earth surface as a sphere
# # of radius given by WGS84
# def boundingBox(latitudeInDegrees, longitudeInDegrees, halfSideInKm):
#     lat = deg2rad(latitudeInDegrees)
#     lon = deg2rad(longitudeInDegrees)
#     halfSide = 1000 * halfSideInKm
#
#     # Radius of Earth at given latitude
#     radius = WGS84EarthRadius(lat)
#     # Radius of the parallel at given latitude
#     pradius = radius * math.cos(lat)
#
#     latMin = lat - halfSide / radius
#     latMax = lat + halfSide / radius
#     lonMin = lon - halfSide / pradius
#     lonMax = lon + halfSide / pradius
#
#     return (rad2deg(latMin), rad2deg(lonMin), rad2deg(latMax), rad2deg(lonMax))


########
# TODO: NDVI is difficult and it might not matter so much
# get vegetation index data
# xmin = datapoints[id]['lonmin']
# xmax = datapoints[id]['lonmax']
# ymin = datapoints[id]['latmin']
# ymax = datapoints[id]['latmax']
# rjson = {
#     "name": str(id),
#     "geo_json": {
#         "type": "Feature",
#         "properties": {},
#         "geometry": {
#             "type": "Polygon",
#             "coordinates": [
#                 [
#                     [xmin, ymin],
#                     [xmax, ymax],
#                     [xmax, ymin],
#                     [xmin, ymin]
#                 ]
#             ]
#         }
#     }
# }
# try:
#     agro_url = 'http://api.agromonitoring.com/agro/1.0/polygons?appid=' + AGRO_KEY
#     print(agro_url)
#     print(rjson)
#     agro_r = requests.post(agro_url, json=rjson)
#     print(agro_r)
#     polygon_id = agro_r.json()['id']
# except requests.exceptions.RequestException as e:
#     print(agro_r.status_code)
#     print(e)
#     # sys.exit(1)
#
# if polygon_id:
#     print('got polygon id:', polygon_id)
#     # use polygon to gather data
#     try:
#         start_secs = int(unix_time_millis(date_o) / 1000)
#         end_secs = start_secs + 86400
#
#         ndvi = ''.join([
#             "https://samples.agromonitoring.com/agro/1.0/ndvi/history?",
#             "polyid=" + polygon_id,
#             "&start=" + str(start_secs),
#             "&end=" + str(end_secs),
#             "&appid=" + AGRO_KEY
#         ])
#         print(ndvi)
#         agro_r2 = requests.get(ndvi)
#         samples = agro_r2.json()
#         tot = []
#         if len(samples) > 0:
#             for s in samples:
#                 tot.append(s['data']['mean'])
#         meanall = np.mean(tot)
#         print(meanall)
#         datapoints[id]['ndvi'] = meanall
#     except requests.exceptions.RequestException as e:
#         print(agro_r.status_code)
#         print(e)
#         # sys.exit(1)
