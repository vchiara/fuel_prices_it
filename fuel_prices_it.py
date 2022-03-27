import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px
from plotly import graph_objects as go
import geojson

import glob
from urllib.request import urlopen
import pickle

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# prepare prices df

#load csv from file
csv_prices = '/fuel_prices/prezzo_alle_8.csv'
df = pd.read_csv(csv_prices, sep = ';', header = 1, parse_dates = ['dtComu'], dayfirst = True)

#change column names
col_names = ['id_station', 'fuel_type', 'price', 'is_self', 'date']
df = df.set_axis(col_names, axis = 1)

#change date col type
df['date'] = df['date'].dt.date

df['service'] = df['is_self'].replace({0 : 'Full service',
                                      1 : 'Self service'})

#move service column
df.insert(4, 'service', df.pop('service'))

# prepare geo info df

#load csv from file
csv_coord = '/content/drive/MyDrive/fuel_prices/anagrafica_impianti_attivi.csv'
df_coord = pd.read_csv(csv_coord, sep = ',', na_values = 'NaN', keep_default_na = False, on_bad_lines = 'warn')
df_coord = df_coord.dropna()

#change column names
col_names_df_coord = ['id_station', 'company', 'parent_company', 'type_station', 'name_station', 'address', 'municipality' , 'province', 'lat', 'lon']
df_coord = df_coord.set_axis(col_names_df_coord, axis = 1)

#explore province column
pr = df_coord.province.unique()

pr = sorted(pr, key = len)

pr_errors = [province for province in pr if len(province) > 2]

#fix errors
df_coord = df_coord.replace({'province' : {
    'POFI' : 'FR',
    'SORA' : 'FR',
    "SALO'" : 'BS',
    'TORINO' : 'TO',
    "FOSSO'" : 'VE',
    'VICENZA' : 'VI',
    'MARSALA' : 'TP',
    'SONNINO' : 'LT',
    "PATERNO'" : 'CT',
    "L'AQUILA" : 'AQ',
    "MONDOVI'" : 'CN',
    'UMBERTIDE' : 'PG',
    'MELENDUGNO' : 'LE',
    'COSTERMANO' : 'VR',
    "SANT'URBANO" : 'PD',
    "TORRE D'ISOLA" : 'PV',
    'MONTEMARCIANO' : 'AN',
    "QUINZANO D'OGLIO" : 'BS',
    'FARA SAN MARTINO' : 'CH',
    "ACI SANT'ANTONIO" : 'CT',
    "MACCHIA D'ISERNIA" : 'IS',
    "ROMANO D'EZZELINO" : 'VI',
    'MOTTA SAN GIOVANNI' : 'RC',
    'AIANO (BO) 3 40034' : 'BO',
    "COLLE DI VAL D'ELSA" : 'SI',
    "MOSCIANO SANT'ANGELO" : 'TE',
    "PALAZZOLO SULL'OGLIO" : 'BS',
    "SANT'ANGELO DI BROLO" : 'ME',
    'CASTELNUOVO BERARDENGA' : 'SI',
    "SANT'AGATA DI MILITELLO" : 'ME',
    ' DELLA PIEVE (PG)  06060' : 'PG',
    "RICCO' DEL GOLFO DI SPEZIA" : 'SP',
    "SANT'AMBROGIO DI VALPOLICELLA" : 'VR',
    'ANGELO S.N. 66050, SAN SALVO (CH)  66050'  : 'CH',
    'ITALIA SNC 73034, GAGLIANO DEL CAPO (LE) 29 73034'  : 'LE'
    }})

#dict with regions and provinces of Italy
it_regions = {
    'Abruzzo' : ['CH', 'AQ', 'PE', 'TE'],
    'Basilicata' : ['MT', 'PZ'],
    'Calabria' : ['CZ', 'CS', 'KR', 'VV', 'RC'],
    'Campania' : ['AV', 'BN', 'CE', 'NA', 'SA'],    
    'Emilia-Romagna': ['PC','PR','RE','MO','BO','FE','RA','FC','RN'],
    'Friuli-Venezia Giulia': ['UD','GO','TS','PN'],
    'Lazio': ['VT','RI','RM','LT','FR'],
    'Liguria': ['IM','SV','GE','SP'],
    'Lombardia': ['VA','CO','SO','MI','BG','BS','PV','CR','MN','LC','LO','MB'],
    'Marche': ['PU','AN','MC','AP','FM'],
    'Molise': ['CB','IS'],
    'Piemonte': ['TO','VC','NO','CN','AT','AL','BI','VB'],
    'Puglia': ['FG','BA','TA','BR','LE','BT'],
    'Sardegna': ['SS','NU','CA','OR','SU', 'OT', 'CI', 'VS', 'OG'], #includes ex-provinces
    'Sicilia': ['TP','PA','ME','AG','CL','EN','CT','RG','SR'],
    'Toscana': ['MS','LU','PT','FI','LI','PI','AR','SI','GR','PO'],
    'Trentino-Alto Adige/Südtirol': ['BZ','TN'],
    'Umbria': ['PG','TR'],
    "Valle d'Aosta/Vallée d'Aoste": ['AO'],
    'Veneto': ['VR','VI','BL','TV','VE','PD','RO']
}

#inverse dictionary values
it_regions_inv = {}

for key in it_regions:
   for value in it_regions[key]:
     it_regions_inv[value] = key

it_regions_inv

#create new column 'region' from province data + dictionary
df_coord['region'] = df_coord['province']
df_coord = df_coord.replace({'region': it_regions_inv})

#check for errors
df_coord.region.value_counts()
'''
Lombardia                       2788
Lazio                           2099
Veneto                          1812
Campania                        1798
Emilia-Romagna                  1722
Sicilia                         1705
Piemonte                        1689
Toscana                         1500
Puglia                          1410
Marche                           741
Calabria                         684
Sardegna                         611
Abruzzo                          597
Liguria                          504
Friuli-Venezia Giulia            474
Umbria                           442
Trentino-Alto Adige/Südtirol     364
Basilicata                       230
Molise                           158
Valle d'Aosta/Vallée d'Aoste      76
Name: region, dtype: int64
'''

#merge dfs
df = df.merge(df_coord, on = 'id_station')

# explore merged df
df['price'].describe()
'''
count    90250.000000
mean         1.875571
std          0.311633
min          0.001000
25%          1.774000
50%          1.859000
75%          2.024000
max          4.000000
'''

#detect outliers with boxplot
box_plot_prices = px.box(df,
                         y = 'price',
                         points = 'suspectedoutliers',
                         color_discrete_sequence = ['rgb(92, 83, 165)'],
                         title = 'Fuel prices boxplot')

box_plot_prices

#remove outliers
df = df[df['price'] > 0.5]
df = df[df['price'] < 4]

#check missing data
df.isnull().sum()
'''
id_station         0
fuel_type          0
price              0
is_self            0
service            0
date               0
company            0
parent_company     6
type_station       6
name_station       6
address            6
municipality      10
province          10
lat               10
lon               10
'''

df = df.dropna(subset = ['parent_company'])

df.isnull().sum()
'''
id_station        0
fuel_type         0
price             0
is_self           0
service           0
date              0
company           0
parent_company    0
type_station      0
name_station      0
address           0
municipality      4
province          4
lat               4
lon               4
'''

df[df['province'].isnull()]

df = df.dropna(subset = ['municipality'])

df.isnull().sum()
'''
id_station        0
fuel_type         0
price             0
is_self           0
service           0
date              0
company           0
parent_company    0
type_station      0
name_station      0
address           0
municipality      0
province          0
region            0
lat               0
lon               0
'''

#fuel prices line plot
pivot_price = df.pivot_table(index = ['date'], 
                       values = 'price')

pivot_price = pivot_price.reset_index()

price_line = px.line(pivot_price, 
                     x = 'date', 
                     y = 'price',
                     color_discrete_sequence = ['rgb(92, 83, 165)'])

price_line = price_line.update_layout(
    title = 'Fuel prices in Italy',
    yaxis_title = 'Price (in EUR)',
    xaxis_title = 'Year')

price_line

#fuel prices histogram

price_hist = px.histogram(df, 
                           x = 'price',
                           nbins = 150,
                           color_discrete_sequence = ['rgb(92, 83, 165)'])

price_hist = price_hist.update_layout(
    title = 'Fuel price distribution in Italy (2020-2022)',
    yaxis_title = 'Observations',
    xaxis_title = 'Price (in EUR)')

price_hist.show()

#main fuels

main_fuels = df.fuel_type.value_counts()

main_fuels = main_fuels.index[:4]
#Index(['Benzina', 'Gasolio', 'Blue Diesel', 'GPL'], dtype='object')

df_main_fuels = df[df.fuel_type.isin(main_fuels)]

#main fuels histogram
main_fuels_hist = px.histogram(df_main_fuels, 
                           x = 'price',
                           nbins = 200,
                           color = 'fuel_type',
                           color_discrete_sequence = px.colors.sequential.Sunset_r,
                           facet_col = 'fuel_type',
                           category_orders = {'fuel_type' : ['Benzina', 'Gasolio', 'Blue Diesel', 'GPL']})

main_fuels_hist = main_fuels_hist.update_layout(
    title = 'Fuel price distribution in Italy (2020-2022)',
    legend_title = 'Fuel type',
    yaxis_title = 'Observations')

main_fuels_hist = main_fuels_hist.for_each_annotation(lambda a: a.update(text = a.text.split("=")[-1]))

main_fuels_hist =  main_fuels_hist.update_xaxes(title = '')
main_fuels_hist =  main_fuels_hist.update_layout(xaxis3 = dict(title = 'Price (in EUR)'))
main_fuels_hist

main_fuels_hist.show()

#main fuels line plot
pivot_main_fuels_date = df_main_fuels.pivot_table(index = ['date', 'fuel_type'], 
                                             values = 'price')

pivot_main_fuels_date = pivot_main_fuels_date.reset_index()

main_fuels_line = px.line(pivot_main_fuels_date, 
                     x = 'date', 
                     y = 'price',
                     color = 'fuel_type',
                     color_discrete_sequence = px.colors.sequential.Sunset_r,
                     facet_col = 'fuel_type',
                     category_orders = {'fuel_type' : ['Benzina', 'Gasolio', 'Blue Diesel', 'GPL']})

main_fuels_line = main_fuels_line.update_layout(
    legend_title = 'Type of fuel',
    title = 'Fuel prices in Italy',
    yaxis_title = 'Price (in EUR)')

main_fuels_line = main_fuels_line.for_each_annotation(lambda a: a.update(text = a.text.split("=")[-1]))

main_fuels_line =  main_fuels_line.update_xaxes(title = '')
main_fuels_line =  main_fuels_line.update_layout(xaxis3 = dict(title = 'Year'))
main_fuels_line

#main companies
main_companies = df_main_fuels.parent_company.value_counts()
main_companies = main_companies.index[:6]
main_companies
#Index(['Agip Eni', 'Api-Ip', 'Pompe Bianche', 'Q8', 'Esso', 'Tamoil'], dtype='object')

df_main_companies = df_main_fuels[df_main_fuels.parent_company.isin(main_companies)]

#main companies histogram
main_companies_hist = px.histogram(df_main_companies, 
                           x = 'price',
                           nbins = 100,
                           color = 'fuel_type',
                           color_discrete_sequence = px.colors.sequential.Sunset_r,
                           facet_col = 'parent_company')

main_companies_hist = main_companies_hist.update_layout(
    title = 'Fuel price distribution in Italy (2020-2022)',
    legend_title = 'Fuel type',
    yaxis_title = 'Observations')

main_companies_hist = main_companies_hist.for_each_annotation(lambda a: a.update(text = a.text.split("=")[-1]))

main_companies_hist =  main_companies_hist.update_xaxes(title = '')
main_companies_hist =  main_companies_hist.update_layout(xaxis3 = dict(title = 'Price (in EUR)'))
main_companies_hist

main_companies_hist.show()

pivot_main_companies = df_main_companies.pivot_table(index = ['parent_company', 'fuel_type'], 
                          values = 'price')
pivot_main_companies = pivot_main_companies.reset_index()

pivot_main_companies = pivot_main_companies.round(3)

pivot_main_companies

#main companies bar plots
company_price = px.bar(pivot_main_companies,
                       x = 'parent_company',
                       y = 'price',
                       text = 'price',
                       color = 'fuel_type',
                       color_discrete_sequence = px.colors.sequential.Sunset_r,
                       facet_col = 'fuel_type',
                       category_orders = {
                           'fuel_type' : ['Benzina', 'Gasolio', 'Blue Diesel', 'GPL'],
                           'parent_company' : ['Agip Eni', 'Api-Ip', 'Pompe Bianche', 'Q8', 'Esso', 'Tamoil']})


company_price = company_price.update_layout(
    legend_title = 'Type of fuel',
    title = 'Fuel prices in Italy by company',
    yaxis_title = 'Price (in EUR)')

company_price = company_price.for_each_annotation(lambda a: a.update(text = a.text.split("=")[-1]))

company_price =  company_price.update_xaxes(title = '')

company_price.update_traces(textposition = 'outside')

#full vs self service
pivot_service = df_main_companies.pivot_table(index = ['fuel_type', 'service'], 
                       values = 'price')

pivot_service = pivot_service.reset_index()

pivot_service = pivot_service.round(3)

service_price_bar = px.bar(pivot_service, 
                   x = 'fuel_type', 
                   y = 'price',
                   text = 'price',
                   color = 'service',
                   color_discrete_sequence = ['rgb(248, 160, 126)', 'rgb(250, 196, 132)'], 
                   barmode = 'group',
                   category_orders = {'fuel_type' : ['Benzina', 'Gasolio', 'Blue Diesel', 'GPL']})

service_price_bar = service_price_bar.update_layout(
    title = 'Fuel prices in Italy',
    legend_title = 'Type of Service',
    yaxis_title = 'Price (in EUR)',
    xaxis_title = 'Type of Fuel')

service_price_bar.update_traces(textposition = 'outside')

#station types
pivot_station_type = df_main_companies.pivot_table(index = ['fuel_type', 'type_station'], 
                       values = 'price')

pivot_station_type = pivot_station_type.reset_index()

pivot_station_type = pivot_station_type.round(3)

station_type_bar = px.bar(pivot_station_type, 
                   x = 'fuel_type', 
                   y = 'price', 
                   text = 'price',
                   color = 'type_station',
                   color_discrete_sequence = ['rgb(248, 160, 126)', 'rgb(250, 196, 132)'], 
                   barmode = 'group',
                   category_orders = {'fuel_type' : ['Benzina', 'Gasolio', 'Blue Diesel', 'GPL']})

station_type_bar = station_type_bar.update_layout(
    title = 'Fuel prices in Italy',
    legend_title = 'Type of Station',
    yaxis_title = 'Price (in EUR)',
    xaxis_title = 'Type of Fuel')

station_type_bar.update_traces(textposition = 'outside')

#analysing each main fuel

#benzina
pivot_benzina = df_main_companies[(df_main_companies['fuel_type'] == 'Benzina')].pivot_table(
    index = ['region', 'fuel_type'], 
    values = 'price')

pivot_benzina = pivot_benzina.reset_index()

pivot_benzina = pivot_benzina.round(3).sort_values(by = 'price', ascending = False)

pivot_benzina.describe()

with open('/content/drive/MyDrive/fuel_prices/limits_IT_regions.geojson') as map:
    regions_it = geojson.load(map)
    features = regions_it['features'][0]

#benzina price by region (map)
benzina_price_region = go.Figure(
    go.Choroplethmapbox(
        geojson = regions_it, #assign geojson file
        featureidkey = 'properties.reg_name', #assign feature key
        locations = pivot_benzina['region'], #assign location data
        z = pivot_benzina['price'], #assign information data
        zauto = False,
        zmin = 1.866000,
        zmax = 1.946000,
        colorscale = 'matter',
        showscale = True
    )
)

benzina_price_region.update_traces(marker_opacity = 0.8)

benzina_price_region.update_layout(
    mapbox_style = 'carto-positron', #decide a style for the map
    mapbox_zoom = 4, #zoom in scale
    mapbox_center = {"lat": 41.87194, "lon": 12.56738}, #center location of the map
    title = 'Benzina prices in Italy')

#benzina price by region (bar plot)
benzina_price = px.bar(pivot_benzina,
                       x = 'region',
                       y = 'price',
                       text = 'price',
                       color = 'fuel_type',
                       color_discrete_sequence = ['rgb(92, 83, 165)'])


benzina_price = benzina_price.update_layout(
    legend_title = 'Type of fuel',
    title = 'Benzina prices in Italy',
    yaxis_title = 'Price (in EUR)',
    xaxis_title = 'Region')

benzina_price.update_traces(textposition = 'outside')

#gasolio
pivot_gasolio = df_main_companies[(df_main_companies['fuel_type'] == 'Gasolio')].pivot_table(
    index = ['region', 'fuel_type'], 
    values = 'price')

pivot_gasolio = pivot_gasolio.reset_index()

pivot_gasolio = pivot_gasolio.round(3).sort_values(by = 'price', ascending = False)

pivot_gasolio.describe()

#gasolio price by region (map)
gasolio_price_region = go.Figure(
    go.Choroplethmapbox(
        geojson = regions_it, #assign geojson file
        featureidkey = 'properties.reg_name', #assign feature key
        locations = pivot_gasolio['region'], #assign location data
        z = pivot_gasolio['price'], #assign information data
        zauto = False,
        zmin = 1.847000,
        zmax = 1.924000,
        colorscale = 'matter',
        showscale = True
    )
)

gasolio_price_region.update_traces(marker_opacity = 0.8)

gasolio_price_region.update_layout(
    mapbox_style = 'carto-positron', #decide a style for the map
    mapbox_zoom = 4, #zoom in scale
    mapbox_center = {"lat": 41.87194, "lon": 12.56738}, #center location of the map
    title = 'Gasolio prices in Italy')


#gasolio price by region (bar plot)
gasolio_price = px.bar(pivot_gasolio,
                       x = 'region',
                       y = 'price',
                       text = 'price',
                       color = 'fuel_type',
                       color_discrete_sequence = ['rgb(160, 89, 160)'])


gasolio_price = gasolio_price.update_layout(
    legend_title = 'Type of fuel',
    title = 'Gasolio prices in Italy',
    yaxis_title = 'Price (in EUR)',
    xaxis_title = 'Region')

gasolio_price.update_traces(textposition = 'outside')

#diesel
pivot_diesel = df_main_companies[(df_main_companies['fuel_type'] == 'Blue Diesel')].pivot_table(
    index = ['region', 'fuel_type'], 
    values = 'price')

pivot_diesel = pivot_diesel.reset_index()

pivot_diesel = pivot_diesel.round(3).sort_values(by = 'price', ascending = False)

pivot_diesel.describe()

#diesel price by region (map)
diesel_price_region = go.Figure(
    go.Choroplethmapbox(
        geojson = regions_it, #assign geojson file
        featureidkey = 'properties.reg_name', #assign feature key
        locations = pivot_diesel['region'], #assign location data
        z = pivot_diesel['price'], #assign information data
        zauto = False,
        zmin = 1.971328,
        zmax = 2.066145,
        colorscale = 'matter',
        showscale = True
    )
)

diesel_price_region.update_traces(marker_opacity = 0.8)

diesel_price_region.update_layout(
    mapbox_style = 'carto-positron', #decide a style for the map
    mapbox_zoom = 4, #zoom in scale
    mapbox_center = {"lat": 41.87194, "lon": 12.56738}, #center location of the map
    title = 'Diesel prices in Italy')

#diesel price by region (bar plot)
diesel_price = px.bar(pivot_diesel,
                       x = 'region',
                       y = 'price',
                       text = 'price',
                       color = 'fuel_type',
                       color_discrete_sequence = ['rgb(206, 102, 147)'])

diesel_price = diesel_price.update_layout(
    legend_title = 'Type of fuel',
    title = 'Diesel prices in Italy',
    yaxis_title = 'Price (in EUR)',
    xaxis_title = 'Region')

diesel_price.update_traces(textposition = 'outside')

#gpl
pivot_gpl = df_main_companies[(df_main_companies['fuel_type'] == 'GPL')].pivot_table(
    index = ['region', 'fuel_type'], 
    values = 'price')

pivot_gpl = pivot_gpl.reset_index()

pivot_gpl = pivot_gpl.round(3).sort_values(by = 'price', ascending = False)

pivot_gpl.describe()

#gpl price by region (map)
gpl_price_region = go.Figure(
    go.Choroplethmapbox(
        geojson = regions_it, #assign geojson file
        featureidkey = 'properties.reg_name', #assign feature key
        locations = pivot_gpl['region'], #assign location data
        z = pivot_gpl['price'], #assign information data
        zauto = False,
        zmin = 0.827000,
        zmax = 0.899000,
        colorscale = 'matter',
        showscale = True
    )
)

gpl_price_region.update_traces(marker_opacity = 0.8)

gpl_price_region.update_layout(
    mapbox_style = 'carto-positron', #decide a style for the map
    mapbox_zoom = 4, #zoom in scale
    mapbox_center = {"lat": 41.87194, "lon": 12.56738}, #center location of the map)

#gpl price by region (bar plot)
gpl_price = px.bar(pivot_gpl,
                       x = 'region',
                       y = 'price',
                       text = 'price',
                       color = 'fuel_type',
                       color_discrete_sequence = ['rgb(235, 127, 134)'])

gpl_price = gpl_price.update_layout(
    legend_title = 'Type of fuel',
    title = 'GPL prices in Italy',
    yaxis_title = 'Price (in EUR)',
    xaxis_title = 'Region')

gpl_price.update_traces(textposition = 'outside')

#predict prices
df_main_fuels.info()
'''
<class 'pandas.core.frame.DataFrame'>
Int64Index: 77163 entries, 1 to 90249
Data columns (total 16 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   id_station      77163 non-null  int64  
 1   fuel_type       77163 non-null  object 
 2   price           77163 non-null  float64
 3   is_self         77163 non-null  int64  
 4   service         77163 non-null  object 
 5   date            77163 non-null  object 
 6   company         77163 non-null  object 
 7   parent_company  77163 non-null  object 
 8   type_station    77163 non-null  object 
 9   name_station    77163 non-null  object 
 10  address         77163 non-null  object 
 11  municipality    77163 non-null  object 
 12  province        77163 non-null  object 
 13  region          77163 non-null  object 
 14  lat             77163 non-null  object 
 15  lon             77163 non-null  object 
dtypes: float64(1), int64(2), object(13)
'''

#save prices as y 
y = df_main_fuels.pop('price')

#prepare x data

#change date format
df_main_fuels['date'] = pd.to_datetime(df_main_fuels['date'],
                           format = '%Y-%m-%dT',
                           errors = 'coerce')

#split date into day, month, and year columns
df_main_fuels['day'] = df_main_fuels['date'].dt.day
df_main_fuels['month'] = df_main_fuels['date'].dt.month
df_main_fuels['year'] = df_main_fuels['date'].dt.year

#columns to select
x_cols = ['id_station', 'fuel_type', 'service', 'parent_company', 'type_station', 'region', 'day', 'month', 'year']

x = df_main_fuels[x_cols]

#split numerical and categorical columns for pipeline
numerical_columns = ['day', 'month', 'year']
categorical_columns = ['id_station', 'fuel_type', 'service', 'parent_company', 'type_station', 'region']

#begin building pipeline for data prep and with MLP as the model
numerical_pipeline = Pipeline([
  ['numerical_imputer', SimpleImputer(strategy = 'median')],
  ['scaler', StandardScaler()]
])

categorical_pipeline = Pipeline([
  ['categorical_imputer', SimpleImputer(strategy = 'constant', fill_value = 'missing')],
  ['one_hot_encoder', OneHotEncoder(handle_unknown = 'ignore')]                              
])

col_transformer = ColumnTransformer([
  ['num_pipe', numerical_pipeline, numerical_columns],
  ['cat_pipe', categorical_pipeline, categorical_columns]
])

pipeline = Pipeline([
  ['data_prep', col_transformer],
  ['model', MLPRegressor(hidden_layer_sizes = (30, 30, 30, 30, 30),
                   random_state = 1,
                   alpha = 0.0001,
                   max_iter = 5000)]
])

#split into training and test
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1, test_size = 0.3)

#train model
model = pipeline.fit(x_train, y_train)

#predict with train and test
pred_train = model.predict(x_train)
pred_test  = model.predict(x_test)

#calculate MAE
mae_train = mean_absolute_error(y_train, pred_train)
mae_test  = mean_absolute_error(y_test, pred_test)

mae_train, mae_test
#(0.017861376561850207, 0.04331751819030215)

print("R2 :", r2_score(pred_test, y_test))
#R2 : 0.9409575511412533

#save model to disk
filename = '/fuel_prices/fuel_prices_model_pipeline.sav'
pickle.dump(model, open(filename, 'wb'))

#load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)
#0.9445291255111631

#new data

#load new csv from url
url_prices = 'https://www.mise.gov.it/images/exportCSV/prezzo_alle_8.csv'
df_pred = pd.read_csv(url_prices, sep = ';', header = 1, parse_dates = ['dtComu'], dayfirst = True)

#change column names
col_names = ['id_station', 'fuel_type', 'price', 'is_self', 'date']
df_pred = df_pred.set_axis(col_names, axis = 1)

df_pred['service'] = df_pred['is_self'].replace({0 : 'Full service',
                                      1 : 'Self service'})

#change date format
df_pred['date'] = df_pred['date'].dt.date

df_pred['date'] = pd.to_datetime(df_pred['date'],
                           format = '%Y-%m-%dT',
                           errors = 'coerce')

#merge with df_coord
df_pred = df_pred.merge(df_coord, on = 'id_station')
df_pred.head()

#select only newest observations
df_pred = df_pred[df_pred['date'] >= '25-03-2022']

#select main fuels
df_pred = df_pred[df_pred.fuel_type.isin(['Benzina', 'Gasolio', 'Blue Diesel', 'GPL'])]

#split date into day, month, and year columns
df_pred['day'] = df_pred['date'].dt.day
df_pred['month'] = df_pred['date'].dt.month
df_pred['year'] = df_pred['date'].dt.year

#select x columns
x_cols = ['id_station', 'fuel_type', 'service', 'parent_company', 'type_station', 'region', 'day', 'month', 'year']

#create new df for new predictions
x_pred = df_pred[x_cols]

y_pred = df_pred.pop('price')

pred_df = x_pred

pred_df['actual_price'] = y_pred

pred_df['predicted_price'] = loaded_model.predict(x_pred).round(3)

#predict prices with loaded model
result_pred = loaded_model.score(x_pred, y_pred)
print(result_pred)
#0.8483561021772759