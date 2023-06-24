import pandas as pd
import re
#Data cleaning

data = pd.read_csv('output.csv')



#
# #to find year of each car, we take the second item from string splitted by commas

data['year'] = data['name'].str.rsplit(' ').str[2]
data['year'] = pd.to_numeric(data['year'], errors='coerce')
data = data[(data['year'] > 1900) & (data['year'] < 2023)]

#converting years number to int
data['year'] = data['year'].astype(int)

#From the dataframe above, the name column was splitted to two - brands and models of cars.
data['brand_of_car'] = data['name'].str.rsplit(' ').str[0]
data['model_of_car'] = data['name'].str.rsplit(' ').str[1]


# volume of car is the 7 character in the car_info column
data['volume_of_car'] = data['car_info'].str.rsplit('\n').str[7]
data['volume_of_car'] = data['volume_of_car'].str.rsplit(' ').str[0]
data = data[data['volume_of_car'].str.contains(r'\d', na=False)]
data['volume_of_car'] = pd.to_numeric(data['volume_of_car'], errors='coerce')

# #converting price to int
data['avg_price'] = data['avg_price'].str.replace(" ","")
data['avg_price'] = data['avg_price'].str.extract(r'(\d+)')
data['avg_price'] = data['avg_price'].str.replace('[^0-9]', '', regex=True)  # Remove non-numeric characters
data['avg_price'] = pd.to_numeric(data['avg_price'], errors='coerce')


#deleting whitespaces and units, converting to float
data = data.drop('name', axis=1)
data = data.drop('car_info', axis=1)




data.to_csv('cars3.csv', index=False)
print(data)
