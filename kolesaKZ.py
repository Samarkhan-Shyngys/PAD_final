from selenium import webdriver
#chrome browser is used
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from time import sleep
import pandas as pd
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

#list for announcement links
links = []

url = "https://kolesa.kz"

# start the Chrome driver

driver = webdriver.Chrome(executable_path = r'C:\Users\Shyngys\Desktop\chromedriver.exe')
driver.maximize_window()

# navigate to the page
driver.get(url)

div_elements = driver.find_elements(By.CLASS_NAME, 'block-links-list')
#div_elements = driver.find_elements("xpath", value="//div[@class='block-links-list']")
for div in div_elements:
    print(div)
    for link in div.find_elements(By.TAG_NAME, 'a'):
        links.append(link.get_attribute('href'))
        if(len(links)>35):
            break
    break

# find all links to individual cars
car_links = []
count = 1


def linker(l, c):
    driver.get(l)
    card_links = WebDriverWait(driver, 4).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, 'a-card__link'))
    )
    for card_link in card_links:
        car_links.append(card_link.get_attribute('href'))
        c+=1
    try:
        next_p = WebDriverWait(driver, 4).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'right-arrow.next_page')))
        next = next_p.get_attribute('href')
        print(next)
        try:
            if c>10:
                return
            linker(next, c)
        except:
            print('link error')
    except:
        print('no next page')

for link in links:
    try:
        count=1
        linker(link, count)
    except:
        print('link Error')

# scrape the name and price of each car
car_info_list = []
car_names = []
for link in car_links:
    try:
        driver.get(link)
        name = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'offer__title'))
        ).text.strip()
        if name in car_names:
            continue
        price = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'price__value'))
        ).text.strip()

        defin = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'offer__parameters'))
        ).text.strip()
        car_info = {}
        car_info['name'] = name
        car_info['avg_price'] = price
        car_info['car_info'] = defin
        car_info_list.append(car_info)
        print(car_info)
    except:
        name = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'offer__title'))
        ).text.strip()
        price = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'offer__price'))
        ).text.strip()
        defin = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'offer__parameters'))
        ).text.strip()
        car_info = {}
        car_info['name'] = name
        car_info['avg_price'] = price
        car_info['car_info'] = defin
        car_info_list.append(car_info)
        print(car_info)
        print('Car link error: no average price')

# close the driver
driver.quit()

df= pd.DataFrame(car_info_list)
df = df.drop_duplicates()
df.to_csv('output.csv', index=False)