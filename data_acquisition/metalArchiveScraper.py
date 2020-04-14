from string import ascii_lowercase
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import requests
import time
import os


urlRoot = "https://www.metal-archives.com/lists/black"
driver = webdriver.Firefox()
counter = 0
bandNameOld = ""

driver.get(urlRoot)
time.sleep(2)
lastLink = ""
while True:
    print("not disabled")
    linkElements = driver.find_elements_by_tag_name('a')
    bandlinks = []
    for e in linkElements:
        link = e.get_attribute("href")
        if not link == None:
            if "/bands/" in link:
                print(link)
                bandlinks.append(link)
    if bandlinks[0] == lastLink: 
        break
    else:
        lastLink = bandlinks[0]

    for bandlink in bandlinks:
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[1])
        driver.get(bandlink)

        logoElements = driver.find_elements_by_id('logo')
        for logoElement in logoElements:
            logoImg = logoElement.find_element_by_css_selector("*")
            logoSrc = logoImg.get_attribute("src")
            bandName = driver.find_element_by_class_name('band_name').text
            bandName = bandName.replace(" ", "_")
            bandName = bandName.replace("/", "_")
            print(bandName)
            r = requests.get(logoSrc)
            outfile = os.getcwd() + '/../dataset/' + bandName
            if bandName == bandNameOld:
                counter += 1
                outfile = outfile + str(counter)
            bandNameOld = bandName
            outfile += ".jpg"
            with open(outfile, 'wb') as filewriter:
                filewriter.write(r.content)
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
    
    time.sleep(1)
    nextButton = driver.find_element_by_id('bandListAlpha_next')
    nextButton.click()
    time.sleep(5)

