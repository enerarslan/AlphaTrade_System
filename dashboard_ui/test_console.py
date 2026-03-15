from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

options = Options()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)

driver.get('http://localhost:5173/trading')
time.sleep(3)

print("CONSOLE LOGS:")
for entry in driver.get_log('browser'):
    print(entry)

driver.quit()
