from bs4 import BeautifulSoup
import urllib.request

#UIUC Dataset

#Individual pages @ UIUC
def indivPage(url, name):
    source = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(source, 'lxml')
    data = soup.get_text()

    f = open(name, "a")
    f.write(data)
    f.close()

#Accessing pages @ UIUC
totalSource = urllib.request.urlopen('https://m-selig.ae.illinois.edu/ads/coord_database.html').read()
totalSoup = BeautifulSoup(totalSource, 'lxml')

links = []

for a in totalSoup.find_all('a', href=True):
    link = a["href"]
    if link[-3:] == "dat":    
        links.append(link)

for l in links:
    url = 'https://m-selig.ae.illinois.edu/ads/' + l
    name = url.split('/')[-1]
    indivPage(url, name)