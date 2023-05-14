from bs4 import BeautifulSoup
import urllib.request

#TUDelft Dataset

#Individual pages @ TUDelft
def indivPage(url, name):
    source = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(source, 'lxml')
    data = soup.find('pre').find(text=True)

    f = open(name, "a")
    f.write(data)
    f.close()

#Accessing pages @ TUDelft
totalSource = urllib.request.urlopen('https://aerodynamics.lr.tudelft.nl/cgi-bin/afCDb').read()
totalSoup = BeautifulSoup(totalSource, 'lxml')

links = []

for a in totalSoup.find_all('a', href=True):
    link = a["href"]
    if link[:5] == "afCDb":    
        links.append(link[6:])

for l in links:
    url = 'https://aerodynamics.lr.tudelft.nl/cgi-bin/afCDb?' + l
    name = l + '.dat'
    indivPage(url, name)