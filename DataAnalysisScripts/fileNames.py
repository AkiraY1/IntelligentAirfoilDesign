import csv

filename = "delft_data.csv"
#uiuc_data.csv
appendText = "airfoilNames.txt"
names = []

with open(appendText, 'a') as foils:
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] not in names:
                names.append(row[0])
                foils.write(row[0] + "\n")