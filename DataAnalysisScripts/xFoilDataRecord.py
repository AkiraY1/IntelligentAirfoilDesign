from csv import writer
import os

DIR = ["C:\\Users\\Akira\\Desktop\\CFD_Automation\\uiuc_polars", "C:\\Users\\Akira\\Desktop\\CFD_Automation\\uiuc2_polars"]

#Getting data from text file
for dir in DIR:
    for f in os.listdir(dir):
        name = dir + "\\" + f
        dataFile = open(name, 'r')
        lines = dataFile.readlines()
        with open('uiuc_data.csv', 'a') as file:
            writer_object = writer(file)
            for l in lines[12:]:
                d = l.split()
                d.insert(0, f.strip(".txt"))
                print(d)
                writer_object.writerow(d)
            file.close()