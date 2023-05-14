import subprocess as sp
import time, os, csv

DIRS = [r"C:\Users\Akira\Desktop\CFD_Automation\uiuc", r"C:\Users\Akira\Desktop\CFD_Automation\uiuc2"]
#Done: r"C:\Users\Akira\Desktop\CFD_Automation\uiuc" r"C:\Users\Akira\Desktop\CFD_Automation\uiuc2" r"C:\Users\Akira\Desktop\CFD_Automation\delft", r"C:\Users\Akira\Desktop\CFD_Automation\delft2"

good = True

for dir in DIRS:
    for f in os.listdir(dir):
        with open("camberThickness.csv", "a") as sheet:
            
            ps = sp.Popen(['C:\\Users\\Akira\\Desktop\\XFOIL6.99\\xfoil.exe'], stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
            def issueCmd(cmd,echo=True):
                ps.stdin.write((cmd+'\n').encode())

            name = os.path.join(dir, f)
            issueCmd(f'LOAD {name}')
            out, err = ps.communicate()
            for line in out.split("\n".encode()):
                if "thickness".encode() in line:
                    l = line.decode()
                    try:
                        thickness = float(l[16:31])
                        thicknessPos = float(l[37:])
                    except:
                        print(name)
                        good = False
                if "camber".encode() in line:
                    l = line.decode()
                    try:
                        camber = float(l[16:31])
                        camberPos = float(l[37:])
                    except:
                        print(name)
                        good = False
            if good:
                writer_object = csv.writer(sheet, lineterminator = '\n')
                writer_object.writerow([f.removesuffix(".dat"), thickness, thicknessPos, camber, camberPos])
            sheet.close()
        good = True