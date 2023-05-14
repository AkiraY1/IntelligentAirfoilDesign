import subprocess as sp
import time, os

#Panel nodes = 160
#Reynold's number = 10000
#Iter = 500
#Mach = 0
#Angle of attack = Sequence
#Alfa values = 0 and 30
#Alfa increment = 0.5
#cm, cl, cd, alpha

#Scan through all files in the directories
DIRS = [r"C:\Users\Akira\Desktop\CFD_Automation\delft"]
#Done: r"C:\Users\Akira\Desktop\CFD_Automation\delft2" r"C:\Users\Akira\Desktop\CFD_Automation\uiuc2" r"C:\Users\Akira\Desktop\CFD_Automation\uiuc" r"C:\Users\Akira\Desktop\CFD_Automation\delft"

for dir in DIRS:
    for f in os.listdir(dir):

        ps = sp.Popen(['C:\\Users\\Akira\\Desktop\\XFOIL6.99\\xfoil.exe'], stdin=sp.PIPE, stdout=None, stderr=None)

        def issueCmd(cmd,echo=True):
            ps.stdin.write((cmd+'\n').encode())
            if echo:
                print(cmd)

        name = os.path.join(dir, f)
        issueCmd(f'LOAD {name}')
        issueCmd('PPAR')
        issueCmd('n')
        issueCmd('160')
        issueCmd('\n')
        issueCmd('oper')
        issueCmd('visc')
        issueCmd('10000')
        issueCmd('iter')
        issueCmd('500')
        issueCmd('mach')
        issueCmd('0')
        issueCmd('pacc')
        fileName = f.replace(".dat", ".txt")
        issueCmd(f'{fileName}')
        issueCmd('')
        issueCmd('aseq')
        issueCmd('0')
        issueCmd('30')
        issueCmd('0.5')
        issueCmd('')
        issueCmd('QUIT')
        issueCmd('QUIT')
        issueCmd('QUIT')
        issueCmd('QUIT')
        issueCmd('QUIT')
        issueCmd('QUIT')
        ps.stdin.close()
        print(name)

#Go through all files didn't go through
