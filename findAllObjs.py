import os
import os.path
import sys

path = str(os.path.dirname(os.path.abspath(__file__))) + "\.objs\\"

objs = []
objs += [each[:-4] for each in os.listdir(".\.objs") if each.endswith('.obj')]

for element in sys.argv:
    objs = [val for val in objs if val.lower() != str(element).lower()]

objs = [path + val + ".obj" for val in objs]

for dirpath, dirnames, filenames in os.walk(str(os.path.dirname(os.path.abspath(__file__)))):
    for filename in [f for f in filenames if f.endswith(".cpp")]:
        if filename[:-4].lower() in [val.lower() for val in sys.argv]:
            print(os.path.join(dirpath, filename))

for val in objs:
    print(val)