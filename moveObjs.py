import os
import shutil

objs = []
objs += [each for each in os.listdir(".") if each.endswith('.obj')]

pdb = []
pdb += [each for each in os.listdir(".") if each.endswith('.pdb')]

for val in pdb:
    os.remove("./" + str(val))

for element in objs:
    os.remove("./.objs/" + str(element))
    shutil.move("./" + str(element), "./.objs/" + str(element))