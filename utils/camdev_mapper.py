import os
import subprocess

for file in os.listdir("/sys/class/video4linux"):
    real_file = os.path.realpath("/sys/class/video4linux/" + file)
    # print(real_file)
    dir = os.path.abspath(real_file + "../../../../")
    # print(os.listdir(dir))
    id = open(dir + "/idVendor").readline().strip() + ":" + open(dir + "/idProduct").readline().strip()
    # serial = open(dir + "/serial").readline().strip()
    name = open(dir + "/product").readline().strip()
    index = file.replace("video", "")
    # serial = str(subprocess.check_output("lsusb -v -d " + id + " | grep -i serial", shell=True))

    print(name, id, index)
    print('-----------------------------------------------')
