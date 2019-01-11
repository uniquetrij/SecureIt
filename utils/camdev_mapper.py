import os
import subprocess

for file in os.listdir("/sys/class/video4linux"):
    real_file = os.path.realpath("/sys/class/video4linux/" + file)
    print(real_file)
    dir = os.path.abspath(real_file + "../../../../")
    id = open(dir + "/idVendor").readline().strip() + ":" + open(dir + "/idProduct").readline().strip()
    name = str(subprocess.check_output("lsusb -d " + id, shell=True)).split(id)[-1].split('\\n')[0].strip()
    index = file.replace("video", "")
    # serial = str(subprocess.check_output("lsusb -d " + id + " | grep -i serial", shell=True))

    print(name, id, index)
