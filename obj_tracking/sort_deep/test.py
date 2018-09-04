
import numpy as np

print ("Generated: ")
# print (np.load('./resources/detections/MOT16_POI_test/MOT16-06.npy').shape)
print (np.load('./resources/detections/MOT16_POI_test/MOT16-06.npy')[10])

print ("Backup/Original: ")
# print (np.load('./resources/detections/MOT16_POI_test/MOT16-06.npy.bkp').shape)
print (np.load('./resources/detections/MOT16_POI_test/MOT16-06.npy.bkp')[10])