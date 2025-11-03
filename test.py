from src.parsers import CocoFreeView
from src.simulation import gen_gaze

data = CocoFreeView()
print('s')
gaze, fixations, fixation_mask = gen_gaze(data,
                                          5164, 
                                          60,
                                          get_scanpath=True,
                                          get_fixation_mask=True)
print('e')

