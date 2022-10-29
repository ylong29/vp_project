import Tracker
import VPStudio
import UE4VirtualProduction

class Main(object):
    def __init__(self, input_1, input_2, output):
        tracker = Tracker(input_1, input_2, output)
        vp_studio = VPStudio(tracker)
        vp = UE4VirtualProduction(vp_studio)

    def process(self):
        vp.forward()
        vp.update()