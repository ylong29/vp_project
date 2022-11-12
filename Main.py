import Tracker
import VPStudio
import UE4VirtualProduction

class Main(object):
    def __init__(self, input_1, input_2, output):
        self.plan = Plan(input_1, input_2, output)
        self.tracker = Tracker(self.plan)
        self.vp_studio = VPStudio(self.tracker)
        self.vp = UE4VirtualProduction(self.vp_studio)

    def process(self):
        self.vp.forward()
        self.vp.update()
