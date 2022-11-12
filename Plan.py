import ShotGrid

class Plan(object):
    def __init__(self, input_1, input_2, output):
        self.shotgrid = ShotGrid()
        self.shotgrid.set_inputs(input_1, input_2)
        self.shotgrid.set_output(output)

    def set_enterprise_resource(self):
        self.shotgrid.set_enterprise_resource()

    def set_production(self):
        self.shotgrid.set_production()
