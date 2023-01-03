from network.cd_branch import get_model as CDBranch
from network.dd_branch import get_model as DDBranch
from network.mid_branch import get_model as MIDBranch
from network.distillation import get_model as Fusion
from network.base_model import LightningBaseModel


class get_model(LightningBaseModel):

    def __init__(self, args):
        super(get_model, self).__init__(args)
        self.args = args
        self.mid_branch = MIDBranch(args).to('cuda')
        self.dd_branch = DDBranch(args).to('cuda')
        if not self.args.test and not self.args.baseline_only:
            self.cd_branch = CDBranch(args).to('cuda')
            self.fusion = Fusion(args).to('cuda')

    def forward(self, input):
        input = self.mid_branch(input)
        input = self.dd_branch(input)
        if not self.args.test and not self.args.baseline_only:
            input = self.cd_branch(input)
            input = self.fusion(input)
        return input
