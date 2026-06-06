from models.icarl import iCaRL
from utils.tal import TAL_Loss


class iCaRLTAL(iCaRL):
    def __init__(self, args):
        super().__init__(args)
        self.tal = TAL_Loss(lambda_=0.995, r=1)

    def _before_train_task(self):
        self.tal.update_class_num(self._total_classes)
        self.tal.to(self._device)

    def _classification_loss(self, logits, targets):
        return self.tal(logits, targets)

    def _feature_dump_dir(self):
        return "[PROJECT_ROOT]/fig/tal"
