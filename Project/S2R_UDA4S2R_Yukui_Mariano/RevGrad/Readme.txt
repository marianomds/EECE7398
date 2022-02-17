Base RevGrad code taken from:
https://github.com/ChrisAllenMing/Mixup_for_UDA/tree/master/Mixup_RevGrad

How to Run:

Before running, comment/uncomment in models.py the corresponding sections of code, according to the model that you want to run (RevGrad, Depthwise Separable RevGrad, or Deeper Depthwise Separable RevGrad).


Without Mixup:
python RevGrad.py --root_path <Datset_Root> --source <Source_Domain> --target <Target_Domain>

With Mixup:
python RevGrad_mixup.py --root_path <Datset_Root> --source <Source_Domain> --target <Target_Domain>