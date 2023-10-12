from typing import Any
from ..victim.classifiers.base import Classifier
from .base import Attacker
from ..attack_assist.goal import ClassifierGoal
from ..tags import *

class ClassificationAttacker(Attacker):
    """
    The base class of all classification attackers.
    """

    def __call__(self, victim: Classifier, input_: Any):
        if not isinstance(victim, Classifier):
            raise TypeError("`victim` is an instance of `%s`, but `%s` expected" % (victim.__class__.__name__, "Classifier"))
        if Tag("get_pred", "victim") not in victim.TAGS:
            raise AttributeError("`%s` needs victim to support `%s` method" % (self.__class__.__name__, "get_pred"))
        self._victim_check(victim)

        if TAG_Classification not in victim.TAGS:
            raise AttributeError("Victim model `%s` must be a classifier" % victim.__class__.__name__)

        if "target" in input_:
            goal = ClassifierGoal(input_["target"], targeted=True)
        else:
             # TODO add ,[ input_["y"] ]
            # origin_x = victim.get_pred([ input_["x"] ])[0]
            # goal = ClassifierGoal( origin_x, targeted=False )
            #将goal变成1, target=1 就是本来是对的
            goal = ClassifierGoal(target=1, targeted=False)

        #TODO 在这里把y带进去
        adversarial_sample = self.attack(victim, input_["x"]+' answer: '+ str(input_["y"]), goal)

        #改成不要check
        # if adversarial_sample is not None:
        #     #add y
        #     y_adv = victim.get_pred([ adversarial_sample + ' answer: '+ str(input_["y"])])[0]
        #     if not goal.check( adversarial_sample, y_adv ):
        #         raise RuntimeError("Check attacker result failed: result ([%d] %s) expect (%s%d)" % ( y_adv, adversarial_sample, "" if goal.targeted else "not ", goal.target))
        return adversarial_sample