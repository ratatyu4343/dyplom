import os
os.environ["DDE_BACKEND"] = "tensorflow"

import deepxde as dde
class SavePredictionsCallback(dde.callbacks.Callback):
    def __init__(self, intervals, X, save_list):
        super().__init__()
        self.intervals = intervals
        self.X = X
        self.save_list = save_list

    def on_epoch_end(self):
        if self.model.train_state.epoch % self.intervals == 0:
            pred = self.model.predict(self.X)
            self.save_list.append(pred)