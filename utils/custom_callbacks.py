from numpy import less, Inf
from tensorflow.keras import callbacks

class EarlyStoppingAtMinLoss(callbacks.Callback):
    """
    Stop training when the val_loss has not decreased for a certain number of epochs, 
    importantly for decaying learning rate: if val_loss remains constant it stops.

    Restores model weights to the one with lowest validated loss.
  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """
    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum val_loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_loss")    
        if less(current, self.best): #Only less than, equality is not enough.
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
            
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("\n Restoring model weights from the end of the best epoch. \n")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
            