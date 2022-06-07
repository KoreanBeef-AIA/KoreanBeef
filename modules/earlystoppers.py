import numpy as np
import logging

class EarlyStopper():
    
    def __init__(self, patience: int, mode: str, logger: logging.RootLogger=None) -> None:
        self.patience = patience
        self.mode = mode
        self.logger = logger
        
        # Initiate
        self.patience_counter = 0
        self.stop = False
        self.best_loss = np.inf
        
        msg = f'Initiated early stopper, mode: {self.mode}, best score: {self.best_loss}, patience: {self.patience}'
        self.logger.info(msg) if self.logger else None
        
    def check_early_stopping(self, loss: float) -> None:
        loss = -loss if self.mode == 'max' else loss
        
        if loss > self.best_loss:
            # Higher loss (worse score)
            self.patience_counter += 1
            
            msg = f"Early stopper, counter {self.patience_counter}/{self.patience}, best:{abs(self.best_loss)} -> now:{abs(loss)}"
            self.logger.info(msg) if self.logger else None
            
        elif loss <= self.best_loss:
            # Lower loss (better score)
            self.patience_counter = 0
            self.best_loss = loss
            
            if self.logger is not None:
                msg = f"Early stopper, counter {self.patience_counter}/{self.patience}, best:{abs(self.best_loss)} -> now:{abs(loss)}"
                self.logger.info(msg)
                self.logger.info(f"Set counter as {self.patience_counter}")
                self.logger.info(f"Update best score as {abs(loss)}")
        
        else:
            print('debug')
 