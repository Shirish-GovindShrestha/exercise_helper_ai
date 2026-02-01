import copy

class EarlyStopping:
    __slots__ = ('mode', 'patience', 'verbose', 'counter', 'best_value', 'early_stop', 'best_model_state', '_is_better')
    
    def __init__(self, mode='min', patience=10, verbose=True):
        assert mode in ('min', 'max'), "mode should be 'min' or 'max'"
        self.mode = mode
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.best_model_state = None
        # Cache comparison function for efficiency
        self._is_better = (lambda curr, best: curr < best) if mode == 'min' else (lambda curr, best: curr > best)

    def __call__(self, current_value, model):
        # Check if this is an improvement
        is_improvement = self.best_value is None or self._is_better(current_value, self.best_value)

        if is_improvement:
            # Improvement detected - save model
            if self.verbose:
                msg = (f"✅ Metric improved ({self.best_value:.6f} → {current_value:.6f}). Saving model..." 
                       if self.best_value is not None 
                       else f"✅ First metric recorded: {current_value:.6f}. Saving model...")
                print(msg)
            
            self.best_value = current_value
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"⏳ EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"🛑 Early stopping triggered. Best metric: {self.best_value:.6f}")

