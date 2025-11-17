class EarlyStopping:
    def __init__(self, mode='min', patience=10, verbose=True):
        """
        Flexible EarlyStopping for either loss or score.
        
        Args:
            mode (str): 'min' for loss (lower is better), 'max' for score (higher is better)
            patience (int): Number of epochs to wait for improvement
            verbose (bool): Whether to print messages
        """
        assert mode in ['min', 'max'], "mode should be 'min' or 'max'"
        self.mode = mode
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, current_value, model):
        is_improvement = False
        if self.best_value is None:
            is_improvement = True
        else:
            if self.mode == 'min' and current_value < self.best_value:
                is_improvement = True
            elif self.mode == 'max' and current_value > self.best_value:
                is_improvement = True

        if is_improvement:
            if self.verbose:
                if self.best_value is not None:
                    print(f"✅ Metric improved ({self.best_value:.6f} → {current_value:.6f}). Saving model...")
                else:
                    print(f"✅ First metric recorded: {current_value:.6f}. Saving model...")
            self.best_value = current_value
            self.best_model_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"⏳ EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
