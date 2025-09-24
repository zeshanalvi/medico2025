# ---------------------------
# Step 5: Task-Specific Predictors
# ---------------------------
import torch.nn as nn
class TaskPredictor(nn.Module):
    def __init__(self, task_type, hidden=512):
        super().__init__()
        if task_type == "yesno":
            self.head = nn.Linear(hidden, 2)
        elif task_type == "single":
            self.head = nn.Linear(hidden, 10)
        elif task_type == "multi":
            self.head = nn.Linear(hidden, 10)
        elif task_type == "color":
            self.head = nn.Linear(hidden, 5)
        elif task_type == "location":
            self.head = nn.Linear(hidden, 6)
        elif task_type == "count":
            self.head = nn.Linear(hidden, 1)
        else:
            raise ValueError("Unknown task")
    
    def forward(self, x):
        return self.head(x)