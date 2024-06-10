import torch
import torch.nn as nn


class AdvancedLMM(nn.Module):
    def __init__(self, n_fixed_effects, n_subjects):
        super(AdvancedLMM, self).__init__()
        # Fixed effects parameters (betas), including intercept
        self.fixed_effects = nn.Parameter(torch.randn(n_fixed_effects))

        # Random intercepts for each subject
        self.random_intercepts = nn.Parameter(torch.randn(n_subjects))

        # Random slopes for each subject, assuming one random slope for simplicity
        self.random_slopes = nn.Parameter(torch.randn(n_subjects))

    def forward(self, X_fixed, subject_indices, X_random_slope):
        # Calculate fixed effects contribution
        fixed_effects = X_fixed @ self.fixed_effects

        # Add random intercepts for each subject
        random_intercepts = self.random_intercepts[subject_indices]

        # Add random slopes for each subject
        # X_random_slope is the design matrix for the variable we have random slopes for
        random_slopes = X_random_slope * self.random_slopes[subject_indices]

        # The final prediction is the sum of fixed effects, random intercepts, and random slopes contributions
        prediction = fixed_effects + random_intercepts + random_slopes.sum(dim=1, keepdim=True)
        return prediction
