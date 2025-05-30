
import torch
import numpy as np
from scipy.interpolate import BSpline
from scipy.stats import norm
import math

def __feat_dims2cov_dims(covariate_dims, covariate_types):
    """Convert covariate dimensions to feature dimensions based on types"""
    feat_dims = []
    for i, (dim, cov_type) in enumerate(zip(covariate_dims, covariate_types)):
        if cov_type == 'categorical':
            # Will be determined by number of unique categories
            feat_dims.append(None)  # To be filled later
        elif cov_type == 'ordinal':
            feat_dims.append(1)  # Single normalized dimension
        elif cov_type == 'continuous':
            feat_dims.append(10)  # Default number of B-spline basis functions
        elif cov_type == 'seasonal':
            feat_dims.append(8)   # Default number of wavelet basis functions
        else:
            feat_dims.append(1)   # Default linear
    return feat_dims

class FeatureMap(object):
    def __init__(self, covariate_dims, covariate_types):
        self.covariate_dims = covariate_dims
        self.covariate_types = covariate_types
    
    def make_feature_map(self, X_train):
        raise NotImplementedError
    
    def featmap(self, X):
        raise NotImplementedError
    
    def __call__(self, X):
        return self.featmap(X)

class LinearFeatureMap(FeatureMap):
    def make_feature_map(self, X_train):
        self.__dim_mean = X_train.mean(axis=0)
        self.__dim_sd = X_train.std(axis=0)
        self.input_dim_indcs = list(range(X_train.shape[1]))
        assert torch.sum(self.__dim_sd == 0) == 0, "Zero variance features"
    
    def featmap(self, X):
        return (X - self.__dim_mean) / self.__dim_sd
    
    def featmap1D(self, x1d, cov_ix):
        trans_feat = (x1d - self.__dim_mean[cov_ix]) / self.__dim_sd[cov_ix]
        return trans_feat.reshape((x1d.shape[0], 1))

class NonlinearFeatureMap(FeatureMap):
    def __init__(self, covariate_dims, covariate_types, n_knots=10, n_wavelets=8):
        super().__init__(covariate_dims, covariate_types)
        self.n_knots = n_knots
        self.n_wavelets = n_wavelets
        self.feature_processors = {}

        
    def make_feature_map(self, X_train):
        self.X_train = X_train
        self.input_dim_indcs = list(range(X_train.shape[1]))
        for i, (dim, cov_type) in enumerate(zip(self.covariate_dims, self.covariate_types)):
            if cov_type == 'categorical':
                self._fit_categorical(X_train[:, i], i)
            elif cov_type == 'ordinal':
                self._fit_ordinal(X_train[:, i], i)
            elif cov_type == 'continuous':
                self._fit_continuous(X_train[:, i], i)
            elif cov_type == 'seasonal':
                self._fit_seasonal(X_train[:, i], i)
    
    def _fit_categorical(self, x, dim_idx):
        """Fit categorical encoding (one-hot)"""
        unique_vals = torch.unique(x)
        self.feature_processors[dim_idx] = {
            'type': 'categorical',
            'categories': unique_vals,
            'n_categories': len(unique_vals)
        }
    
    def _fit_ordinal(self, x, dim_idx):
        """Fit ordinal scaling (mean-std normalization)"""
        mean_val = x.mean()
        std_val = x.std()
        assert std_val > 0, f"Zero variance in ordinal feature {dim_idx}"
        self.feature_processors[dim_idx] = {
            'type': 'ordinal',
            'mean': mean_val,
            'std': std_val
        }
    
    def _fit_continuous(self, x, dim_idx):
        """Fit B-spline basis using quantile knots"""
        x_np = x.detach().numpy() if isinstance(x, torch.Tensor) else x
        
        # Create quantile-based knots
        quantiles = np.linspace(0, 1, self.n_knots - 2)
        interior_knots = np.quantile(x_np, quantiles[1:-1])
        
        # Add boundary knots
        x_min, x_max = x_np.min(), x_np.max()
        knots = np.concatenate([[x_min], interior_knots, [x_max]])
        
        # B-spline degree
        degree = 3
        
        self.feature_processors[dim_idx] = {
            'type': 'continuous',
            'knots': knots,
            'degree': degree,
            'n_basis': len(knots) + degree - 1
        }
    
    def _fit_seasonal(self, x, dim_idx):
        """Fit wavelet basis for seasonal patterns"""
        # Simple Fourier basis for seasonal patterns
        x_np = x.detach().numpy() if isinstance(x, torch.Tensor) else x
        period = 2 * np.pi  # Assume normalized seasonal period
        
        self.feature_processors[dim_idx] = {
            'type': 'seasonal',
            'period': period,
            'n_harmonics': self.n_wavelets // 2
        }
    
    def _transform_categorical(self, x, processor):
        """Transform categorical variable to one-hot encoding"""
        categories = processor['categories']
        n_categories = processor['n_categories']
        
        # Create one-hot encoding
        one_hot = torch.zeros(len(x), n_categories)
        for i, cat in enumerate(categories):
            mask = (x == cat)
            one_hot[mask, i] = 1.0
            
        return one_hot
    
    def _transform_ordinal(self, x, processor):
        """Transform ordinal variable with mean-std scaling"""
        mean_val = processor['mean']
        std_val = processor['std']
        return ((x - mean_val) / std_val).reshape(-1, 1)
    
    def _transform_continuous(self, x, processor):
        """Transform continuous variable using B-spline basis"""
        knots = processor['knots']
        degree = processor['degree']
        x_np = x.detach().numpy() if isinstance(x, torch.Tensor) else x
        
        # Extend knots for B-spline
        extended_knots = np.concatenate([
            [knots[0]] * degree,
            knots,
            [knots[-1]] * degree
        ])
        
        # Evaluate B-spline basis functions
        n_basis = len(knots) + degree - 1
        basis_matrix = np.zeros((len(x_np), n_basis))
        
        for i in range(n_basis):
            # Create B-spline for i-th basis function
            coeffs = np.zeros(n_basis)
            coeffs[i] = 1.0
            spline = BSpline(extended_knots, coeffs, degree, extrapolate=True)
            basis_matrix[:, i] = spline(x_np)
        
        return torch.tensor(basis_matrix, dtype=torch.float32)
    
    def _transform_seasonal(self, x, processor):
        """Transform seasonal variable using Fourier basis"""
        period = processor['period']
        n_harmonics = processor['n_harmonics']
        x_np = x.detach().numpy() if isinstance(x, torch.Tensor) else x
        
        # Create Fourier basis
        basis_matrix = np.zeros((len(x_np), 2 * n_harmonics))
        
        for k in range(1, n_harmonics + 1):
            # Sine and cosine components
            basis_matrix[:, 2*(k-1)] = np.sin(2 * np.pi * k * x_np / period)
            basis_matrix[:, 2*(k-1) + 1] = np.cos(2 * np.pi * k * x_np / period)
        
        return torch.tensor(basis_matrix, dtype=torch.float32)
    
    def featmap(self, X):
        """Transform all features"""
        transformed_features = []
        
        for i, (dim, cov_type) in enumerate(zip(self.covariate_dims, self.covariate_types)):
            processor = self.feature_processors[i]
            x_col = X[:, i]
            
            if cov_type == 'categorical':
                feat = self._transform_categorical(x_col, processor)
            elif cov_type == 'ordinal':
                feat = self._transform_ordinal(x_col, processor)
            elif cov_type == 'continuous':
                feat = self._transform_continuous(x_col, processor)
            elif cov_type == 'seasonal':
                feat = self._transform_seasonal(x_col, processor)
            else:
                # Default linear transformation
                feat = x_col.reshape(-1, 1)
            
            transformed_features.append(feat)
        
        return torch.cat(transformed_features, dim=1)
    
    def featmap1D(self, x1d, cov_ix):
        """Transform single covariate"""
        processor = self.feature_processors[cov_ix]
        cov_type = self.covariate_types[cov_ix]
        
        if cov_type == 'categorical':
            return self._transform_categorical(x1d, processor)
        elif cov_type == 'ordinal':
            return self._transform_ordinal(x1d, processor)
        elif cov_type == 'continuous':
            return self._transform_continuous(x1d, processor)
        elif cov_type == 'seasonal':
            return self._transform_seasonal(x1d, processor)
        else:
            return x1d.reshape(-1, 1)
        
        ### need to add one with cut off frequency
        
    def get_feature_processors(self):
        return self.feature_processors

class AutoFeatureMap(FeatureMap):
    """Automatically choose between linear and nonlinear based on covariate types"""
    
    def __init__(self, covariate_dims, covariate_types, mode='auto-nonlinear'):
        super().__init__(covariate_dims, covariate_types)
        self.mode = mode
        
        # Determine if we need nonlinear features
        needs_nonlinear = any(ctype in ['continuous', 'seasonal', 'categorical'] 
                             for ctype in covariate_types)
        
        if mode == 'auto-linear' or not needs_nonlinear:
            self.feature_map = LinearFeatureMap(covariate_dims, covariate_types)
        else:
            self.feature_map = NonlinearFeatureMap(covariate_dims, covariate_types)
    
    def make_feature_map(self, X_train):
        return self.feature_map.make_feature_map(X_train)
    
    def featmap(self, X):
        return self.feature_map.featmap(X)
    
    def featmap1D(self, x1d, cov_ix):
        return self.feature_map.featmap1D(x1d, cov_ix)

if __name__ == "__main__":
    print("Testing Linear Feature Map:")
    p = 10
    X_train = torch.normal(mean=0., std=1., size=(1000, p))
    X_test = torch.normal(mean=0., std=1., size=(100, p))
    covariate_dims = list(range(p))
    covariate_types = ['continuous'] * p
    print("Covariate types:", covariate_types)
    
    linfeatmap = LinearFeatureMap(covariate_dims, covariate_types)
    linfeatmap.make_feature_map(X_train)
    linear_result = linfeatmap(X_test)
    print("Linear feature map output shape:", linear_result.shape)
    print("First few rows:", linear_result[:3])
    
    print("\nTesting Nonlinear Feature Map:")
    # Test with mixed covariate types
    mixed_types = ['continuous', 'ordinal', 'categorical', 'seasonal'] + ['continuous'] * (p-4)
    
    # Create some categorical data for testing
    X_train_mixed = X_train.clone()
    X_test_mixed = X_test.clone()
    X_train_mixed[:, 2] = torch.randint(0, 3, (1000,)).float()  # 3 categories
    X_test_mixed[:, 2] = torch.randint(0, 3, (100,)).float()
    
    nonlin_featmap = NonlinearFeatureMap(covariate_dims, mixed_types)
    nonlin_featmap.make_feature_map(X_train_mixed)
    nonlinear_result = nonlin_featmap(X_test_mixed)
    print("Nonlinear feature map output shape:", nonlinear_result.shape)
    
    print("\nTesting Auto Feature Map:")
    auto_featmap = AutoFeatureMap(covariate_dims, mixed_types, mode='auto-nonlinear')
    auto_featmap.make_feature_map(X_train_mixed)
    auto_result = auto_featmap(X_test_mixed)
    print("Auto feature map output shape:", auto_result.shape)