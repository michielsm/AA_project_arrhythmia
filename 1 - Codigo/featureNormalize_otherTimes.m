function X_norm = _normalize_mu_sigma(X, mu, sigma)
X_norm = bsxfun(@minus, X, mu);
X_norm = bsxfun(@rdivide, X_norm, sigma);

endfunction
