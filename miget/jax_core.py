import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial
import numpy as np

# Enable 64-bit precision for numerical stability in scientific computing
jax.config.update("jax_enable_x64", True)

@jit
def ridge_loss(x, A, b, reg_matrix):
    """
    Computes the Ridge Regression Loss:
    L(x) = ||Ax - b||^2 + ||reg_matrix * x||^2
    """
    residuals = jnp.dot(A, x) - b
    measurement_loss = jnp.sum(residuals ** 2)
    
    reg_term = jnp.dot(reg_matrix, x)
    regularization_loss = jnp.sum(reg_term ** 2)
    
    return measurement_loss + regularization_loss

@jit
def pgd_step(x, A, b, reg_matrix, step_size):
    """
    Performs one step of Projected Gradient Descent.
    x_new = max(0, x - step_size * grad(L))
    """
    # Calculate gradient of the loss function w.r.t x
    grads = grad(ridge_loss)(x, A, b, reg_matrix)
    
    # Gradient Descent Step
    x_new = x - step_size * grads
    
    # Projection (Non-negativity constraint)
    x_new = jnp.maximum(0.0, x_new)
    
    return x_new

@partial(jit, static_argnames=['max_iter'])
def solve_nnls_jax(A, b, reg_matrix, x_init=None, max_iter=10000, step_size=None, tol=1e-9):
    """
    Solves NNLS with Tikhonov Regularization using Projected Gradient Descent.
    Min ||Ax - b||^2 + ||Rx||^2  s.t. x >= 0
    """
    n_params = A.shape[1]
    if x_init is None:
        x_init = jnp.zeros(n_params)

    # Pre-calculate Lipschitz constant for step size
    H = jnp.dot(A.T, A) + jnp.dot(reg_matrix.T, reg_matrix)
    
    # Power iteration for spectral radius
    def power_iteration(Matrix, num_iters=10):
        v = jnp.ones(Matrix.shape[0])
        def body(i, v):
             v_new = jnp.dot(Matrix, v)
             return v_new / jnp.linalg.norm(v_new)
        v_final = jax.lax.fori_loop(0, num_iters, body, v)
        eig = jnp.dot(v_final, jnp.dot(Matrix, v_final)) / jnp.dot(v_final, v_final)
        return eig
        
    L_const = power_iteration(H)
    actual_step = 1.0 / L_const if step_size is None else step_size
    
    # Use scan for differentiation compatibility
    # Structure: (carry, stack_output). carry = x
    def scan_body(x, _):
        x_new = pgd_step(x, A, b, reg_matrix, actual_step)
        return x_new, None

    # Run for max_iter steps
    # We do not early stop in scan, ensuring constant graph for JIT/AD
    x_final, _ = jax.lax.scan(scan_body, x_init, None, length=max_iter)
    
    return x_final

def compute_covariance(A, b, reg_matrix, x_sol, measurement_sigma):
    """
    Computes the covariance matrix using implicit differentiation theorem or AD through solver.
    Using AD through 'solve_nnls_jax' (unrolled loop via scan) is numerically exact for the computed graph.
    """
    # Define a function 'solve_wrapper' that takes 'b' and returns 'x'.
    def solve_wrapper(b_vec):
        return solve_nnls_jax(A, b_vec, reg_matrix, max_iter=2000) # Smaller iter for AD speed
    
    # Jacobian of solution w.r.t b: shape (n_params, n_measurements)
    J_sol = jax.jacrev(solve_wrapper)(b) 
    
    # Sigma_b is diagonal with entries measurement_sigma^2.
    Sigma_b = jnp.diag(measurement_sigma**2)
    
    # Sigma_x = J * Sigma_b * J.T
    Sigma_x = jnp.dot(J_sol, jnp.dot(Sigma_b, J_sol.T))
    
    return Sigma_x

# ---- Moment Calculations (Differentiable) ----

def calculate_moments(q_dist, vq_ratios):
    """
    Calculates Mean and SD of the distribution in Log Domain.
    """
    # Filter for valid compartments (avoid log(0))
    # In JAX, we need masks.
    valid_mask = (vq_ratios > 1e-6)
    
    # Use jnp.where to handle invalid values safely (replace with 1.0 so log is 0)
    safe_ratios = jnp.where(valid_mask, vq_ratios, 1.0)
    log_ratios = jnp.log(safe_ratios)
    
    total_flow = jnp.sum(q_dist * valid_mask)
    
    # Normalize (locally for moment calc)
    w_dist = jnp.where(valid_mask, q_dist, 0.0)
    weights = w_dist / (total_flow + 1e-9) # Avoid division by zero
    
    # Mean Log V/Q
    mean_log = jnp.sum(weights * log_ratios)
    
    # Variance Log V/Q
    var_log = jnp.sum(weights * (log_ratios - mean_log)**2)
    sd_log = jnp.sqrt(var_log)
    
    # Convert Mean back to linear scale (Geometric Mean)
    mean_linear = jnp.exp(mean_log)
    
    return mean_linear, sd_log

def get_moment_errors(q_dist, vq_ratios, Sigma_x):
    """
    Calculates the standard error for Mean and SD using Delta Method.
    Var(f(x)) ~ grad_f.T * Sigma_x * grad_f
    """
    
    # Define vector function for moments
    def moment_fn(x):
        m, s = calculate_moments(x, vq_ratios)
        return jnp.stack([m, s])
        
    # Jacobian of moments w.r.t q_dist: Shape (2, n_check)
    J_moments = jax.jacfwd(moment_fn)(q_dist)
    
    # Covariance of moments: (2, n) * (n, n) * (n, 2) -> (2, 2)
    Cov_moments = jnp.dot(J_moments, jnp.dot(Sigma_x, J_moments.T))
    
    # Extract variances
    var_mean = Cov_moments[0, 0]
    var_sd = Cov_moments[1, 1]
    
    return jnp.sqrt(var_mean), jnp.sqrt(var_sd)

