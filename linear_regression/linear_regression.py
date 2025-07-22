import numpy as np
import matplotlib.pyplot as plt
try:
    from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
    has_lab_utils = True
except ImportError:
    has_lab_utils = False
    print("lab_utils_uni not found. Using fallback plotting functions.")

# Set matplotlib backend to interactive mode
# Use 'qt' backend for VS Code (works well in Python scripts)
plt.switch_backend('qt5agg')  # or 'tkagg' if Qt5Agg is unavailable

# Data for the first part
x_train_1 = np.array([1.0, 2.0])           # Size in 1000 square feet
y_train_1 = np.array([300.0, 500.0])       # Price in 1000s of dollars

# Data for the second part
x_train_2 = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train_2 = np.array([250, 300, 480, 430, 630, 730])

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray (m,)): Data, m examples 
        y (ndarray (m,)): target values
        w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
    """
    m = x.shape[0] 
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum  
    return total_cost

# Fallback plotting function if lab_utils_uni is unavailable
def plot_data(x, y, w=200, b=100, title="House Prices vs Size"):
    plt.figure()
    plt.scatter(x, y, color='blue', label='Data points')
    # Plot linear fit
    x_line = np.array([min(x), max(x)])
    y_line = w * x_line + b
    plt.plot(x_line, y_line, color='orange', label=f'Fit: w={w}, b={b}')
    plt.xlabel('Size (1000 sq ft)')
    plt.ylabel('Price (1000s of $)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    

# Fallback cost function visualization (contour plot)
def plot_cost_contour(x, y, w_range=(100, 300, 50), b_range=(0, 200, 50)):
    w_vals = np.arange(w_range[0], w_range[1], w_range[2])
    b_vals = np.arange(b_range[0], b_range[1], b_range[2])
    W, B = np.meshgrid(w_vals, b_vals)
    costs = np.zeros_like(W, dtype=float)
    
    for w, b in zip(W.ravel(), B.ravel()):
        costs[W == w, B == b] = compute_cost(x, y, w, b)
    
    costs = costs.reshape(W.shape)
    
    plt.figure()
    contour = plt.contour(W, B, costs, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Cost J(w,b)')
    plt.xlabel('w')
    plt.ylabel('b')
    plt.title('Cost Function Contour')
    plt.grid(True)

# Plotting
if has_lab_utils:
    # Original plotting with lab_utils_uni
    plt_intuition(x_train_1, y_train_1)
    
    plt.close('all')  # Close previous plots
    fig, ax, dyn_items = plt_stationary(x_train_2, y_train_2)
    updater = plt_update_onclick(fig, ax, x_train_2, y_train_2, dyn_items)
    
    soup_bowl()
else: 
    # Fallback plots
    plot_data(x_train_1, y_train_1, w=200, b=100, title="House Prices (Small Dataset)")
    plot_data(x_train_2, y_train_2, w=200, b=100, title="House Prices (Larger Dataset)")
    plot_cost_contour(x_train_2, y_train_2)

plt.show()  # Display all plots