---
title: 💀 Домашка
order: 3
toc: true
---

### Linear algebra basics


1. [5 points] **Sensitivity Analysis in Linear Systems** Consider a nonsingular matrix $A \in \mathbb{R}^{n \times n}$ and a vector $b \in \mathbb{R}^n$. Suppose that due to measurement or computational errors, the vector $b$ is perturbed to $\tilde{b} = b + \delta b$.  
    1. Derive an upper bound for the relative error in the solution $x$ of the system $Ax = b$ in terms of the condition number $\kappa(A)$ and the relative error in $b$.  
    1. Provide a concrete example using a $2 \times 2$ matrix where $\kappa(A)$ is large (say, $\geq 100500$).

1. [5 points] **Effect of Diagonal Scaling on Rank** Let $A \in \mathbb{R}^{n \times n}$ be a matrix with rank $r$. Suppose $D \in \mathbb{R}^{n \times n}$ is a diagonal matrix. Determine the rank of the product $DA$. Explain your reasoning.

1. [8 points] **Unexpected SVD** Compute the Singular Value Decomposition (SVD) of the following matrices:
    * $A_1 = \begin{bmatrix} 2 \\ 2 \\ 8 \end{bmatrix}$
    * $A_2 = \begin{bmatrix} 0 & x \\ x & 0 \\ 0 & 0 \end{bmatrix}$, where $x$ is the sum of your birthdate numbers (day + month).

1. [10 points] **Effect of normalization on rank** Assume we have a set of data points $x^{(i)}\in\mathbb{R}^{n},\,i=1,\dots,m$, and decide to represent this data as a matrix
    $$
    X =
    \begin{pmatrix}
     | & & | \\
     x^{(1)} & \dots & x^{(m)} \\
     | & & | \\
    \end{pmatrix} \in \mathbb{R}^{n \times m}.
    $$

    We suppose that $\text{rank}\,X = r$.

    In the problem below, we ask you to find the rank of some matrix $M$ related to $X$.
    In particular, you need to find relation between $\text{rank}\,X = r$ and $\text{rank}\,M$, e.g., that the rank of $M$ is always larger/smaller than the rank of $X$ or that $\text{rank}\,M = \text{rank}\,X \big / 35$.
    Please support your answer with legitimate arguments and make the answer as accurate as possible.

    Note that border cases are possible depending on the structure of the matrix $X$. Make sure to cover them in your answer correctly.

    In applied statistics and machine learning, data is often normalized.
    One particularly popular strategy is to subtract the estimated mean $\mu$ and divide by the square root of the estimated variance $\sigma^2$. i.e.
    $$
    x \rightarrow (x - \mu) \big / \sigma.
    $$
    After the normalization, we get a new matrix
    $$
    \begin{split}
    Y &:=
    \begin{pmatrix}
     | & & | \\
     y^{(1)} & \dots & y^{(m)} \\
     | & & | \\
    \end{pmatrix},\\
    y^{(i)} &:= \frac{x^{(i)} - \frac{1}{m}\sum_{j=1}^{m} x^{(j)}}{\sigma}.
    \end{split}
    $$
    What is the rank of $Y$ if $\text{rank} \; X = r$? Here $\sigma$ is a vector and the division is element-wise. The reason for this is that different features might have different scales. Specifically:
    $$
    \sigma_i = \sqrt{\frac{1}{m}\sum_{j=1}^{m} \left(x_i^{(j)}\right)^2 - \left(\frac{1}{m}\sum_{j=1}^{m} x_i^{(j)}\right)^2}.
    $$

1. [20 points] **Image Compression with Truncated SVD** Explore image compression using Truncated Singular Value Decomposition (SVD). Understand how varying the number of singular values affects the quality of the compressed image.
    Implement a Python script to compress a grayscale image using Truncated SVD and visualize the compression quality.
    
    * **Truncated SVD**: Decomposes an image $A$ into $U, S,$ and $V$ matrices. The compressed image is reconstructed using a subset of singular values.
    * **Mathematical Representation**: 
        $$
        A \approx U_k \Sigma_k V_k^T
        $$
        * $U_k$ and $V_k$ are the first $k$ columns of $U$ and $V$, respectively.
        * $\Sigma_k$ is a diagonal matrix with the top $k$ singular values.
        * **Relative Error**: Measures the fidelity of the compressed image compared to the original. 
            $$
            \text{Relative Error} = \frac{\| A - A_k \|}{\| A \|}
            $$

    ```python
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    from skimage import io, color
    import requests
    from io import BytesIO

    def download_image(url):
        response = requests.get(url)
        img = io.imread(BytesIO(response.content))
        return color.rgb2gray(img)  # Convert to grayscale

    def update_plot(i, img_plot, error_plot, U, S, V, original_img, errors, ranks, ax1, ax2):
        # Adjust rank based on the frame index
        if i < 70:
            rank = i + 1
        else:
            rank = 70 + (i - 69) * 10

        reconstructed_img = ... # YOUR CODE HERE 

        # Calculate relative error
        relative_error = ... # YOUR CODE HERE
        errors.append(relative_error)
        ranks.append(rank)

        # Update the image plot and title
        img_plot.set_data(reconstructed_img)
        ax1.set_title(f"Image compression with SVD\n Rank {rank}; Relative error {relative_error:.2f}")

        # Remove axis ticks and labels from the first subplot (ax1)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Update the error plot
        error_plot.set_data(ranks, errors)
        ax2.set_xlim(1, len(S))
        ax2.grid(linestyle=":")
        ax2.set_ylim(1e-4, 0.5)
        ax2.set_ylabel('Relative Error')
        ax2.set_xlabel('Rank')
        ax2.set_title('Relative Error over Rank')
        ax2.semilogy()

        # Set xticks to show rank numbers
        ax2.set_xticks(range(1, len(S)+1, max(len(S)//10, 1)))  # Adjust the step size as needed
        plt.tight_layout()

        return img_plot, error_plot


    def create_animation(image, filename='svd_animation.mp4'):
        U, S, V = np.linalg.svd(image, full_matrices=False)
        errors = []
        ranks = []

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))
        img_plot = ax1.imshow(image, cmap='gray', animated=True)
        error_plot, = ax2.plot([], [], 'r-', animated=True)  # Initial empty plot for errors

        # Add watermark
        ax1.text(1, 1.02, '@fminxyz', transform=ax1.transAxes, color='gray', va='bottom', ha='right', fontsize=9)

        # Determine frames for the animation
        initial_frames = list(range(70))  # First 70 ranks
        subsequent_frames = list(range(70, len(S), 10))  # Every 10th rank after 70
        frames = initial_frames + subsequent_frames

        ani = animation.FuncAnimation(fig, update_plot, frames=len(frames), fargs=(img_plot, error_plot, U, S, V, image, errors, ranks, ax1, ax2), interval=50, blit=True)
        ani.save(filename, writer='ffmpeg', fps=8, dpi=300)

        # URL of the image
        url = ""

        # Download the image and create the animation
        image = download_image(url)
        create_animation(image)
    ```


### Convergence rates

1. [6 points] Determine (it means to prove the character of convergence if it is convergent) the convergence or divergence of a given sequences
    * $r_{k} = \frac{1}{\sqrt{k+5}}$.
    * $r_{k} = 0.101^k$.
    * $r_{k} = 0.101^{2^k}$.

1. [8 points] Let the sequence $\{r_k\}$ be defined by
    $$
    r_{k+1} = 
    \begin{cases}
    \frac{1}{2}\,r_k, & \text{if } k \text{ is even}, \\
    r_k^2, & \text{if } k \text{ is odd},
    \end{cases}
    $$
    with initial value $0 < r_0 < 1$. Prove that $\{r_k\}$ converges to 0 and analyze its convergence rate. In your answer, determine whether the overall convergence is linear, superlinear, or quadratic.

1. [6 points] Determine the following sequence $\{r_k\}$ by convergence rate (linear, sublinear, superlinear). In the case of superlinear convergence, determine whether there is quadratic convergence.
    $$
    r_k = \dfrac{1}{k!}
    $$

1. [8 points] Consider the recursive sequence defined by
    $$
    r_{k+1} = \lambda\,r_k + (1-\lambda)\,r_k^p,\quad k\ge0,
    $$
    where $\lambda\in [0,1)$ and $p>1$. Which additional conditions on $r_0$ should be satisfied for the sequence to converge? Show that when $\lambda>0$ the sequence converges to 0 with a linear rate (with asymptotic constant $\lambda$), and when $\lambda=0$ determine the convergence rate in terms of $p$. In particular, for $p=2$ decide whether the convergence is quadratic.

### Line search

1. [10 points] Consider a strongly convex quadratic function $f: \mathbb{R}^n \rightarrow \mathbb{R}$, and let us start from a point $x_k \in \mathbb{R}^n$ moving in the direction of the antigradient $-\nabla f(x_k)$, note that $\nabla f(x_k)\neq 0$. Show that the minimum of $f$ along this direction as a function of the step size $\alpha$, for a decreasing function at $x_k$, satisfies Armijo's condition for any $c_1$ in the range $0 \leq c_1 \leq \frac{1}{2}$. Specifically, demonstrate that the following inequality holds at the optimal $\alpha^*$:
    $$
    \varphi(\alpha) = f(x_{k+1}) = f(x_k - \alpha \nabla f(x_k)) \leq f(x_k) - c_1 \alpha \|\nabla f(x_k)\|_2^2
    $$

1. **Implementing and Testing Line Search Conditions in Gradient Descent** [36 points] 
    $$
    x_{k+1} = x_k - \alpha \nabla f(x_k)
    $$
    In this assignment, you will modify an existing Python code for gradient descent to include various line search conditions. You will test these modifications on two functions: a quadratic function and the Rosenbrock function. The main objectives are to understand how different line search strategies influence the convergence of the gradient descent algorithm and to compare their efficiencies based on the number of function evaluations.

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize_scalar
    np.random.seed(214)

    # Define the quadratic function and its gradient
    def quadratic_function(x, A, b):
        return 0.5 * np.dot(x.T, np.dot(A, x)) - np.dot(b.T, x)

    def grad_quadratic(x, A, b):
        return np.dot(A, x) - b

    # Generate a 2D quadratic problem with a specified condition number
    def generate_quadratic_problem(cond_number):
        # Random symmetric matrix
        M = np.random.randn(2, 2)
        M = np.dot(M, M.T)

        # Ensure the matrix has the desired condition number
        U, s, V = np.linalg.svd(M)
        s = np.linspace(cond_number, 1, len(s))  # Spread the singular values
        A = np.dot(U, np.dot(np.diag(s), V))

        # Random b
        b = np.random.randn(2)

        return A, b

    # Gradient descent function
    def gradient_descent(start_point, A, b, stepsize_func, max_iter=100):
        x = start_point.copy()
        trajectory = [x.copy()]

        for i in range(max_iter):
            grad = grad_quadratic(x, A, b)
            step_size = stepsize_func(x, grad)
            x -= step_size * grad
            trajectory.append(x.copy())

        return np.array(trajectory)

    # Backtracking line search strategy using scipy
    def backtracking_line_search(x, grad, A, b, alpha=0.3, beta=0.8):
        def objective(t):
            return quadratic_function(x - t * grad, A, b)
        res = minimize_scalar(objective, method='golden')
        return res.x

    # Generate ill-posed problem
    cond_number = 30
    A, b = generate_quadratic_problem(cond_number)

    # Starting point
    start_point = np.array([1.0, 1.8])

    # Perform gradient descent with both strategies
    trajectory_fixed = gradient_descent(start_point, A, b, lambda x, g: 5e-2)
    trajectory_backtracking = gradient_descent(start_point, A, b, lambda x, g: backtracking_line_search(x, g, A, b))

    # Plot the trajectories on a contour plot
    x1, x2 = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 2, 400))
    Z = np.array([quadratic_function(np.array([x, y]), A, b) for x, y in zip(x1.flatten(), x2.flatten())]).reshape(x1.shape)

    plt.figure(figsize=(10, 8))
    plt.contour(x1, x2, Z, levels=50, cmap='viridis')
    plt.plot(trajectory_fixed[:, 0], trajectory_fixed[:, 1], 'o-', label='Fixed Step Size')
    plt.plot(trajectory_backtracking[:, 0], trajectory_backtracking[:, 1], 'o-', label='Backtracking Line Search')

    # Add markers for start and optimal points
    plt.plot(start_point[0], start_point[1], 'ro', label='Start Point')
    optimal_point = np.linalg.solve(A, b)
    plt.plot(optimal_point[0], optimal_point[1], 'y*', markersize=15, label='Optimal Point')

    plt.legend()
    plt.title('Gradient Descent Trajectories on Quadratic Function')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig("linesearch.svg")
    plt.show()
    ```

    ![The code above plots this](linesearch.svg)

    Start by reviewing the provided Python code. This code implements gradient descent with a fixed step size and a backtracking line search on a quadratic function. Familiarize yourself with how the gradient descent function and the step size strategies are implemented.

    1. [10/36 points] Modify the gradient descent function to include the following line search conditions:

        1. Dichotomy 
        1. Sufficient Decrease Condition
        1. Wolfe Condition
        1. Polyak step size
            $$
            \alpha_k = \frac{f(x_k) - f^*}{\|\nabla f(x_k)\|_2^2},
            $$
            where $f^*$ is the optimal value of the function. It seems strange to use the optimal value of the function in the step size, but there are options to estimate it even without knowing the optimal value.       
        1. Sign Gradient Method:
            $$
            \alpha_k = \frac{1}{\|\nabla f(x_k)\|_2},
            $$
        Test your modified gradient descent algorithm with the implemented step size search conditions on the provided quadratic function. Plot the trajectories over iterations for each condition. Choose and specify hyperparameters for inexact line search conditions. Choose and specify the **termination criterion**. Start from the point $x_0 = (-1, 2)^T$.

    1. [8/36 points] Compare these 7 methods from the budget perspective. Plot the graph of function value from the number of function evaluations for each method on the same graph.
    1. [10/36 points] Plot trajectory for another function with the same set of methods
        $$
        f(x_1, x_2) =  10(x_2 − x_1^2)^2 + (x_1 − 1)^2
        $$
        with $x_0 = (-1, 2)^T$. You might need to adjust hyperparameters.

    1. [8/36 points] Plot the same function value from the number of function calls for this experiment.

### Matrix calculus

1. [6 points] Find the gradient $\nabla f(x)$ and hessian $f^{\prime\prime}(x)$, if $f(x) = \frac{1}{2}\Vert A - xx^T\Vert ^2_F, A \in \mathbb{S}^n$

1. [6 points] Find the gradient $\nabla f(x)$ and hessian $f''(x)$, if $f(x) = \dfrac{1}{2} \Vert Ax - b\Vert^2_2$.

1. [8 points] Find the gradient $\nabla f(x)$ and hessian $f''(x)$, if 
    $$
    f(x) = \frac1m \sum\limits_{i=1}^m \log \left( 1 + \exp(a_i^{T}x) \right) + \frac{\mu}{2}\Vert x\Vert _2^2, \; a_i, x \in \mathbb R^n, \; \mu>0
    $$

1. [8 points] Compute the gradient $\nabla_A f(A)$ of the trace of the matrix exponential function $f(A) = \text{tr}(e^A)$ with respect to $A$. Hint: Use the definition of the matrix exponential. Use the definition of the differential $df = f(A + dA) - f(A) + o(\Vert dA \Vert)$ with the limit $\Vert dA \Vert \to 0$.

1. [20 points] **Principal Component Analysis through gradient calculation.** Let there be a dataset $\{x_i\}_{i=1}^N, x_i \in \mathbb{R}^D$, which we want to transform into a dataset of reduced dimensionality $d$ using projection onto a linear subspace defined by the matrix $P \in \mathbb{R}^{D \times d}$. The orthogonal projection of a vector $x$ onto this subspace can be computed as $P(P^TP)^{-1}P^Tx$. To find the optimal matrix $P$, consider the following optimization problem:
    $$
    F(P) = \sum_{i=1}^N \|x_i - P(P^TP)^{-1}P^Tx_i\|^2 = N \cdot \text{tr}\left((I - P(P^TP)^{-1}P^T)^2 S\right) \to \min_{P \in \mathbb{R}^{D \times d}},
    $$
    where $S = \frac{1}{N} \sum_{i=1}^N x_i x_i^T$ is the sample covariance matrix for the normalized dataset.

    1. Find the gradient $\nabla_P F(P)$, calculated for an arbitrary matrix $P$ with orthogonal columns, i.e., $P : P^T P = I$. 
    
        *Hint: When calculating the differential $dF(P)$, first treat $P$ as an arbitrary matrix, and then use the orthogonality property of the columns of $P$ in the resulting expression.*
    1. Consider the eigendecomposition of the matrix $S$: 
        $$
        S = Q \Lambda Q^T,
        $$
        where $\Lambda$ is a diagonal matrix with eigenvalues on the diagonal, and $Q = [q_1 | q_2 | \ldots | q_D] \in \mathbb{R}^{D \times D}$ is an orthogonal matrix consisting of eigenvectors $q_i$ as columns. Prove the following:

        1. The gradient $\nabla_P F(P)$ equals zero for any matrix $P$ composed of $d$ distinct eigenvectors $q_i$ as its columns.
        2. The minimum value of $F(P)$ is achieved for the matrix $P$ composed of eigenvectors $q_i$ corresponding to the largest eigenvalues of $S$.


### Automatic differentiation and jax

1. **Benchmarking Hessian-Vector Product (HVP) Computation in a Neural Network via JAX** [22 points]

    You are given a simple neural network model (an MLP with several hidden layers using a nonlinearity such as GELU). The model’s parameters are defined by the weights of its layers. Your task is to compare different approaches for computing the Hessian-vector product (HVP) with respect to the model’s loss and to study how the computation time scales as the model grows in size.

    **Model and Loss Definition:** [2/22 points] Here is the code for the model and loss definition. Write a method `get_params()` that returns the flattened vector of all model weights.

    ```python
    import jax
    import jax.numpy as jnp
    import time
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from tqdm.auto import tqdm
    from jax.nn import gelu

    # Определение MLP модели
    class MLP:
        def __init__(self, key, layer_sizes):
            self.layer_sizes = layer_sizes
            keys = jax.random.split(key, len(layer_sizes) - 1)
            self.weights = [
                jax.random.normal(k, (layer_sizes[i], layer_sizes[i + 1]))
                for i, k in enumerate(keys)
                ]
        
        def forward(self, x):
            for w in self.weights[:-1]:
                x = gelu(jnp.dot(x, w))
            return jnp.dot(x, self.weights[-1])
        
        def get_params(self):
            ### YOUR CODE HERE ###
            return None
    ```

    **Hessian and HVP Implementations:** [2/22 points] Write a function 
    
    ```python
    # Функция для вычисления Гессиана
    def calculate_hessian(model, params):
        def loss_fn(p):
            x = jnp.ones((1, model.layer_sizes[0]))  # Заглушка входа
            return jnp.sum(model.forward(x))
        
        ### YOUR CODE HERE ###
        #hessian_fn =           
        return hessian_fn(params)
    ```
    that computes the full Hessian $H$ of the loss function with respect to the model parameters using JAX’s automatic differentiation.
    
    **Naive HVP via Full Hessian:** [2/22 points] Write a function ```naive_hvp(hessian, vector)``` that, given a precomputed Hessian $H$ and a vector $v$ (of the same shape as the parameters), computes the Hessian-vector product using a straightforward matrix-vector multiplication.

    **Efficient HVP Using Autograd:** [4/22 points] Write a function 
    ```python
     def hvp(f, x, v):
         return jax.grad(lambda x: jnp.vdot(jax.grad(f)(x), v))(x)
    ```
    that directly computes the HVP without explicitly forming the full Hessian. This leverages the reverse-mode differentiation capabilities of JAX.
     
    **Timing Experiment:** Consider a family of models with an increasing number of hidden layers. 

    ```python
    ns = np.linspace(50, 1000, 15, dtype=int)  # The number of hidden layers
    num_runs = 10  # The number of runs for averaging
    ```

    For each model configuration:
    * Generate the model and extract its parameter vector.
    * Generate a random vector $v$ of the same dimension as the parameters.
    * Measure (do not forget to use ```.block_until_ready()``` to ensure accurate timing and proper synchronization) the following:
       1. **Combined Time (Full Hessian + Naive HVP):** The total time required to compute the full Hessian and then perform the matrix-vector multiplication.
       2. **Naive HVP Time (Excluding Hessian Computation):** The time required to perform the matrix-vector multiplication given a precomputed Hessian.
       3. **Efficient HVP Time:** The time required to compute the HVP using the autograd-based function.
    * Repeat each timing measurement for a fixed number of runs (e.g., 10 runs) and record both the mean and standard deviation of the computation times. 
    
    **Visualization and Analysis:** [12/22 points]
    * Plot the timing results for the three methods on the same graph. For each method, display error bars corresponding to the standard deviation over the runs.
    * Label the axes clearly (e.g., “Number of Layers” vs. “Computation Time (seconds)”) and include a legend indicating which curve corresponds to which method.
    * Analyze the scaling behavior. Try analytically derive the scaling of the methods and compare it with the experimental results.

1. [15 points] We can use automatic differentiation not only to calculate necessary gradients but also for tuning hyperparameters of the algorithm like learning rate in gradient descent (with gradient descent 🤯). Suppose, we have the following function $f(x) = \frac{1}{2}\Vert x\Vert^2$, select a random point $x_0 \in \mathbb{B}^{1000} = \{0 \leq x_i \leq 1 \mid \forall i\}$. Consider $10$ steps of the gradient descent starting from the point $x_0$:
    $$
    x_{k+1} = x_k - \alpha_k \nabla f(x_k)
    $$
    Your goal in this problem is to write the function, that takes $10$ scalar values $\alpha_i$ and return the result of the gradient descent on function $L = f(x_{10})$. And optimize this function using gradient descent on $\alpha \in \mathbb{R}^{10}$. Suppose that each of $10$ components of $\alpha$ is uniformly distributed on $[0; 0.1]$.
    $$
    \alpha_{k+1} = \alpha_k - \beta \frac{\partial L}{\partial \alpha}
    $$
    Choose any constant $\beta$ and the number of steps you need. Describe the obtained results. How would you understand, that the obtained schedule ($\alpha \in \mathbb{R}^{10}$) becomes better than it was at the start? How do you check numerically local optimality in this problem? 

### Convexity

1. [10 points] Show that this function is convex.:
    $$
    f(x, y, z) = z \log \left(e^{\frac{x}{z}} + e^{\frac{y}{z}}\right) + (z - 2)^2 + e^{\frac{1}{x + y}}
    $$
    where the function $f : \mathbb{R}^3 \to \mathbb{R}$ has its domain defined as:
    $$
    \text{dom } f = \{ (x, y, z) \in \mathbb{R}^3 : x + y > 0, \, z > 0 \}.
    $$

1. [5 points] The center of mass of a body is an important concept in physics (mechanics). For a system of material points with masses $m_i$ and coordinates $x_i$, the center of mass is given by:
    $$
    x_c = \frac{\sum_{i=1}^k m_i x_i}{\sum_{i=1}^k m_i}
    $$
    The center of mass of a body does not always lie inside the body. For example, the center of mass of a doughnut is located in its hole. Prove that the center of mass of a system of material points lies in the convex hull of the set of these points.
1. [8 points] Show, that $\mathbf{conv}\{xx^\top: x \in \mathbb{R}^n, \Vert x\Vert  = 1\} = \{A \in \mathbb{S}^n_+: \text{tr}(A) = 1\}$.
1. [5 points] Prove that the set of $\{x \in \mathbb{R}^2 \mid e^{x_1}\le x_2\}$ is convex.
1. [8 points] Consider the function $f(x) = x^d$, where $x \in \mathbb{R}_{+}$. Fill the following table with ✅ or ❎. Explain your answers (with proofs).

    | $d$ | Convex | Concave | Strictly Convex | $\mu$-strongly convex |
    |:-:|:-:|:-:|:-:|:-:|
    | $-2, x \in \mathbb{R}_{++}$| | | | |
    | $-1, x \in \mathbb{R}_{++}$| | | | |
    | $0$| | | | |
    | $0.5$ | | | | |
    |$1$ | | | | |
    | $\in (1; 2)$ | | | | |
    | $2$| | | | |
    | $> 2$| | | | 
    
    : {.responsive}

1. [6 points] Prove that the entropy function, defined as
    $$
    f(x) = -\sum_{i=1}^n x_i \log(x_i),
    $$
    with $\text{dom}(f) = \{x \in \R^n_{++} : \sum_{i=1}^n x_i = 1\}$, is strictly concave.  

1. [8 points] Show that the maximum of a convex function $f$ over the polyhedron $P = \text{conv}\{v_1, \ldots, v_k\}$ is achieved at one of its vertices, i.e.,
    $$
    \sup_{x \in P} f(x) = \max_{i=1, \ldots, k} f(v_i).
    $$

    A stronger statement is: the maximum of a convex function over a closed bounded convex set is achieved at an extreme point, i.e., a point in the set that is not a convex combination of any other points in the set. (you do not have to prove it). *Hint:* Assume the statement is false, and use Jensen’s inequality.

1. [6 points] Show, that the two definitions of $\mu$-strongly convex functions are equivalent:
    1. $f(x)$ is $\mu$-strongly convex $\iff$ for any $x_1, x_2 \in S$ and $0 \le \lambda \le 1$ for some $\mu > 0$:
        $$
        f(\lambda x_1 + (1 - \lambda)x_2) \le \lambda f(x_1) + (1 - \lambda)f(x_2) - \frac{\mu}{2} \lambda (1 - \lambda)\|x_1 - x_2\|^2
        $$

    1. $f(x)$ is $\mu$-strongly convex $\iff$ if there exists $\mu>0$ such that the function $f(x) - \dfrac{\mu}{2}\Vert x\Vert^2$ is convex.

### Optimality conditions. KKT. Duality

In this section, you can consider either the arbitrary norm or the Euclidian norm if nothing else is specified.

1. **Toy example** [10 points] 
    $$
    \begin{split}
    & x^2 + 1 \to \min\limits_{x \in \mathbb{R} }\\
    \text{s.t. } & (x-2)(x-4) \leq 0
    \end{split}
    $$

    1. Give the feasible set, the optimal value, and the optimal solution.
    1.  Plot the objective $x^2 +1$ versus $x$. On the same plot, show the feasible set, optimal point, and value, and plot the Lagrangian $L(x,\mu)$ versus $x$ for a few positive values of $\mu$. Verify the lower bound property ($p^* \geq \inf_x L(x, \mu)$for $\mu \geq 0$). Derive and sketch the Lagrange dual function $g$.
    1. State the dual problem, and verify that it is a concave maximization problem. Find the dual optimal value and dual optimal solution $\mu^*$. Does strong duality hold?
    1.  Let $p^*(u)$ denote the optimal value of the problem

    $$
    \begin{split}
    & x^2 + 1 \to \min\limits_{x \in \mathbb{R} }\\
    \text{s.t. } & (x-2)(x-4) \leq u
    \end{split}
    $$

    as a function of the parameter $u$. Plot $p^*(u)$. Verify that $\dfrac{dp^*(0)}{du} = -\mu^*$ 

1. Consider a smooth convex function $f(x)$ at some point $x_k$. One can define the first-order Taylor expansion of the function as:
    $$
    f^I_{x_k}(x) = f(x_k) + \nabla f(x_k)^\top (x - x_k),
    $$
    where we can define $\delta x = x - x_k$ and $g = \nabla f(x_k)$. Thus, the expansion can be rewritten as:
    $$
    f^I_{x_k}(\delta x) = f(x_k) + g^\top \delta x.
    $$
    Suppose, we would like to design the family of optimization methods that will be defined as:
    $$
    x_{k+1} = \text{arg}\min_{\delta x} \left\{f^I_{x_k}(\delta x) + \frac{\lambda}{2} \|\delta x\|^2\right\},
    $$
    where $\lambda > 0$ is a parameter.

    1. [5 points] Show, that this method is equivalent to the gradient descent method with the choice of Euclidean norm of the vector $\|\delta x\| = \|\delta x\|_2$. Find the corresponding learning rate.
    1. [5 points] Prove, that the following holds:
        $$
        \text{arg}\min_{\delta x \in \mathbb{R}^n} \left\{ g^T\delta x + \frac{\lambda}{2} \|\delta x\|^2\right\} = - \frac{\|g\|_*}{\lambda} \text{arg}\max_{\|t\|=1} \left\{ t^T g \right\},
        $$
        where $\|g\|_*$ is the [dual norm](https://fmin.xyz/docs/theory/Dual%20norm.html) of $g$.
    1. [3 points] Consider another vector norm $\|\delta x\| = \|\delta x\|_\infty$. Write down exact expression for the corresponding method.
    1. [2 points] Consider induced operator matrix norm for any matrix $W \in \mathbb{R}^{d_{out} \times d_{in}}$
        $$
        \|W\|_{\alpha \to \beta} = \max_{x \in \mathbb{R}^{d_{in}}} \frac{\|Wx\|_{\beta}}{\|x\|_{\alpha}}.
        $$
        Typically, when we solve optimization problems in deep learning, we stack the weight matrices for all layers $l = [1, L]$ into a single vector.
        $$
        w = \text{vec}(W_1, W_2, \ldots, W_L) \in \mathbb{R}^{n},
        $$
        Can you write down the exact expression, that relates
        $$
        \|w\|_\infty \qquad \text{ and } \qquad \|W_l\|_{\alpha \to \beta}, \; l = [1, L]?
        $$

1. [10 points] Derive the dual problem for the Ridge regression problem with $A \in \mathbb{R}^{m \times n}, b \in \mathbb{R}^m, \lambda > 0$:

    $$
    \begin{split}
    \dfrac{1}{2}\|y-b\|^2 + \dfrac{\lambda}{2}\|x\|^2 &\to \min\limits_{x \in \mathbb{R}^n, y \in \mathbb{R}^m }\\
    \text{s.t. } & y = Ax
    \end{split}
    $$

1. [20 points] Derive the dual problem for the support vector machine problem with $A \in \mathbb{R}^{m \times n}, \mathbf{1} \in \mathbb{R}^m \in \mathbb{R}^m, \lambda > 0$:

    $$
    \begin{split}
    \langle \mathbf{1}, t\rangle + \dfrac{\lambda}{2}\|x\|^2 &\to \min\limits_{x \in \mathbb{R}^n, t \in \mathbb{R}^m }\\
    \text{s.t. } & Ax \succeq \mathbf{1} - t \\
    & t \succeq 0
    \end{split}
    $$

1. [10 points] Give an explicit solution to the following LP.
    
    $$
    \begin{split}
    & c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
    \text{s.t. } & 1^\top x = 1, \\
    & x \succeq 0 
    \end{split}
    $$

    This problem can be considered the simplest portfolio optimization problem.

1. [20 points] Show, that the following problem has a unique solution and find it:

    $$
    \begin{split}
    & \langle C^{-1}, X\rangle - \log \det X \to \min\limits_{x \in \mathbb{R}^{n \times n} }\\
    \text{s.t. } & \langle Xa, a\rangle \leq 1,
    \end{split}
    $$

    where $C \in \mathbb{S}^n_{++}, a \in \mathbb{R}^n \neq 0$. The answer should not involve inversion of the matrix $C$.

1. [20 points] Give an explicit solution to the following QP.
    
    $$
    \begin{split}
    & c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
    \text{s.t. } & (x - x_c)^\top A (x - x_c) \leq 1,
    \end{split}
    $$

    where $A \in \mathbb{S}^n_{++}, c \neq 0, x_c \in \mathbb{R}^n$.

1. [10 points] Consider the equality-constrained least-squares problem
    
    $$
    \begin{split}
    & \|Ax - b\|_2^2 \to \min\limits_{x \in \mathbb{R}^n }\\
    \text{s.t. } & Cx = d,
    \end{split}
    $$

    where $A \in \mathbb{R}^{m \times n}$ with $\mathbf{rank }A = n$, and $C \in \mathbb{R}^{k \times n}$ with $\mathbf{rank }C = k$. Give the KKT conditions, and derive expressions for the primal solution $x^*$ and the dual solution $\lambda^*$.



1. **Supporting hyperplane interpretation of KKT conditions**. [10 points]  Consider a **convex** problem with no equality constraints
    
    $$
    \begin{split}
    & f_0(x) \to \min\limits_{x \in \mathbb{R}^n }\\
    \text{s.t. } & f_i(x) \leq 0, \quad i = [1,m]
    \end{split}
    $$

    Assume, that $\exists x^* \in \mathbb{R}^n, \mu^* \in \mathbb{R}^m$ satisfy the KKT conditions
    
    $$
    \begin{split}
    & \nabla_x L (x^*, \mu^*) = \nabla f_0(x^*) + \sum\limits_{i=1}^m\mu_i^*\nabla f_i(x^*) = 0 \\
    & \mu^*_i \geq 0, \quad i = [1,m] \\
    & \mu^*_i f_i(x^*) = 0, \quad i = [1,m]\\
    & f_i(x^*) \leq 0, \quad i = [1,m]
    \end{split}
    $$

    Show that

    $$
    \nabla f_0(x^*)^\top (x - x^*) \geq 0
    $$

    for all feasible $x$. In other words, the KKT conditions imply the simple optimality criterion or $\nabla f_0(x^*)$ defines a supporting hyperplane to the feasible set at $x^*$.
    
1. **A penalty method for equality constraints.** [10 points] We consider the problem of minimization

    $$
    \begin{split}
    & f_0(x) \to \min\limits_{x \in \mathbb{R}^{n} }\\
    \text{s.t. } & Ax = b,
    \end{split}
    $$
    
    where $f_0(x): \mathbb{R}^n \to\mathbb{R} $ is convex and differentiable, and $A \in \mathbb{R}^{m \times n}$ with $\mathbf{rank }A = m$. In a quadratic penalty method, we form an auxiliary function

    $$
    \phi(x) = f_0(x) + \alpha \|Ax - b\|_2^2,
    $$
    
    where $\alpha > 0$ is a parameter. This auxiliary function consists of the objective plus the penalty term $\alpha \Vert Ax - b\Vert_2^2$. The idea is that a minimizer of the auxiliary function, $\tilde{x}$, should be an approximate solution to the original problem. Intuition suggests that the larger the penalty weight $\alpha$, the better the approximation $\tilde{x}$ to a solution of the original problem. Suppose $\tilde{x}$ is a minimizer of $\phi(x)$. Show how to find, from $\tilde{x}$, a dual feasible point for the original problem. Find the corresponding lower bound on the optimal value of the original problem.

### Linear programming

1. **📱🎧💻 Covers manufacturing.** [20 points] Lyzard Corp is producing covers for the following products: 
    * 📱 phones
    * 🎧 headphones
    * 💻 laptops

    The company’s production facilities are such that if we devote the entire production to headphone covers, we can produce 5000 of them in one day. If we devote the entire production to phone covers or laptop covers, we can produce 4000 or 2000 of them in one day. 

    The production schedule is one week (6 working days), and the week’s production must be stored before distribution. Storing 1000 headphone covers (packaging included) takes up 30 cubic feet of space. Storing 1000 phone covers (packaging included) takes up 50 cubic feet of space, and storing 1000 laptop covers (packaging included) takes up 200 cubic feet of space. The total storage space available is 1500 cubic feet. 
    
    Due to commercial agreements with Lyzard Corp has to deliver at least 6000 headphone covers and 4000 laptop covers per week to strengthen the product’s diffusion. 

    The marketing department estimates that the weekly demand for headphones covers, phone, and laptop covers does not exceed 15000, 12000 and 8000 units, therefore the company does not want to produce more than these amounts for headphones, phone, and laptop covers. 

    Finally, the net profit per headphone cover, phone cover, and laptop cover are \$5, \$7, and \$12, respectively.

    The aim is to determine a weekly production schedule that maximizes the total net profit.

    1. Write a Linear Programming formulation for the problem.  Use the following variables:

        * $y_1$ = number of headphones covers produced over the week,  
        * $y_2$ = number of phone covers produced over the week,  
        * $y_3$ = number of laptop covers produced over the week. 

    1. Find the solution to the problem using [PyOMO](http://www.pyomo.org)
    
        ```python
        !pip install pyomo
        ! sudo apt-get install glpk-utils --quiet  # GLPK
        ! sudo apt-get install coinor-cbc --quiet  # CoinOR
        ```

    1. Perform the sensitivity analysis. Which constraint could be relaxed to increase the profit the most? Prove it numerically.

1. Prove the optimality of the solution [10 points] 
    
    $$
    x = \left(\frac{7}{3} , 0, \frac{1}{3}\right)^T
    $$
    
    to the following linear programming problem:
    
    $$
    \begin{split}
    & 9x_1 + 3x_2 + 7x_3 \to \max\limits_{x \in \mathbb{R}^3 }\\
    \text{s.t. } & 2x_1 + x_2 + 3x_3 \leq 6 \\
    & 5x_1 + 4x_2 + x_3 \leq 12 \\
    & 3x_3 \leq 1,\\
    & x_1, x_2, x_3 \geq 0
    \end{split}
    $$

    but you cannot use any numerical algorithm here.

1. [10 points] Economic interpretation of the dual problem: Suppose a small shop makes wooden toys, where each toy train requires one piece of wood and $2$ tins of paint, while each toy boat requires one piece of wood and $1$ tin of paint. The profit on each toy train is $\$30$, and the profit on each toy boat is $\$20$. Given an inventory of $80$ pieces of wood and $100$ tins of paint, how many of each toy
should be made to maximize the profit?
    1. Write out the optimization problem in standard form, writing all constraints as inequalities.
    1. Sketch the feasible set and determine $p^*$ and $x^*$
    1. Find the dual problem, then determine $d^*$ and $\lambda^*$. Note that we can interpret the Lagrange multipliers $\lambda_k$ associated with the constraints on wood and paint as the prices for each piece of wood and tin of paint, so that $−d^*$ is how much money would be obtained from selling the inventory for those prices. Strong duality says a buyer should not pay more for the inventory than what the toy store would make by producing and selling toys from it, and that the toy store should not sell the inventory for less than that.
    1. The other interpretation of the Lagrange multipliers is as sensitivities to changes in the constraints. Suppose the toymaker found some more pieces of wood; the $\lambda_k$ associated with the wood constraint will equal the partial derivative of $−p^*$ with respect to how much more wood became available. Suppose the inventory increases by one piece of wood. Use $\lambda^*$ to estimate how much the profit would increase, without solving the updated optimization problem. How is this consistent with the price interpretation given above for the Lagrange multipliers? [source](https://tleise.people.amherst.edu/Math294Spring2017/TeXfiles/LagrangeDualityHW.pdf) 

### Gradient Descent

### Accelerated methods

### Subgradient method

### Conjugate gradients

### Conditional gradient methods

### Proximal gradient method

### Newton and quasi-Newton methods

### Stochastic gradient methods

### Neural network training

### Big models
