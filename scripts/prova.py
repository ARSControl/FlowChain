import numpy as np
import casadi as ca
from scipy.spatial import Voronoi
from shapely import Polygon, Point
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


def gmm_cost_expr(x0, q_points, q_weights):
  p = ca.repmat(x0.T, q_points.size1(), 1)
  sq_dists = ca.sum2((p - q_points)**2)
  
  return ca.sum1(sq_dists * q_weights)

def gauss_pdf(x, y, mean, covariance):
  points = np.column_stack([x.flatten(), y.flatten()])
  # Calculate the multivariate Gaussian probability
  exponent = -0.5 * np.sum((points - mean) @ np.linalg.inv(covariance) * (points - mean), axis=1)
  coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance))
  prob = coefficient * np.exp(exponent)
  return prob

def pdf_func(x, mean, covariance):
  coeff = 1 / ca.sqrt((2 * ca.pi) ** 2 * ca.det(covariance))
  mahalanobis_dist = []
  for i in range(x.shape[0]):
    diff = x[i, :] - mean
    mahalanobis_dist.append(diff @ ca.inv(covariance) @ diff.T)
  mahalanobis_dist = ca.vertcat(*mahalanobis_dist)
  exponent = -0.5 * mahalanobis_dist
  return coeff * ca.exp(exponent)
  

def voronoi_cost(pi, q, mean, covariance):
  """
  pi: 2D position of the robot
  q: (N, 2) discretized points inside i-th Voronoi cell
  """

  dists = np.linalg.norm(pi - q, axis=1)
  pdfs = gauss_pdf(q[:, 0], q[:, 1], mean, covariance)
  cost = np.sum(np.square(dists) * pdfs)

  return -cost

def voronoi_cost_func(pi, q, mean, covariance):
  p = ca.reshape(pi, 1, q.shape[1])
  p = ca.repmat(p, q.shape[0], 1)
  sq_dists = ca.sum2((p - q)**2)
  pdfs = pdf_func(q, mean, covariance)
  cost = ca.sum1(sq_dists * pdfs)
  return cost


def mirror(points):
    mirrored_points = []

    # Define the corners of the square
    square_corners = [(-0.5*AREA_W, -0.5*AREA_W), (0.5*AREA_W, -0.5*AREA_W), (0.5*AREA_W, 0.5*AREA_W), (-0.5*AREA_W, 0.5*AREA_W)]

    # Mirror points across each edge of the square
    for edge_start, edge_end in zip(square_corners, square_corners[1:] + [square_corners[0]]):
        edge_vector = (edge_end[0] - edge_start[0], edge_end[1] - edge_start[1])

        for point in points:
            # Calculate the vector from the edge start to the point
            point_vector = (point[0] - edge_start[0], point[1] - edge_start[1])

            # Calculate the mirrored point by reflecting across the edge
            mirrored_vector = (point_vector[0] - 2 * (point_vector[0] * edge_vector[0] + point_vector[1] * edge_vector[1]) / (edge_vector[0]**2 + edge_vector[1]**2) * edge_vector[0],
                               point_vector[1] - 2 * (point_vector[0] * edge_vector[0] + point_vector[1] * edge_vector[1]) / (edge_vector[0]**2 + edge_vector[1]**2) * edge_vector[1])

            # Translate the mirrored vector back to the absolute coordinates
            mirrored_point = (edge_start[0] + mirrored_vector[0], edge_start[1] + mirrored_vector[1])

            # Add the mirrored point to the result list
            mirrored_points.append(mirrored_point)

    return mirrored_points

# Parameters
PARTICLES_NUM = 100
STD_DEV = 0.3
T = 20
nx, nu = 2, 2 
dt = 0.1
sim_time = 15.0
NUM_STEPS = int(sim_time / dt)
R = 5.0
r = R / 2
AREA_W = 15.0
ROBOTS_NUM = 6
OBS_NUM = 5       # obstacles
Ds = 0.5          # safety distance to obstacles
points = -0.5*AREA_W + AREA_W * np.random.rand(ROBOTS_NUM, 2) 
#np.random.seed(42)
mean1 = -0.5*AREA_W + AREA_W * np.random.rand(2)
cov1 = 2.0 * np.eye(2) # np.diag([1.0, 1.0])
mean2 = -0.5*AREA_W + AREA_W * np.random.rand(2)
cov2 = 2.0 * np.eye(2) # np.diag([1.0, 1.0])
x_obs = -0.5*AREA_W + AREA_W * np.random.rand(OBS_NUM, 2)

# Gaussiane
xg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, 100)
yg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, 100)
X, Y = np.meshgrid(xg, yg)
Z1 = gauss_pdf(X, Y, mean1, cov1)
Z2 = gauss_pdf(X, Y, mean2, cov2)



plt.ion() # Turn on interactive mode
fig, ax = plt.subplots(figsize=(8, 8))

robots_hist = np.zeros((NUM_STEPS+1, ROBOTS_NUM, 2))
robots_hist[0, :, :] = points

def dynamics(state, ctrl):
  return state + ctrl * dt

# Human trajectory
start = np.array([-0.5*AREA_W + 2.0, -0.5*AREA_W + 2.0])
n_points = 100
human_trajectory = np.linspace(start, mean1, n_points)
"""
d1 = start - mean1
d2 = start - mean2
# uomo va verso gaussiana più vicina, robot va nell'altra
if np.linalg.norm(d1) < np.linalg.norm(d2):
  human_trajectory = np.linspace(start, mean1, n_points)
  mean_r = mean2 
  cov_r = cov2
  Z_r = Z2
else:
  human_trajectory= np.linspace(start, mean2, n_points)
  mean_r = mean1
  cov_r = cov1
  Z_r = Z1
"""



# Voronoi partitioning
for s in range(NUM_STEPS):
  planned_trajectories = np.zeros((T+1, ROBOTS_NUM, nx))
  polygons = []
  positions_now = points.copy()


  human_pos = human_trajectory[min(s, len(human_trajectory)-1)]
  dist_to_Z1 = np.linalg.norm(human_pos - mean1)
  dist_to_Z2 = np.linalg.norm(human_pos - mean2)

  # PDF dell'umano
  samples = human_pos + STD_DEV * np.random.randn(PARTICLES_NUM, 2)
  mu = np.mean(samples, axis=0)
  cov = np.cov(samples, rowvar=False)
  Z_h = gauss_pdf(X, Y, mu, cov)


  # # Penalizzazione basata sulla distanza
  # base_weight = 0.45  # Peso iniziale
  # alpha = 0.6  # Intensità della penalizzazione
  # threshold = 5.0  
  # threshold2 = 10.0  

  # # Penalizzazione graduale
  # if dist_to_Z1 > threshold:
  #     w1 = base_weight
  # if threshold <= dist_to_Z1 <= threshold2:
  #     w1 = base_weight * (1.0 - alpha * (1.0 - dist_to_Z1 / threshold2))
  # else:
  #     w1 = base_weight * (1.0 - alpha * (1.0 - dist_to_Z1 / threshold))

  # w1 = np.clip(w1, 0.0, 1.0)  # Limita w1 tra 0 e 1
  # w2 = 1.0 - w1  

  # Penalizzazione locale per Z1 e Z2 dove Z_h si sovrappone
  penal1 = Z_h * gauss_pdf(X, Y, mean1, cov1) / np.max(Z1) 
  penal2 = Z_h * gauss_pdf(X, Y, mean2, cov2) / np.max(Z2) 


  w1 = 0.4
  w2 = 0.6


  # if dist_to_Z1 < 5.0:
  #   Z1_eff = w1 * np.clip(Z1 - penal1, 0.0, None)
  #   Z2_eff = Z2
  # elif dist_to_Z2 < 5.0:
  #   Z1_eff = Z1
  #   Z2_eff = w2 * np.clip(Z2 - penal2, 0.0, None)
  # else:
  #   Z1_eff = Z1
  #   Z2_eff = Z2

  # scale_factor1 = 1.0 - penal1 
  # scale_factor2 = 1.0 - penal2

  # Z1_eff = w1 * np.clip(Z1 - penal1, 0.0, None)
  # Z2_eff = w2 * np.clip(Z2 - penal2, 0.0, None)
  # Z1_eff = w1 * Z1 * np.clip(scale_factor1, 0.0, 1.0)
  # Z2_eff = w2 * Z2 * np.clip(scale_factor2, 0.0, 1.0)

  alpha = 0.6  # intensità della penalizzazione
  sigma = 2.0  # raggio di influenza

  scale1 = 1.0 - alpha * np.exp(-dist_to_Z1**2 / (2 * sigma**2))
  scale2 = 1.0 - alpha * np.exp(-dist_to_Z2**2 / (2 * sigma**2))

  Z1_eff = w1 * Z1 * scale1
  Z2_eff = w2 * Z2 * scale2


  # penalizzazione globale in base alla distanza
  scale1 = 1.0 - alpha * np.exp(-dist_to_Z1**2 / (2 * sigma**2))
  scale2 = 1.0 - alpha * np.exp(-dist_to_Z2**2 / (2 * sigma**2))

  Z1_eff = w1 * Z1 * scale1
  Z2_eff = w2 * Z2 * scale2


  # Controllo per abbassare Z1 globalmente
  if np.linalg.norm(human_pos - mean1) < 0.5:  # L'umano è vicino al centro di Z1
      global_penalty = 0.5  # Fattore di penalizzazione globale
      Z1_eff *= global_penalty


  # GMM risultante
  Z_gmm = Z1_eff + Z2_eff

  Z_gmm = np.clip(Z_gmm, 1e-6, None)  # Evita valori nulli
  Z_gmm /= (np.sum(Z_gmm) * (xg[1] - xg[0]) * (yg[1] - yg[0]))


  for idx in range(ROBOTS_NUM):
    dummy_points = np.zeros((5*ROBOTS_NUM, 2))
    dummy_points[:ROBOTS_NUM, :] = positions_now
    mirrored_points = mirror(positions_now)
    mir_pts = np.array(mirrored_points)
    dummy_points[ROBOTS_NUM:, :] = mir_pts

    #dummy_points = np.vstack([positions_now] + [mirror(positions_now)])
    vor = Voronoi(dummy_points)
    region = vor.point_region[idx]
    #poly = Polygon([vor.vertices[i] for i in vor.regions[region]])
    poly_vert = []
    for vert in vor.regions[region]:
        v = vor.vertices[vert]
        poly_vert.append(v)

    poly = Polygon(poly_vert)
    
    # Limited range cell
    range_vert = []
    for th in np.arange(0, 2*np.pi, np.pi/10):
      vx = positions_now[idx, 0] + r * np.cos(th)
      vy = positions_now[idx, 1] + r * np.sin(th)
      range_vert.append((vx, vy))
    range_poly = Polygon(range_vert)
    lim_region = poly.intersection(range_poly)
    polygons.append(lim_region)
    robot = vor.points[idx]

    xmin, ymin, xmax, ymax = poly.bounds
    discr_points = 20
    qs = []
    for i in np.linspace(xmin, xmax, discr_points):
        for j in np.linspace(ymin, ymax, discr_points):
            pt_i = Point(i, j)
            if lim_region.contains(pt_i):
                qs.append(np.array([i, j]))

    qs = np.array(qs)
# ---------------- End Voronoi partitioning ----------------

    # 1. Variables
    U = ca.SX.sym('U', nu, T)
    x0 = ca.SX.sym('x0', nx) # posizione iniziale robot 
    p0 = ca.SX.sym('p0', nx) # posizione iniziale uomo

    # 2. Dynamics: single integrator
    vmax = 1.5

    # 3. Cost function
    #cost_expr = voronoi_cost_func(x0, qs, mean_gmm, cov_gmm) # MEAN, COV
    #cost_fn = ca.Function('cost', [x0], [cost_expr])

    #qs_vals = w1 * gauss_pdf(qs[:,0], qs[:,1], mean1, cov1) + w2 * gauss_pdf(qs[:,0], qs[:,1], mean2, cov2) - 0.5 * gauss_pdf(qs[:,0], qs[:,1], mu, cov)
    qs_vals = w1 * gauss_pdf(qs[:, 0], qs[:, 1], mean1, cov1) * scale1 + w2 * gauss_pdf(qs[:, 0], qs[:, 1], mean2, cov2) * scale2
    qs_vals = np.clip(qs_vals, 1e-6, None)

    qs_vals_casadi = ca.SX(qs_vals.tolist())
    
    qs_points_casadi = ca.SX(qs.tolist())

    cost_expr = gmm_cost_expr(x0, qs_points_casadi, qs_vals_casadi)
    cost_fn = ca.Function('cost', [x0], [cost_expr])

    # 4. Build optimization problem
    obj = 0.0
    x_curr = x0
    g_list = []
    for k in range(T):
      obj += cost_fn(x_curr)

      # obstacle avoidance constraint: Ds**2 - ||x - x_obs||**2 < 0
      for obs in x_obs:
        g_list.append(Ds**2 - ca.sumsqr(x_curr - obs))
      x_curr = dynamics(x_curr, U[:, k])

    # 5. Constraints
    g = ca.vertcat(*g_list) # constraints

    # 6. Problem
    opt_vars = ca.vec(U)
    nlp = {'x': opt_vars, 'f': obj, 'g': g, 'p': x0}
    
    solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0})

    x_init = robot.copy()

    # 7. bounds and initial guess
    u_min = np.array([-vmax, -vmax])
    u_max = np.array([ vmax, vmax])
    lbx = np.tile(u_min, T)
    ubx = np.tile(u_max, T)
    
    # bounds on g: g <= 0
    lbg = -np.inf * np.ones(g.shape)
    ubg = np.zeros(g.shape)               # Ds**2 - ||x - x_obs||**2 < 0
    u0_guess = np.zeros(opt_vars.shape)

    # solve
    sol = solver(x0=u0_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=x_init)
    u_opt = sol['x'].full().reshape(T, nu)
    planned_traj = np.zeros((T+1, nx))
    planned_traj[0, :] = x_init
    for k in range(T):
      planned_traj[k+1, :] = dynamics(planned_traj[k, :], u_opt[k, :])
    planned_trajectories[:, idx, :] = planned_traj
    points[idx, :] = dynamics(positions_now[idx, :], u_opt[0, :])
    robots_hist[s+1, idx, :] = points[idx, :]

  ax.cla()
  #ax.contourf(X, Y, Z_r.reshape(X.shape), levels=30, cmap='YlOrRd', alpha=0.75) # levels=10
  ax.plot(human_trajectory[:, 0], human_trajectory[:, 1], 'b-') # traiettoria umano
  #ax.scatter(mean_r[0], mean_r[1], c='blue', label='Centro Gaussiana Robot', marker='*')
  ax.scatter(mean2[0], mean2[1], c='green', label='Centro Gaussiana 2', marker='*')
  ax.scatter(human_trajectory[-1, 0], human_trajectory[-1, 1], c='red', label='Posizione finale Umano')
  for idx in range(ROBOTS_NUM):
    ax.plot(robots_hist[:s+2, idx, 0], robots_hist[:s+2, idx, 1], color='tab:blue')
    ax.plot(planned_trajectories[:, idx, 0], planned_trajectories[:, idx, 1], color='tab:green')
    x, y = polygons[idx].exterior.xy
    ax.plot(x, y, c='tab:red')
  for obs in x_obs:
    ax.scatter(*obs, marker='x', color='k')
    circle = obs + Ds * np.column_stack((np.cos(np.linspace(0, 2*np.pi, 20)),
                                         np.sin(np.linspace(0, 2*np.pi, 20))))
    ax.plot(circle[:, 0], circle[:, 1], c='k')


  # PARTICLES_NUM = 100
  # STD_DEV = 0.3

  # # Prendi la posizione attuale dell'umano
  # human_pos = human_trajectory[min(s, len(human_trajectory)-1)]

  # samples = human_pos + STD_DEV * np.random.randn(PARTICLES_NUM, 2)

  # # gaussiana per human 
  # mu = np.mean(samples, axis=0)
  # cov = np.cov(samples, rowvar=False)
  # Z_h = gauss_pdf(X, Y, mu, cov)

  #ax.contour(X, Y, Z_h.reshape(X.shape), levels=10, colors='black', alpha=0.7)
  ax.scatter(samples[:,0], samples[:,1], c='red', alpha=0.7, marker='.')


  ax.contour(X, Y, Z1.reshape(X.shape), levels=5, colors='purple', linestyles='dotted')
  ax.contour(X, Y, Z2.reshape(X.shape), levels=5, colors='green', linestyles='dotted')
  ax.contourf(X, Y, Z_gmm.reshape(X.shape), levels=10, cmap='YlOrRd', alpha=0.7)
  #ax.contour(X, Y, Z_diff.reshape(X.shape), levels=10, colors='blue')


  ax.set_title(f"Timestep: {s}", fontsize=14)
  ax.legend(loc='upper right', fontsize=8)
  ax.grid(True, alpha=0.3)
  ax.set_xlim([-0.5*AREA_W, 0.5*AREA_W])
  ax.set_ylim([-0.5*AREA_W, 0.5*AREA_W])
  ax.set_aspect('equal')
  plt.pause(0.01)

plt.ioff()
plt.show()
