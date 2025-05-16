import numpy as np
import casadi as ca
from scipy.spatial import Voronoi
from shapely import Polygon, Point
import matplotlib.pyplot as plt

def gauss_pdf(x, y, mean, covariance):

  points = np.column_stack([x.flatten(), y.flatten()])
  # Calculate the multivariate Gaussian probability
  exponent = -0.5 * np.sum((points - mean) @ np.linalg.inv(covariance) * (points - mean), axis=1)
  coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance))
  prob = coefficient * np.exp(exponent)
  return prob


def pdf_func(x, mean, covariance):
  coeff = 1 / ca.sqrt((2 * ca.pi) ** 2 * ca.det(covariance))
  # exponent = -0.5 * ((x[0] - mean[0])**2 + (x[1] - mean[1])**2)
  # m = ca.reshape(mean, 2, 1)
  # expanded_mean = ca.repmat(m.T, x.shape[0], 1)
  # exponent = -0.5 * ((x - expanded_mean) @ ca.inv(covariance) @ ((x - expanded_mean)).T)
  mahalanobis_dist = []
  for i in range(x.shape[0]):
    diff = x[i, :] - mean
    mahalanobis_dist.append(diff @ ca.inv(covariance) @ diff.T)
  mahalanobis_dist = ca.vertcat(*mahalanobis_dist)
  exponent = -0.5 * mahalanobis_dist
  # print("Mahal shape: ", mahalanobis_dist.shape)
  # print("exponent shape: ", exponent.shape)
  # print("coeff shape: ", coeff.shape)
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


# 0. parameters
T = 10
dt = 0.1
sim_time = 20.0
NUM_STEPS = int(sim_time / dt)
# x1 = np.zeros(2)
# x2 = np.array([2.5, 3.0])
# mean = np.array([5.0, -2.5])
R = 5.0
r = R/2
AREA_W = 10.0
ROBOTS_NUM = 6
# points = np.concatenate((x1.reshape(1, -1), x2.reshape(1, -1)),axis=0)
points = -0.5*AREA_W + AREA_W * np.random.rand(ROBOTS_NUM, 2)
mean = -0.5*AREA_W + AREA_W * np.random.rand(2)
cov = 2.0 * np.diag([1.0, 1.0])

nx = 2
nu = 2

# plotting stuff
xg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, 100)
yg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, 100)
X, Y = np.meshgrid(xg, yg)
Z = gauss_pdf(X, Y, mean, cov)
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(8, 8))

def dynamics(state, ctrl):
  return state + ctrl * dt


robots_hist = np.zeros((NUM_STEPS+1, ROBOTS_NUM, 2))
robots_hist[0, :, :] = points
for s in range(NUM_STEPS):
  planned_trajectories = np.zeros((T+1, ROBOTS_NUM, nx))
  polygons = []
  positions_now = points.copy()
  for idx in range(ROBOTS_NUM):
    # ------- Voronoi partitioning ---------
    dummy_points = np.zeros((5*ROBOTS_NUM, 2))
    dummy_points[:ROBOTS_NUM, :] = positions_now
    mirrored_points = mirror(positions_now)
    mir_pts = np.array(mirrored_points)
    dummy_points[ROBOTS_NUM:, :] = mir_pts

    # Voronoi partitioning
    vor = Voronoi(dummy_points)

    region = vor.point_region[idx]
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
    x0 = ca.SX.sym('x0', nx)

    # 2. Dynamics: single integrator
    vmax = 1.5

    # 3. cost
    cost_expr = voronoi_cost_func(x0, qs, mean, cov)
    cost_fn = ca.Function('cost', [x0], [cost_expr])
    # print("Cost function: ", cost_fn)

    # 4. Build optimizaiton problem
    obj = 0.0
    x_curr = x0
    for k in range(T):
      # print("Planning step: ", k)
      obj += cost_fn(x_curr)
      x_curr = dynamics(x_curr, U[:, k])

    # print("cost: ", obj)

    # 5. Constraints
    # g = ca.vertcat(*g)

    # 7. problem
    opt_vars = ca.vec(U)
    nlp = {
      'x': opt_vars,
      'f': obj,
      'g': ca.SX.zeros(1),
      'p': x0
    }

    opts = {'ipopt': {'print_level': 0, 'max_iter': 1000, 'tol': 1e-5}}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    x_init = robot.copy()

    # 8. bounds and initial guess
    u_min = np.array([-vmax, -vmax])
    u_max = np.array([ vmax, vmax])
    lbx = np.tile(u_min, T)
    ubx = np.tile(u_max, T)
    u0_guess = np.zeros(opt_vars.shape)

    # solve
    sol = solver(x0=u0_guess, lbx=lbx, ubx=ubx, p=x_init)
    u_opt = sol['x'].full().reshape(T, nu)
    # print("uopt: ", u_opt)
    planned_traj = np.zeros((T+1, nx))
    planned_traj[0, :] = x_init
    for k in range(T):
      planned_traj[k+1, :] = dynamics(planned_traj[k, :], u_opt[k, :])
    planned_trajectories[:, idx, :] = planned_traj
    points[idx, :] = dynamics(positions_now[idx, :], u_opt[0, :])
    robots_hist[s+1, idx, :] = points[idx, :]
  
  ax.cla()
  ax.contourf(X, Y, Z.reshape(X.shape), levels=10, cmap='YlOrRd', alpha=0.75)
  for idx in range(ROBOTS_NUM):
    ax.plot(robots_hist[:s+2, idx, 0], robots_hist[:s+2, idx, 1], label='Robot Trajectory', color='tab:blue')
    ax.scatter(robots_hist[s+1, idx, 0], robots_hist[s+1, idx, 1], color='tab:blue')
    ax.plot(planned_trajectories[:, idx, 0], planned_trajectories[:, idx, 1], label='Planned Trajectory', color='tab:green')
    x, y = polygons[idx].exterior.xy
    ax.plot(x, y, c='tab:red')
  # ax.legend()
  plt.pause(0.01)

plt.ioff()
plt.show()


"""
nx = 2
nu = 2

# plotting stuff
xg = np.linspace(0, 10, 100)
yg = np.linspace(0, 10, 100)
X, Y = np.meshgrid(xg, yg)
Z = gauss_pdf(X, Y, mean, covariance)
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(8, 8))

def dynamics(state, ctrl):
  return state + ctrl * dt

robot = np.zeros(2)
robot_hist = np.zeros((NUM_STEPS+1, 2))
robot_hist[0, :] = robot
for s in range(NUM_STEPS):
  # 1. Variables
  U = ca.SX.sym('U', nu, T)
  x0 = ca.SX.sym('x0', nx)

  # 2. Dynamics: single integrator
  # f = ca.Function('f', [x, u], [x + u * dt])
  vmax = 1.5
  amax = 1.0

  # 3. cost
  cost_expr = -pdf_func(x0, mean, covariance)
  cost_fn = ca.Function('cost', [x0], [cost_expr])

  # 4. Build optimizaiton problem
  g = []
  obj = 0.0
  x_curr = x0

  for k in range(T):
    obj += cost_fn(x_curr)
    x_curr = dynamics(x_curr, U[:, k])

  # 5. Constraints
  # g = ca.vertcat(*g)

  # 7. problem
  opt_vars = ca.vec(U)
  nlp = {
    'x': opt_vars,
    'f': obj,
    'g': ca.SX.zeros(1),
    'p': x0
  }
  opts = {'ipopt': {'print_level': 0, 'max_iter': 1000, 'tol': 1e-9}}
  solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

  x_init = robot.copy()

  # 8. bounds and initial guess
  # lbx = -ca.inf * np.ones(opt_vars.shape)
  # ubx = ca.inf * np.ones(opt_vars.shape)
  # lbx[nx*(T+1):] = -vmax
  # ubx[nx*(T+1):] = vmax
  u_min = np.array([-vmax, -vmax])
  u_max = np.array([ vmax, vmax])
  lbx = np.tile(u_min, T)
  ubx = np.tile(u_max, T)
  # lbg = np.zeros(g.shape)
  # ubg = np.zeros(g.shape)
  u0_guess = np.zeros(opt_vars.shape)

  # solve
  sol = solver(x0=u0_guess, lbx=lbx, ubx=ubx, p=x_init)
  u_opt = sol['x'].full().reshape(T, nu)
  planned_traj = np.zeros((T+1, nx))
  planned_traj[0, :] = x_init
  for k in range(T):
      planned_traj[k+1, :] = dynamics(planned_traj[k, :], u_opt[k, :])
  
  # update robot state (apply 1st input)
  robot = dynamics(robot, u_opt[0, :])
  robot_hist[s+1, :] = robot

  ax.cla()
  ax.set_xlim(-1, 10)
  ax.set_ylim(-1, 10)
  ax.contourf(X, Y, Z.reshape(X.shape), levels=10, cmap='viridis', alpha=0.5)
  ax.plot(robot_hist[:s+2, 0], robot_hist[:s+2, 1], label='Robot Trajectory', color='blue')
  ax.scatter(robot_hist[s+1, 0], robot_hist[s+1, 1], color='blue')
  ax.plot(planned_traj[:, 0], planned_traj[:, 1], label='Planned Trajectory', color='red')
  ax.legend()
  plt.pause(0.01)

plt.ioff()
plt.show()

"""