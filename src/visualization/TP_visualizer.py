from yacs.config import CfgNode
from typing import Dict, List, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from joblib import delayed, Parallel
import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from torch.distributions.normal import Normal

from visualization.density_plot import plot_density
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.interpolate import griddata, RBFInterpolator, interp2d, RegularGridInterpolator
from scipy.interpolate import bisplrep, bisplev
from scipy.stats import multivariate_normal
import matplotlib

from sklearn.mixture import GaussianMixture

def gauss_pdf(x, y, mean, covariance):

  points = np.column_stack([x.flatten(), y.flatten()])
  # Calculate the multivariate Gaussian probability
  exponent = -0.5 * np.sum((points - mean) @ np.linalg.inv(covariance) * (points - mean), axis=1)
  coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance))
  prob = coefficient * np.exp(exponent)

  return prob

def gmm_pdf(x, y, means, covariances, weights):
  prob = 0.0
  s = len(means)
  for i in range(s):
    prob += weights[i] * gauss_pdf(x, y, means[i], covariances[i])

  return prob


class Visualizer(ABC):
    def __init__(self, cfg: CfgNode):
        pass

    @abstractmethod
    def __call__(self, dict_list: List[Dict]) -> None:
        pass

    def to_numpy(self, tensor) -> np.ndarray:
        return tensor.cpu().numpy()


class TP_Visualizer(Visualizer):
    def __init__(self, cfg: CfgNode, env=None): # def __init__(self, cfg: CfgNode):  per altri dataset
        self.model_name = cfg.MODEL.TYPE

        self.output_dir = Path(cfg.OUTPUT_DIR) / "visualize"
        self.output_dir.mkdir(exist_ok=True)
        self.dataset = cfg.DATA.DATASET_NAME

        self.env = env # aggiunto

        
        if self.env is None:
            import dill
            env_path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / \
                "processed_data" / f"{cfg.DATA.DATASET_NAME}_test.pkl"
            with open(env_path, 'rb') as f:
                self.env = dill.load(f, encoding='latin1')


        # TODO: params for other nodes
        from data.TP.trajectron_dataset import hypers
        node = 'PEDESTRIAN'
        self.state = hypers[cfg.DATA.TP.PRED_STATE][node]
        mean, std = self.env.get_standardize_params(self.state, node)
        self.mean = mean
        self.std = std
        self.std = self.env.attention_radius[('PEDESTRIAN', 'PEDESTRIAN')]

        self.num_grid = 100
        if hasattr(self.env, "gt_dist"):
            self.gt_dist = self.env.gt_dist
        else:
            self.gt_dist = None

        self.observe_length = cfg.DATA.OBSERVE_LENGTH


    def __call__(self, dict_list: List[Dict]) -> None: # dict_list è una lista di dizionari
        index = dict_list[0]['index']
        min_pos, max_pos = self.get_minmax(index)

        # (batch, timesteps, [x,y])
        obs = self.to_numpy(dict_list[0]['obs'][:, :, 0:2]) # prende x,y dei primi 8 pti
        gt = self.to_numpy(dict_list[0]['gt'])

        pred = []
        for d in dict_list:
            pred.append(self.to_numpy(d[("pred", 0)][:, :, None]))
            #assert verifica che i dati siano consistenti tra i dizionari
            assert np.all(obs == self.to_numpy(d["obs"][:, :, 0:2]))
            assert np.all(gt == self.to_numpy(d["gt"]))

        # (batch, timesteps, num_trials, [x,y])
        pred = np.concatenate(pred, axis=2) # combina predizioni raccolte in unico array
        for i in range(len(obs)):
            self.plot2d_trajectories(obs[i:i+1],
                                     gt[i:i+1],
                                     pred[i:i+1],
                                     index[i],
                                     max_pos,
                                     min_pos)

        if ("prob", 0) in dict_list[0]:
            path_density_map = self.output_dir / "density_map"
            path_density_map.mkdir(exist_ok=True)

            xx, yy = self.get_grid(index)

            for k in dict_list[0].keys():
                if k[0] == "prob":
                    update_step = k[1] 
                    prob = dict_list[0][k]

                    bs, _, timesteps = prob.shape

                    obs = self.to_numpy(dict_list[0]['obs'])[..., :2]
                    gt = self.to_numpy(dict_list[0]['gt'])
                    traj = np.concatenate([obs, gt], axis=1)
                    obs = traj[:, :self.observe_length + update_step]
                    gt = traj[:, self.observe_length + update_step-1:]

                    zz_list = []
                    # n_components = 4
                    # data = np.zeros((timesteps, n_components, 7))
                    data = np.zeros((timesteps, 6))
                    for j in range(timesteps):
                        zz = prob[0, :, j].reshape(xx.shape)
                        total = np.sum(zz)
                        if total > 1e-12:
                            pdf = zz / np.sum(zz)
                        else:
                            print("[WARNING] pdf sum is 0. Replacing with uniform distribution")
                            pdf = np.full_like(zz, 1/zz.size)
                        pdf_flat = pdf.ravel()
                        pdf_flat /= np.sum(pdf_flat)
                        zz /= np.max(zz)
                        # print(zz)
                        
                        # Get GMM
                        # pdf = zz / np.sum(zz)
                        # pdf = zz.clone()
                        coords = np.array([xx.ravel(), yy.ravel()])
                        n_samples = 100
                        # if np.isnan(pdf_flat).any():
                        #     print("[WARNING]: Found nans in pdf!")
                        #     pdf_flat = np.nan_to_num(pdf_flat, 0.0)
                        #     if np.isnan(pdf_flat).any():
                        #         print("nans still present")
                            
                        #     pdf_flat /= np.sum(pdf_flat)
                        #     if np.isnan(pdf_flat).any():
                        #         print("[WARNING] Found nans after normalizing")
                        indices = np.random.choice(len(pdf_flat), n_samples, p=pdf_flat)
                        samples = coords[:, indices].T
                        # gmm = GaussianMixture(n_components=n_components, covariance_type='full', max_iter=1000)
                        # gmm.fit(samples)
                        
                        # for c in range(n_components):
                        #     data[j, c] = np.hstack((gmm.means_[c],
                        #                          gmm.covariances_[c].flatten(),
                        #                          gmm.weights_[c]))

                        # 2D gaussian distribution
                        mean = np.mean(samples, axis=0)
                        # cov = np.cov(samples, rowvar=False)
                        if samples.shape[0] < 2:
                            cov = np.eye(samples.shape[1]) * 1e-3  # Small regularization
                        else:
                            cov = np.cov(samples, rowvar=False)
                            # Regularize if needed
                            if np.linalg.cond(cov) > 1e12 or np.isnan(cov).any():
                                print("[WARNING] Covariance is ill-conditioned or NaN, regularizing.")
                                cov = cov + np.eye(cov.shape[0]) * 1e-3
                        data[j] = np.hstack((mean, cov.flatten()))
                        



                        # fig, ax = plt.subplots()
                        # Z = gmm_pdf(xx, yy, gmm.means_, gmm.covariances_, gmm.weights_)
                        # ax.pcolormesh(xx, yy, Z.reshape(xx.shape), shading='auto',
                        #                cmap=plt.cm.get_cmap("Greens"),
                        #                norm=matplotlib.colors.Normalize())                        
                        # plt.show()
                        # plot_density(xx, yy, zz, path=path_density_map / f"update{update_step}_{index[i][0]}_{index[i][1]}_{index[i][2].strip('PEDESTRIAN/')}_{j}.png",
                        #              traj=[obs[i], gt[i]])
                        zz_list.append(zz)
                    filename = f"sq_pdf_{update_step}_{index[i][1]}"
                    np.save(path_density_map/f'gmms/{filename}.npy', data)
                    

                    zz_sum = sum(zz_list)
                    plot_density(xx, yy, zz_sum,
                                 path=path_density_map /
                                 f"update{update_step}_{index[i][0]}_{index[i][1]}_{index[i][2].strip('PEDESTRIAN/')}_sum.png",
                                 traj=[obs[i], gt[i]])


    def prob_to_grid(self, dict_list: List[Dict]) -> List:
        if ("prob", 0) in dict_list[0]:
            index = dict_list[0]['index']
            min_pos, max_pos = self.get_minmax(index)
            xx, yy = self.get_grid(index)
            

            for data_dict in dict_list:
                data_dict["grid"] = [xx, yy]
                data_dict["minmax"] = [min_pos, max_pos]
                for k in list(data_dict.keys()):
                    if k[0] == "prob":
                        prob = data_dict[k]
                        if type(prob) == torch.Tensor:
                            prob = self.to_numpy(prob)
                            batch, _, timesteps, _ = prob.shape

                            zz_batch = []
                            for i in range(batch):
                                zz_timesteps = Parallel(n_jobs=1)(delayed(self.griddata_on_cluster)(i, prob, xx, yy, max_pos, min_pos, j)
                                                                          for j in range(timesteps))
                                #zz_timesteps = [self.griddata_on_cluster(i, prob, xx, yy, max_pos, min_pos, j) for j in range(timesteps)]
                                zz_timesteps = np.stack(
                                    zz_timesteps, axis=-1).reshape(-1, timesteps)
                                zz_batch.append(zz_timesteps)

                            zz_batch = np.stack(zz_batch, axis=0)
                            data_dict[k] = zz_batch

                        elif self.model_name == "Trajectron" or self.model_name == "GT_Dist":
                            value = torch.Tensor(np.array([xx.flatten(), yy.flatten()])).transpose(0, 1)[
                                None, :, None].tile(1, 1, prob.mus.shape[2], 1).cuda()
                            zz_batch = torch.exp(prob.log_prob(value))
                            data_dict[k] = self.to_numpy(zz_batch)

                        else:
                            zz_batch = []
                            for i in range(len(prob)):
                                zz_timesteps = []
                                for kernel in prob[i]:
                                    zz = kernel(torch.Tensor(
                                        np.array([xx.flatten(), yy.flatten()]).T).cuda())
                                    zz_timesteps.append(zz.cpu().numpy())
                                zz_timesteps = np.stack(zz_timesteps, axis=1)
                                zz_batch.append(zz_timesteps)
                            zz_batch = np.stack(zz_batch, axis=0)
                            data_dict[k] = zz_batch

                        if ("gt_traj_log_prob", k[1]) not in data_dict:
                            gt = data_dict["gt"][:, k[1]:].cpu()
                            timesteps = gt.shape[1]

                            
                            # gt_traj_prob = np.array([interp2d(xx, yy, zz_batch[:, :, t])(gt[:, t, 0], gt[:, t, 1]) for t in range(timesteps)])
                            x_flat = xx.flatten()
                            y_flat = yy.flatten()
                            ny, nx = xx.shape
                            gt_traj_prob = []
                            for t in range(timesteps):
                                z_flat = zz_batch[:, :, t].ravel()
                                tck = bisplrep(x_flat, y_flat, z_flat, s=0)

                                # Evaluate at all (x, y) pairs in gt[:, t, :]
                                xq = gt[:, t, 0]
                                yq = gt[:, t, 1]

                                # Create fine grid for evaluation
                                x_dense = np.linspace(np.min(xx), np.max(xx), 200)
                                y_dense = np.linspace(np.min(yy), np.max(yy), 200)
                                z_dense = bisplev(y_dense, x_dense, tck)
                                
                                # Interpolate manually using bilinear interpolation
                                interp_func = RegularGridInterpolator((y_dense, x_dense), z_dense)
                                vals = interp_func(np.column_stack((yq, xq)))

                                gt_traj_prob.append(vals)
                            gt_traj_prob = np.array(gt_traj_prob)


                            # gt_traj_prob = []
                            # for t in range(timesteps):
                            #     print(f"t: {t}")
                            #     print(f"zz_batch[:, :, t] shape before reshape: {zz_batch[:, :, t].shape}")
                                
                            #     # Rimodella zz_batch[:, :, t] per adattarlo alla griglia
                            #     zz_batch_t = zz_batch[:, :, t].reshape(len(ys), len(xs))
                            #     print(f"zz_batch_t shape after reshape: {zz_batch_t.shape}")
                                
                            #     # Interpolazione
                            #     interp_func = interp2d(xx, yy, zz_batch_t)
                            #     gt_traj_prob.append(interp_func(gt[:, t, 0], gt[:, t, 1]))

                            # gt_traj_prob = np.array(gt_traj_prob)
                            # print(gt_traj_prob)


                            # gt_traj_prob = np.array([
                            #     RegularGridInterpolator((xs, ys), zz_batch[:, :, t])(  # xx[:, 0], yy[0, :]
                            #         (gt[:, t, 0], gt[:, t, 1])
                            #     )
                            #     for t in range(timesteps)
                            #     #if print(f"t: {t}, xs: {xs}, ys: {ys}, zz_batch[:, :, t] shape: {zz_batch[:, :, t].shape}, gt[:, t, 0]: {gt[:, t, 0]}, gt[:, t, 1]: {gt[:, t, 1]}")
                            # ])

                        

                            # gt_traj_log_prob = torch.log(
                            #     torch.Tensor(gt_traj_prob)).squeeze()
                            gt_traj_prob_safe = np.clip(gt_traj_prob, 1e-12, None)
                            gt_traj_log_prob = torch.log(torch.Tensor(gt_traj_prob_safe)).squeeze()
                            if torch.sum(torch.isnan(gt_traj_log_prob) + torch.isinf(gt_traj_log_prob)) > 0:
                                
                                # mask = torch.isnan(gt_traj_log_prob) + torch.isinf(gt_traj_log_prob)
                                # value = torch.min(gt_traj_log_prob[~mask])
                                # gt_traj_log_prob = torch.nan_to_num(gt_traj_log_prob, nan=value, neginf=value)
                                
                                mask = torch.isnan(gt_traj_log_prob) + torch.isinf(gt_traj_log_prob)

                                if torch.sum(~mask) > 0:  # Se ci sono valori validi
                                    value = torch.min(gt_traj_log_prob[~mask])
                                else:
                                    print("Warning: gt_traj_log_prob has only NaN or inf values. Setting default value to 0.")
                                    value = torch.tensor(0.0)  # Valore neutro o scelto a seconda del tuo caso

                                gt_traj_log_prob = torch.nan_to_num(gt_traj_log_prob, nan=value, neginf=value)


                            data_dict[("gt_traj_log_prob", k[1])
                                      ] = gt_traj_log_prob[None]
                            

                            # gt_traj_log_prob = torch.log(torch.Tensor(gt_traj_prob)).squeeze()
                            # if torch.sum(torch.isnan(gt_traj_log_prob) + torch.isinf(gt_traj_log_prob)) > 0:
                            #     mask = torch.isnan(gt_traj_log_prob) + torch.isinf(gt_traj_log_prob)
                            #     if torch.sum(~mask) > 0:  # Controlla se ci sono valori validi
                            #         value = torch.min(gt_traj_log_prob[~mask])
                            #     else:
                            #         print("Warning: All values in gt_traj_log_prob are NaN or inf. Using default value.")
                            #         value = torch.tensor(0.0)  # Valore predefinito
                            #     gt_traj_log_prob = torch.nan_to_num(gt_traj_log_prob, nan=value, neginf=value)
                            # data_dict[("gt_traj_log_prob", k[1])] = gt_traj_log_prob[None]


                if self.gt_dist is not None:  # assume simfork
                    bs, timesteps, d = data_dict["gt"].shape
                    split = True
                    value = torch.Tensor(np.array([xx.flatten(), yy.flatten()])).transpose(0, 1)[
                        None, :, None].tile(1, 1, timesteps, 1).cuda()
                    if split:
                        gaussians = []
                        for i in range(len(self.gt_dist)):
                            gt_base_traj = torch.Tensor(
                                self.gt_dist[[i] * bs, -timesteps:]).cuda()
                            gaussians.append(
                                Normal(gt_base_traj[..., :2], gt_base_traj[..., 2:]))
                        data_dict["gt_prob"] = sum(
                            [torch.exp(g.log_prob(value).sum(dim=-1)) for g in gaussians])
                    else:
                        gt_base_traj = torch.Tensor(
                            self.gt_dist[data_dict["gt"][:, -1, 1] < 0, -timesteps:]).cuda()
                        gaussian = Normal(
                            gt_base_traj[:, :2], gt_base_traj[:, 2:])
                        data_dict["gt_prob"] = torch.exp(
                            gaussian.log_prob(value).sum(dim=-1))

        return dict_list


    def get_grid(self, index):
        min_pos, max_pos = self.get_minmax(index)
        xs = np.linspace(min_pos[0], max_pos[0], num=self.num_grid)
        ys = np.linspace(min_pos[1], max_pos[1], num=self.num_grid)


        xx, yy = np.meshgrid(xs, ys)

        return xx, yy
        

    def get_minmax(self, index):
        idx = [s.name for s in self.env.scenes].index(index[0][0])
        max_pos, min_pos = self.env.scenes[idx].calculate_pos_min_max()
        max_pos += 0.05 * (max_pos - min_pos)
        min_pos -= 0.05 * (max_pos - min_pos)
        return min_pos, max_pos


    def griddata_on_cluster(self, i, prob, xx, yy, max_pos, min_pos, j):
        prob_ = prob[i, :, j]
        prob_ = prob_[np.where(np.isinf(prob_).sum(axis=1) == 0)]
        prob_ = prob_[np.where(np.isnan(prob_).sum(axis=1) == 0)]

        # Controlla se prob_ è vuoto
        if prob_.shape[0] == 0:
            print(f"Warning: prob_ is empty for i={i}, j={j}. Returning zeros.")
            return np.zeros_like(xx)
    
        lnk = linkage(prob_[:, :-1],
                      method='single',
                      metric='euclidean')
        idx_cls = fcluster(lnk, t=np.linalg.norm(max_pos-min_pos)*0.003,
                           criterion='distance')
        idx_cls -= 1

        zz_ = []
        for c in range(np.max(idx_cls)+1):
            try:
                zz_.append(griddata(prob_[idx_cls == c, :-1],
                                    prob_[idx_cls == c, -1],
                                    (xx, yy), method='linear',
                                    fill_value=0.0)
                           )
            except:
                pass

        zz = sum(zz_)

        intp = RBFInterpolator(np.array([xx, yy]).reshape(
            2, -1).T, zz.flatten(), smoothing=10, kernel='linear', neighbors=8)
        zz = intp(np.array([xx, yy]).reshape(2, -1).T).reshape(xx.shape)

        prob_ = prob[i, :, j, -1]
        prob_ = prob_[~np.isnan(prob_)]
        zz = np.clip(zz, a_min=0.0, a_max=np.percentile(prob_, 95))
        return zz


    def plot2d_trajectories(self,
                            obs:  np.ndarray,
                            gt:   np.ndarray,
                            pred: np.ndarray,
                            index: Tuple,
                            max_pos: np.ndarray,
                            min_pos: np.ndarray) -> None:
        """plot 2d trajectories

        Args:
            obs (np.ndarray): (N_seqs, N_timesteps, [x, y])
            gt (np.ndarray): (N_seqs, N_timesteps, [x,y])
            pred (np.ndarray): (N_seqs, N_timesteps, N_trials, [x,y])
            img_path (Path): Path
        """

        N_seqs, N_timesteps, N_trials, N_dim = pred.shape
        gt_vis = np.zeros([N_seqs, N_timesteps+1, N_dim])
        gt_vis[:, 0] = obs[:, -1]
        gt_vis[:, 1:] = gt

        pred_vis = np.zeros([N_seqs, N_timesteps+1, N_trials, N_dim])
        # (num_seqs, num_dim) -> (num_seqs, 1, num_dim)
        pred_vis[:, 0] = obs[:, -1][:, None]
        pred_vis[:, 1:] = pred

        f, ax = plt.subplots(1, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(min_pos[0], max_pos[0])
        ax.set_ylim(min_pos[1], max_pos[1])

        for j in range(N_seqs):
            sns.lineplot(x=obs[j, :, 0], y=obs[j, :, 1], color='black',
                         legend='brief', label="obs", marker='o', linestyle='None')
            sns.lineplot(x=gt_vis[j, :, 0], y=gt_vis[j, :, 1],
                         color='blue', legend='brief', label="GT", marker='o', linestyle='None')
            for i in range(pred.shape[2]):
                if i == 0:
                    sns.lineplot(x=pred_vis[j, :, i, 0], y=pred_vis[j, :, i, 1],
                                 color='green', legend='brief', label="pred", marker='o', linestyle='None')
                else:
                    sns.lineplot(
                        x=pred_vis[j, :, i, 0], y=pred_vis[j, :, i, 1], color='green', marker='o', linestyle='None')

        img_path = self.output_dir / \
            f"{index[0]}_{index[1]}_{index[2].strip('PEDESTRIAN/')}.png"
        plt.savefig(img_path)
        plt.close()


def plot2d_trajectories_samples(
        obs:  np.ndarray,
        gt:   np.ndarray,
        max_pos: np.ndarray,
        min_pos: np.ndarray) -> None:
    """plot 2d trajectories

    Args:
        obs (np.ndarray): (N_seqs, N_timesteps, [x, y])
        gt (np.ndarray): (N_seqs, N_timesteps, [x,y])
        pred (np.ndarray): (N_seqs, N_timesteps, N_trials, [x,y])
        img_path (Path): Path
    """

    N_seqs = len(gt)
    _, N_timesteps, N_dim = gt[0].shape

    f, ax = plt.subplots(1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_pos[0], max_pos[0])
    ax.set_ylim(min_pos[1], max_pos[1])

    for j in range(N_seqs):
        gt_vis = np.zeros([N_timesteps+1, N_dim])
        gt_vis[0] = obs[j][0, -1]
        gt_vis[1:] = gt[j][0]
        if j == 0:
            sns.lineplot(x=obs[j][0, :, 0], y=obs[j][0, :, 1], color='green',
                         legend='brief', label="obs", marker='o')
            sns.lineplot(x=gt_vis[:, 0], y=gt_vis[:, 1],
                         color='black', legend='brief', label="GT", marker='o')
        else:
            sns.lineplot(x=obs[j][0, :, 0], y=obs[j][0, :, 1], color='green',
                         marker='o')
            sns.lineplot(x=gt_vis[:, 0], y=gt_vis[:, 1],
                         color='black', marker='o')

    img_path = "gts.png"
    plt.axis('off')
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()
