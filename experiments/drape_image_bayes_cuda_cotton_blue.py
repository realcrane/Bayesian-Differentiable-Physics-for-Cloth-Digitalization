import math
import os
import sys
from time import time

import torch
import torchvision
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.checkpoint import checkpoint

import kaolin as kal

from diffsim import Simulation, Cloth, Mesh
from diffsim import read_obj, save_obj
from diffsim import num_bend_edges

# ---- Selecting GPU -----
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # [Change This Value]
device = torch.device('cuda:0')

train_log_folder = "./train_bnn_log/train1/" # The folder for saving training logs [Change if you need]

loss_log_file = train_log_folder + "heter_cotton_blue_samples.txt"  # The filename for saving losses [Change if you need]
grad_log_file = train_log_folder + "heter_cotton_blue_samples_grad.txt"  # The filename for saving parameter gradients [Change if you need]
parameter_log_file = train_log_folder + "heter_cotton_blue_samples_params.txt"   # The filename for saving parameter while training [Change if you need]
gt_images_filename_log = train_log_folder + "gt_image_filenames.txt"    # The filaname for saving the used training image name [Change if you need]
image_log_path = train_log_folder + "images/"   # The folder for saving the simulated drape cloth silhouette while training [Change if you need]
mesh_log_path = train_log_folder + "meshes/"    # The folder for saving the simulated drape cloth meshes while training [Change if you need]
checkpoint_log_path = train_log_folder + "checkpoint/"  # The folder for saving the traning models (can be loaded later) while training [Change if you need]

cloth_mesh_path = "./mesh_in/circle3.obj"   # Used initial mesh [Change to select your needed mesh]
cloth_image_gt_path = "./bnn_image_gt/cotton_blue/" # Folder containing ground truth images [Change according to your folder for saving training silhouettes]

# ---- Create the folders for saving the training log --------
if os.path.exists(train_log_folder):
    print("Folder exists, confirm if you intend to overwrite it.")
    sys.exit(0)
else:
    print("Create the Log Folder.")
    os.mkdir(train_log_folder)
    os.mkdir(image_log_path)
    os.mkdir(mesh_log_path)
    os.mkdir(checkpoint_log_path)

NUM_GT = 1 # Number of GT images （used for training）
EPOCH = 1000    # Number of epochs
SIM_STEPS = 99  # Number of simulation steps [NO need to change when following the paper]
BNN_NUM_SAMPLES = NUM_GT # When 1, learning from single training sample. Otherwise, learning from multiple samples.
IS_SELECT_SAMPLES = True    # Select training samples or not
SELECTED_SAMPLE_IDXS = [2] # Use when train by one sample (Sample 2)

# read the ground truth images path
gt_images_walk = os.walk(cloth_image_gt_path)
gt_images = []

for path, dir_lst, file_list in gt_images_walk:
    for file_name in file_list:
        if IS_SELECT_SAMPLES and eval(file_name.split("_")[3]) in SELECTED_SAMPLE_IDXS:
            gt_images.append(os.path.join(path, file_name))
        elif not IS_SELECT_SAMPLES:
            gt_images.append(os.path.join(path, file_name))

gt_images = sorted(gt_images, key=lambda gt_images: gt_images.split("/")[-1].split("_")[1]) # Sort image path by sample index

# Save the used training images in the file "gt_images_filename_log.txt"
with open(gt_images_filename_log, "w") as gt_file_log:    
    for s in range(NUM_GT):
        gt_file_log.write(gt_images[s] + "\n")

# Differentiable rending related constants
IMAGE_W = IMAGE_L = 256
camera_proj = kal.render.camera.generate_perspective_projection(0.5).cuda()
cam_transform = kal.render.camera.generate_transformation_matrix( 
    torch.tensor([[0.0, 0.0, 0.7]]), 
    torch.tensor([[0.0, 0.0, -1.0]]),
    torch.tensor([[0.0, 1.0, 0.0]])).cuda()

# Cloth physical parameters
# Initial parameters
SMALL_STD = 1e-13   # For setting the standard deviation which is very close to zero

density = torch.tensor( [0.192], dtype=torch.double, device=device)

#-------------------- Posteriors ---------------------
c11_mean = torch.tensor([150.0862, 189.2275, 51.2988, 36.7329, 58.9375, 207.3079], dtype=torch.double, device=device)  # Stretch u
c12_mean = torch.tensor([0.3456, 2.3716, 4.9362, 6.3891, 5.1495, 2.2163], dtype=torch.double, device=device)   # Poisson
c22_mean = torch.tensor([150.4856, 200.7334, 60.3071, 32.7261, 49.8735, 193.4768], dtype=torch.double, device=device)   # Stretch v
c33_mean = torch.tensor([10.189, 20.905, 25.393, 30.462, 25.055, 20.669], dtype=torch.double, device=device)   # Shear (or Stretch uv)

c11_std = torch.log(torch.exp(torch.tensor([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4], dtype=torch.double, device=device)) - 1.0)  # Stretch u
c12_std = torch.log(torch.exp(torch.tensor([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4], dtype=torch.double, device=device)) - 1.0)   # Poisson
c22_std = torch.log(torch.exp(torch.tensor([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4], dtype=torch.double, device=device)) - 1.0)   # Stretch v
c33_std = torch.log(torch.exp(torch.tensor([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4], dtype=torch.double, device=device)) - 1.0)   # Shear (or Stretch uv)

bending_mean = torch.tensor((
    (1.07101021e-05, 2.91414065e-05, 9.28662269e-05, 9.49818339e-05, 8.14479132e-05),
    (2.75335216e-05, 4.83758496e-05, 3.83801280e-05, 7.39827734e-05, 6.98164992e-05),
    (2.46016914e-05, 1.02834500e-04, 1.23397066e-04, 1.00201198e-04, 1.02161645e-04)), dtype=torch.double, device=device).unsqueeze(0).unsqueeze(1)

bending_std = torch.log(torch.exp(torch.tensor((
    (4.63300752e-05, 2.28590446e-06, 2.12511344e-06, 1.10795624e-06, 1.83805267e-06),
    (4.24540374e-05, 3.82856794e-06, 3.41387528e-06, 2.08860323e-06, 2.02380604e-06),
    (5.38681744e-05, 2.04742644e-06, 2.11579862e-06, 1.06534624e-06, 1.54218522e-06)), dtype=torch.double, device=device)) - 1.0).unsqueeze(0).unsqueeze(1)

# -----------------------  Priors (fixed) -------------------------
c11_mean_prior = torch.tensor([150.0862, 189.2275, 51.2988, 36.7329, 58.9375, 207.3079], dtype=torch.double, device=device)  # Stretch u
c12_mean_prior = torch.tensor([0.3456, 2.3716, 4.9362, 6.3891, 5.1495, 2.2163], dtype=torch.double, device=device)   # Poisson
c22_mean_prior = torch.tensor([150.4856, 200.7334, 60.3071, 32.7261, 49.8735, 193.4768], dtype=torch.double, device=device)   # Stretch v
c33_mean_prior = torch.tensor([10.189, 20.905, 25.393, 30.462, 25.055, 20.669], dtype=torch.double, device=device)   # Shear (or Stretch uv)

c11_std_prior = torch.log(torch.exp(torch.tensor([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4], dtype=torch.double, device=device)) - 1.0)  # Stretch u
c12_std_prior = torch.log(torch.exp(torch.tensor([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4], dtype=torch.double, device=device)) - 1.0)   # Poisson
c22_std_prior = torch.log(torch.exp(torch.tensor([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4], dtype=torch.double, device=device)) - 1.0)   # Stretch v
c33_std_prior = torch.log(torch.exp(torch.tensor([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4], dtype=torch.double, device=device)) - 1.0)   # Shear (or Stretch uv)

bending_prior_mean = torch.tensor((
    (1.07101021e-05, 2.91414065e-05, 9.28662269e-05, 9.49818339e-05, 8.14479132e-05),
    (2.75335216e-05, 4.83758496e-05, 3.83801280e-05, 7.39827734e-05, 6.98164992e-05),
    (2.46016914e-05, 1.02834500e-04, 1.23397066e-04, 1.00201198e-04, 1.02161645e-04)), dtype=torch.double, device=device).unsqueeze(0).unsqueeze(1)

bending_prior_std = torch.log(torch.exp(torch.tensor((
    (1.63300752e-05, 2.28590446e-06, 2.12511344e-06, 1.10795624e-06, 1.83805267e-06),
    (1.24540374e-05, 3.82856794e-06, 3.41387528e-06, 2.08860323e-06, 2.02380604e-06),
    (1.38681744e-05, 2.04742644e-06, 2.11579862e-06, 1.06534624e-06, 1.54218522e-06)), dtype=torch.double, device=device)) - 1.0).unsqueeze(0).unsqueeze(1)

# Networks

# The Gaussian distribution for cloth physical parameters
class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(
            torch.tensor(0.0, dtype=torch.double, device=device), 
            torch.tensor(1.0, dtype=torch.double, device=device))

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.rho.size())
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        # Log of Gaussian PDF
        return (-math.log(math.sqrt( 2 * math.pi)) - torch.log(self.sigma) - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).mean()

    def sample_batch(self, batch_size):
        sample_size = torch.Size([batch_size] + list(self.rho.shape))
        epsilon_batch = self.normal.sample(sample_size)
        return self.mu.expand(sample_size) + self.sigma.expand(sample_size) * epsilon_batch

    def log_prob_batch(self, input_batch):
        return (-math.log(math.sqrt( 2 * math.pi)) - torch.log(self.sigma).expand(input_batch.shape) - ((input_batch - self.mu.expand(input_batch.shape)) ** 2) / (2 * self.sigma.expand(input_batch.shape) ** 2)).mean()

# Bayesian Cloth Simulator
class BayesCloth(nn.Module):
    def __init__(self):
        super().__init__()

        self.sample_seed = 0    # For sampling the same parameters in the checkpoint function

        # Initializing Simulator
        self.sim = Simulation()

        self.sim.dev_wordy = False

        self.sim.gravity = torch.tensor([0.0, 0.0, -9.8], dtype=torch.double, device=device)

        self.sim.step_time_cuda = torch.tensor(0.05, dtype=torch.double, device=device) # Simulation step size 0.05s

        cloth = Cloth()

        cloth.handle_stiffness_cuda = torch.tensor(1e3, dtype=torch.double, device=device)  # Set cloth handle stiffness

        read_obj(cloth_mesh_path, cloth.mesh)

        self.num_nodes = len(cloth.mesh.nodes)

        self.num_faces = len(cloth.mesh.faces)  # Number of faces (number of stretching stiffness)

        self.num_bending_edges = num_bend_edges(cloth.mesh) # Number of bending edges (number of bending stiffness)

        self.sim.cloths.append(cloth)

        self.sim.init_vectorization_cuda()   # Initialize the matrices

        # # For Differentiable Rendering
        faces_uv, faces = self.sim.cloths[0].faces_uv_idx()

        self.faces_uv_render = faces_uv.to(torch.float32).unsqueeze(0).cuda()

        self.faces_render = faces.cuda()

        self.face_attributes = [self.faces_uv_render, torch.ones((1, faces.shape[0], 3, 1), device=device)]

        # Cloth Physical Parameter Distributions
        self.densities = [density.clone() for _ in range(self.num_faces)]

        self.c11_mu = nn.Parameter(c11_mean.clone())
        self.c12_mu = nn.Parameter(c12_mean.clone())
        self.c22_mu = nn.Parameter(c22_mean.clone())
        self.c33_mu = nn.Parameter(c33_mean.clone())

        self.c11_sigma = nn.Parameter(c11_std.clone())
        self.c12_sigma = nn.Parameter(c12_std.clone())
        self.c22_sigma = nn.Parameter(c22_std.clone())
        self.c33_sigma = nn.Parameter(c33_std.clone())

        self.bending_mu = nn.Parameter(bending_mean.clone())
        self.bending_rho = nn.Parameter(bending_std.clone())

        self.c11_gaussian = Gaussian(self.c11_mu, self.c11_sigma)   # Stretching Gaussian Distribution (u direction)
        self.c12_gaussian = Gaussian(self.c12_mu, self.c12_sigma)   # Stretching Gaussian Distribution (Poisson)
        self.c22_gaussian = Gaussian(self.c22_mu, self.c22_sigma)   # Stretching Gaussian Distribution (v direction)
        self.c33_gaussian = Gaussian(self.c33_mu, self.c33_sigma)   # Stretching Gaussian Distribution (shear)

        self.bending_gaussian = Gaussian(self.bending_mu, self.bending_rho) # Bending Gaussian Distribution

        self.c11_prior = Gaussian(c11_mean_prior, c11_std_prior)  # Stretching Prior Distribution (u direction)
        self.c12_prior = Gaussian(c12_mean_prior, c12_std_prior)  # Stretching Prior Distribution (Poisson)
        self.c22_prior = Gaussian(c22_mean_prior, c22_std_prior)  # Stretching Prior Distribution (v direction)
        self.c33_prior = Gaussian(c33_mean_prior, c33_std_prior)  # Stretching Prior Distribution (shear)

        self.bending_prior = Gaussian(bending_prior_mean, bending_prior_std)   # Bending Prior Distribution

        self.log_variational_posterior = 0  # The first term in the Eq.2 
        self.log_prior = 0  # The second term in the Eq.2

    def reset(self, new_sample_seed=0):

        self.sample_seed = new_sample_seed  # Update sample_seed

        torch.manual_seed(self.sample_seed)    # Make sure the sampled parameter is the same in every checkpoint
      
        c11s = self.c11_gaussian.sample_batch(self.num_faces) # Sampling stretches list (u direction)
        c12s = self.c12_gaussian.sample_batch(self.num_faces) # Sampling stretches list (Poisson)
        c22s = self.c22_gaussian.sample_batch(self.num_faces) # Sampling stretches list (v direction)
        c33s = self.c33_gaussian.sample_batch(self.num_faces) # Sampling stretches list (Shearing uv)

        bendings = self.bending_gaussian.sample_batch(self.num_bending_edges) # the bendings list

        self.log_variational_posterior = self.c11_gaussian.log_prob_batch(c11s) +\
            self.c12_gaussian.log_prob_batch(c12s) +\
            self.c22_gaussian.log_prob_batch(c22s) +\
            self.c33_gaussian.log_prob_batch(c33s) +\
            self.bending_gaussian.log_prob_batch(bendings)

        self.log_prior = self.c11_prior.log_prob_batch(c11s) +\
            self.c12_prior.log_prob_batch(c12s) +\
            self.c22_prior.log_prob_batch(c22s) +\
            self.c33_prior.log_prob_batch(c33s) +\
            self.bending_prior.log_prob_batch(bendings)

        self.sim.clean_mesh()    # Clean Cloth and Obstacle Mesh

        read_obj(cloth_mesh_path, self.sim.cloths[0].mesh)

        self.sim.cloths[0].set_densities_cuda(self.densities)

        self.sim.cloths[0].set_stretches_cuda(list(c11s), list(c12s), list(c22s), list(c33s))

        self.sim.cloths[0].set_bendings_cuda(list(bendings))

        self.sim.prepare_cuda()

    def checkpoint_advance(self, pos, vel):

        # Make sure the sampled parameters are the same as the initially sampled parameters and are the same in every checkpoint
        torch.manual_seed(self.sample_seed)

        chk_c11s = list(self.c11_gaussian.sample_batch(self.num_faces)) # Sampling stretches list (u direction)
        chk_c12s = list(self.c12_gaussian.sample_batch(self.num_faces)) # Sampling stretches list (Poisson)
        chk_c22s = list(self.c22_gaussian.sample_batch(self.num_faces)) # Sampling stretches list (v direction)
        chk_c33s = list(self.c33_gaussian.sample_batch(self.num_faces)) # Sampling stretches list (Shearing uv)

        chk_bendings = list(self.bending_gaussian.sample_batch(self.num_bending_edges)) # the bendings list

        self.sim.cloths[0].set_stretches_cuda(chk_c11s, chk_c12s, chk_c22s, chk_c33s)
        self.sim.cloths[0].set_bendings_cuda(chk_bendings)

        return self.sim.advance_step_cuda(pos, vel)

    def forward(self, steps):

        self.sim.cloths[0].update_pos_vel_cuda()
        
        pos, vel = self.sim.cloths[0].get_pos_vel_cuda() # Get cloth initial pos and vel

        pos, vel = self.sim.advance_step_cuda(pos, vel)  # Simulate one step for "checkpoint"

        for _ in range(steps):
            # Checkpoint Forward simulation
            pos, vel = checkpoint(self.checkpoint_advance, pos, vel)

        # Update cloth mesh from position and velocity vector (For saving mesh later)
        self.sim.cloths[0].update_mesh_cuda(pos.clone().detach().cpu().view(self.num_nodes, 3), vel.clone().detach().cpu().view(self.num_nodes, 3))

        # Differentiable Rendering DIB-R
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            pos.view(1, -1, 3).to(torch.float32), self.faces_render, camera_proj, camera_transform=cam_transform)

        _, soft_mask, _ = kal.render.mesh.dibr_rasterization(
            IMAGE_W, IMAGE_L, face_vertices_camera[:, :, :, -1], face_vertices_image, self.face_attributes, face_normals[:, :, -1])

        # Return render silhouette (CUDA)
        return soft_mask[0]

# --------------- Loading GT ------------------
# Single Training Image
# image_gt = torch.abs(1.0 - torch.mean(torchvision.io.read_image(gt_images[SELECTED_SAMPLE_IDX]).to(torch.float), 0) / 255.0)   # Iterative over all GT images

# Multiple Training Images 
image_gts = [torch.abs(1.0 - torch.mean(torchvision.io.read_image(image_path).to(torch.float).to(device), 0) / 255.0)  for image_path in gt_images]

# Save Ground truth images for debug
# for i in range(len(image_gts)):
#     torchvision.utils.save_image(image_gts[i], f"./save_image/{i}.jpg")

# --------------- Training --------------------
net = BayesCloth()

# Optimizer and Parameters' learning rate
optim = Adam([
	{"params": net.c11_mu, "lr": 1e-14}, 
	{"params": net.c11_sigma, "lr": 1e-14},
    {"params": net.c12_mu, "lr": 1e-14}, 
	{"params": net.c12_sigma, "lr": 1e-14},
    {"params": net.c22_mu, "lr": 1e-14}, 
	{"params": net.c22_sigma, "lr": 1e-14},
    {"params": net.c33_mu, "lr": 1e-14}, 
	{"params": net.c33_sigma, "lr": 1e-14},
	{"params": net.bending_mu, "lr": 3e-6},
	{"params": net.bending_rho, "lr": 0.1}
])

# --------------- For Loading Trained Models ------------------
# model_dict = torch.load("./train_bnn_log/train1/checkpoint/train_dict_epoch0.pt")
# net.load_state_dict(model_dict['model_state_dict'])
# optim.load_state_dict(model_dict['optimizer_state_dict'])

loss_log_f = open(loss_log_file, "w")
para_log_f = open(parameter_log_file, "w")

grad_log_f = open(grad_log_file, "w")   # Print gradients for debugging

for e in range(EPOCH):

    optim.zero_grad()   # Reset optimizor

    log_priors = torch.zeros(1, device=device)
    log_variational_posteriors = torch.zeros(1, device=device)
    log_likelihoods = torch.zeros(1, device=device)

    # Iterative over all GT images (One Gt Per Epoch)
    # image_gt = torch.abs(1.0 - torch.mean(torchvision.io.read_image(gt_images[e%NUM_GT]).to(torch.float), 0) / 255.0)   

    # # Follow the original paper: the third term in Eq.2 is a log likelihood
    # sigma_obs = 0.01	# Noise or Error tolerance
    # likelihood_dist = torch.distributions.normal.Normal(image_gt, sigma_obs)

    for s in range(BNN_NUM_SAMPLES):

        sys_time_int = int(time()) # Use system time.

        net.reset(sys_time_int) # Reset cloth simulator (use s as the sample seed) 

        # Follow the original paper: the third term in Eq.2 is a log likelihood
        sigma_obs = 0.01	# Noise or Error tolerance
        likelihood_dist = torch.distributions.normal.Normal(image_gts[s], sigma_obs)

        image_pred = net.forward(SIM_STEPS) # Simulate and render to silhouette

        torchvision.utils.save_image(image_pred, image_log_path + f"Epoch_{e}_Sample_{s}.jpg") # Save images for Training Log

        save_obj(mesh_log_path + f"Epoch_{e}_Sample_{s}.obj", net.sim.cloths[0].mesh) # Save mesh

        log_variational_posteriors += net.log_variational_posterior # The first term in Eq.2

        log_priors += net.log_prior # The second term in Eq.2

        log_likelihoods += torch.mean(likelihood_dist.log_prob(image_pred))    # The third term in Eq.2

    # The loss defined in the paper
    log_variational_posterior = log_variational_posteriors/BNN_NUM_SAMPLES  # The (average) first term in Eq. 2
    log_prior = log_priors/BNN_NUM_SAMPLES  # The (average) second term in Eq.2
    log_likelihood = log_likelihoods/BNN_NUM_SAMPLES   # The (average) third term in Eq.2

    loss = log_variational_posterior - log_prior - log_likelihood   # Loss function

    loss.backward() # Compute gradient
    
    # Log Gradient Information
    stretching_mean_grad = torch.stack((
        net.c11_mu.grad, net.c12_mu.grad, 
        net.c22_mu.grad, net.c33_mu.grad), dim=1)

    stretching_std_grad = torch.stack((
        net.c11_sigma.grad, net.c12_sigma.grad, 
        net.c22_sigma.grad, net.c33_sigma.grad, ), dim=1)

    grad_log_f.write(f"Epoch: {e}\nStretch Mean Grad:\n{stretching_mean_grad}\nStretch Var Grad:\n{stretching_std_grad}\nBend Mean Grad:\n{net.bending_mu.grad}\nBend Var Grad:\n{net.bending_rho.grad}\n")
    grad_log_f.flush() # Write to the file immediately

    optim.step()    # Optimize parameter

    # Writing Training Log 
    # One Sample Per Epoch
    # loss_log_f.write(f"Epoch: {e}\nGT Image: {gt_images[e%NUM_GT]}\nLoss: {loss.item()}\nLog Variational Posterior : {log_variational_posterior.item()}\nLog Prior: {log_prior.item()}\nLog Likelihood: {log_likelihood.item()}\n")  
    # All Samples (Average loss) Per Epoch (*Sampling (num_gt times) in BNN*)

    # Combining stretching stiffness tensor's mean and standard deviation
    stretching_mean = torch.stack((
        net.c11_mu.clone().detach(), net.c12_mu.clone().detach(), 
        net.c22_mu.clone().detach(), net.c33_mu.clone().detach()), dim=1)

    stretching_std = torch.log1p(torch.exp(torch.stack((
        net.c11_sigma.clone().detach(), net.c12_sigma.clone().detach(), 
        net.c22_sigma.clone().detach(), net.c33_sigma.clone().detach()), dim=1)))
    
    loss_log_f.write(f"Epoch: {e}\nLoss: {loss.item()}\nLog Variational Posterior : {log_variational_posterior.item()}\nLog Prior: {log_prior.item()}\nLog Likelihood: {log_likelihood.item()}\n")     
    para_log_f.write(f"Epoch: {e}\nStretches Mean:\n{stretching_mean}\nStretches Var:\n{stretching_std}\nBendings Mean:\n{net.bending_mu}\nBendings Var:\n{net.bending_gaussian.sigma}\n")

    loss_log_f.flush()  # For immediately viewing result and saving memory
    para_log_f.flush()  # For immediately viewing result and saving memory

    torch.save({
        'epoch':EPOCH,
        'model_state_dict':net.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss':loss}, checkpoint_log_path + f"train_dict_epoch_{e}.pt")

loss_log_f.close()
para_log_f.close() 

grad_log_f.close()