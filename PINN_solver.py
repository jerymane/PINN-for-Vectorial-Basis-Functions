import os
import sys
import tensorflow as tf
import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt 
import deepxde as dde


from atexit import register
import timeit
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable warnings from tensorflow
os.environ["TF_GPU_ALLOCATOR"] = 'cuda_malloc_async'

# tf.compat.v1.enable_eager_execution()


NN_WIDTH = 30
NN_DEPTH = 2
EPOCHS = 10000
BATCH_SIZE = 10
TEST_SET_SIZE = 10
LR = 1e-3                   # learning rate (can also be adjusted to be dynamic)
w_PDE = 10                  # PDE training weight
w_BCs = 1.0                 # BCS training weight

train = 0
load_pretrained_model = 1
plot_histories = 0


training_quads_dir = r'./training_sets_quads/'
# quads_filename = 'quads100a_sp1_ir5'
quads_filename = 'slices40_sp01_ir05_1-40'
# quads_filename = 'quads50a_sp01_ir05'
bary_coords_path = 'barycentric_unit_square_d500_b200.dat'

quads_path = training_quads_dir + quads_filename
dir_path = './output/' + quads_filename + '_nn' + str(NN_WIDTH) + 'x' + str(NN_DEPTH) + '_e' \
            + str(EPOCHS) + '_bt' + str(BATCH_SIZE) + '_lr_1e-3' + '_wPDE100' # + '_lr' + "{:.2e}".format(LR)
dir_path = 'tryout'

load_model_dir_path = 'tryout/'

print('dir path: ', dir_path)

if train:
    if os.path.exists(dir_path) and os.path.exists(dir_path + '/model/'):
        sys.exit('Proceeding will rewrite the already trained model')
    else:
        os.makedirs(dir_path)


shutil.copy(quads_path, dir_path+'/')

DTYPE='float64'
dde.config.set_default_float(DTYPE)
tf.keras.backend.set_floatx(DTYPE)


class PINN_NeuralNet(tf.keras.Model):                                               
    """ Set basic architecture of the PINN model."""

    def __init__(self, 
            output_dim=1,
            num_hidden_layers=NN_DEPTH, 
            num_neurons_per_layer=NN_WIDTH,
            activation='tanh',
            kernel_initializer='glorot_normal',
            **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim

        # Define NN architecture

        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                             activation=tf.keras.activations.get(activation),
                             kernel_initializer=kernel_initializer)
                           for _ in range(self.num_hidden_layers)]
        self.out = tf.keras.layers.Dense(output_dim)
        
    def call(self, X):
        """Forward-pass through neural network."""
        Z=X
        for i in range(self.num_hidden_layers):
            Z = self.hidden[i](Z)
        return self.out(Z)



class PINNSolver():
    def __init__(self, model, collocation_pts, quads):
        self.model = model
        
        # # Store collocation points
        self.collocation_pts = None
        self.x = None
        self.y = None
        self.quads = None
        
        self.train_set_quads = quads[TEST_SET_SIZE:]
        self.train_set_pts = collocation_pts[TEST_SET_SIZE:, :, :]
        self.num_bcs_pts = np.sum(quads[0].num_bcs)


        self.test_set_quads = quads[:TEST_SET_SIZE]
        self.test_set_pts = collocation_pts[:TEST_SET_SIZE, :, :]

        self.u = None
        self.u_x = None
        self.u_y = None
        # Initialize history of losses and global iteration counter
        self.hist = []
        self.iter = 0
        self.hist_error_pde = []
        self.hist_error_bcs = []
        self.current_error_pde = 0
        self.current_error_bcs = 0
        self.current_error_distr = []
        self.current_test_error_distr = []

        self.test_hist = []
        self.test_hist_error_pde = []
        self.test_hist_error_bcs = []
        self.test_current_error_pde = 0
        self.test_current_error_bcs = 0
        self.quad_error_distr_hist = []
        self.quad_test_error_distr_hist = []

    def get_residual(self):

        with tf.GradientTape(persistent=True) as tp1, tf.GradientTape(persistent=True) as tp2:
            tp1.watch(self.x)
            tp1.watch(self.y)

            tp2.watch(self.x)  
            tp2.watch(self.y)

            #u = self.model(self.x, self.y)       
            self.u = self.model(tf.stack([self.x, self.y], axis=2))

            self.u_x = tp1.gradient(self.u, self.x)
            self.u_y = tp1.gradient(self.u, self.y)

        u_xx = tp2.gradient(self.u_x, self.x)
        u_yy = tp2.gradient(self.u_y, self.y)

        del tp1
        del tp2

        return u_xx + u_yy
    

    def get_residual_eager(self):
        #u = self.model(self.x, self.y)       
        self.u = self.model(tf.stack([self.x, self.y], axis=2))
        print('u shape ', self.u.shape)
        self.u_x = tf.gradients(self.u, self.x)[0]
        self.u_y = tf.gradients(self.u, self.y)[0]
        print('u_x shape ', self.u_x.shape)
        u_xx = tf.gradients(self.u_x, self.x)
        u_yy = tf.gradients(self.u_y, self.y)

        return u_xx + u_yy
    

    def loss_fn(self, arg):
        return tf.reduce_mean(tf.square(arg))


    def get_pde_losses(self):
        
        r = self.get_residual()
        r = r[:, self.num_bcs_pts::]
        loss_pde = self.loss_fn(r)
        return loss_pde, r

    def boundary_derivative(self, inputs, beg, end):
        with tf.GradientTape(persistent=True) as tp:
            tp.watch(inputs)
            out = self.model(inputs)
        dydx = tp.gradient(out, inputs)[beg:end]
        del tp
        return dydx


    def get_bcs_losses(self):
        losses = []
        bcs_errors = []
        
        for qi, quad in enumerate(self.quads):
            single_quad_bcs_errors = []
            bcs_start = np.cumsum([0] + quad.num_bcs)
            for i, bc in enumerate(quad.bcs):
                beg, end = bcs_start[i], bcs_start[i + 1]
                value = bc.func(self.collocation_pts[qi,:,:], beg, end, None)

                n = bc.boundary_normal(self.collocation_pts[qi,:,:], beg, end, None)
                bc_d = self.boundary_derivative(tf.convert_to_tensor(self.collocation_pts[qi, :, :], dtype=DTYPE), beg, end)
                errors = tf.reduce_sum(bc_d * n, axis=1, keepdims=True) - value
                single_quad_bcs_errors.append(errors)
                losses.append(self.loss_fn(errors))

            bcs_errors.append(tf.concat(single_quad_bcs_errors, 0))
        return losses, bcs_errors
        

    def get_grad(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            pde_loss, pde_errors = self.get_pde_losses()
            bcs_losses, bcs_errors = self.get_bcs_losses()
            loss = w_BCs*sum(bcs_losses) + w_PDE*pde_loss
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        
        return loss, g, bcs_errors, pde_errors


    # @timeit
    def solve_with_TFoptimizer_batches(self, optimizer, epochs=1001):

        @tf.function
        def train_step():
            loss, grad_theta, bcs_errors, pde_errors = self.get_grad()
            
            # Perform gradient descent step
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss, bcs_errors, pde_errors
        
        @tf.function
        def test_step():
            loss, grad_theta, bcs_error, pde_errors = self.get_grad()

            return loss, bcs_error, pde_errors


        for ep in range(epochs):
            
            self.epoch = ep
            batch_step = 0
            train_set_size = self.train_set_pts.shape[0]
            #print('epoch ', ep)
            self.current_error_distr = []
            self.quad_error_distr_hist = []
            while batch_step*BATCH_SIZE < train_set_size:
                batch_start = batch_step*BATCH_SIZE
                batch_end = (batch_step+1)*BATCH_SIZE if (batch_step+1)*BATCH_SIZE < train_set_size else train_set_size
                batch_range = slice(batch_start, batch_end, 1)

                self.x = tf.convert_to_tensor( self.train_set_pts[batch_range, :, 0], dtype=DTYPE )
                self.y = tf.convert_to_tensor( self.train_set_pts[batch_range, :, 1], dtype=DTYPE )
                self.collocation_pts = self.train_set_pts[batch_range, :,:]
                self.quads = self.train_set_quads[batch_range]
                loss, bcs_errors, pde_errors = train_step()
                if ep % 100 == 0:
                    self.current_error_distr.extend(tf.concat((tf.squeeze(bcs_errors,[-1]), pde_errors), 1))

                self.current_error_bcs = tf.reduce_mean(tf.abs(bcs_errors)).numpy() # TODO: separate bcs_errors to get better insight
                self.current_error_pde = tf.reduce_mean(tf.abs(pde_errors)).numpy()
                self.current_loss = loss.numpy()
                self.callback(step=100)

                if TEST_SET_SIZE != 0:
                    self.current_test_error_distr = []
                    self.quad_test_error_distr_hist = []
                    self.x = tf.convert_to_tensor( self.test_set_pts[:, :, 0], dtype=DTYPE )
                    self.y = tf.convert_to_tensor( self.test_set_pts[:, :, 1], dtype=DTYPE )
                    self.collocation_pts = self.test_set_pts
                    self.quads = self.test_set_quads
                    
                    test_loss, test_bcs_errors, test_pde_errors = test_step()
                    if ep % 100 == 0:
                        self.current_test_error_distr.extend(tf.concat((tf.squeeze(test_bcs_errors,[-1]), test_pde_errors), 1))
                    self.test_current_error_bcs = np.mean(np.abs(test_bcs_errors)) # TODO: separate bcs_errors to get better insight
                    self.test_current_error_pde = np.mean(np.abs(test_pde_errors))
                    self.test_current_loss = test_loss.numpy()
                    self.callback_save_test()
            
                batch_step += 1
            
            
            if ep % 100 == 0:
                print('Epoch: ', ep)
                self.quad_error_distr_hist.append(self.current_error_distr)
                self.quad_test_error_distr_hist.append(self.current_test_error_distr)
                self.callback_end_of_epoch_save()
            
           

    # @timeit
    def solve_with_ScipyOptimizer(self, method='L-BFGS-B', **kwargs):
        """This method provides an interface to solve the learning problem
        using a routine from scipy.optimize.minimize.
        (Tensorflow 1.xx had an interface implemented, which is not longer
        supported in Tensorflow 2.xx.)
        Type conversion is necessary since scipy-routines are written in Fortran
        which requires 64-bit floats instead of 32-bit floats."""
        
        def get_weight_tensor():
            """Function to return current variables of the model
            as 1d tensor as well as corresponding shapes as lists."""
            
            weight_list = []
            shape_list = []
            
            # Loop over all variables, i.e. weight matrices, bias vectors and unknown parameters
            for v in self.model.trainable_variables:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())
                
            weight_list = tf.convert_to_tensor(weight_list)
            return weight_list, shape_list

        x0, shape_list = get_weight_tensor()
        

        def set_weight_tensor(weight_list):
            """Function which sets list of weights
            to variables in the model."""
            idx = 0
            for v in self.model.variables:
                vs = v.shape
                
                # Weight matrices
                if len(vs) == 2:  
                    sw = vs[0]*vs[1]
                    new_val = tf.reshape(weight_list[idx:idx+sw],(vs[0],vs[1]))
                    idx += sw
                
                # Bias vectors
                elif len(vs) == 1:
                    new_val = weight_list[idx:idx+vs[0]]
                    idx += vs[0]
                    
                # Variables (in case of parameter identification setting)
                elif len(vs) == 0:
                    new_val = weight_list[idx]
                    idx += 1
                    
                # Assign variables (Casting necessary since scipy requires float64 type)
                v.assign(tf.cast(new_val, DTYPE))
        
    def callback(self, step=1):
        if self.iter % step == 0:
            print('It {:05d}: loss = {:10.8e}'.format(self.iter,self.current_loss))
        self.hist.append(self.current_loss)
        self.hist_error_bcs.append(self.current_error_bcs)
        self.hist_error_pde.append(self.current_error_pde)
        
        self.iter+=1


    def callback_save_test(self):
        self.test_hist.append(self.test_current_loss)
        self.test_hist_error_bcs.append(self.test_current_error_bcs)
        self.test_hist_error_pde.append(self.test_current_error_pde)


    def callback_end_of_epoch_save(self):
        self.model.save(dir_path + '/model', save_format="tf")
        np.savetxt(dir_path + '/loss_history.dat', self.hist)
        np.savetxt(dir_path + '/test_loss_history.dat', self.test_hist)
        np.savetxt(dir_path + '/error_history.dat', (self.test_hist_error_pde, self.test_hist_error_bcs))
        np.savetxt(dir_path + '/test_error_history.dat', (self.test_hist_error_pde, self.test_hist_error_bcs))
        
        if not os.path.exists(dir_path + '/error_distribution_history/' + str(self.epoch)): 
            os.makedirs(dir_path + '/error_distribution_history/' + str(self.epoch))
        np.savetxt(dir_path + '/error_distribution_history/' + str(self.epoch) + '/error_distribution_history.dat', np.array(self.quad_error_distr_hist)[0])
        
        if not os.path.exists(dir_path + '/test_error_distribution_history/' + str(self.epoch)): 
            os.makedirs(dir_path + '/test_error_distribution_history/' + str(self.epoch))
        np.savetxt(dir_path + '/test_error_distribution_history/' + str(self.epoch) + '/test_error_distribution_history.dat', np.array(self.quad_test_error_distr_hist)[0])
        

    def plot_errors_history(self, save=True, show=True):
        fig = plt.figure()
        ax_l = fig.add_subplot(111)
        ax_l.plot(self.hist_error_pde, label='pde_error', color='tab:orange')
        ax_l.plot(self.test_hist_error_pde, '--', label='pde_error', color='tab:orange')
        ax_r = ax_l.twinx()
        ax_r.plot(self.hist_error_bcs, label='bcs_error', color='tab:blue')
        ax_r.plot(self.test_hist_error_bcs, '--', label='bcs_error', color='tab:blue')
        ax_l.set_xlabel('Epochs')
        ax_l.set_ylabel('Errors_PDE', color='tab:orange')
        ax_r.set_ylabel('Errors_BCs', color='tab:blue')
        np.savetxt(dir_path + '/error_history.dat', (self.hist_error_pde, self.hist_error_bcs))
        if save: fig.savefig(dir_path + '/error_history.png')
        if show: plt.show()
        return


    def plot_solution(self, x, y, ax=None, save=True, show=True, **kwargs):
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        X_grid = tf.reshape(tf.stack([x, y], axis=1), [1, 1456])
        upred = self.model(X_grid)
        ax.plot_surface(x, y, upred, cmap='viridis', **kwargs)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$u$')
        if show: plt.show()
        return ax
        

    def plot_loss_history(self, ax=None, save=True, show=True):
        if not ax:
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111)
        ax.semilogy(range(len(self.hist)), self.hist,'k-')
        ax.semilogy(range(len(self.test_hist)), self.test_hist,'k--')
        np.savetxt(dir_path + '/loss_history.dat', self.hist)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Total Loss')
        fig.savefig(dir_path + '/loss_history.png')
        return ax


    def plot_model_weights(self, ep):
        # plot weights every 100th training step
        print(self.model.trainable_variables)
        kernel0 = self.model.trainable_variables[0][25:50]
        plt.imshow(kernel0)
        plt.colorbar()
        plt.savefig('Epoch: ', str(ep))
        plt.close()


    #@register
    def emergency_save_model(self):
        """ Save training data when training is interrupted """

        print('Training interrupted, saving training data.')
        self.model.save(dir_path + '/model', save_format="tf")
        np.savetxt(dir_path + '/loss_history.dat', self.hist)
        np.savetxt(dir_path + '/error_history.dat', (self.hist_error_pde, self.hist_error_bcs))


## graph implementation
def first_derivative(model, x, y):
    with tf.GradientTape(persistent=True) as tp:
        tp.watch(x)
        tp.watch(y)
        u = model(tf.stack([x, y], axis=1))

    u_x = tp.gradient(u, x)
    u_y = tp.gradient(u, y)
    del tp
    return u_x, u_y


def second_derivative(model, x, y):
    with tf.GradientTape(persistent=True) as tp:
        tp.watch(x)
        tp.watch(y)
        u = model(tf.stack([x, y], axis=1))

        u_x = tp.gradient(u, x)
        u_y = tp.gradient(u, y)
    u_xx = tp.gradient(u_x, x)
    u_yy = tp.gradient(u_y, y)
    del tp
    return u_xx, u_yy


class Quad(dde.geometry.Polygon):
    def __init__(self, vertices, sample_points):
        super().__init__(vertices)
        if sample_points != []:
            print('>>> Setting predetermined sample points')
            self.sample_points = sample_points
        else:
            print('>>> Setting random sample points')
            self.sample_points = self.random_points(500)
        ## sort sample points 
        self.num_bcs = None
        self.bcs, self.bcs_vals = self.set_bcs()
        self.train_points = [] # input to the PINN
        self.train_x_bc = self.bc_points()
        self.main_L = np.linalg.norm(self.vertices[0] - self.vertices[1])
        

    def set_bcs(self):
        from deepxde.geometry.geometry_2d import is_on_line_segment
        bcs = []
        bc_vals = []
        line_int = 0.0
        normal_sign = -1.0

        for i in range(-1, len(self.vertices)-1):
            # print("at indices " + str(i) + ", " + str(i+1) + " -> coords " + str(self.vertices[i]) + ", " + str(self.vertices[i+1]))
            
            if i == 0: # bottom edge
                L1 = np.linalg.norm(self.vertices[i] - self.vertices[i+1]) # need to haave this one inside 'if' because can't bind it locally in lambda
                # print("This is the source edge, L = " + str(L1))
                center = (self.vertices[i] + self.vertices[i+1]) / 2
                u_n = (normal_sign)*self.boundary_normal(center).T
                bc_val = 1/L1 + 1 / (2*self.area) * np.dot( center, u_n)
                bc_vals.append(bc_val)
                line_int += L1*bc_val
                # print(" BC val is " + str(bc_val))
                bc = dde.icbc.NeumannBC(super(),
                                    lambda x: 1/(L1) + 1 / (2*self.area) *  np.sum(x * (normal_sign)*self.boundary_normal(x), axis=1, keepdims=1), # note that this lambda works correctly only if L and S are not changing
                                    #lambda x: 1/(L) + 1 / (2*S) * np.dot( x, (normal_sign)*geom.boundary_normal(x).T),
                                    lambda X, on_boundary, vertices=self.vertices, i=i:  on_boundary and is_on_line_segment(vertices[i], vertices[i+1], X))

            else:
                L = np.linalg.norm(self.vertices[i] - self.vertices[i+1])
                center = (self.vertices[i] + self.vertices[i+1]) / 2
                u_n = (normal_sign)*self.boundary_normal(center).T
                bc_val = 1 / (2*self.area) * np.dot( center, u_n)
                bc_vals.append(bc_val)
                line_int += L*bc_val
                # print(" BC val is " + str(bc_val))
                bc = dde.icbc.NeumannBC(super(),
                        lambda x:  1 / (2*self.area) *  np.sum(x * (normal_sign)*self.boundary_normal(x), axis=1, keepdims=1),
                        #lambda x:  1 / (2*S) * np.dot( x, (normal_sign)*geom.boundary_normal(x).T),
                        lambda X, on_boundary, vertices=self.vertices, i=i: on_boundary and is_on_line_segment(vertices[i], vertices[i+1], X))
            
            
            bcs.append(bc)
        
        # print('Boundary line integral check = ' + str(line_int))
        return (bcs, bc_vals)


    def bc_points(self):
        x_bcs = [bc.collocation_points(self.sample_points) for bc in self.bcs]
        # print('x_bcs_0 ', x_bcs[0].shape)
        self.num_bcs = list(map(len, x_bcs))
        self.train_x_bc = (
            np.vstack(x_bcs)
            if x_bcs
            else np.empty([0, self.sample_points.shape[-1]], dtype=DTYPE)
        )
        self.train_points = np.vstack((self.train_x_bc, self.sample_points))
        return self.train_x_bc


    def plot_B(self, model, fig=None, ax=None, save=False, show=False):
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        # print('sample points ', self.train_points.shape)
        #du = first_derivative(u, tf.convert_to_tensor(self.train_points, dtype=DTYPE))
        x = tf.convert_to_tensor(self.train_points[:,0], dtype=DTYPE)
        y = tf.convert_to_tensor(self.train_points[:,1], dtype=DTYPE)
        dudx, dudy = first_derivative(model, x, y)
        # print('first derivative ', dudx.shape)
        U = -(1/(2*self.area)*self.train_points[:,0] + dudx)
        V = -(1/(2*self.area)*self.train_points[:,1] + dudy)

        # projection =  abs(np.sum(np.array([U,V]).T * self.boundary_normal(self.train_points), axis=1, keepdims=1))
        # sctr = ax.scatter(self.train_points[:,0], self.train_points[:,1], c=projection, s=0.8)
        # fig.colorbar(sctr, ax=ax)
        # plt.title("BC values, 1/L = " + str(1/self.main_L))
        # if save: plt.savefig('BC_values.png')
        # if show: plt.show()

        colour = np.sqrt(U**2 + V**2)
        #sctr = ax.scatter(data.test_x[:,0], data.test_x[:,1], s=1.0, c=proj)
        qvr = ax.quiver(self.train_points[:,0], self.train_points[:,1], U, V, colour, width = 0.01)
        fig.colorbar(qvr, ax=ax)
        ax.set_title("B(r)")# 1/L =", 1/self.main_L)
        if save: plt.savefig(dir_path + '/B_r.png')
        if show: plt.show() 
        return ax


    def plot_error_distribution(self, model, fig=None, ax=None, save=False, show=False):
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        x = tf.convert_to_tensor(self.train_points[:,0], dtype=DTYPE)
        y = tf.convert_to_tensor(self.train_points[:,1], dtype=DTYPE)
        u_x, u_y = first_derivative(model, x, y)
        u_xx, u_yy = second_derivative(model, x, y)
        domain_error = u_xx + u_yy
        bcs_start = np.cumsum([0] + self.num_bcs)
        
        bcs_pts = []
        bc_errors = []
        for j, bc in enumerate(self.bcs):
            beg, end = bcs_start[j], bcs_start[j + 1]
            pts = self.train_points[beg:end]
            bcs_pts.extend(pts)
            bc_val = self.bcs_vals[j]
            bc_val_predict = np.sum(np.vstack((u_x, u_y)).T[beg:end] * self.boundary_normal(pts), axis=1, keepdims=1)
            bc_errors.extend(abs(bc_val-bc_val_predict))
        
        # print(domain_pts.shape, np.array(bcs_pts).shape, prediction_error_domain.shape, np.array(bc_errors).shape)
        # print('bc_errors.shape ', np.array(bc_errors).shape)
        # print('domain_error.shape ', domain_error.shape)
        #error_distribution = np.vstack((np.array(np.array(bc_errors), domain_error)))
        # print(np.array(bc_errors)[:,0].shape)
        # print(domain_error[0:len(bc_errors)].shape)
        error_distribution = np.hstack((np.array(bc_errors)[:,0], domain_error[len(bc_errors)::]))
        # np.savetxt(dir_path + 'predict_error.dat', np.c_[geom_pts, geom_errors])
        # with open(dir_path + 'predict_error_stats_over_meps.txt', "ab") as f:
        #     np.savetxt(f, np.c_[np.mean(geom_errors), np.max(geom_errors), np.min(geom_errors), np.std(geom_errors)])
        
        # print("Mean prediction error on test points: %.3e" % np.mean(error_distribution))
        # print("Median prediction error on test points: %.3e" % np.median(error_distribution))

        sctr = ax.scatter(self.train_points[:,0], self.train_points[:,1], s=1.0, c=abs(error_distribution))
        fig.colorbar(sctr, ax=ax)
        ax.set_title("Test points prediction error")
        # plt.savefig(dir_path + 'bc_error_' + str(mep) + '.png')
        if show: plt.show()
        return ax



def plot_quad_Bs(dde_polys, model, save=True, show=True):
    fig, axes = plt.subplots(5, round(len(dde_polys)/5))
    for i, (ax, poly) in enumerate(zip(axes.flat, dde_polys)):
        ax = poly.plot_B(model, fig=fig, ax=ax)
        ax.set_title(str(i+1))
    if show: plt.show()
    return


def plot_quad_errors(dde_polys, model, save=True, show=True):
    fig, axes = plt.subplots(5, round(len(dde_polys)/5))
    for i, (ax, poly) in enumerate(zip(axes.flat, dde_polys)):
        ax = poly.plot_error_distribution(model, fig=fig, ax=ax)
        ax.set_title(str(i+1))
    if show: plt.show()
    return


def plot_history(dir_path):
    (error_hist_pde, error_hist_bcs) = np.genfromtxt(dir_path + '/error_history.dat')
    loss_hist = np.genfromtxt(dir_path + '/loss_history.dat')
    test_loss_hist = np.genfromtxt(dir_path + '/test_loss_history.dat')


    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111)
    ax.semilogy(loss_hist,'k-')
    ax.semilogy(test_loss_hist, 'k--')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Total Loss')
    fig.savefig(dir_path + '/loss_history.png')

    fig = plt.figure()
    ax_l = fig.add_subplot(111)

    ax_l.plot(error_hist_pde, label='pde_error', color='tab:orange')
    ax_r = ax_l.twinx()
    ax_r.plot(error_hist_bcs, label='bcs_error', color='tab:blue')
    ax_l.set_xlabel('Epochs')
    ax_l.set_ylabel('Errors_PDE', color='tab:orange')
    ax_r.set_ylabel('Errors_BCs', color='tab:blue')
    fig.savefig(dir_path + '/error_history.png')
    plt.show()
    return


def plot_error_history(dir_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    test_error_distr_hist = []
    train_error_distr_hist = []
    ephs = []
    show_every = 1
    for train_ep_dir, test_ep_dir in zip(os.listdir(dir_path + '/error_distribution_history/'), os.listdir(dir_path + '/test_error_distribution_history/')):
        ephs.append(int(test_ep_dir))

        train_error_distribution_data = np.genfromtxt(dir_path + '/error_distribution_history/' + train_ep_dir + '/error_distribution_history.dat')
        train_error_distr_hist.append(np.abs(train_error_distribution_data.flatten()))

        test_error_distribution_data = np.genfromtxt(dir_path + '/test_error_distribution_history/' + test_ep_dir + '/test_error_distribution_history.dat')
        test_error_distr_hist.append(np.abs(test_error_distribution_data.flatten()))

    ephs=np.array(ephs)
    ax.boxplot(train_error_distr_hist[::show_every], positions=ephs[::show_every], showfliers=False, widths=10)
    ax.boxplot(test_error_distr_hist[::show_every], positions=ephs[::show_every]+20, showfliers=False, widths=10)
    ## if ticks get cramped:
    # for i, label in enumerate(ax.xaxis.get_ticklabels()):
    #     if i%2 != 0: label.set_visible(False)
    # for i, line in enumerate(ax.xaxis.get_ticklines()):
    #     if i%2 != 0: line.set_visible(False)

    plt.show()
    return


def get_results():
    select_every = 2
    quads_vertices = np.genfromtxt(load_model_dir_path + '/' + quads_filename)[0::select_every]
    quads_vertices = quads_vertices.reshape(len(quads_vertices), 4, 2)

    bary_coords = np.genfromtxt(bary_coords_path)
   
    dde_polys = []
    for qv in quads_vertices:
        sample_points = np.dot(bary_coords, qv)
        quad_el = Quad(qv, sample_points)
        dde_polys.append(quad_el)

    sh = dde_polys[0].train_points.shape

    model = PINN_NeuralNet(output_dim=1,
                        num_hidden_layers=NN_DEPTH, 
                        num_neurons_per_layer=NN_WIDTH,
                        activation='tanh',
                        kernel_initializer='glorot_normal',)
    
    model.build(input_shape=(None, sh[0], sh[1]))

    ####### load model
    model = tf.keras.models.load_model(load_model_dir_path + 'model/', 
                                            compile=False,
                                            custom_objects={"PINN_NeuralNet": PINN_NeuralNet}
                                            )

    ####### model predict
    # plot_quad_Bs(dde_polys, model)
    # plot_quad_errors(dde_polys, model)
    if plot_histories:
        plot_history(load_model_dir_path)
    
    plot_error_history(load_model_dir_path)


def train_model():
    shutil.copy(quads_path, dir_path+'/')
    quads_vertices = np.genfromtxt(quads_path)
    quads_vertices = quads_vertices.reshape(len(quads_vertices), 4, 2)

    bary_coords = np.genfromtxt(bary_coords_path)

    dde_polys = []
    collocation_pts = []
    for qv in quads_vertices:
        sample_points = np.dot(bary_coords, qv)
        quad_el = Quad(qv, sample_points)
        collocation_pts.append(quad_el.train_points)
        dde_polys.append(quad_el)

    collocation_pts = np.array(collocation_pts)
    sh = collocation_pts.shape

    print('col points shape: ',  collocation_pts.shape)

    if load_pretrained_model:
        print('>>> Loading pretrained model')
        model = tf.keras.models.load_model(load_model_dir_path + 'model/', 
                                            compile=False,
                                            custom_objects={"PINN_NeuralNet": PINN_NeuralNet})
    else:
        print('>>> Creating new model')
        model = PINN_NeuralNet(output_dim=1,
                                    num_hidden_layers=NN_DEPTH, 
                                    num_neurons_per_layer=NN_WIDTH,
                                    activation='tanh',
                                    kernel_initializer='glorot_normal')


    model.build(input_shape=(None, sh[1], sh[2]))
    # tf.keras.utils.plot_model(model, show_dtype=True, 
    #                    show_layer_names=True, show_shapes=True,  
    #                    to_file='model.png')


    print(model.summary())

    solver = PINNSolver(model, collocation_pts, dde_polys)

    # Choose optimizer
    # lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000,3000],[1e-2,1e-3,5e-4])
    lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,
                                                        decay_steps=1000,
                                                        decay_rate=0.9)
    ### parameter names
    # names = [weight.name for layer in model.layers for weight in layer.weights]
    
    # ####### TRAIN
    
    optim = tf.keras.optimizers.Adam(learning_rate=lr)
    solver.solve_with_TFoptimizer_batches(optim, epochs=EPOCHS)

    # solver.solve_with_ScipyOptimizer()
    solver.model.save(dir_path + '/model', save_format="tf")
    solver.plot_loss_history()
    solver.plot_errors_history()


def main():

    if train:
        print('***********Training******************')
        train_model()
    else:
        print('*********Loading model***************')
        get_results()

if __name__=="__main__":
    main()