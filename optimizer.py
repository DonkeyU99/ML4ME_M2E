import json
import os
import cv2

import numpy as np
import numpy.fft as fft
import scipy.signal
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from local_prior import smooth_region
from utils import compute_gradient, Phi_func, toeplitz_matrix
from optimizier_utils import gradient_filter, omega


class Optimizer:
    def __init__(self, image, kernel_size, sigma=None, max_iterations=15):
        self.I = image
        self.I_flat = self.I.reshape(-1, 3).T
        self.I_grad_x = compute_gradient(image, "x")
        self.I_grad_y = compute_gradient(image, "y")
        self.I_grad_x_flat = self.I_grad_x.reshape(-1, 3).T
        self.I_grad_y_flat = self.I_grad_y.reshape(-1, 3).T
        self.I_grad_xx_flat = compute_gradient(image, "xx").reshape(-1, 3).T
        self.I_grad_xy_flat = compute_gradient(image, "xy").reshape(-1, 3).T
        self.I_grad_yy_flat = compute_gradient(image, "yy").reshape(-1, 3).T
        self.kernel_size = kernel_size

        self.height = image.shape[0] + kernel_size - 1
        self.width = image.shape[1] + kernel_size - 1
        self.F_I = fft.fft2(self.I, (self.height, self.width), axes=(0, 1))

        # initialize L, psi_X, psi_y
        self.L = image
        self.Psi_x = compute_gradient(self.L, "x")
        self.Psi_y = compute_gradient(self.L, "y")
        self.f = np.diag((1 / kernel_size) * np.full(kernel_size, 1))
        self.f_flat = self.f.flatten()

        """
        Hyperparameters
        usually 1/(ζ**2 * τ) = 50  
        """
        self.delta_f = None
        self.delta_L = None
        self.delta_Psi_x = None
        self.delta_Psi_y = None

        self.zeta_0 = 1  # TODO
        # tau : 2 ~ 500
        self.tau = 100  # TODO
        self.gamma = 10  # TODO

        # 0.002 ~ 0.5 / 10 ~ 25
        # self.lambda_1 = 1/self.tau
        # self.lambda_2 = 1/(sigma**2*self.tau)

        self.lambda_1 = 0.5
        self.lambda_2 = 25

        self.k1 = 1.3
        self.k2 = 1.5

        self.weight = np.array(
            [omega(grad) for grad in ["0", "x", "y", "xx", "xy", "yy"]]
        )
        self.sigma_star = np.array(
            [
                fft.fft2(gradient_filter(grad), (self.height, self.width))
                for grad in ["0", "x", "y", "xx", "xy", "yy"]
            ]
        )
        self.delta = np.einsum(
            "i,ijk->jk",
            self.weight,
            np.einsum("ijk,ijk->ijk", self.sigma_star, np.conjugate(self.sigma_star)),
        )

        self.max_iterations = max_iterations
        self.threshold_smooth_region = np.array([5, 5, 5])
        self.threshold_Phi_func = 5

    def target_equation_x(self, x, mask_3d):
        return (
            self.lambda_1 * np.abs(Phi_func(x, self.threshold_Phi_func))
            + self.lambda_2 * mask_3d * (x - self.I_grad_x) ** 2
            + self.gamma * (x - compute_gradient(self.L, "x")) ** 2
        )

    def target_equation_y(self, x, mask_3d):
        return (
            self.lambda_1 * np.abs(Phi_func(x, self.threshold_Phi_func))
            + self.lambda_2 * mask_3d * (x - self.I_grad_y) ** 2
            + self.gamma * (x - compute_gradient(self.L, "y")) ** 2
        )

    def get_argmin_Psi(self, mask_3d):
        a = 6.1e-4
        b = 5
        k = 2.7
        L_x = compute_gradient(self.L, "x")
        positive_solution_x = (
            self.lambda_2 * mask_3d * self.I_grad_x
            + self.gamma * L_x
            + (k * self.lambda_1 / 2)
        ) / (
            self.lambda_2 * mask_3d + self.gamma
        )  # - threshold < phi < 0일 때
        negative_solution_x = (
            self.lambda_2 * mask_3d * self.I_grad_x
            + self.gamma * L_x
            - (k * self.lambda_1 / 2)
        ) / (
            self.lambda_2 * mask_3d + self.gamma
        )  # 0 < phi < threshold 일 때
        quadratic_solution_x = (
            self.lambda_2 * mask_3d * self.I_grad_x + self.gamma * L_x
        ) / (
            self.lambda_2 * mask_3d + self.gamma + self.lambda_1 * a
        )  # |phi| > threshold 일 때

        positive_values = self.target_equation_x(positive_solution_x, mask_3d)
        negative_values = self.target_equation_x(negative_solution_x, mask_3d)
        quadratic_values = self.target_equation_x(quadratic_solution_x, mask_3d)

        # Find the index (0, 1, or 2) of the array that gives the minimum value at each position
        # Use the index to select the corresponding values from the three arrays
        solution_x = np.choose(
            np.argmin([positive_values, negative_values, quadratic_values], axis=0),
            [positive_solution_x, negative_solution_x, quadratic_solution_x],
        )
        print(self.Psi_x.shape)
        print(solution_x.shape)
        new_Psi_x = np.zeros_like(self.Psi_x)
        new_Psi_x = np.minimum(self.Psi_x, solution_x, out=new_Psi_x)
        # print(solution_x.size)

        L_y = compute_gradient(self.L, "y")
        positive_solution_y = (
            self.lambda_2 * mask_3d * self.I_grad_y
            + self.gamma * L_y
            + (k * self.lambda_1 / 2)
        ) / (
            self.lambda_2 * mask_3d + self.gamma
        )  # - threshold < phi < 0일 때
        negative_solution_y = (
            self.lambda_2 * mask_3d * self.I_grad_y
            + self.gamma * L_y
            - (k * self.lambda_1 / 2)
        ) / (
            self.lambda_2 * mask_3d + self.gamma
        )  # 0 < phi < threshold 일 때
        quadratic_solution_y = (
            self.lambda_2 * mask_3d * self.I_grad_y + self.gamma * L_y
        ) / (
            self.lambda_2 * mask_3d + self.gamma + self.lambda_1 * a
        )  # |phi| > threshold 일 때

        positive_values = self.target_equation_y(positive_solution_y, mask_3d)
        negative_values = self.target_equation_y(negative_solution_y, mask_3d)
        quadratic_values = self.target_equation_y(quadratic_solution_y, mask_3d)

        # Find the index (0, 1, or 2) of the array that gives the minimum value at each position
        # Use the index to select the corresponding values from the three arrays
        solution_y = np.choose(
            np.argmin([positive_values, negative_values, quadratic_values], axis=0),
            [positive_solution_y, negative_solution_y, quadratic_solution_y],
        )
        new_Psi_y = np.zeros_like(self.Psi_y)
        new_Psi_y = np.minimum(self.Psi_y, solution_y, out=new_Psi_y)
        return new_Psi_x, new_Psi_y

    def update_Psi(self):
        smth_bln = smooth_region(self.L, self.kernel_size, self.threshold_smooth_region)
        mask = 1 * smth_bln
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        new_Psi_x, new_Psi_y = self.get_argmin_Psi(mask_3d)
        new_Psi_x[~smth_bln] = np.array(self.Psi_x[~smth_bln])
        new_Psi_y[~smth_bln] = np.array(self.Psi_y[~smth_bln])

        self.delta_Psi_x = new_Psi_x - self.Psi_x
        self.delta_Psi_y = new_Psi_y - self.Psi_y

        self.Psi_x = new_Psi_x
        self.Psi_y = new_Psi_y

    def update_L(self):
        F_psi_x = fft.fft2(self.Psi_x, (self.height, self.width), axes=(0, 1))
        F_psi_y = fft.fft2(self.Psi_y, (self.height, self.width), axes=(0, 1))

        F_f = fft.fft2(
            self.f, (self.height, self.width), axes=(0, 1)
        )  # 3x3 -> (i+f-1)x(i+f-1)
        self.F_I  # 30x30x3(for img 28, kernel 3)
        self.delta  # 3x3

        denom = np.conjugate(F_f) * F_f * self.delta + self.gamma * (
            np.conjugate(self.sigma_star[1]) * self.sigma_star[1]
            + np.conjugate(self.sigma_star[2]) * self.sigma_star[2]
        )
        new_L = np.zeros((self.height, self.width, 3))
        for i in range(3):
            numer = np.conjugate(F_f) * self.F_I[:, :, i] * self.delta + self.gamma * (
                np.conjugate(self.sigma_star[1]) * F_psi_x[:, :, i]
                + np.conjugate(self.sigma_star[2]) * F_psi_y[:, :, i]
            )

            new_L[:, :, i] = np.abs(
                fft.ifft2(numer / denom, (self.height, self.width), axes=(0, 1))
            )
        # image cropping
        st = (self.kernel_size - 1) // 2 - 1
        ed = self.I.shape[0]
        ed_ = self.I.shape[1]  # if image is not square
        new_L = new_L[st : ed + st, st : ed_ + st, :]
        self.delta_L = new_L - self.L
        self.L = new_L
        # print("delta_L:", self.delta_L.shape)
        # print("self.L:", self.L.shape)

    def update_f(self):
        self.f_flat = self.f.flatten()
        A0 = toeplitz_matrix(self.L, self.kernel_size)
        Ax = toeplitz_matrix(
            compute_gradient(self.L, "x"), kernel_size=self.kernel_size
        )
        Ay = toeplitz_matrix(
            compute_gradient(self.L, "y"), kernel_size=self.kernel_size
        )
        # Axx = toeplitz_matrix(
        #    compute_gradient(self.L, "xx"), kernel_size=self.kernel_size
        # )
        # Axy = toeplitz_matrix(
        #    compute_gradient(self.L, "xy"), kernel_size=self.kernel_size
        # )
        # Ayy = toeplitz_matrix(
        #    compute_gradient(self.L, "yy"), kernel_size=self.kernel_size
        # )
        B0 = self.I_flat
        Bx = self.I_grad_x_flat
        By = self.I_grad_y_flat
        # Bxx = self.I_grad_xx_flat
        # Bxy = self.I_grad_xy_flat
        # Byy = self.I_grad_yy_flat
        print("Toeplitz construction Done")
        # print(A0.shape)
        # print(self.f.shape)
        # print(B0.shape)
        # print(A0@self.f.flatten())
        # print((A0@self.f.flatten()).shape)

        # left_mat = 0
        # right_mat = 0
        # for i in [(A0,B0),(Ax,Bx),(Ay,By)]:
        #    for j in range(3):
        #        left_mat += i[0][:,:,j].T@i[0][:,:,j]
        #        right_mat += i[1][:,:,j].T@i[1][:,:,j]
        # left_mat += np.identity(left_mat.shape[0])
        #
        # self.f_flat = np.inv(left_mat)@right_mat

        objective_fun = (
            lambda x: omega("0") * np.linalg.norm(A0 @ x - B0)
            + omega("x") * np.linalg.norm(Ax @ x - Bx)
            + omega("y") * np.linalg.norm(Ay @ x - By)
            # + omega("xx") * np.linalg.norm(Axx @ x - Bxx)
            # + omega("xy") * np.linalg.norm(Axy @ x - Bxy)
            # + omega("yy") * np.linalg.norm(Ayy @ x - Byy)
            + np.sum(np.abs(x))
        )
        initial_guess = self.f.flatten()

        self.f_flat = minimize(objective_fun, initial_guess, method="BFGS").x
        print("Optimization Done")
        new_f = self.f_flat.reshape((self.kernel_size, self.kernel_size))
        self.delta_f = new_f - self.f
        self.f = new_f

    def optimize(self):
        print("------- start optimizing image -------")
        iteration = 0
        json_file_path = "./fig_save/results.json"

        if os.path.exists(json_file_path):
            os.remove(json_file_path)
        # Inner loop to optimize L
        plt.imshow(self.L.astype(np.uint8))
        plt.savefig(f"./fig_save/fig_start.jpg")
        plt.imshow(self.f, cmap="gray")
        plt.savefig(f"./fig_save/kernel_start.jpg")
        while iteration < self.max_iterations:
            print(f"------- Iteraton {iteration} -------")
            rep = 0
            while True:  # while(norm_L > 1e-5):
                print("------Updating Psi-------")
                self.update_Psi()
                norm_delta_psi_x = np.linalg.norm(self.delta_Psi_x)
                norm_delta_psi_y = np.linalg.norm(self.delta_Psi_y)
                norm_delta_psi = norm_delta_psi_x + norm_delta_psi_y
                print(f"Norm of delta_Psi : {norm_delta_psi}")

                print("------Updating L-------")
                self.update_L()
                norm_delta_L = np.linalg.norm(self.delta_L)
                print(f"Norm of delta L: {norm_delta_L}")

                plt.imshow(self.L.astype(np.uint8))
                plt.savefig(f"./fig_save/fig{iteration}_{rep}.jpg")
                result_data = {
                    "iteration": iteration,
                    "rep": rep,
                    "norm_delta_psi_x": float(norm_delta_psi_x),
                    "norm_delta_psi_y": float(norm_delta_psi_y),
                    "norm_delta_L": float(norm_delta_L),
                    "gamma": float(self.gamma),
                    "lambda_1": float(self.lambda_1),
                    "lambda_2": float(self.lambda_2),
                }
                with open(json_file_path, "a", newline="") as json_file:
                    json.dump(result_data, json_file, indent=2)
                    json_file.write(",\n")
                rep += 1
                if (norm_delta_psi < 1.0e-1 and norm_delta_L < 2.0e-1) or rep > 10:
                    break

            print(f"------- Iteraton {iteration} finished -------")
            print("-----updating f------")
            # Update f
            self.update_f()
            norm_delta = np.linalg.norm(self.delta_f)
            print(f"f_delta : {norm_delta}")
            plt.imshow(self.f, cmap="gray")
            plt.savefig(f"./fig_save/kernel_{iteration}.jpg")
            if norm_delta < 1e-5:
                break

            self.gamma *= 2
            self.lambda_1 /= self.k1
            self.lambda_2 /= self.k2
            iteration += 1
        # Return L and f after optimization
        return self.L, self.f


img = cv2.imread("data/toy_dataset/6_HUAWEI-MATE20_M_small.JPG")

# img = np.random.randint(0, 256, (28, 28, 3)).astype(float)

a = Optimizer(img, 7, max_iterations=10)
# a.optimize()
# update L & update psi 부분 확인


L, f = a.optimize()
print(L)
deblurred_image = L.astype(np.uint8)
# plt.imshow(deblurred_image)
# plt.show()
