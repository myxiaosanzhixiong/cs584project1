import numpy as np
from scipy import linalg


class LassoHomotopyModel():
    """
    实现使用Homotopy方法的LASSO回归。

    Homotopy方法是一种高效算法，用于计算LASSO的整个正则化路径。
    它通过从使所有系数为零的最高正则化参数值开始，
    然后逐渐减小该参数，跟踪活动集（非零系数）的变化。

    参数:
    ------
    max_iter : int, 默认=1000
        Homotopy路径的最大迭代次数
    tol : float, 默认=1e-6
        收敛的容差
    lambda_max : float, 可选
        正则化参数的最大值。如果为None，将计算为特征与目标间的最大相关性。
    lambda_min_ratio : float, 默认=1e-6
        lambda_min/lambda_max的比率，决定路径结束的位置
    standardize : bool, 默认=False
        是否在拟合前标准化特征
    """

    def __init__(self, max_iter=1000, tol=1e-6, lambda_max=None, lambda_min_ratio=1e-6, standardize=False):
        """初始化LASSO Homotopy模型。"""
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_max = lambda_max
        self.lambda_min_ratio = lambda_min_ratio
        self.standardize = standardize

        # 存储内部状态
        self.coef_ = None
        self.active_set_ = None
        self.lambda_path_ = None
        self.coef_path_ = None
        self.intercept_ = 0.0
        self.feature_means_ = None
        self.feature_stds_ = None
        self.target_mean_ = None

    def _standardize_data(self, X, y):
        """标准化特征和目标"""
        # 保存原始特征均值和标准差，用于后续转换系数
        self.feature_means_ = np.mean(X, axis=0)
        self.feature_stds_ = np.std(X, axis=0)
        self.feature_stds_[self.feature_stds_ == 0] = 1.0  # 避免除以零

        # 标准化特征
        X_scaled = (X - self.feature_means_) / self.feature_stds_

        # 只中心化目标（不缩放）
        self.target_mean_ = np.mean(y)
        y_centered = y - self.target_mean_

        return X_scaled, y_centered

    def fit(self, X, y):
        """
        使用Homotopy方法拟合LASSO模型。

        参数:
        ------
        X : array-like, shape (n_samples, n_features)
            训练数据
        y : array-like, shape (n_samples,)
            目标值

        返回:
        ------
        results : LassoHomotopyResults
            包含拟合模型结果的对象
        """
        # 转换输入为numpy数组
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # 如果需要，对y进行重塑
        if y.ndim > 1:
            y = y.ravel()

        # 如果需要，标准化数据
        if self.standardize:
            X_scaled, y_centered = self._standardize_data(X, y)
        else:
            X_scaled = X.copy()
            y_centered = y.copy()
            self.feature_means_ = np.zeros(X.shape[1])
            self.feature_stds_ = np.ones(X.shape[1])
            self.target_mean_ = 0.0

        n_samples, n_features = X_scaled.shape

        # 计算lambda_max（如果未提供）
        correlation = np.abs(X_scaled.T @ y_centered)
        if self.lambda_max is None:
            self.lambda_max = np.max(correlation)

        if self.lambda_max <= 0:
            raise ValueError("lambda_max必须为正值，得到 {}".format(self.lambda_max))

        # 初始化lambda和活动集
        lambda_current = self.lambda_max
        lambda_min = self.lambda_max * self.lambda_min_ratio

        # 初始化beta（系数）为零
        beta = np.zeros(n_features)

        # 初始化活动集及其符号
        active_set = []
        active_signs = []

        # 存储lambda和系数路径
        lambda_path = [lambda_current]
        coef_path = [beta.copy()]

        # Homotopy算法
        for iteration in range(self.max_iter):
            # 计算残差
            residual = y_centered - X_scaled @ beta

            # 计算相关性
            correlation = X_scaled.T @ residual

            # 如果是第一次迭代或者需要添加变量到活动集
            if len(active_set) == 0:
                # 找到具有最大绝对相关性的特征
                j = np.argmax(np.abs(correlation))
                active_set.append(j)
                active_signs.append(np.sign(correlation[j]))

            # 活动集矩阵
            X_active = X_scaled[:, active_set]

            # 使用活动集计算方向
            signs = np.array(active_signs)

            try:
                # 使用更稳健的矩阵计算方法
                gram_matrix = X_active.T @ X_active

                # 添加小正则化项以提高数值稳定性
                reg_term = self.tol * np.eye(gram_matrix.shape[0])
                gram_matrix_reg = gram_matrix + reg_term

                # 使用伪逆以提高稳定性
                inverse_gram = linalg.pinv(gram_matrix_reg)
                direction = inverse_gram @ signs

                # 计算特征空间中的方向
                delta_beta = np.zeros(n_features)
                for i, idx in enumerate(active_set):
                    delta_beta[idx] = direction[i]

                # 计算lambda减少的步长
                delta_correlation = X_scaled.T @ (X_scaled @ delta_beta)

                # 计算变量进入的步长
                lambda_gamma = []
                for j in range(n_features):
                    if j not in active_set:
                        if abs(delta_correlation[j]) > self.tol:  # 阈值以避免数值问题
                            gamma1 = (lambda_current - correlation[j]) / (delta_correlation[j])
                            if gamma1 > 0:
                                lambda_gamma.append((gamma1, j, 1))  # 1表示添加到活动集

                            gamma2 = (lambda_current + correlation[j]) / (delta_correlation[j])
                            if gamma2 > 0:
                                lambda_gamma.append((gamma2, j, -1))  # -1表示以负号添加

                # 计算变量离开活动集的步长
                beta_gamma = []
                for i, idx in enumerate(active_set):
                    if delta_beta[idx] * active_signs[i] < 0:  # 如果方向与符号相反
                        gamma = -beta[idx] / delta_beta[idx]
                        if gamma > 0:
                            beta_gamma.append((gamma, i, 0))  # 0表示从活动集中移除

                # 合并并排序步长
                gamma_list = lambda_gamma + beta_gamma
                if not gamma_list:
                    # 没有更多事件，尝试小步长并检查收敛
                    small_step = lambda_current * 0.1
                    beta_new = beta + small_step * delta_beta
                    if np.max(np.abs(beta_new - beta)) < self.tol:
                        break  # 已收敛
                    else:
                        lambda_current -= small_step
                        beta = beta_new
                else:
                    min_gamma, min_idx, min_type = min(gamma_list)

                    # 更新beta
                    beta += min_gamma * delta_beta

                    # 更新lambda
                    lambda_current -= min_gamma

                    # 更新活动集
                    if min_type == 0:  # 从活动集中移除
                        i_remove = min_idx
                        idx_remove = active_set[i_remove]
                        beta[idx_remove] = 0  # 将系数置零

                        # 从活动集和符号中移除
                        active_set.pop(i_remove)
                        active_signs.pop(i_remove)

                    elif min_type in [1, -1]:  # 添加到活动集
                        j_add = min_idx
                        active_set.append(j_add)
                        active_signs.append(min_type)

                # 存储当前lambda和系数
                lambda_path.append(lambda_current)
                coef_path.append(beta.copy())

                # 检查lambda是否足够小以终止
                if lambda_current <= lambda_min:
                    break

            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"矩阵计算错误: {e}")
                # 如果计算失败，使用当前beta并终止
                break

        # 存储结果
        self.coef_ = beta.copy()  # 确保是副本
        self.active_set_ = active_set.copy()  # 确保是副本
        self.lambda_path_ = np.array(lambda_path)
        self.coef_path_ = np.array(coef_path)

        # 如果数据已标准化，计算截距和转换系数回到原始比例
        if self.standardize:
            self._rescale_coefficients()
        else:
            # 如果没有标准化，直接计算截距
            self.intercept_ = np.mean(y - X @ self.coef_)

        return LassoHomotopyResults(self)

    def _rescale_coefficients(self):
        """将标准化系数转换回原始比例"""
        # 首先计算截距项
        self.intercept_ = self.target_mean_
        self.intercept_ -= np.sum(self.coef_ * self.feature_means_ / self.feature_stds_)

        # 然后将系数转换回原始尺度
        self.coef_ = self.coef_ / self.feature_stds_

        # 同样转换系数路径
        if hasattr(self, 'coef_path_'):
            path_intercepts = []
            for coef in self.coef_path_:
                intercept = self.target_mean_
                intercept -= np.sum(coef * self.feature_means_ / self.feature_stds_)
                path_intercepts.append(intercept)
            self.intercept_path_ = np.array(path_intercepts)

            # 将每个路径点的系数转换回原始比例
            for i in range(len(self.coef_path_)):
                self.coef_path_[i] = self.coef_path_[i] / self.feature_stds_


class LassoHomotopyResults():
    """存储LASSO Homotopy模型拟合结果的类。"""

    def __init__(self, model):
        """用模型参数初始化。"""
        self.coef_ = model.coef_
        self.active_set_ = model.active_set_
        self.lambda_path_ = model.lambda_path_
        self.intercept_ = model.intercept_

        if hasattr(model, 'coef_path_'):
            self.coef_path_ = model.coef_path_

        if hasattr(model, 'intercept_path_'):
            self.intercept_path_ = model.intercept_path_

    def predict(self, X):
        """使用拟合的LASSO模型进行预测。"""
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """
        返回预测的R^2分数。

        R^2 = 1 - SS_res / SS_tot
        其中SS_res是残差平方和，SS_tot是总平方和。
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

    def summary(self):
        """返回模型的摘要信息"""
        active_coeffs = [(i, self.coef_[i]) for i in self.active_set_]

        print("LASSO Homotopy 模型摘要:")
        print(f"活动特征数量: {len(self.active_set_)}/{len(self.coef_)}")
        print(f"最终lambda值: {self.lambda_path_[-1]:.6f}")
        print(f"截距: {self.intercept_:.6f}")
        print("\n活动系数:")
        for idx, coef in active_coeffs:
            print(f"特征 {idx}: {coef:.6f}")

        return {
            "active_features": len(self.active_set_),
            "total_features": len(self.coef_),
            "final_lambda": self.lambda_path_[-1],
            "intercept": self.intercept_,
            "active_coefficients": active_coeffs
        }
