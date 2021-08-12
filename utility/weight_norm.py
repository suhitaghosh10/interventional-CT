# This code is from following repository
# https://github.com/openai/weightnorm/blob/master/keras/weightnorm.py

from utility.common_imports import *
from keras.optimizers import Adam


# adapted from keras.optimizers.Adam
class AdamWithWeightnorm(Adam):
    # def get_updates(self, params, constraints, loss):
    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [tfkb.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * tfkb.cast(self.iterations,
                                                     tfkb.dtype(self.decay))))

        t = tfkb.cast(self.iterations, tfkb.floatx()) + 1
        lr_t = lr * tfkb.sqrt(1. - tfkb.pow(self.beta_2, t)) / (1. - tfkb.pow(self.beta_1, t))

        shapes = [tfkb.get_variable_shape(p) for p in params]
        ms = [tfkb.zeros(shape) for shape in shapes]
        vs = [tfkb.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):

            # if a weight tensor (len > 1) use weight normalized parameterization
            # this is the only part changed w.r.t. keras.optimizers.Adam
            ps = tfkb.get_variable_shape(p)
            if len(ps)>1:

                # get weight normalization parameters
                V, V_norm, V_scaler, g_param, grad_g, grad_V = get_weightnorm_params_and_grads(p, g)

                # Adam containers for the 'g' parameter
                V_scaler_shape = tfkb.get_variable_shape(V_scaler)
                m_g = tfkb.zeros(V_scaler_shape)
                v_g = tfkb.zeros(V_scaler_shape)

                # update g parameters
                m_g_t = (self.beta_1 * m_g) + (1. - self.beta_1) * grad_g
                v_g_t = (self.beta_2 * v_g) + (1. - self.beta_2) * tfkb.square(grad_g)
                new_g_param = g_param - lr_t * m_g_t / (tfkb.sqrt(v_g_t) + self.epsilon)
                self.updates.append(tfkb.update(m_g, m_g_t))
                self.updates.append(tfkb.update(v_g, v_g_t))

                # update V parameters
                m_t = (self.beta_1 * m) + (1. - self.beta_1) * grad_V
                v_t = (self.beta_2 * v) + (1. - self.beta_2) * tfkb.square(grad_V)
                new_V_param = V - lr_t * m_t / (tfkb.sqrt(v_t) + self.epsilon)
                self.updates.append(tfkb.update(m, m_t))
                self.updates.append(tfkb.update(v, v_t))

                # if there are constraints we apply them to V, not W
                # if p in constraints:
                #     c = constraints[p]
                #     new_V_param = c(new_V_param)

                # wn param updates --> W updates
                add_weightnorm_param_updates(self.updates, new_V_param, new_g_param, p, V_scaler)

            else: # do optimization normally
                m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
                v_t = (self.beta_2 * v) + (1. - self.beta_2) * tfkb.square(g)
                p_t = p - lr_t * m_t / (tfkb.sqrt(v_t) + self.epsilon)

                self.updates.append(tfkb.update(m, m_t))
                self.updates.append(tfkb.update(v, v_t))

                new_p = p_t
                # apply constraints
                # if p in constraints:
                #     c = constraints[p]
                #     new_p = c(new_p)
                self.updates.append(tfkb.update(p, new_p))
        return self.updates


def get_weightnorm_params_and_grads(p, g):
    ps = tfkb.get_variable_shape(p)

    # construct weight scaler: V_scaler = g/||V||
    V_scaler_shape = (ps[-1],)  # assumes we're using tensorflow!
    V_scaler = tfkb.ones(V_scaler_shape)  # init to ones, so effective parameters don't change

    # get V parameters = ||V||/g * W
    norm_axes = [i for i in range(len(ps) - 1)]
    V = p / tf.reshape(V_scaler, [1] * len(norm_axes) + [-1])

    # split V_scaler into ||V|| and g parameters
    V_norm = tf.sqrt(tf.reduce_sum(tf.square(V), norm_axes))
    g_param = V_scaler * V_norm

    # get grad in V,g parameters
    grad_g = tf.reduce_sum(g * V, norm_axes) / V_norm
    grad_V = tf.reshape(V_scaler, [1] * len(norm_axes) + [-1]) * \
             (g - tf.reshape(grad_g / V_norm, [1] * len(norm_axes) + [-1]) * V)

    return V, V_norm, V_scaler, g_param, grad_g, grad_V


def add_weightnorm_param_updates(updates, new_V_param, new_g_param, W, V_scaler):
    ps = tfkb.get_variable_shape(new_V_param)
    norm_axes = [i for i in range(len(ps) - 1)]

    # update W and V_scaler
    new_V_norm = tf.sqrt(tf.reduce_sum(tf.square(new_V_param), norm_axes))
    new_V_scaler = new_g_param / new_V_norm
    new_W = tf.reshape(new_V_scaler, [1] * len(norm_axes) + [-1]) * new_V_param
    updates.append(tfkb.update(W, new_W))
    updates.append(tfkb.update(V_scaler, new_V_scaler))
