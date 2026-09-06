import tensorflow as tf
import numpy as np

# -------------------------------------------------
# Utility
# -------------------------------------------------

def depth_to_space(x):
    return tf.nn.depth_to_space(x, 2)

def lrelu(x):
    return tf.nn.leaky_relu(x, alpha=0.1)

# -------------------------------------------------
# Conv layer
# -------------------------------------------------

class Conv(tf.keras.layers.Layer):
    def __init__(self, weight, bias, stride=1):
        super().__init__()
        self.w = tf.constant(weight, dtype=tf.float32)
        self.b = tf.constant(bias, dtype=tf.float32)
        self.stride = stride

    def call(self, x):
        x = tf.nn.conv2d(
            x,
            self.w,
            strides=[1, self.stride, self.stride, 1],
            padding="SAME"
        )
        x = tf.nn.bias_add(x, self.b)
        return x

# -------------------------------------------------
# Dense
# -------------------------------------------------

class Dense(tf.keras.layers.Layer):
    def __init__(self, weight, bias):
        super().__init__()
        self.w = tf.constant(weight, dtype=tf.float32)
        self.b = tf.constant(bias, dtype=tf.float32)

    def call(self, x):
        return tf.matmul(x, self.w) + self.b

# -------------------------------------------------
# Downscale
# -------------------------------------------------

class Downscale(tf.keras.layers.Layer):
    def __init__(self, w, b):
        super().__init__()
        self.conv = Conv(w, b, stride=2)

    def call(self, x):
        return lrelu(self.conv(x))

# -------------------------------------------------
# Upscale
# -------------------------------------------------

class Upscale(tf.keras.layers.Layer):
    def __init__(self, w, b):
        super().__init__()
        self.conv = Conv(w, b)

    def call(self, x):
        x = lrelu(self.conv(x))
        return depth_to_space(x)

# -------------------------------------------------
# Residual
# -------------------------------------------------

class Residual(tf.keras.layers.Layer):
    def __init__(self, w1, b1, w2, b2):
        super().__init__()
        self.c1 = Conv(w1, b1)
        self.c2 = Conv(w2, b2)

    def call(self, x):
        y = lrelu(self.c1(x))
        y = self.c2(y)
        return lrelu(x + y)

# -------------------------------------------------
# Main model
# -------------------------------------------------

class LIAEModel(tf.keras.Model):

    def __init__(self, encoder, inter_ab, inter_b, decoder):
        super().__init__()

        # ---------------- Encoder ----------------

        self.down1 = Downscale(
            encoder["down1/downs_0/conv1/weight:0"],
            encoder["down1/downs_0/conv1/bias:0"],
        )

        self.down2 = Downscale(
            encoder["down1/downs_1/conv1/weight:0"],
            encoder["down1/downs_1/conv1/bias:0"],
        )

        self.down3 = Downscale(
            encoder["down1/downs_2/conv1/weight:0"],
            encoder["down1/downs_2/conv1/bias:0"],
        )

        self.down4 = Downscale(
            encoder["down1/downs_3/conv1/weight:0"],
            encoder["down1/downs_3/conv1/bias:0"],
        )

        # ---------------- Inter ----------------

        self.ab_dense1 = Dense(
            inter_ab["dense1/weight:0"],
            inter_ab["dense1/bias:0"],
        )

        self.ab_dense2 = Dense(
            inter_ab["dense2/weight:0"],
            inter_ab["dense2/bias:0"],
        )

        self.ab_up = Upscale(
            inter_ab["upscale1/conv1/weight:0"],
            inter_ab["upscale1/conv1/bias:0"],
        )

        self.b_dense1 = Dense(
            inter_b["dense1/weight:0"],
            inter_b["dense1/bias:0"],
        )

        self.b_dense2 = Dense(
            inter_b["dense2/weight:0"],
            inter_b["dense2/bias:0"],
        )

        self.b_up = Upscale(
            inter_b["upscale1/conv1/weight:0"],
            inter_b["upscale1/conv1/bias:0"],
        )

        # ---------------- Decoder ----------------

        self.up0 = Upscale(
            decoder["upscale0/conv1/weight:0"],
            decoder["upscale0/conv1/bias:0"],
        )

        self.up1 = Upscale(
            decoder["upscale1/conv1/weight:0"],
            decoder["upscale1/conv1/bias:0"],
        )

        self.up2 = Upscale(
            decoder["upscale2/conv1/weight:0"],
            decoder["upscale2/conv1/bias:0"],
        )

        self.res0 = Residual(
            decoder["res0/conv1/weight:0"],
            decoder["res0/conv1/bias:0"],
            decoder["res0/conv2/weight:0"],
            decoder["res0/conv2/bias:0"],
        )

        self.res1 = Residual(
            decoder["res1/conv1/weight:0"],
            decoder["res1/conv1/bias:0"],
            decoder["res1/conv2/weight:0"],
            decoder["res1/conv2/bias:0"],
        )

        self.res2 = Residual(
            decoder["res2/conv1/weight:0"],
            decoder["res2/conv1/bias:0"],
            decoder["res2/conv2/weight:0"],
            decoder["res2/conv2/bias:0"],
        )

        self.rgb = Conv(
            decoder["out_conv/weight:0"],
            decoder["out_conv/bias:0"],
        )

        # mask branch

        self.m0 = Upscale(
            decoder["upscalem0/conv1/weight:0"],
            decoder["upscalem0/conv1/bias:0"],
        )

        self.m1 = Upscale(
            decoder["upscalem1/conv1/weight:0"],
            decoder["upscalem1/conv1/bias:0"],
        )

        self.m2 = Upscale(
            decoder["upscalem2/conv1/weight:0"],
            decoder["upscalem2/conv1/bias:0"],
        )

        self.mask = Conv(
            decoder["out_convm/weight:0"],
            decoder["out_convm/bias:0"],
        )

    # -------------------------------------------------

    def encode(self, x):

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        return tf.reshape(x, [tf.shape(x)[0], -1])

    # -------------------------------------------------

    def inter(self, flat, which="AB"):

        if which == "AB":
            x = self.ab_dense1(flat)
            x = self.ab_dense2(x)
            x = tf.reshape(x, [-1, 8, 8, 256])
            return self.ab_up(x)

        x = self.b_dense1(flat)
        x = self.b_dense2(x)
        x = tf.reshape(x, [-1, 8, 8, 256])
        return self.b_up(x)

    # -------------------------------------------------

    def decode(self, z):

        x = self.up0(z)
        x = self.res0(x)

        x = self.up1(x)
        x = self.res1(x)

        x = self.up2(x)
        x = self.res2(x)

        rgb = tf.nn.sigmoid(self.rgb(x))

        m = self.m0(z)
        m = self.m1(m)
        m = self.m2(m)
        m = tf.nn.sigmoid(self.mask(m))

        return rgb, m

    # -------------------------------------------------

    def call(self, src, dst):

        src_flat = self.encode(src)
        dst_flat = self.encode(dst)

        src_code = self.inter(src_flat, "AB")
        dst_code = self.inter(dst_flat, "B")

        merged = tf.concat([src_code, dst_code], axis=-1)

        return self.decode(merged)
