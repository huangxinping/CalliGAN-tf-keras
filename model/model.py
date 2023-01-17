import tensorflow as tf
import tensorflow_addons as tfa


class TransformerEncoder(tf.keras.layers.Layer):
    """
    https://keras.io/examples/nlp/ner_transformers/#build-the-ner-model-class-as-a-kerasmodel-subclass
    https://keras.io/examples/nlp/neural_machine_translation_with_transformer/
    """
    
    def __init__(self, embed_dim, dense_dim, num_heads, rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dense_dim, activation='relu'), 
                tf.keras.layers.Dense(embed_dim)
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-8)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-8)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, inputs, mask=None, training=False):
        if mask is not None:
            mask = mask[: tf.newaxis, :]
        else:
            mask = self.get_causal_attention_mask(inputs)
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        attention_output = self.dropout1(attention_output, training=training)
        proj_input = self.layernorm1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        proj_output = self.dropout2(proj_output, training=training)
        return self.layernorm2(proj_input + proj_output)
    
    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)
    
    def get_config(self):
        config = super().get_confog()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim
        })
        return config    

class CalliComponent(tf.keras.models.Model):
    
    def __init__(self, **kwargs):
        super(CalliComponent, self).__init__(**kwargs)
        vc = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(input_dim=518, output_dim=128, input_length=28),
            
            # Optinal 1 or 2
            # 1. with LSTM layer
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128)),
            
            # 2. with Tranformer layer
#             TransformerEncoder(128, 32, 8),
            # tf.keras.layers.Flatten(),
#             tf.keras.layers.Dropout(0.3),
            # tf.keras.layers.Dense(256),
            
            tf.keras.layers.Reshape(target_shape=(1, 1, 256)),
        ]) # [batch_size, 256]
        self.model = tf.keras.models.Model(inputs=vc.input, outputs=vc.output)
        super(CalliComponent, self).__init__(inputs=self.model.input, outputs=self.model.output, **kwargs)
        
    def call(self, x, training=None):
        return self.model(x, training=training)
    
    
class CalliEncoder(tf.keras.models.Model):
    
    def __init__(self, **kwargs):
        super(CalliEncoder, self).__init__(**kwargs)
        
        self.e_input = tf.keras.layers.Input(shape=(256, 256, 1))
        
        self.e_l1_conv2d = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(self.e_input)
        self.e_l1_bn = tf.keras.layers.BatchNormalization()(self.e_l1_conv2d)
        self.e_l1_leakyrelu = tf.keras.layers.LeakyReLU(alpha=0.2)(self.e_l1_bn)
        
        self.e_l2_conv2d = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding='same', use_bias=False)(self.e_l1_leakyrelu)
        self.e_l2_bn = tf.keras.layers.BatchNormalization()(self.e_l2_conv2d)
        self.e_l2_leakyrelu = tf.keras.layers.LeakyReLU(alpha=0.2)(self.e_l2_bn)

        self.e_l3_conv2d = tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=2, padding='same', use_bias=False)(self.e_l2_leakyrelu)
        self.e_l3_bn = tf.keras.layers.BatchNormalization()(self.e_l3_conv2d)
        self.e_l3_leakyrelu = tf.keras.layers.LeakyReLU(alpha=0.2)(self.e_l3_bn)

        self.e_l4_conv2d = tf.keras.layers.Conv2D(filters=512, kernel_size=5, strides=2, padding='same', use_bias=False)(self.e_l3_leakyrelu)
        self.e_l4_bn = tf.keras.layers.BatchNormalization()(self.e_l4_conv2d)
        self.e_l4_leakyrelu = tf.keras.layers.LeakyReLU(alpha=0.2)(self.e_l4_bn)

        self.e_l5_conv2d = tf.keras.layers.Conv2D(filters=512, kernel_size=5, strides=2, padding='same', use_bias=False)(self.e_l4_leakyrelu)
        self.e_l5_bn = tf.keras.layers.BatchNormalization()(self.e_l5_conv2d)
        self.e_l5_leakyrelu = tf.keras.layers.LeakyReLU(alpha=0.2)(self.e_l5_bn)
        
        self.e_l6_conv2d = tf.keras.layers.Conv2D(filters=512, kernel_size=5, strides=2, padding='same', use_bias=False)(self.e_l5_leakyrelu)
        self.e_l6_bn = tf.keras.layers.BatchNormalization()(self.e_l6_conv2d)
        self.e_l6_leakyrelu = tf.keras.layers.LeakyReLU(alpha=0.2)(self.e_l6_bn)
        
        self.e_l7_conv2d = tf.keras.layers.Conv2D(filters=512, kernel_size=5, strides=2, padding='same', use_bias=False)(self.e_l6_leakyrelu)
        self.e_l7_bn = tf.keras.layers.BatchNormalization()(self.e_l7_conv2d)
        self.e_l7_leakyrelu = tf.keras.layers.LeakyReLU(alpha=0.2)(self.e_l7_bn)
        
        self.e_l8_conv2d = tf.keras.layers.Conv2D(filters=512, kernel_size=5, strides=2, padding='same', use_bias=False)(self.e_l7_leakyrelu)
        self.e_l8_bn = tf.keras.layers.BatchNormalization()(self.e_l8_conv2d)
        self.e_l8_leakyrelu = tf.keras.layers.LeakyReLU(alpha=0.2)(self.e_l8_bn)
        
        self.model = tf.keras.models.Model(inputs=self.e_input, outputs=[self.e_l1_leakyrelu, self.e_l2_leakyrelu, self.e_l3_leakyrelu, self.e_l4_leakyrelu, self.e_l5_leakyrelu, self.e_l6_leakyrelu, self.e_l7_leakyrelu, self.e_l8_leakyrelu])
        super(CalliEncoder, self).__init__(inputs=self.model.input,  outputs=self.model.output,  **kwargs)

    def call(self, x, training=None):
        return self.model(x, training=training)

    
class CalliDecoder(tf.keras.models.Model):
    
    def __init__(self, **kwargs):
        super(CalliDecoder, self).__init__(**kwargs)
        
        self.d_l1_conv2d = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=5, strides=2, padding='same', use_bias=False)
        self.d_l1_bn = tf.keras.layers.BatchNormalization() 
#         self.d_l1_bn = tfa.layers.InstanceNormalization() # The CalliGAN's Github code uses InstanceNormalization, but the paper uses BN.
                                                            # The BatchNormalization is inefficent if batch_size is too small.
        self.d_l1_relu = tf.keras.layers.ReLU()
        self.d_l1_dropout = tf.keras.layers.Dropout(0.5)
        
        self.d_l2_conv2d = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=5, strides=2, padding='same', use_bias=False)
        self.d_l2_bn = tf.keras.layers.BatchNormalization()
        self.d_l2_relu = tf.keras.layers.ReLU()
        self.d_l2_dropout = tf.keras.layers.Dropout(0.5)
        
        self.d_l3_conv2d = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=5, strides=2, padding='same', use_bias=False)
        self.d_l3_bn = tf.keras.layers.BatchNormalization()
        self.d_l3_relu = tf.keras.layers.ReLU()
#         self.d_l3_bn = tfa.layers.InstanceNormalization()
        self.d_l3_dropout = tf.keras.layers.Dropout(0.5)

        self.d_l4_conv2d = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=5, strides=2, padding='same', use_bias=False)
        self.d_l4_bn = tf.keras.layers.BatchNormalization()
        self.d_l4_relu = tf.keras.layers.ReLU()
#         self.d_l4_bn = tfa.layers.InstanceNormalization()
        self.d_l4_dropout = tf.keras.layers.Dropout(0.5)

        self.d_l5_conv2d = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=5, strides=2, padding='same', use_bias=False)
        self.d_l5_bn = tf.keras.layers.BatchNormalization()
        self.d_l5_relu = tf.keras.layers.ReLU()
#         self.d_l5_bn = tfa.layers.InstanceNormalization()
        
        self.d_l6_conv2d = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same', use_bias=False)
        self.d_l6_bn = tf.keras.layers.BatchNormalization()
        self.d_l6_relu = tf.keras.layers.ReLU()
#         self.d_l6_bn = tfa.layers.InstanceNormalization()

        self.d_l7_conv2d = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same', use_bias=False)
        self.d_l7_bn = tf.keras.layers.BatchNormalization()
        self.d_l7_relu = tf.keras.layers.ReLU()
#         self.d_l7_bn = tfa.layers.InstanceNormalization()
        
        self.d_l8_conv2d = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=5, strides=2, padding='same', use_bias=False)
        self.d_l8_bn = tf.keras.layers.BatchNormalization()
        self.d_l8_tanh = tf.keras.layers.Activation('tanh')
        
    def call(self, x, encoder_output, training=None):
        e_l1, e_l2, e_l3, e_l4, e_l5, e_l6, e_l7 = encoder_output[0], encoder_output[1], encoder_output[2], encoder_output[3], encoder_output[4], encoder_output[5], encoder_output[6]

        x = self.d_l1_conv2d(x)
        x = self.d_l1_bn(x)
        x = self.d_l1_relu(x) 
        if training:
            x = self.d_l1_dropout(x)
        x = tf.concat([x, e_l7], -1)
        # x = tf.keras.layers.Add()([x, e_l7])
        
        x = self.d_l2_conv2d(x)
        x = self.d_l2_bn(x)
        x = self.d_l2_relu(x) 
        if training:
            x = self.d_l2_dropout(x)
        x = tf.concat([x, e_l6], -1)
        # x = tf.keras.layers.Add()([x, e_l6])
        
        x = self.d_l3_conv2d(x)
        x = self.d_l3_bn(x)
        x = self.d_l3_relu(x) 
        if training:
            x = self.d_l3_dropout(x)
        x = tf.concat([x, e_l5], -1)
        # x = tf.keras.layers.Add()([x, e_l5])
        
        x = self.d_l4_conv2d(x)
        x = self.d_l4_bn(x)
        x = self.d_l4_relu(x) 
#         if training:
#             x = self.d_l4_dropout(x)
        x = tf.concat([x, e_l4], -1)
        # x = tf.keras.layers.Add()([x, e_l4])
        
        x = self.d_l5_conv2d(x)
        x = self.d_l5_bn(x)
        x = self.d_l5_relu(x) 
        x = tf.concat([x, e_l3], -1)
        # x = tf.keras.layers.Add()([x, e_l3])
        
        x = self.d_l6_conv2d(x)
        x = self.d_l6_bn(x)
        x = self.d_l6_relu(x) 
        x = tf.concat([x, e_l2], -1)
        # x = tf.keras.layers.Add()([x, e_l2])

        x = self.d_l7_conv2d(x)
        x = self.d_l7_bn(x)
        x = self.d_l7_relu(x) 
        x = tf.concat([x, e_l1], -1)
        # x = tf.keras.layers.Add()([x, e_l1])
        
        x = self.d_l8_conv2d(x)
        x = self.d_l8_bn(x)
        output = self.d_l8_tanh(x)
        
        return output
    

class CalliGenerator(tf.keras.Model):
    
    def __init__(self, nb_classes=7, **kwargs):
        super(CalliGenerator, self).__init__(**kwargs)
        self.nb_classes = nb_classes
        self.component = CalliComponent() # input: [batch_size, 28] output: [batch, 1, 1, 256]
        self.encoder = CalliEncoder() # input: [batch_size, 256, 256, 1] output: [batch, 1, 1, 512]
        self.decoder = CalliDecoder() # input: [batch_size, 1, 1, 775] output: [batch, 256, 256, 1]
        
    def call(self, x, training=None):
        ec, images, s = x[0], x[1], x[2] # ec: [batch_size, 28] images: [batch_size, 256, 256, 1] s: [batch_size, 7]
        
        vc = self.component(ec, training=training)
        vi_l1_output, vi_l2_output, vi_l3_output, vi_l4_output, vi_l5_output, vi_l6_output, vi_l7_output, vi_l8_output = self.encoder(images, training=training)
        vs = tf.reshape(s, [-1, 1, 1, self.nb_classes]) # input: [batch, 7] output: [batch, 1, 1, 7]
        
        d_input = tf.keras.layers.Concatenate()([vc, vi_l8_output, vs]) # [batch_size, 1, 1, 775=256+512+7]
        d_output = self.decoder(d_input, [vi_l1_output, vi_l2_output, vi_l3_output, vi_l4_output, vi_l5_output, vi_l6_output, vi_l7_output], training=training)

        # print(f'vc shape: {vc.shape}, vi shape: {vi.shape}, vs shape: {vs.shape}, d_input shape: {d_input.shape}, d_output shape: {d_output.shape}')
        return d_output
    

class CalliDiscriminator(tf.keras.models.Model):
    
    def __init__(self, nb_classes=7, **kwargs):
        super(CalliDiscriminator, self).__init__(**kwargs)
        self.stem = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', use_bias=False, input_shape=(256, 256, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            
            tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            
            tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            
            tf.keras.layers.Conv2D(filters=512, kernel_size=5, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            
            tf.keras.layers.GlobalMaxPooling2D()
        ])
        
        self.binary = tf.keras.layers.Dense(1, activation='sigmoid')(self.stem.output)
        self.category = tf.keras.layers.Dense(nb_classes, activation='softmax')(self.stem.output)
        
        self.model = tf.keras.models.Model(inputs=self.stem.input, outputs=[self.binary, self.category])
        super(CalliDiscriminator, self).__init__(inputs=self.model.inputs, outputs=self.model.outputs)
        
    def call(self, inputs, training=None):
        return self.model(inputs, training=training)


class CalliGAN(tf.keras.models.Model):
    
    def __init__(self, nb_classes=7, image_size=256, global_batch_size=0,**kwargs):
        super(CalliGAN, self).__init__(**kwargs)
        self.nb_classes = nb_classes
        self.image_size = image_size
        self.global_batch_size = global_batch_size # Using Multi-GPU if global_batch_size!=0, else Single-GPU
        
        self.generator = CalliGenerator(nb_classes=nb_classes)
        self.discriminator = CalliDiscriminator(nb_classes=nb_classes)
        
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
        
        self.L1_penalty = 100
        self.Lconst_penalty = 15
        self.Lcategory_penalty = 1.0
        self.Ltv_penalty = 0.0
        
    @property
    def metrics(self):
        return [self.g_loss_tracker, self.d_loss_tracker, self.val_loss_tracker]

    def compile(self, g_optimizer, d_optimizer, pixel_wise_loss_fn, constancy_loss_fn, category_loss_fn, adversarial_loss_fn):
        super(CalliGAN, self).compile(run_eagerly=False)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.pixel_wise_loss_fn = pixel_wise_loss_fn
        self.constancy_loss_fn = constancy_loss_fn
        self.category_loss_fn = category_loss_fn
        self.adversarial_loss_fn = adversarial_loss_fn
        
    def __train_discriminator(self, real_components, real_images, real_categories, real_y):
        with tf.GradientTape() as tape:
            # Decode the image (guided by components and categories) to fake images
            generated_images = self.generator([real_components, real_images, real_categories], training=True)
        
            real_dis_opinions, real_dis_categories = self.discriminator(tf.concat([real_images, real_y], axis=3), training=True)
            real_dis_adversarial_loss = self.adversarial_loss_fn(tf.ones_like(real_dis_opinions), real_dis_opinions)
            real_dis_category_loss = self.category_loss_fn(real_categories, real_dis_categories)
            
            fake_dis_opinions, fake_dis_categories = self.discriminator(tf.concat([real_images, generated_images], axis=3), training=True)
            fake_dis_adversarial_loss = self.adversarial_loss_fn(tf.zeros_like(fake_dis_opinions), fake_dis_opinions)
            fake_dis_category_loss = self.category_loss_fn(real_categories, fake_dis_categories)
            
            d_loss = (real_dis_adversarial_loss+fake_dis_adversarial_loss)+self.Lcategory_penalty*(real_dis_category_loss+fake_dis_category_loss)*0.5
            if self.global_batch_size != 0:
                d_loss = tf.nn.compute_average_loss(d_loss, global_batch_size=self.global_batch_size)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
        return d_loss
    
    def __train_generator(self, real_components, real_images, real_categories, real_y):
        with tf.GradientTape() as tape:
            fake_images = self.generator([real_components, real_images, real_categories], training=True)
            _, _, _, _, _, _, _, encode_real_images = self.generator.encoder(real_images, training=True)
            _, _, _, _, _, _, _, encode_fake_images = self.generator.encoder(fake_images, training=True)
            
            # constancy loss between encoded real image and encoded fake image
            const_loss =  self.constancy_loss_fn(encode_real_images, encode_fake_images)
            
            # pixel_wise loss between real image and fake image
            pixel_wise_loss = self.pixel_wise_loss_fn(fake_images, real_y)
            
            # maximize the chance generator fool the discriminator
            fake_fool_dis_opinions, fake_dis_categories = self.discriminator(tf.concat([real_y, fake_images], axis=3), training=True)
            fake_gen_adversarial_loss = self.adversarial_loss_fn(tf.ones_like(fake_fool_dis_opinions), fake_fool_dis_opinions)
            fake_dis_category_loss = self.category_loss_fn(real_categories, fake_dis_categories)

            # total variation loss
            tv_loss = (tf.nn.l2_loss(fake_images[:, 1:, :, :] - fake_images[:, :self.image_size - 1, :, :]) / self.image_size
                   + tf.nn.l2_loss(fake_images[:, :, 1:, :] - fake_images[:, :, :self.image_size - 1, :]) / self.image_size) * self.Ltv_penalty

            if self.global_batch_size != 0:
#                 g_loss = self.L1_penalty*pixel_wise_loss+self.Lconst_penalty*const_loss+tv_loss
#                 g_loss += tf.math.reduce_sum(fake_gen_adversarial_loss+self.Lcategory_penalty*fake_dis_category_loss)
#                 g_loss = tf.nn.compute_average_loss(g_loss, global_batch_size=self.global_batch_size)
                pixel_wise_loss = tf.reduce_sum(self.L1_penalty*pixel_wise_loss)                
                const_loss = tf.reduce_sum(self.Lconst_penalty*const_loss)
                tv_loss = tf.reduce_sum(tv_loss)
                fake_gen_adversarial_loss = tf.reduce_sum(fake_gen_adversarial_loss)
                fake_dis_category_loss = tf.reduce_sum(self.Lcategory_penalty*fake_dis_category_loss)
                g_loss = (pixel_wise_loss+const_loss+tv_loss+fake_gen_adversarial_loss+fake_dis_category_loss)/self.global_batch_size
            else:
                g_loss = self.L1_penalty*pixel_wise_loss+self.Lconst_penalty*const_loss+fake_gen_adversarial_loss+self.Lcategory_penalty*fake_dis_category_loss+tv_loss
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        return g_loss
        
    def train_step(self, data):
        # Unpack the data
        real_X, real_y = data
        real_components, real_images, real_categories = real_X[0], real_X[1], real_X[2] 
        
        # Train the discriminator
        d_loss = self.__train_discriminator(real_components, real_images, real_categories, real_y)
        
        # Train the generator (note that we should *not* update the discriminator weights!)
        g_loss = self.__train_generator(real_components, real_images, real_categories, real_y)
        
        # magic move to Optimize G again
        # according to https://github.com/carpedm20/DCGAN-tensorflow (to make sure that d_loss does not go to zero)
        g_loss = self.__train_generator(real_components, real_images, real_categories, real_y)
        
        # Monitor loss.
        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.g_loss_tracker.result(),
            "d_loss": self.d_loss_tracker.result()
        }
    
    def test_step(self, data):
        # Unpack the data
        real_X, real_y = data
        real_components, real_images, real_categories = real_X[0], real_X[1], real_X[2] 
        generated_images = self.generator([real_components, real_images, real_categories], training=False)
        pixel_wise_loss = self.pixel_wise_loss_fn(generated_images, real_y)
        if self.global_batch_size != 0:
            pixel_wise_loss = tf.nn.compute_average_loss(pixel_wise_loss, global_batch_size=self.global_batch_size)
        
        # Monitor loss.
        self.val_loss_tracker.update_state(pixel_wise_loss)
        return {
            "loss": self.val_loss_tracker.result(),
        }


if __name__ == '__main__':
    # component = CalliComponent()
    # component(tf.zeros([1, 28]))
    # component.summary()
    # encoder = CalliEncoder()
    # encoder.summary()
    
    gen = CalliGenerator()
    inputs = [
        tf.zeros([1, 28]), 
        tf.zeros([1, 256, 256, 1]),
        tf.zeros([1, 7])
    ]
    output = gen(inputs)
    print(output.shape)
    gen.summary()
    # # tf.keras.utils.plot_model(gen, 'generator.png')
    # # gen.load_weights('CalliGAN_generator.h5')
    
    dis = CalliDiscriminator()
    dis.summary()
    # # tf.keras.utils.plot_model(dis, 'discriminator.png')
    # # dis.load_weights('CalliGAN_discriminator.h5')
    