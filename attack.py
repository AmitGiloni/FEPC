from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import random
import warnings
warnings.filterwarnings("ignore")
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import *
from keras import optimizers
from keras.callbacks import Callback, ReduceLROnPlateau


class auc_callback(Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        auc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        auc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rauc: %s - val_auc: %s' % (str(round(auc, 4)), str(round(auc_val, 4))), end=100 * ' ' + '\n')
        logs['auc'] = auc
        logs['val_auc'] = auc_val
        return


class Attack:
    def __init__(self, train_data_x, train_data_y, val_x, val_y, o_model, CONSTRAINTS, scaler=None, gb=None, model=None):
        super().__init__()
        self.seed = 123456789
        tf.random.set_random_seed(self.seed)
        np.random.seed(self.seed)
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
            self.scaler.fit(train_data_x)
        self.CONSTRAINTS = CONSTRAINTS
        self.columns = train_data_x.columns.tolist()
        if gb is None:
            self.gb = GradientBoostingClassifier(max_depth=5, n_estimators=275, learning_rate=0.01, random_state=self.seed)
            self.gb.fit(self.scaler.transform(train_data_x), o_model.predict(train_data_x))
        else:
            self.gb = gb
        if model is None:
            self.model = self.create_nn_end2end(train_data_x, train_data_y, val_x, val_y)
        else:
            self.model = model
        self.tf_max_value = tf.constant(np.inf)
        self.tf_min_value = tf.constant(-np.inf)

    def create_model(self, input_shape, embedding_size):
        model = Sequential()
        model.add(Dense(32, input_shape=input_shape, activation='relu', name='dense1'))
        model.add(BatchNormalization())
        model.add(Dense(16, activation='relu', name='dense2'))
        model.add(Dropout(0.4))
        model.add(Dense(embedding_size, activation='linear', kernel_initializer='TruncatedNormal', name='embeddings'))
        return model

    def create_classifier(self, nn_embedding, input_shape, embedding_size):
        cred2vec = self.create_model(input_shape, embedding_size)
        cred2vec.set_weights(nn_embedding.get_weights())
        for layer in cred2vec.layers:
            layer.trainable = False
        cred2vec.add(Dense(52, activation='sigmoid'))
        cred2vec.add(Dense(1, activation='sigmoid'))
        return cred2vec

    def cosine_dist(self, x, y):
        x = K.expand_dims(K.l2_normalize(x, axis=-1), axis=1)
        y = K.expand_dims(K.l2_normalize(y, axis=-1), axis=0)
        return -K.mean(x * y, axis=-1, keepdims=True)

    def bh_triplet_loss(self,dists, labels):
        same_identity_mask = tf.equal(tf.expand_dims(labels, axis=1),
                                      tf.expand_dims(labels, axis=0))
        negative_mask = tf.logical_not(same_identity_mask)
        positive_mask = tf.logical_xor(same_identity_mask,
                                       tf.eye(tf.shape(labels)[0], dtype=tf.bool))
        furthest_positive = tf.reduce_max(dists * tf.cast(positive_mask, tf.float32), axis=1)
        closest_negative = tf.map_fn(lambda x: tf.reduce_min(tf.boolean_mask(x[0], x[1])),
                                     (dists, negative_mask), tf.float32)
        diff = furthest_positive - closest_negative
        TL_MARGIN = 1
        return tf.maximum(diff + TL_MARGIN, 0.0)

    def transform_data(self, x, y, num_classes):
        if type(x) == pd.DataFrame:
            x = x.values
        if type(y) == pd.Series:
            y = y.values
        dataset = {i: [] for i in range(num_classes)}
        for idx in range(x.shape[0]):
            dataset[y[idx]].append(x[idx])
        return dataset

    def get_batch(self, dataset_x, dataset_y, k, num_classes):
        """
        Sample BATCH_K random images from each category,
        returning the data along with its labels.
        """
        dataset = self.transform_data(dataset_x, dataset_y, num_classes)
        batch = []
        labels = []
        for l in range(num_classes):
            indices = random.sample(range(len(dataset[l])), k)
            indices = np.array(indices)
            batch.append([dataset[l][i] for i in indices])
            labels += [l] * k
        batch = np.array(batch).reshape(num_classes * k, -1)
        labels = np.array(labels)
        s = np.arange(batch.shape[0])
        np.random.shuffle(s)
        batch = batch[s]
        labels = labels[s]
        return batch, labels

    def creat_embeddings(self, input_shape, embedding_size, EPOCH_SIZE, train_set_x, train_set_y, BATCH_SIZE):
        num_classes = 2
        n_epochs = 100 * EPOCH_SIZE
        train_set_x = pd.DataFrame(self.scaler.transform(train_set_x), columns=train_set_x.columns)
        labels = tf.placeholder(tf.int32, [None], name='labels_ph')
        cred2vec = self.create_model(input_shape, embedding_size)
        dists = self.cosine_dist(cred2vec.output, cred2vec.output)
        loss = tf.reduce_mean(self.bh_triplet_loss(dists, labels))
        global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        train_step = optimizer.minimize(loss=loss, global_step=global_step)
        sess = tf.keras.backend.get_session()
        with sess.as_default():
            tf.global_variables_initializer().run()
            loss_hist = []
            for i in range(n_epochs):
                x, y = self.get_batch(train_set_x, train_set_y, BATCH_SIZE, num_classes)
                feed_dict = {cred2vec.input: x, labels: y}
                _, raw_loss, embeddings, final_dists = sess.run([train_step,
                                                                 loss, cred2vec.output, dists], feed_dict)
                loss_hist.append(raw_loss)
                if i % 100 == 0:
                    print('Training - Batch: ', i, '/', n_epochs, 'Loss: ', raw_loss)
            print('Finished training for ', n_epochs, ' epochs')
        return cred2vec

    def create_nn_end2end(self, x_train, y_train, x_val, y_val):
        n_features = self.columns
        BATCH_SIZE = 32
        EPOCH_SIZE = (x_train.shape[0] // BATCH_SIZE)
        n_epochs = 200
        embedding_size = 8 # changed to 2 in the synthetic experiments manually
        input_shape = (len(n_features),)
        cred2vec = self.creat_embeddings(input_shape, embedding_size, EPOCH_SIZE, x_train, y_train, BATCH_SIZE)
        nn_end2end = self.create_classifier(cred2vec, input_shape, embedding_size)
        opt = optimizers.Adam(lr=0.0005)
        nn_end2end.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
        val_data = (self.scaler.transform(x_val), y_val)
        auc_cb = auc_callback((self.scaler.transform(x_train), y_train), (self.scaler.transform(x_val), y_val))
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, cooldown=3, min_lr=0.00001)
        history = nn_end2end.fit(self.scaler.transform(x_train), y_train, epochs=n_epochs, validation_data=val_data,
                                 callbacks=[auc_cb, reduce_lr])
        return nn_end2end

    def print_pert(self, x, x_tag, mask):
        computed_perts = np.where(np.isclose(x, x_tag) == False)[1]
        if len(computed_perts) == 0:
            print('No changes were made.')
        else:
            for idx in computed_perts:
                print('Column {} -- Benign: {:.4f} -- Adv.: {:.4f}'.format(
                    self.columns[idx], x[0, idx], x_tag[0, idx]
                ))
                if mask[idx] == 0:
                    print('print("\x1B[4m\x1b[31m\x1B[1mWARNING"): !!!!!!!!!!!!! THIS COLUMN IS IMMUTABLE')

    def rand_choice_multidim(self, arr, n_to_sample):
        res = []
        for sample in arr:
            tmp = np.random.choice(sample, n_to_sample, replace=False)
            res.append(tmp)
        return np.array(res)

    def process_gradients(self, g, m_mutables, m_previously_chosen, o_model, records, top_k=1):
        if np.count_nonzero(g) == 0:
            print('WARNING: Gradient vector is all zeros.')
        g = o_model.feature_importances_.reshape(g.shape)
        masked_g = g * m_mutables * (1 - m_previously_chosen)
        if masked_g.max() == 0 or (masked_g > 0).sum() < top_k:
            masked_g[masked_g < 0] = masked_g[masked_g < 0] + np.abs(masked_g.min())
        if np.count_nonzero(masked_g) == 0:
            masked_g = np.ones_like(masked_g) * m_mutables
        if np.all(m_mutables == m_previously_chosen):
            masked_g = g * m_mutables
            top_k = sum(m_mutables)
        else:
            top_k = 1 if np.count_nonzero(masked_g) == 1 else top_k
        records = records.corr()
        records_np = np.array(records)
        top_k_idx = np.argpartition(masked_g, -top_k)[:, -top_k:]
        chosen_idx = self.rand_choice_multidim(top_k_idx, 1).flatten()
        print(chosen_idx)
        top_3 = np.argpartition(records_np[chosen_idx[0]], -3)[-3:]
        g_onehot = np.zeros_like(g)
        g_onehot[np.arange(g_onehot.shape[0]), top_3] = 1
        top_3 = g_onehot * masked_g
        top_3 = np.where(top_3 > 0)[1]
        if np.count_nonzero(m_mutables * g_onehot) == 0:
            print('print("\x1B[4m\x1b[31m\x1B[1mWARNING"): !!!!!!!!!! IMMUTABLE FEATURE')
            print(g)
            print(m_previously_chosen)
            print(masked_g)
        return top_3, g_onehot * masked_g

    def spatial_l2(self,truth, pred):
        return tf.reduce_mean(tf.square(tf.subtract(truth, pred)))

    def get_col_idx(self,colname):
        return self.columns.index(colname)

    def apply_constraints(self, tf_x):
        cols = self.columns
        new_df = []
        for feature in cols:
            tensor = tf_x[..., self.get_col_idx(feature)]
            if feature in self.CONSTRAINTS:
                for constraint in self.CONSTRAINTS[feature]:
                    tensor = constraint(tensor)
            tf_new_col = tf.expand_dims(tensor, -1)
            new_df.append(tf_new_col)
        return tf.concat(new_df, axis=1)

    def cat_wrapper(self, records, o_model, critic_threshold, prototype_threshold, x_train, protected_place=-1):
        """
        The method compute for the given record how many iterations took to make it change
        the original model prediction.
        :param record: The original record to use.
        :param o_model: The original model
        :return:new_record - The record after noise was added.
                iter_num - The number of iteration took to create new_record.

        """
        iter_num = -1
        sess = tf.keras.backend.get_session()
        if protected_place == -1:
            feature_mask = [1 for x in range((records.shape[1]))]
        else:
            feature_mask = [1 for x in range((records.shape[1]))]
            feature_mask[protected_place] = 0
        n_features = records.shape[1]
        n_adversarial = 1

        emb_layer_name = 'embeddings'
        emb_size = self.model.get_layer(emb_layer_name).output.shape[1].value
        emb = Model(self.model.inputs, self.model.get_layer(emb_layer_name).output)

        input_x = tf.placeholder(tf.float32, (n_adversarial, n_features))
        input_y = tf.placeholder(tf.float32, (n_adversarial, 1))
        input_emb_vector = tf.placeholder(tf.float32, (n_adversarial, emb_size))
        scale_ = tf.constant(self.scaler.scale_, dtype=tf.float32)
        mean_ = tf.constant(self.scaler.mean_, dtype=tf.float32)
        mask = tf.constant(feature_mask, dtype=tf.float32)
        global_step = tf.Variable(0, trainable=False)
        x_tag = tf.Variable(tf.zeros((n_adversarial, n_features)))
        x_clipped = self.apply_constraints(x_tag)
        x_assign_op = tf.assign(x_tag, input_x)
        x_scaled = tf.divide(tf.subtract(x_tag, mean_), scale_)
        output = self.model(x_scaled)
        output_emb = emb(x_scaled)
        cross_entropy = tf.keras.backend.binary_crossentropy(input_y, output)
        loss = tf.reduce_mean(cross_entropy)
        cross_entropy_adv = tf.keras.backend.binary_crossentropy(1 - input_y, output)
        loss_adv = tf.reduce_mean(cross_entropy_adv)
        loss_adv_full = loss_adv + self.spatial_l2(input_emb_vector, output_emb)
        t_grads = tf.gradients(loss, x_tag)[0]
        optimizer = tf.train.AdamOptimizer(learning_rate=1.)
        grad_apply_op = optimizer.minimize(loss_adv_full, var_list=[x_tag], global_step=global_step)

        proto_critic_list = []
        iter_num_list = []
        x_adv = []
        curr_example = []
        for ex_idx in range(0, len(records)):
            if records[ex_idx:ex_idx + 1].values.tolist()[0][0:3] == [1,1,1]:
                print()
            x_benign = records[ex_idx:ex_idx + 1].copy().values
            y_benign = o_model.predict(records[ex_idx:ex_idx + 1])
            session_input_x = x_benign.copy()
            session_input_y = y_benign.copy()
            session_input_emb = emb.predict(session_input_x)
            perturbed_features = np.zeros_like(session_input_x)
            with sess.as_default():
                tf.global_variables_initializer().run()
                is_record_changed = False
                iter_num = 0
                for iteration in range(1, 20):
                    if is_record_changed:
                        is_record_changed = False
                        iter_num = iter_num + 1
                    feed_dict = {
                        input_x: session_input_x,
                        input_y: session_input_y.reshape(1,1),
                        input_emb_vector: session_input_emb
                    }
                    _ = sess.run(x_assign_op, feed_dict)
                    gradients = sess.run(t_grads, feed_dict)
                    chosen_feature_idx, chosen_feature_mask = self.process_gradients(gradients, feature_mask, perturbed_features, o_model, x_train)
                    try:
                        print('Iteration #{} -- Next Chosen Feature: {}'.format(iteration,
                                                                                self.columns[chosen_feature_idx[0]]))
                        _ = sess.run(grad_apply_op, feed_dict)
                        new_x = sess.run(x_clipped)
                        features_to_perturb = np.clip(chosen_feature_mask + perturbed_features, 0, 1).astype(bool)
                        session_input_x[features_to_perturb] = new_x[features_to_perturb]
                        feed_dict.update({input_x: session_input_x})
                        new_losses = sess.run([loss, loss_adv], feed_dict)
                        self.print_pert(x_benign, session_input_x, feature_mask)
                        perturbed_features = np.clip(perturbed_features + chosen_feature_mask, 0, 1)
                        print('Benign Loss: {:.3f} -- Adv. Loss: {:.3f}'.format(new_losses[0], new_losses[1]))
                        curr_example.append(session_input_x)
                        print('Original prediction: {} New prediction: {}'.format(
                            int(o_model.predict(records[ex_idx:ex_idx + 1])),
                            int(o_model.predict(pd.DataFrame(session_input_x, columns=records.columns)))))
                        print("-----------------")
                        if all([records[ex_idx:ex_idx + 1].values[0] != session_input_x[0]][0].tolist()):
                            is_record_changed = True
                        if int(o_model.predict(pd.DataFrame(session_input_x, columns=records.columns))) != int(
                                o_model.predict(records[ex_idx:ex_idx + 1])):
                            if o_model.predict_proba(pd.DataFrame(session_input_x, columns=records.columns))[0][
                                1] > 0.6 or \
                                    o_model.predict_proba(pd.DataFrame(session_input_x, columns=records.columns))[0][
                                        1] < 0.4:
                                # set as 0.55 and 0.45 for the synthetic experiments
                                iter_num = iteration
                                new_record = session_input_x
                                break
                        reset_optimizer_op = tf.variables_initializer(optimizer.variables())
                        _ = sess.run(reset_optimizer_op)
                    except:
                        break

            new_record = session_input_x
            x_adv.append(new_record)
            iter_num_list.append(iter_num)
            if iter_num <= critic_threshold:
                proto_critic_list.append("critic")
            elif iter_num >= prototype_threshold:
                proto_critic_list.append("prototype")
            else:
                proto_critic_list.append("non")

            if ex_idx % 200 == 0:
                print("TMP TMP TMP TMP we are at sample {} out of {}".format(ex_idx, len(records)))

        return proto_critic_list, iter_num_list, pd.DataFrame(np.concatenate(x_adv),columns=records.columns)


class factory_cat():
    def __init__(self, min_, max_):
        self.min = min_
        self.max = max_

    def integer(self, tensor):
        res = tf.round(tensor)
        return res

    def bound(self,tensor, t_min, t_max):
        res = tf.clip_by_value(tensor, t_min, t_max)
        return res

    def method(self, tensor):
        tensor = self.integer(tensor)
        return self.bound(tensor, self.min, self.max)

    def get_method(self):
        return self.method


tf_max_value = tf.constant(np.inf)
tf_min_value = tf.constant(-np.inf)


def bound(tensor, t_min, t_max):
    res = tf.clip_by_value(tensor, t_min, t_max)
    return res


def positive(tensor):
    res = bound(tensor, 0, tf_max_value)
    return res


def negative(tensor):
    res = bound(tensor,  tf_min_value, 0)
    return res


def integer(tensor):
    res = tf.round(tensor)
    return res


def categorical(tmin, tmax):
    return factory_cat(tmin, tmax).get_method()


def binary(tensor):
    tensor = integer(tensor)
    tensor = bound(tensor, 0, 1)
    return tensor


def normalized(tensor):
    return bound(tensor, 0, 1)
