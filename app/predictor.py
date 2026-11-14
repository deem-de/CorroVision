"""CorroVision - model loading, preprocessing, prediction and Grad-CAM.

Model: ImageNet ResNet50 (frozen) + the two classification heads trained in this project
(app/models/head_stage1.keras, head_stage2.keras).  Test accuracy: Stage 1 99.56%, Stage 2 74.06%.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = 224
MODELS = Path(__file__).resolve().parent / 'models'
CLASSES = json.loads((MODELS / 'labels.json').read_text())['stage2_classes']   # 8 corrosion types


# ----------------------------------------------------------------------------- preprocessing (same as training)
def preprocessing_steps(rgb):
    """Returns the image after each of the four steps of the training pipeline (Part 2)."""
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    denoised = cv2.bilateralFilter(resized, 5, 20, 3)
    f = denoised.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0) + 1e-6
    corrected = np.clip(f * np.clip(means.mean() / means, 0.5, 3.0), 0, 255).astype(np.uint8)   # Gray-World
    lab = cv2.cvtColor(corrected, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return {'resize': resized, 'denoise': denoised, 'color': corrected, 'clahe': enhanced}


class Predictor:
    def __init__(self):
        self.base = keras.applications.ResNet50(weights='imagenet', include_top=False,
                                                input_shape=(IMG_SIZE, IMG_SIZE, 3))
        self.head1 = keras.models.load_model(MODELS / 'head_stage1.keras')
        self.head2 = keras.models.load_model(MODELS / 'head_stage2.keras')

    @staticmethod
    def _logits(head, feat):
        """Head = Dropout, Dense(256, relu), Dropout, Dense(out). Returns the values before sigmoid/softmax,
        which keeps Grad-CAM gradients alive even when the model is very confident."""
        dense1, dense2 = head.layers[1], head.layers[3]
        return tf.matmul(dense1(feat), dense2.kernel) + dense2.bias

    def _forward(self, batch):
        conv = self.base(keras.applications.resnet50.preprocess_input(batch), training=False)   # (N, 7, 7, 2048)
        feat = tf.reduce_mean(conv, axis=(1, 2))                                                # global average pool
        return conv, self._logits(self.head1, feat)[:, 0], self._logits(self.head2, feat)

    def predict(self, rgb, threshold=0.5, tta=False, cam=True):
        steps = preprocessing_steps(np.asarray(rgb)[..., :3].astype(np.uint8))
        x = steps['clahe'].astype(np.float32)
        views = [x, x[:, ::-1], x[::-1]] if tta else [x]           # TTA: original + horizontal flip + vertical flip
        batch = tf.convert_to_tensor(np.stack(views))
        with tf.GradientTape() as tape:
            conv, z1, z2 = self._forward(batch)
            p_cor_all, p_type_all = tf.sigmoid(z1), tf.nn.softmax(z2, axis=-1)
            p_cor = float(tf.reduce_mean(p_cor_all)); p_type = tf.reduce_mean(p_type_all, axis=0).numpy()
            corroded = p_cor >= threshold
            top = int(p_type.argmax())
            # Grad-CAM target: the predicted corrosion type if corroded, otherwise the "healthy" evidence
            score = z2[0, top] if corroded else -z1[0]
        heat = None
        if cam:
            grads = tape.gradient(score, conv)[0]
            weights = tf.reduce_mean(grads, axis=(0, 1))
            m = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
            m = m / (m.max() + 1e-8)
            m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC).clip(0, 1)
            colour = cv2.cvtColor(cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            heat = (0.55 * steps['clahe'] + 0.45 * colour).astype(np.uint8)
        return {'steps': steps, 'p_corroded': p_cor, 'corroded': bool(corroded),
                'type_probs': {c: float(p) for c, p in zip(CLASSES, p_type)},
                'top_type': CLASSES[top], 'gradcam': heat, 'tta': tta, 'threshold': threshold}
