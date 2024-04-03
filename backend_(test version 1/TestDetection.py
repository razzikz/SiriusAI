from LoadData import x_tensor, y_tensor

import keras.models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.4, random_state=8)
model = keras.models.load_model('model_v1')
result = model.predict(x_test)
print(result[0])
plt.plot(y_test, 'g', label='Accuracy')
plt.plot(result, 'r', label='Loss')
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()