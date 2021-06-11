from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as  tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np

#Вводим данные для обучения модели
degrees = np.array([0, 30, 45, 60, 90, 180, 360], dtype=float)
radians = np.array([0, 0.52, 0.79, 1.05, 1.57, 3.14, 6.28], dtype=float)


# Содаем модель
# Используем модель плотной сети (Dense-сеть),
# которая будет состоять из единственного слоя с еднственым нейроном

# Создаем слой l0 количесвто нейронов (units) равно 1,
# размерность входного параметра (input_shape) - единичное значение
# разменость входных данных = размерность всей модели

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

# Преобразуем слой в модель

model = tf.keras.Sequential([l0])

# Компилируем модель с функцией потерь и оптимизаций
# Функция потерь - среденквалратичная ошибка
# Для функции оптимизации параметр, коэфициент скорости ибучения, равен 0.1
# - это размер шага при корректировке внутренних значений переменных

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

# Тренируем модель
# используем метод fit, первый аргумент - входные значения, второй арумент - желаемые выходные значения
# epochs - количество итераций цыкла обучения
# verbose - контроль уровня логирования

history = model.fit(degrees, radians, epochs=1000, verbose=False)

for i,c in enumerate(degrees):
  print("{} градусов = {} радиан".format(c, radians[i]))

print("Завершили тренировку модели")

# Выводим график обучения
import matplotlib.pyplot as plt
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(history.history['loss'])
plt.show

# Используем модель для предсказаний
print("720 градусов =", model.predict([720.0]), "радиан")

