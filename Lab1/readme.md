## Бинаризация

На вход поступает изображение, 
программа отрисовывает окно, в которое выводится в черно-белое изображение.

Пороговое значение используется для сегментации изображения путем установки для всех пикселей, 
значения интенсивности которых выше порогового значения, значения переднего плана, 
а для всех оставшихся пикселей значения фона.

В работе представлены три варианта реализации бинаризации с адаптивным порогом

Файл lib_task.py содержит библиотечну реализацию посредством библиотеки cv2

Файл native_task.py реализует адаптивную бинаризацию посредством python. 
В скрипте реализовано скользящее окно, проходящее по изображению, и вычисляющее пороговое окно.
## Реализация скользящего окна

```python
for i in range(h_k - 1, M + h_k):
    for j in range(h_k - 1, N + h_k):
        m = pdimg[i - h_k + 1:i + h_k, j - h_k + 1:j + h_k].mean() - C
        if pdimg[i, j] < m:
            cp_img[i, j] = 0
        else:
            cp_img[i, j] = 255
```

Файл with_numba.py использует Numba для ускорения вычислений.

## Ускорение
```python
@njit
def adapt_bin(img_to_bin, dest):
    for i in range(h_k - 1, M + h_k):
        for j in range(h_k - 1, N + h_k):
            m = img_to_bin[i - h_k + 1:i + h_k, j - h_k + 1:j + h_k].mean() - C
            if img_to_bin[i, j] < m:
                dest[i, j] = 0
            else:
                dest[i, j] = 255
    return dest
```

## Сравнение скорости работы алгоритмов

`lib_task.py`: 0.001001119613647461

`native_task.py`: 3.311195135116577

`with_numba.py`: 0.05890679359436035

## Результат рабооты
Исходное изображение:

![plot](./take_the_frog.jpg)

Результат работы нативной реализации:

![plot](./result.jpg)
