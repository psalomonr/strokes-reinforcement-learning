# Aprendiendo la transferencia de estilos pictóricos mediante entrenamiento por refuerzo

La pintura, lleva simbolizando la sabiduría y la creatividad humana desde hace miles de años, donde los artistas han ido utilizando una variedad infinita de técnicas y herramientas para crear y dar forma a sus obras. Y aunque parece una tarea fácil de abordar, en realidad esta habilidad requiere mucho esfuerzo y dedicación hasta poder ser desarrollada de forma profesional. Es por ello que, lograr el objetivo de enseñar a pintar a máquinas es una tarea muy desafiante e interesante. 
 
Recientemente se ha publicado un artículo [Zhewei Huang, Wen Heng, Shuchang Zhou, Megvii In, Learning to
Paint With Model-based Deep Reinforcement Learning, 2019.] en el que un agente es capaz de replicar una imagen usando figuras geómetricas. Dicho agente ha sido entrenado mediante técnicas de entrenamiento por refuerzo de manera que el agente aprende a determinar el color y la posición de cada uno de los trazos. Este trabajo se reproduce usando la librería Keras y además, se implementa un nuevo generador de trazos con un aspecto más realista y artístico, consiguiendo que el agente sea capaz de pintar con este nuevo conjunto de pinceladas.


## Entrenamiento del agente y pruebas
```
!python3 train.py --max_step=40 --debug --batch_size=96 

!python3 test.py --max_step=80 --img={img-name} --divide=4
```

## Demos

![](https://github.com/psalomonr/strokes-reinforcement-learning/blob/develop/2.reinforcement-learningv1.0/demo/demo.gif)
