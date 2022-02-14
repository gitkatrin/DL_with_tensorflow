# 1. Einführung in Tensoren
### Definitionen:
  - Skalar: eine einzige Zahl
  - Vektor: Zahlen mit Richtungen (z.B. Windgeschwindigkeit und Richtung)
  - Matrix: ein 2-dimensionales Array aus Zahlen
  - Tensor: ein n- dimensionales Array aus Zahlen (indem n jede Zahl sein kann, ein 0-dimansionaler Tensor ist ein Skalar, ein 1-dimensionaler Tensor ist ein Vektor)

### tf.constant() und tf.Variable():
  - ```tf.constant```: unveränderbarer Tensor
  - ```tf.Variable```: veränderbarer Tensor
  - 🔑 **Keynode:** In der Praxis werden Sie nur selten entscheiden müssen, ob Sie tf.constant oder tf.Variable zum Erzeugen von Tensoren verwenden sollen, da TensorFlow dies für Sie erledigt. Wenn Sie jedoch Zweifel haben, verwenden Sie tf.constant und ändern Sie es später, wenn Sie es brauchen.

### Random Tensoren erstellen:
  - "If both the global and the operation seed are set: Both seeds are used in conjunction to determine the ransom sequence."
  - Wenn mehrere geshufflte Tensoren die gleiche Ordnung haben sollen, muss *global-level random seed* und auch *operation level random seed* verwendet werden

### Weitere Möglichkeiten um Tensoren zu erstellen:
- Der Hauptunterschied zwischen NumPy-Arrays und TensorFlow-Tensoren ist, dass Tensoren auf einer GPU ausgeführt werden können (viel schneller für numerische Berechnungen).
- großer Buchstabe für Matrizen und Tensoren (```X = tf.constant(some_matrix)```)
- kleiner Buchstabe für Vektoren (```y = tf.constant(vector)```)

# 2. Informationen über Tensoren
### wichtigste Attribute:
|Attribut            |Bedeutung                                                                          |Code                            |
|--------------------|-----------------------------------------------------------------------------------|--------------------------------|
|Shape               |Die Länge (Anzahl an Elementen) jeder Dimension eines Tensors.                     |```tensor.shape```              |
|Rank                |Die Anzahl der Dimensionen eines Tensors. (Skalar=0, Vektor=1, Matrix=2, Tensor=n) |```tensor.ndim```               |
|Axis oder dimension |Eine spezielle Dimension eines Tensors.                                            |```tensor[0], tensor[1], ...``` |
|Size                |Die gesamte Anzahl an Elementen in einem Tensor.                                   |```tf.size(tensor)```           |

  - Tensoren können genau wie Python list indexiert werden.

# 3. Tensoren maipulieren (Tensor Operations)
### Basic operations +, -, *, /
  - Tensorflow hat build-in Operationen

|Operation|Code                               |
|---------|-----------------------------------|
|+        |```tf.math.add(tensor, 10)```      |
|-        |```tf.math.subtract(tensor, 10)``` |
|*        |```tf.multiply(tensor, 10)```      |
|/        |```tf.math.divide(tensor, 10)```   |

### Matrix Multiplikation
  - Im Bereich des maschinellen lernens ist die Matrixmultiplikation die meist verwendete Tensoroperation.
  - Regeln:
    1. Die inneren Dimensionen der Tensoren müssen zusammenpassen.
    2. Die resultierende Matrix hat die Form/Größe, der inneren Dimensionen.
  - Skalar (Punktprodukt):
    - TensorFlow: ```tf.linalg.matmul(tensor1, tensor2)```
    - Python: ```tensor @ tensor```

# 4. Tensoren und NumPy
