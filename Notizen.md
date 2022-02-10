# 1. Einführung in Tensoren
Definitionen:
  - Skalar: eine einzige Zahl
  - Vektor: Zahlen mit Richtungen (z.B. Windgeschwindigkeit und Richtung)
  - Matrix: ein 2-dimensionales Array aus Zahlen
  - Tensor: ein n- dimensionales Array aus Zahlen (indem n jede Zahl sein kann, ein 0-dimansionaler Tensor ist ein Skalar, ein 1-dimensionaler Tensor ist ein Vektor)


# 2. Informationen über Tensoren gewinnen
  - tf.constant: unveränderbarer Tensor
  - tf.Variable: veränderbarer Tensor

🔑 **Keynode:** In der Praxis werden Sie nur selten entscheiden müssen, ob Sie tf.constant oder tf.Variable zum Erzeugen von Tensoren verwenden sollen, da TensorFlow dies für Sie erledigt. Wenn Sie jedoch Zweifel haben, verwenden Sie tf.constant und ändern Sie es später, wenn Sie es brauchen.
