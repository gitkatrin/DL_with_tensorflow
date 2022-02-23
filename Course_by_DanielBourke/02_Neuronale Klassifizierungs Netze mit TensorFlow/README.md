# Neuronale Klassifizierungs Netze mit TensorFlow

## Klassifizierungsprobleme
  - Binary classification (2 Klassen)
  - Multiclass classification (>2 Klassen, Vorhersage ist eine Klasse)
  - Multilabel classification (>2 Label, Vorhersage sind mehrere Labels)

## Architektur



## Form des Inputs und Outputs
  - Input: Höhe, Breite, Colour channel der Bilder normalisiert als Pixelwerte
  - Output: Wahrscheinlichkeit jeder Klasse
  - Tensoren:
      - Input Shape: [batch_size, width, height, colour_channels] -> Beispiel: Shape = [32, 224, 224, 3]
      - Output Shape: [number_classes]

## Eigene Daten erstellen

## Modell erstellen, kompilieren, anpassen und evalieren

## Verschiedene Evaluierungsmethoden für Klassifizierungsmodelle

## Speichern und Laden eines Modells
