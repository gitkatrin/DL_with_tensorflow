# Einführung in die Regression mit neuronalen Netzen in Tensorflow

## Modelling in Tensorflow
  1. **Model erstellen:** Input-, Output- und Hidden Layer definieren
    - verbessern durch:
      - weitere Schichten hinzufügen
      - Anzahl der Neuronen innerhalb jeder Schicht erhöhen
      - Aktivierungsfunktionen ändern
  3. **Model kompilieren:** Loss Funktion, Optimierer und Evaluationsmetriken definieren
    - verbessern durch:
      - Optimiernunsgfunktion anpassen
      - Lernrate ändern
  5. **Model anpassen:** Model ausprobieren und Muster zwischen X und y erkennen (zwischen Features und Labeln)
    - verbessern durch:
      - mehr Epochen (länger tranieren)
      - mehr Daten (mehr Beispiele zum Lernen)

## Modell Evaluieren
  - 3 Datensäze:
    1. Trainings-Datensatz: Das Modell lernt von diesen Daten (70-80% aller Daten)
    2. Validierungs-Datensatz: Das Modell wird mit diesen Daten angepasst (10-15% aller Daten)
    3. Test-Datensatz: Das Modell wird mit diesen Daten evaluiert. Es wird getestet, was das Modell gelernt hat (10-15% aller Daten)
  - Daten visualisieren
  - Model visualisieren
  - Training des Models visualisieren
  - Vorhersagen des Modells visualisieren
