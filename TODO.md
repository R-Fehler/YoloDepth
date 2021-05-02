TODOS
=============

- Depth Maps aus Ostring projizieren als GT.
    - dabei Werte in Meter zwischen 1 und 255 als Float in meter
- Die Depth Maps in val und training splitten
- Detection gleichzeitig?:
    - Wie lade ich PASCAL pre trained, wenn die Tensoren nicht mehr 75 outputs haben sondern 75 + 3 depth values?

Eigene Daten
----------------
- Datensatz interpolieren anstatt nearest neighbour?
    - fuer das target dann einfach down scalen?
- Auf Ostring Depth Maps mal Adabins trainieren.
- depth imgs in train val sortieren