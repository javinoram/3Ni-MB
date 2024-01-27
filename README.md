# DMRG en cadenas de moleculas 3Ni

Repositorio con los codigos para calcular la magnetizacion y la entropia de von Neumann de cadenas de moleculas 3Ni en diferentes configuraciones.


# Ejecutar codigo
Para ejecutar los codigos, se tiene que estar en el directiorio base del proyecto y seguir lo siguientes pasos.
1. Se tienen que calcular los estados de minima energia para cadenas desacopladas, ya que, estas se usan como base para los calculos de moleculas acopladas. Para ejecutar estos codigos hay que usar el siguiente comando.

`
python3 <archivo> <numero de pares de moleculas>
`

2. Una vez que se tienen los datos de moleculas desacopladas, se pueden ejecutar los otros, la ejecucion es similar
`
python3 <archivo> <numero de pares de moleculas> <valor de exchange entre moleculas>
`

En cada carpeta se tienen carpetas de datos donde se almacenan los resultados.

# Estructuras de los archivos
Las partes mas importantes son base.py y trimeric_molecule_model.py en donde se definen todas las funciones para construir y calcular elementos. 

# Editar archivos
No recomiendo editar estos archivos, en especial trimeric_molecule_model.py, ya que, se requiere mucho conocimiento de la libreria tenpy.

Agradecimiento especial a Emilio Cortes por crear las partes mas importantes de estos flujos de calculo (trimeric_molecule_model.py y base.py).