# src_clasificacion_vistas

En este repositorio están las funciones y scripts para clasificar vistas de frutos pequeños (olivas, arandanos).

Para cada producto debe haber otro repositorio con DVC que contendrá:
* Datos
  * Imágenes
  * Anotaciones
* Pipelines, etc relacionadas con DVC
* Modelos creados
* logs de los experimentos



## Instalación entorno virtual
A partir de la instalación de *Anaconda*

```
conda env create -f environment_mscandvc.yml
````

Luego para usar el entorno

```
conda activate mscandvc
````


# Obtener información sobre un modelo entrenado

```
 python ../src_clasificacion_vistas/tools/print_model_info.py modelfile.pkl
```

# Evaluar un modelo entrenado

En el fichero de configuración se especifica un directorio de salida

* Se generan scores y groundtruth de training y val dataset

* Se miden AUCS de cada tipo de defecto

```
 python ../src_clasificacion_vistas/evaluate/evaluate.py --config config/config1.yaml
```

# Predecir imágenes en una carpeta