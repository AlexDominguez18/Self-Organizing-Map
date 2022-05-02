import SOM as SOM

if __name__ == "__main__":
    som = SOM(max_epocas=5000)
    som.entrenar()
    som.graficar()