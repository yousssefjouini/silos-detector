from silos.classification.prediction.prediction import SilosClassification
import click
import os


@click.command()
@click.argument("filepath")
def main(filepath):
    """Given a filepath, returns the output of SilosClassification"""
    sc = SilosClassification()
    prediction = sc.predict(filepath)
    if os.path.isfile(filepath):
        print(f"Probability of presence of a silo : {prediction}")
    if os.path.isdir(filepath):
        print(prediction)


if __name__ == "__main__":
    main()
