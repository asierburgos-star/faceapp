import typer
from rich import print
from .app import run_enroll, run_recognize

app = typer.Typer(add_completion=False, help="FaceApp CLI")

@app.command()
def enroll(name: str = typer.Option(..., "--name", "-n", help="Nombre de usuario a registrar")):
    """Da de alta a un usuario y guarda su embedding en la DB."""
    run_enroll(name)

@app.command()
def recognize():
    """Ejecuta el reconocimiento en vivo desde la cámara."""
    run_recognize()

if __name__ == "__main__":
    app()
