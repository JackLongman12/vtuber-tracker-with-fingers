"""Avatar Studio: webcam motion capture for VTuber avatars.

  python app.py                  the app window (development: .venv\\Scripts\\python.exe app.py)
  python app.py <command> ...    a tool: studio, mocap, train-face, gestures, tune-smoothing, ... (avatarstudio/main.py)

The packaged AvatarStudio.exe is built from this file (tools/build_dist.py). The code is in avatarstudio/;
docs/DEVELOPMENT.md explains the layout.
"""
from avatarstudio.main import main

if __name__ == "__main__":
    main()
