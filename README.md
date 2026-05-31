# Spatial Notebook

An infinite 2D spatial canvas for designers. Users can upload image and 3D files as floating blocks, then pan around the canvas with a webcam-tracked hand gesture.

## Features

- MediaPipe Tasks Vision Hand Landmarker integration through CDN modules
- Open palm and closed fist gesture states
- Closed fist motion pans the canvas like grabbing space
- Image and 3D file upload through one floating action button
- OBJ and STL wireframe previews without extra rendering libraries
- GLB and GLTF file cards for spatial placement
- Tiny webcam picture-in-picture preview
- Custom hand cursor with grab feedback
- Plain HTML, CSS, and JavaScript for GitHub Pages

## Run Locally

Camera access requires `localhost` or HTTPS.

```bash
python3 -m http.server 4173
```

Then open:

```text
http://localhost:4173
```
