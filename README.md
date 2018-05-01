# godot-export-gltf

Python script to export [glTF 2.0][gltf] scenes from the [Godot] game engine

## To use:
You'll need [Godot] v3.0.2 or later, and [godot-python].
You'll also need a local numpy installation in your godot-python.
(This script assumes it's in the python/ subdir, but if not, edit the
line in the script.)

Once you have the godot-python pythonscript symlink in your project dir,
follow the instructions at https://github.com/touilleMan/godot-python
to autoload this script. Specifically, add this to your project.godot:
```
[autoload]
Export="*res://export_gltf.py"
[gdnative]
singletons=[ "res://pythonscript.gdnlib" ]
```
Then in your gdscript, call export like this:
```
Export.export_scene('/tmp/godot-scene/godot-scene.gltf', get_node('/root'))
```

## Documentation
What is handled:
* `PrimitiveMesh`, `ArrayMesh`
* cameras
* lights (using KHR_lights extension)
* index and non-index mode
* transforms, materials (simple PBR only), UVs, textures, per-vertex colors
* renormalizes all normals
* coalesces vertices with "close" normals and colors, converting to index form


## TO DO:
* `MultiMeshInstance`
* animations
* cache/dedup textures
* ambient lights
* hdr backgrounds (not in glTF yet)

## HARDER:
* shaders

[gltf]: https://www.khronos.org/gltf/
[Godot]: https://github.com/godotengine/godot
[godot-python]: https://github.com/touilleMan/godot-python
