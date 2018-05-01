#! /usr/bin/env python

# This software is released under the MIT License:

# Copyright 2018 Dark Star Systems, Inc.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Export a Godot scene (v3.0 or later) as glTF v2.
#
# To use:
# You'll need godot v3.0 or later, and godot-python.
# You'll also need a local numpy installation in your godot-python.
# (This file assumes it's in the python/ subdir, but if not, edit the line below.)
#
# Once you have the godot-python pythonscript symlink in your project dir,
# follow the instructions at https://github.com/touilleMan/godot-python
# to autoload this script.
# Specifically, add this to your project.godot:
#  [autoload]
#  Export="*res://export_gltf.py"
#  [gdnative]
#  singletons=[ "res://pythonscript.gdnlib" ]
# Then in your gdscript, call export like this:
#  Export.export_scene('/tmp/godot-scene/godot-scene.gltf', get_node('/root'))

# Documentation
# What is handled:
# * PrimitiveMesh, ArrayMesh
# * cameras
# * lights (using KHR_lights extension)
# * index and non-index mode
# * transforms, materials (simple PBR only), UVs, textures, per-vertex colors
# * renormalizes all normals
# * coalesces vertices with "close" normals and colors, converting to index form
#

# TO DO:
# * MultiMeshInstance
# * animations
# * cache/dedup textures
# * ambient lights
# * hdr backgrounds (not in glTF yet)


# HARDER:
# * shaders

from godot import exposed, export
import godot.bindings as g

import sys, os, os.path
import math
import json
sys.path.append('python')       # for numpy (local)
import numpy as np

verbose = True

def indent(n):
    return ' '*(n*2)

def iprint(level, *args):
    if verbose:
        print(indent(level), *args)

def deg_to_rad(x):
    return x * 3.1415926535 / 180
def color4_to_list(color):
    return [color.r, color.g, color.b, color.a]
def color3_to_list(color):
    return [color.r, color.g, color.b]

def PoolIntArray_to_list(array):
    """Return a copy of the PoolIntArray as a list of floats.
    This is not efficient."""
    result = []
    for elt in array:
        result.append(elt)
    return result

def PoolVec3Array_to_list(array):
    """Return a copy of the array as a list of 3-element lists of floats.
    This is not efficient."""
    result = []
    for elt in array:
        v3 = [elt.x, elt.y, elt.z]
        result.append(v3)
    return result

def PoolVec2Array_to_list(array):
    """Return a copy of the array as a list of 2-element lists of floats.
    This is not efficient."""
    result = []
    for elt in array:
        v2 = [elt.x, elt.y]
        result.append(v2)
    return result

def PoolColorArray_to_list(array):
    """Return a copy of the array as a list of 3-element lists of floats.
    This is not efficient."""
    result = []
    for elt in array:
        color = [elt.r, elt.g, elt.b] # XXX: should we be exporting a? Would be a v4 then.
        result.append(color)
    return result

def Basis_to_ndarray(basis):
    """Return a 2d array from a Godot basis"""
    bx = basis.x
    by = basis.y
    bz = basis.z
    # This is the correct ordering -- tested with various rotations.
    # Seems a bit odd but it works.
    return np.array([[bx.x, by.x, bz.x],
                     [bx.y, by.y, bz.y],
                     [bx.z, by.z, bz.z]])

def normalize_normals(normals):
    """glTF validator complains if normals are even a little bit off normalized"""
    normals_array = np.array(normals, dtype=np.float32)
    return (normals_array / np.linalg.norm(normals_array, axis=1, keepdims=1)).tolist()

def swap_tri_winding_order(tri_list):
    """Interchange the 0th and 1st elements of each triangle to invert the
    winding order. This works for vertices and normals as well as
    indices, as long as the vertices/normals are stored as
    sub-lists.
    """
    for i in range(1, len(tri_list), 3):
        tri_list[i], tri_list[i-1] = tri_list[i-1], tri_list[i]

def get_quat(basis):
    """Get a rotation quaternion from a Godot matrix.
    Following godot/core/math/matrix3.cpp Basis::get_quat()
    """
    orig_matrix = Basis_to_ndarray(basis)
    # First remove scale factor and orthonormalize so it's a rotation-only matrix
    scale = basis.get_scale()
    rbasis = basis.scaled(g.Vector3(1.0/scale.x, 1.0/scale.y, 1.0/scale.z)).orthonormalized()
    matrix = Basis_to_ndarray(rbasis)
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    quat = [0,0,0,0]

    if trace > 0.0:
        s = math.sqrt(trace + 1.0)
        quat[3] = s * 0.5
        s = 0.5 / s

        quat[0] = ((matrix[2][1] - matrix[1][2]) * s)
        quat[1] = ((matrix[0][2] - matrix[2][0]) * s)
        quat[2] = ((matrix[1][0] - matrix[0][1]) * s)
    else:
        i = ((2 if matrix[1][1] < matrix[2][2] else 1)
             if matrix[0][0] < matrix[1][1]
             else (2 if matrix[0][0] < matrix[2][2] else 0))
        j = (i + 1) % 3
        k = (i + 2) % 3

        s = math.sqrt(matrix[i][i] - matrix[j][j] - matrix[k][k] + 1.0)
        quat[i] = s * 0.5
        s = 0.5 / s

        quat[3] = (matrix[k][j] - matrix[j][k]) * s
        quat[j] = (matrix[j][i] + matrix[i][j]) * s
        quat[k] = (matrix[k][i] + matrix[i][k]) * s

    #print(f"get_quat: matrix=\n{matrix}\norig matrix=\n{orig_matrix}\nquat={quat}")
    return [quat[0], quat[1], quat[2], quat[3]]


@exposed
class Export(g.Node):

    def store_ints_in_buffer(self, ints):
        """Store some ints in the int_buffer. Create a bufferView and accessor;
        return the accessor idx."""

        assert isinstance(ints[0], int)
        byte_length = len(ints) * np.int32().itemsize
        start_idx = len(self.int_buffer)
        self.int_buffer.extend(ints)

        view = {"buffer": self.int_buffer_idx,
                "byteLength": byte_length,
                "byteOffset": start_idx * np.int32().itemsize
                # "target": ???
                }
        view_idx = len(self.world['bufferViews'])
        self.world['bufferViews'].append(view)

        accessor = {
            "bufferView": view_idx,
            "byteOffset": 0,
            "componentType": 5125, # 4-byte unsigned int
            "type": "SCALAR",      # single component
            "count": len(ints),
            # can store min/max here
            }
        accessor_idx = len(self.world['accessors'])
        self.world['accessors'].append(accessor)
        return accessor_idx

    def store_vecs_in_buffer(self, vecs, buffer, buffer_idx, compute_minmax = False):
        """Store some float vecs in the buffer. Create a bufferView and accessor;
        return the accessor idx."""

        assert isinstance(vecs[0][0], float)
        n_floats_per_elt = len(vecs[0])
        assert n_floats_per_elt in (1,2,3,4)

        elt_size = n_floats_per_elt * np.float32().itemsize
        byte_length = len(vecs) * elt_size
        start_idx = len(buffer)
        buffer.extend(vecs)

        view = {"buffer": buffer_idx,
                "byteLength": byte_length,
                "byteOffset": start_idx * elt_size,
                # "target": ???
                }
        view_idx = len(self.world['bufferViews'])
        self.world['bufferViews'].append(view)

        vectype = {
            1: 'SCALAR',
            2: 'VEC2',
            3: 'VEC3',
            4: 'VEC4'
            }
        accessor = {
            "bufferView": view_idx,
            "byteOffset": 0,
            "componentType": 5126, # 4-byte float
            "type": vectype[n_floats_per_elt],
            "count": len(vecs)
            }

        if compute_minmax:
            # Compute min/max. Not necessary for all, but might as well.
            np_array = np.array(vecs, dtype=np.float32) # should be Nx3 or Nx2
            minval = np.amin(np_array, axis=0)
            maxval = np.amax(np_array, axis=0)
            if n_floats_per_elt == 4:
                accessor['min'] = [ minval[0], minval[1], minval[2], minval[3] ]
                accessor['max'] = [ maxval[0], maxval[1], maxval[2], maxval[3] ]
            if n_floats_per_elt == 3:
                accessor['min'] = [ minval[0], minval[1], minval[2] ]
                accessor['max'] = [ maxval[0], maxval[1], maxval[2] ]
            elif n_floats_per_elt == 2:
                accessor['min'] = [ minval[0], minval[1] ]
                accessor['max'] = [ maxval[0], maxval[1] ]
            elif n_floats_per_elt == 1: # not sure about this
                accessor['min'] = [ minval[0] ]
                accessor['max'] = [ maxval[0] ]

        accessor_idx = len(self.world['accessors'])
        self.world['accessors'].append(accessor)
        return accessor_idx

    def export_vertices(self, arrays, level):
        """Export all properties of the vertices of the given vertex arrays
        arrays should be e.g. PrimitiveMesh.get_mesh_arrays() or ArrayMesh.surface_get_arrays().
        Adds the vertices to the end of the buffer,
        creates one or more buffer_views pointing to that area,
        and one accessor for each buffer_view.
        Returns a named dict to be merged into a glTF mesh descriptor.
        """
        ArrayFormat = {
            "ARRAY_FORMAT_VERTEX"  : 1,
            "ARRAY_FORMAT_NORMAL"  : 2,
            "ARRAY_FORMAT_TANGENT" : 4,
            "ARRAY_FORMAT_COLOR"   : 8,
            "ARRAY_FORMAT_TEX_UV"  : 16,
            "ARRAY_FORMAT_TEX_UV2" : 32,
            "ARRAY_FORMAT_BONES"   : 64,
            "ARRAY_FORMAT_WEIGHTS" : 128,
            "ARRAY_FORMAT_INDEX"   : 256,
        }
        # These are the sub-arrays of a surface's array
        ArrayType = {
            "ARRAY_VERTEX"  : 0,
            "ARRAY_NORMAL"  : 1,
            "ARRAY_TANGENT" : 2,
            "ARRAY_COLOR"   : 3,
            "ARRAY_TEX_UV"  : 4,
            "ARRAY_TEX_UV2" : 5,
            "ARRAY_BONES"   : 6,
            "ARRAY_WEIGHTS" : 7,
            "ARRAY_INDEX"   : 8,
            "ARRAY_MAX"     : 9,
        }

        result = {"attributes": {}}

        # If we're in index mode, have to do some things differently
        index_mode = not not arrays[ArrayType['ARRAY_INDEX']]

        # output the vertex array
        vertices = PoolVec3Array_to_list(arrays[ArrayType['ARRAY_VERTEX']]) # list of 3-lists
        if not index_mode:
            swap_tri_winding_order(vertices)
        accessor_idx = self.store_vecs_in_buffer(vertices, self.vec3_buffer, self.vec3_buffer_idx,
                                                 compute_minmax=True)
        result['attributes']['POSITION'] = accessor_idx
        # iprint(level, "VERTEX:", arrays[ArrayType['ARRAY_VERTEX']])

        # Others are optional
        if (arrays[ArrayType['ARRAY_NORMAL']]):
            normals = PoolVec3Array_to_list(arrays[ArrayType['ARRAY_NORMAL']])
            normals = normalize_normals(normals)
            if not index_mode:
                swap_tri_winding_order(normals)
            accessor_idx = self.store_vecs_in_buffer(normals, self.vec3_buffer, self.vec3_buffer_idx)
            result['attributes']['NORMAL'] = accessor_idx
            # iprint(level, "NORMAL:", arrays[ArrayType['ARRAY_NORMAL']])
        if (arrays[ArrayType['ARRAY_COLOR']]):
            colors = PoolColorArray_to_list(arrays[ArrayType['ARRAY_COLOR']])
            if not index_mode:
                swap_tri_winding_order(colors)
            accessor_idx = self.store_vecs_in_buffer(colors, self.vec3_buffer, self.vec3_buffer_idx)
            result['attributes']['COLOR_0'] = accessor_idx
            # iprint(level, "COLOR:", arrays[ArrayType['ARRAY_COLOR']])
        if (arrays[ArrayType['ARRAY_TEX_UV']]):
            uv = PoolVec2Array_to_list(arrays[ArrayType['ARRAY_TEX_UV']])
            if not index_mode:
                swap_tri_winding_order(uv)
            accessor_idx = self.store_vecs_in_buffer(uv, self.vec2_buffer, self.vec2_buffer_idx)
            result['attributes']['TEXCOORD_0'] = accessor_idx
            # iprint(level, "COLOR:", arrays[ArrayType['ARRAY_COLOR']])
        if (arrays[ArrayType['ARRAY_TEX_UV2']]):
            uv = PoolVec2Array_to_list(arrays[ArrayType['ARRAY_TEX_UV2']])
            if not index_mode:
                swap_tri_winding_order(uv)
            accessor_idx = self.store_vecs_in_buffer(uv, self.vec2_buffer, self.vec2_buffer_idx)
            result['attributes']['TEXCOORD_1'] = accessor_idx
            # iprint(level, "COLOR:", arrays[ArrayType['ARRAY_COLOR']])
        # Indices are special: ints, and they go in a special place in the gltf
        if (arrays[ArrayType['ARRAY_INDEX']]):
            ints = PoolIntArray_to_list(arrays[ArrayType['ARRAY_INDEX']])
            swap_tri_winding_order(ints)
            accessor_idx = self.store_ints_in_buffer(ints)
            result['indices'] = accessor_idx
            # iprint(level, "INDEX:", ints)
        return result

    def export_texture(self, texture, image_name):
        """Export a texture and its image data"""

        try:
            # This is supposed to help with a Godot bug with textures in VRAM mode,
            # according to Godot bug #18109:
            # (this is GDscript but could be ported to python)
            # func _ready():
            #   var filename = "texture-image-saved.png"
            #   print("Saving PNG for ", $Floor, $Floor.get_surface_material(0))
            #   var img = $Floor.get_surface_material(0).albedo_texture.get_data()
            #   img.clear_mipmaps()
            #   var x = img.save_png(filename)
            #   print("Saved.")
            # So basically, just call img.clear_mipmaps.
            godot_image = texture.get_data()
            godot_image.clear_mipmaps()
        except AttributeError as e:
            print(f"Can't get texture data for tex {texture}, ID {texture.get_instance_id()}, class {texture.get_class()}, type {type(texture)}: {e}")
            return

        sampler = {
            # see OpenGL types for these
            # Note: none of these is required.
            "magFilter": 9729,  # linear
            "minFilter": 9987,  # linear mipmap
            "wrapS": 10497,     # repeat in S(U) - 10497 (repeat) is default
            "wrapT": 10497      # repeat in T(V) - 10497 (repeat) is default
        }
        sampler_idx = len(self.world['samplers'])
        self.world['samplers'].append(sampler)

        # XXX: cache/dedup textures to save space (how?)
        image_name = image_name + '.png'
        image_pathname = os.path.join(self.export_dir, image_name)
        # XXX Images don't look right -- tried flip_y(); didn't help.
        # Don't think I need decompress() here unless flipping or doing something else fancy
        # godot_image.decompress()
        stat = godot_image.save_png(image_pathname)
        print(f"Saved image {image_pathname}, status={stat}")
        image = {"uri": image_name}
        image_idx = len(self.world['images'])
        self.world['images'].append(image)

        texture = {"sampler": sampler_idx,
                   "source": image_idx}
        texture_idx = len(self.world['textures'])
        self.world['textures'].append(texture)
        return texture_idx

    def export_material(self, material):
        """Export a material; returns the index in the materials array.
        Returns -1 if can't export this material."""
        try:
            rid = material.get_rid().get_id() # resource ID, should be unique I hope
        except AttributeError:
            # it's probably null
            return -1
        # maybe it's already in the dedup cache
        if rid in self.materials_to_export:
            return self.materials_to_export[rid]
        else:
            # Not present; assign the next index
            idx = len(self.world['materials'])
            # add it to the deduplication cache
            self.materials_to_export[rid] = idx
            material_name = f"material{rid}.{material.get_instance_id()}"
            if isinstance(material, g.ShaderMaterial):
                print("XXX: Implement ShaderMaterial!")
                return -1
            gltf_material = {"name":material_name,
                             "pbrMetallicRoughness": {
                                 "baseColorFactor": color4_to_list(material.albedo_color),
                                 "metallicFactor": material.metallic,
                                 "roughnessFactor": material.roughness
                             }}
            # textures:
            if material.albedo_texture:
                texture_idx = self.export_texture(material.albedo_texture, "material-" + material_name)
                if texture_idx is not None: # can be None if invalid texture found
                    gltf_material["pbrMetallicRoughness"]["baseColorTexture"] = {
                        "index": texture_idx,
                        "texCoord": 0 # use mesh's TEXCOORD_0
                    }
            # XXX other textures here e.g. metallicRoughnessTexture, which btw
            # is metallic and roughness packed into R and G channels;
            # that will take some work.

            self.world['materials'].append(gltf_material)
            return idx

    def export_arraymesh_surface(self, mesh, surf_idx, level):
        """Creates a primitive for the surface"""
        iprint(level, "Exporting mesh surf %s"%surf_idx)
        material_idx = self.export_material(mesh.surface_get_material(surf_idx))
        iprint(level, "Surface %d: " % surf_idx,
              mesh.surface_get_name(surf_idx), ", ",
              "mat: ", mesh.surface_get_material(surf_idx), ", ",
              "%d vertices"% mesh.surface_get_array_len(surf_idx))
        info = self.export_vertices(mesh.surface_get_arrays(surf_idx), level)
        prim = {"mode": 4}      # mode 4 = TRIANGLES. XXX support TRIANGLE_STRIP etc.
        if material_idx >= 0:
            prim["material"] = material_idx
        prim = {**prim, **info} # merge dicts
        return prim

    def export_primitive_mesh(self, mesh, node, level):
        """Creates a glTF primitive for the Godot primitive mesh.

        For PrimitiveMesh, it can have a material, but if the node
        (MeshInstance) has get_surface_material(), that overrides
        it.
        """
        iprint(level, f"Exporting primitive mesh {mesh}")
        prim = {"mode": 4}      # mode 4 = TRIANGLES. XXX support TRIANGLE_STRIP etc.

        # Three levels of override: GeometryInstance.material_override, per-surface, and mesh itself.
        material = node.material_override
        if not isinstance(material, g.Material):
            material = node.get_surface_material(0)
            if not isinstance(material, g.Material):
                material = mesh.material
        if isinstance(material, g.Material):
            material_idx = self.export_material(material)
            if material_idx >= 0:
                prim["material"] = material_idx
        else:
            print(f"Ignoring material for {node}, can't get valid material")
        info = self.export_vertices(mesh.get_mesh_arrays(), level)
        prim = {**prim, **info} # merge dicts
        return prim

    def export_mesh(self, mesh, node, level):
        """Returns mesh index"""
        idx_list = []
        gltf_mesh = {"primitives": []}
        if isinstance(mesh, g.PrimitiveMesh):
            iprint(level, "PrimitiveMesh")
            prim = self.export_primitive_mesh(mesh, node, level+1)
            gltf_mesh['primitives'].append(prim)
        elif isinstance(mesh, g.ArrayMesh):
            iprint(level, "ArrayMesh(%d surfaces):"%mesh.get_surface_count())
            for s in range(mesh.get_surface_count()):
                prim = self.export_arraymesh_surface(mesh, s, level+1)
                gltf_mesh['primitives'].append(prim)
        else:
            iprint(level, f"Unknown mesh type {type(mesh)} in {mesh}")
        mesh_idx = self.add_mesh(gltf_mesh)
        return mesh_idx

    def export_mesh_instance(self, node, level):
        """Returns list of mesh index for each surface"""
        # Note: don't have to deal with get_surface_material here.
        return self.export_mesh(node.mesh, node, level)

    def export_light(self, node, level):
        """Export a light. Base glTF 2.0 doesn't have lights, it's a proposed extension:
        https://github.com/UX3D-nopper/glTF/tree/master_lights_blinnphong/extensions/Khronos/KHR_lights
        """

        light_color = color3_to_list(node.light_color)
        light_type = 'ambient'
        light = {}
        if isinstance(node, g.SpotLight):
            light_type = 'spot'
            light = {}
            # XXX: how to use node.spot_angle_attenuation?
            spot_angle = deg_to_rad(node.spot_angle)
            light['spot'] = {'innerConeAngle': spot_angle * 0.5, # XXX: hack, for softness
                             'outerConeAngle': spot_angle}
        if isinstance(node, g.OmniLight):
            light_type = 'point'
            light['positional'] = {}
            light['positional']['linearAttenuation'] = node.omni_attenuation
            light['positional']['constantAttenuation'] = 1.0
        if isinstance(node, g.DirectionalLight):
            light_type = 'directional'
        # XXX: Godot has lots more light properties; we could put them in extras perhaps
        light["color"] = light_color
        light["intensity"] = node.light_energy
        light["type"] = light_type # ambient, directional, point, spot
        iprint(level, f"Exporting light {node.name}: {light}")

        light_idx = len(self.world['extensions']['KHR_lights']['lights'])
        self.world['extensions']['KHR_lights']['lights'].append(light)
        return light_idx

    def export_camera(self, node, level):
        """Export a camera. Returns index."""
        camera = node
        iprint(level, "Camera ", node.name, ": ", node)
        if camera.projection == g.Camera.PROJECTION_PERSPECTIVE: # PERSPECTIVE=0, ORTHOGONAL=1
            matrix = {#"aspectRatio": XXX, # optional
                      "yfov": deg_to_rad(camera.fov if camera.keep_aspect==g.Camera.KEEP_HEIGHT
                                         else camera.fov), # XXX
                      "zfar": camera.far,
                      "znear": camera.near}
        else:                   # orthogonal
            matrix = {"xmag": 1.0, # XXX
                      "ymag": 1.0, # XXX
                      "zfar": camera.far,
                      "znear": camera.near}
        camera = {"name": node.name,
                  "type": "perspective", # perspective or orthographic
                  "perspective": matrix,
                  }
        camera_idx = len(self.world['cameras'])
        self.world['cameras'].append(camera)
        return camera_idx

    def add_node_transform(self, node, transform, level):
        iprint(level, f"Transform is {transform}")
        basis = transform.basis
        origin = transform.origin
        # NOTE: Although not recommended, we could just output "matrix" rather than TRS
        # XXX: get_quat is missing in godot/gdnative, so implemented in python above.
        rotation = get_quat(basis)
        if not np.all(np.isclose(np.array(rotation) - np.array([0,0,0,1]), 0)):
            node['rotation'] = rotation
        # scale is length of each row
        # XXX: this could be crosswise; need to confirm
        scale_v3 = basis.get_scale()
        scale = [scale_v3.x, scale_v3.y, scale_v3.z]
        if not np.all(np.isclose(scale, 1)):
            node['scale'] = scale
        origin = [origin.x, origin.y, origin.z]
        if not np.all(np.isclose(origin, 0)):
            node['translation'] = origin

    def export_node(self, node, parent_i, level):
        """Export a node, and all its properties.
        Return node_i, and recurse: if false, don't consider any of
        this node's children.
        """

        iprint(level, "Node ", node.name, ": ", node)
        if isinstance(node, g.CanvasItem):
            iprint(level+1, "(2d)")
            return -1, False # don't care about 2d UI elements
        has_transform = False
        if isinstance(node, g.Spatial): # all Spatial nodes have a transform
            if not node.visible:
                iprint(level+1, "(invisible)")
                return -1, False
            has_transform = True

        gltf_node, i = self.add_node(node.name, parent_i)
        if has_transform:
            self.add_node_transform(gltf_node, node.transform, level+1)

        # Now handle specific Node subclasses
        if isinstance(node, g.MeshInstance):
            idx = self.export_mesh_instance(node, level+1)
            gltf_node['mesh'] = idx
        if isinstance(node, g.Camera):
            idx = self.export_camera(node, level+1)
            # A camera has its own transform in node.get_camera_transform().
            # What to do with that?
            # I thought we should create a sub-node, but now I think get_camera_transform()
            # just gives the _world-space_ transform of the camera so we should ignore it.
            #gltf_subnode, subnode_idx = self.add_node(node.name+"__camera__", i)
            #self.add_node_transform(gltf_subnode, node.get_camera_transform(), level+1)
            gltf_node['camera'] = idx # set camera instance on this node
        elif isinstance(node, g.SpotLight) or isinstance(node, g.OmniLight) or isinstance(node, g.DirectionalLight):
            idx = self.export_light(node, level+1)
            # add light to this node
            light = {"extensions": {
                "KHR_lights": {
                    "light": idx
                    }}}
            gltf_node.update(light)
        # XXX Implement ambient lights; they're different, they get applied to the scene.

        elif node.get_class() in ('Spatial', 'Node'):
            pass                # we've already handled everything for these
        else:
            print(f"Unknown node type {type(node)}, ignoring.")

        return i, True

    def export_scene_nodes(self, node, parent_i, level):
        node_i, recurse = self.export_node(node, parent_i, level)
        if parent_i == -1:
            # top-level node: add it to the latest scene
            self.world['scenes'][-1]['nodes'].append(node_i)
        if recurse:
            for n in node.get_children():
                self.export_scene_nodes(n, node_i, level+1)

    def add_mesh(self, mesh_dict):
        """Add a mesh to the world.
        Return the mesh's index in 'meshes'."""

        self.world['meshes'].append(mesh_dict)
        return len(self.world['meshes'])-1

    def add_asset(self, world):
        world['asset'] = {'version': '2.0',
                          'generator': 'godot-gltf-export.darkstarsystems.com',
                          'copyright': '2018 (c) Dark Star Systems, Inc.'}

    def add_node(self, name, parent=-1):
        """Add a node to the world and its parent if parent >= 0.
        Return node and its index"""

        node = {'name': name}
        self.world['nodes'].append(node)
        node_i = len(self.world['nodes'])-1

        # Add to its parent:
        if parent >= 0:
            self.world['nodes'][parent].setdefault('children', []).append(node_i)
        return node, node_i

    def add_scene(self, name):
        """Add an empty, named scene to the world.
        Return the scene (dict) and its index in 'scenes'."""

        scene = {'name': name, 'nodes': []}
        self.world['scenes'].append(scene)
        return scene, len(self.world['scenes'])-1

    def export_buffers(self, filebase):
        """Export the buffers to bin files
        Note: using struct may be faster here but this is decent for now.
        Also note: these have to be in the right order, see *_buffer_idx.
        """
        # ints
        int_filename = filebase + '-int.bin'
        ints = np.array(self.int_buffer, dtype=np.int32)
        ints.tofile(int_filename)
        elt_size = np.int32().itemsize
        buffer = { "byteLength": len(ints) * elt_size,
                   "uri": os.path.basename(int_filename) }
        self.world['buffers'].append(buffer)

        # vec2
        vec2_filename = filebase + '-vec2.bin'
        floats = np.array(self.vec2_buffer, dtype=np.float32)
        floats.tofile(vec2_filename)
        elt_size = np.float32().itemsize * 2
        buffer = { "byteLength": len(floats) * elt_size,
                   "uri": os.path.basename(vec2_filename) }
        self.world['buffers'].append(buffer)

        # vec3
        vec3_filename = filebase + '-vec3.bin'
        floats = np.array(self.vec3_buffer, dtype=np.float32)
        floats.tofile(vec3_filename)
        elt_size = np.float32().itemsize * 3
        buffer = { "byteLength": len(floats) * elt_size,
                   "uri": os.path.basename(vec3_filename) }
        self.world['buffers'].append(buffer)


    def setup_world(self):
        """Set up the basic structure of the world"""
        # data sources:
        self.world = {}
        self.world['buffers'] = []
        self.world['bufferViews'] = []
        self.world['accessors'] = []
        # models:
        self.world['scenes'] = []
        self.world['nodes'] = []
        self.world['meshes'] = []
        self.world['skins'] = []
        # textures:
        self.world['textures'] = []
        self.world['images'] = []
        self.world['samplers'] = []
        # misc:
        self.world['materials'] = []
        self.world['cameras'] = []
        self.world['animations'] = []

        # Lights are an extension, not in base 2.0
        self.world['extensions'] = {
            "KHR_lights" : {
                "lights": []
                }}
        self.world['extensionsUsed'] = ["KHR_lights"]

        self.world['scene'] = 0      # default scene is scene 0
        self.add_asset(self.world)

        # Buffers for storing binary/numeric data
        # These will be saved as .bin files
        # Could put them all in one, but this simplifies bookkeeping
        self.int_buffer = []
        self.vec2_buffer = []
        self.vec3_buffer = []
        self.int_buffer_idx = 0
        self.vec2_buffer_idx = 1
        self.vec3_buffer_idx = 2

        # materials cache: dict for deduplication
        self.materials_to_export = {}

    def clean_world(self):
        """Remove empty entities"""
        for k in list(self.world.keys()):
            v = self.world[k]
            if v is None or v == []:
                print(f"Cleaning {k}: {v}")
                del self.world[k]
        for n in self.world['nodes']:
            if n.get('children', None) == []:
                del n['children']
            # Don't try to delete empty nodes (without children or mesh);
            # it would move the others and refs would break.

    def export_scene(self, file, root):
        """Exports the scene to file, which must end in .gltf.
        Binary buffers are exported as file-vec3.bin and file-int.bin.
        """
        try:
            self.export_dir = os.path.dirname(file) or "."
            self.setup_world()
            self.add_scene('main')
            self.export_scene_nodes(root, -1, 0)
            self.export_buffers(file.replace('.gltf',''))

            self.clean_world()

            class myJSONEncoder(json.JSONEncoder):
                def default(self, o):
                    if type(o) in (np.float32, np.int32):
                        return np.asscalar(o)
                    print(f"WARNING: can't JSON-encode type {type(o)}; see generated file")
                    return f"DEFAULT: {str(o)} type {type(o)}"
            with open(file, 'w') as f:
                json.dump(self.world, f, indent=2, cls=myJSONEncoder)
        except IOError as e:
            print(f"Can't write to output file {file}: {e}")
            raise
        print(f"glTF file {file} exported.")

    def _ready(self):
        """
        Called every time the node is added to the scene.
        Initialization here.
        """
        self.setup_world()
        print(f"export_gltf.py: READY")

# end of export_gltf.py
