import os
import string

import chainer
import numpy as np
import skimage.io
import time
from skimage.transform import rescale


def load_mtl(filename_mtl):
    # load color (Kd) and filename of textures from *.mtl
    texture_filenames = {}
    colors = {}
    material_name = ''
    for line in open(filename_mtl).readlines():
        if len(line.split()) != 0:
            if line.split()[0] == 'newmtl':
                material_name = line.split()[1]
            if line.split()[0] == 'map_Kd':
                if os.path.exists(line.split()[1]):
                    texture_filenames[material_name] = line.split()[1]
            if line.split()[0] == 'Kd':
                colors[material_name] = np.array(map(float, line.split()[1:4]))
    return colors, texture_filenames

class texture_loader():
    def __init__(self):
        self.chainer_func = chainer.cuda.elementwise(
            'int32 j, raw float32 image, raw float32 faces, raw float32 textures, raw int32 is_update, raw int32 sizes',
            '',
            string.Template('''
                const int ts = sizes[0];//${texture_size};
                const int fn = i / (ts * ts * ts);
                float dim0 = ((i / (ts * ts)) % ts) / (ts - 1.) ;
                float dim1 = ((i / ts) % ts) / (ts - 1.);
                float dim2 = (i % ts) / (ts - 1.);
                if (1 < dim0 + dim1 + dim2) {
                    float sum = dim0 + dim1 + dim2;
                    dim0 /= sum;
                    dim1 /= sum;
                    dim2 /= sum;
                }
                const float* face = &faces[fn * 3 * 2];
                float* texture = &textures[i * 3];
                if (is_update[fn] == 0) return;

                const float pos_x = (
                    (face[2 * 0 + 0] * dim0 + face[2 * 1 + 0] * dim1 + face[2 * 2 + 0] * dim2) * (sizes[1] - 1)); 
                const float pos_y = (
                    (face[2 * 0 + 1] * dim0 + face[2 * 1 + 1] * dim1 + face[2 * 2 + 1] * dim2) * (sizes[2] - 1));
                if (1) {
                    /* bilinear sampling */
                    const float weight_x1 = pos_x - (int)pos_x;
                    const float weight_x0 = 1 - weight_x1;
                    const float weight_y1 = pos_y - (int)pos_y;
                    const float weight_y0 = 1 - weight_y1;
                    for (int k = 0; k < 3; k++) {
                        float c = 0;
                        c += image[((int)pos_y * sizes[1] + (int)pos_x) * 3 + k] * (weight_x0 * weight_y0);
                        c += image[((int)(pos_y + 1) * sizes[1] + (int)pos_x) * 3 + k] * (weight_x0 * weight_y1);
                        c += image[((int)pos_y * sizes[1] + ((int)pos_x) + 1) * 3 + k] * (weight_x1 * weight_y0);
                        c += image[((int)(pos_y + 1)* sizes[1] + ((int)pos_x) + 1) * 3 + k] * (weight_x1 * weight_y1);
                        texture[k] = c;
                    }
                } else {
                    /* nearest neighbor */
                    const int pos_xi = round(pos_x);
                    const int pos_yi = round(pos_y);
                    for (int k = 0; k < 3; k++) {
                        texture[k] = image[(pos_yi * sizes[1] + pos_xi) * 3 + k];
                    }
                }
            ''').substitute(
                texture_size=0),
            'function',
        )

    def load_textures(self,vertices, faces, material_names, filename_obj, filename_mtl, texture_size):
        # load vertices
        if len(vertices) > 0:
            vertices = np.vstack(vertices).astype('float32')

        # load faces for textures
        faces = np.vstack(faces).astype('int32') - 1
        textures = np.zeros((faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32') + 0.5

        if len(vertices) == 0:
            return textures

        faces = vertices[faces]
        faces = chainer.cuda.to_gpu(faces)
        faces[1 < faces] = faces[1 < faces] % 1

        #
        colors, texture_filenames = load_mtl(filename_mtl)

        textures = chainer.cuda.to_gpu(textures)

        #
        for material_name, color in colors.items():
            color = chainer.cuda.to_gpu(color)
            for i, material_name_f in enumerate(material_names):
                if material_name == material_name_f:
                    textures[i, :, :, :, :] = color[None, None, None, :]

        #
        for material_name, filename_texture in texture_filenames.items():
            filename_texture = os.path.join(os.path.dirname(filename_obj), filename_texture)
            image = skimage.io.imread(filename_texture) / 255.
            if image.shape[0] * image.shape[1] > 100*100:
                largesize = image.shape[0] if image.shape[0] > image.shape[1] else image.shape[1]
                scale = 100./largesize
                image = rescale(image, scale, anti_aliasing=False)#resize(image, (200, 200 * image.shape[1] / image.shape[0]), anti_aliasing=True)

            # print (image.shape)
            image = chainer.cuda.to_gpu(image.astype('float32'))
            image = image[::-1, ::1]
            is_update = np.array(material_names) == material_name
            is_update = chainer.cuda.to_gpu(is_update).astype('int32')

            loop = np.arange(textures.size / 3).astype('int32')
            loop = chainer.cuda.to_gpu(loop)
            # .substitute(
            #     texture_size=texture_size,
            #     image_height=image.shape[0],
            #     image_width=image.shape[1],
            # )
            sizes = chainer.cuda.to_gpu(np.array([texture_size, image.shape[0], image.shape[1]], dtype=np.int32))
            self.chainer_func(loop, image, faces, textures, is_update, sizes)
        return textures.get()


def load_obj(filename_obj, texture_size=4, load_texture=False, maxnverts=10000, maxntris=10000):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """

    # load vertices
    vertices = []
    faces = []

    #texture
    vertices_texture = []
    faces_texture = []
    material_names = []
    material_name = ''
    filename_mtl = None
    # print (filename_obj)

    tic = time.time()
    lines = open(filename_obj, "r").read().replace("\\\n", " ").splitlines()    
    for line in lines:#open(filename_obj).readlines():

        if len(line.split()) == 0:
            continue

        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])

        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            if '//' in vs[0]:
                v0 = int(vs[0].split('//')[0])
            elif '/' in vs[0]:
                v0 = int(vs[0].split('/')[0])
            # v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):

                if '//' in vs[i + 1]:
                    v1 = int(vs[i + 1].split('//')[0])
                elif '/' in vs[i + 1]:
                    v1 = int(vs[i + 1].split('/')[0])

                if '//' in vs[i + 2]:
                    v2 = int(vs[i + 2].split('//')[0])
                elif '/' in vs[i + 2]:
                    v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))

            if load_texture:
                if '/' in vs[0] and '//' not in vs[0]:
                    v0 = int(vs[0].split('/')[1])
                else:
                    v0 = 0
                for i in range(nv - 2):
                    if '/' in vs[i + 1] and '//' not in vs[i + 1]:
                        v1 = int(vs[i + 1].split('/')[1])
                    else:
                        v1 = 0
                    if '/' in vs[i + 2] and '//' not in vs[i + 2]:
                        v2 = int(vs[i + 2].split('/')[1])
                    else:
                        v2 = 0
                    faces_texture.append((v0, v1, v2))
                    material_names.append(material_name)

        if load_texture:
            if line.split()[0] == 'vt':
                vertices_texture.append([float(v) for v in line.split()[1:3]])

            if line.startswith('mtllib'):
                filename_mtl = os.path.join(os.path.dirname(filename_obj), line.split()[1])

            if line.split()[0] == 'usemtl':
                material_name = line.split()[1]

    faces = np.vstack(faces).astype('int32') - 1
    vertices = np.vstack(vertices).astype('float32')

    print(len(vertices), len(faces))
    
    if len(vertices) > maxnverts or len(faces) > maxntris:
        return None

    if load_texture and (filename_mtl is not None):
        tloader = texture_loader()
        textures = tloader.load_textures(vertices_texture, faces_texture, material_names, filename_obj, filename_mtl, texture_size)

    # print ('obj',time.time() - tic)
    if load_texture:
        return vertices, faces, textures
    else:
        return vertices, faces