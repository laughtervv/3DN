### 3DN: 3D Deformation Network [<a href="">Arxiv</a>]

### Installation
Install <a href="https://www.tensorflow.org/">Tensorflow (1.10.0)</a>, <a href="https://pymesh.readthedocs.io/en/latest/">PyMesh</a>.

The mesh sampling and mesh laplacian operations are under folder `models/tf_ops`. To build them, simply use `cd models/tf_ops` and `sh make.sh` to compile. You may need to set CUDA path in each subfolder.

### Training

```bash
cd shapenet/3D
python train.py
```
```bash
cd shapenet/2D
python train_vgg.py
```
Pretrained Model can be found <a href="">here</a>.
### Testing 

```bash
cd shapenet/3D
python test.py
```
```bash
cd shapenet/2D
python test.py
```

### Citation
If you find this work useful, please consider citing:

	@inproceedings{wang20193dn,
	    title={3DN: 3D Deformation Network},
	    author={Wang, Weiyue and Ceylan, Duygu and Mech, Radomir and Neumann, Ulrich},
	    booktitle={CVPR},
	    year={2019}
	}

    
### Acknowledgemets
<a href="https://github.com/charlesq34/pointnet">PointNet</a>
<a href="https://github.com/charlesq34/pointnet-autoencoder">PointNet AutoEncoder</a>
<a href="https://github.com/fanhqme/PointSetGeneration">Point Set Generation</a>


