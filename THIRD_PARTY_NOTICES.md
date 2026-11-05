# Third-party notices

No license is granted here for project-authored mercymek code. Publishing this
repository does not relicense the third-party components below. Their original
copyright and license notices remain applicable.

## Bundled browser libraries

| Component | Included files | Notice |
| --- | --- | --- |
| three.js r180 and its controls/loaders/exporters | `frontend/vendor/three*`, `OrbitControls.js`, `addons/`, `loaders/`, `exporters/`, `utils/` | [MIT notice](frontend/vendor/three.LICENSE); [upstream](https://github.com/mrdoob/three.js) |
| Spark Gaussian-splat renderer | `frontend/vendor/spark/` | [MIT notice](frontend/vendor/spark/LICENSE); [upstream](https://github.com/sparkjsdev/spark) |
| Leaflet 1.9.4, styles and marker images | `frontend/vendor/leaflet/` | [BSD-2-Clause notice](frontend/vendor/leaflet/LICENSE); [upstream](https://github.com/Leaflet/Leaflet/tree/v1.9.4) |

Do not remove these notices when distributing the frontend. Leaflet CSS includes
local style adjustments. Locally cloned third-party repositories under `vendor/`
are excluded from Git; the small compatibility shims are project-authored code.

## Models, services and data

Python packages are installed separately through `pyproject.toml`; retain their
own notices in any redistributed environment. Model weights are not shipped in
this source repository. Code licenses and checkpoint licenses can differ: check
the exact checkpoint and revision before using or redistributing a model.

- VGGT: [upstream license and acceptable-use policy](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt).
  The default `facebook/VGGT-1B` checkpoint declares CC-BY-NC-4.0; its
  non-commercial weight terms are separate from the source-code license.
- AnySplat: [upstream MIT license](https://github.com/InternRobotics/AnySplat/blob/main/LICENSE).
- RAFT/torchvision: [upstream license](https://github.com/pytorch/vision/blob/main/LICENSE).
- Real-ESRGAN: [upstream license](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE).
- FastGS and its underlying 3D Gaussian Splatting dependencies: check each upstream
  license before redistribution; they are not bundled with the public source.
- Meta Model API, ElevenLabs, Hugging Face and Kaggle are external services with
  their own terms. OpenStreetMap attribution remains visible in the map.

Documentation images include derived Nexar demonstration footage. Attribution:
Moura, Daniel C., and Zvitia, Orly. “Nexar Collison Dataset.” Hugging Face, 2025.
[Dataset and Nexar Open Data License](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/LICENSE).
The complete notice is retained in [third_party/NEXAR_LICENSE.txt](third_party/NEXAR_LICENSE.txt)
and copied into `data/licenses/` by the data packer. Keep its attribution, use
restrictions and disclaimers with redistributed
Nexar data. It does not permit resale for profit or re-identification of anonymized
people or vehicles. Estimated/generated reconstruction is labelled in the application.

The README screenshots and final film use the London CC BY 3.0 example. Their full creator, source and
modification credits are in [docs/demo-credits.md](docs/demo-credits.md). Each data
case retains its own source/provenance metadata; the cases do not all have the
same license. The project-authored-code licensing decision does not override any
of these notices or source-media terms.
