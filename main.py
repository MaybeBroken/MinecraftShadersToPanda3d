import glob
import os
from parser import *

SHADERS_PACK_FOLDER = "./Shaders/"
SOURCES_TO_PARSE = [
    "**/composite*",
    "**/final.fsh",
    "**/lib/*",
]
SHADERPACKS: dict[str, list[list[str]]] = {}

scanned_files = 0


for shader_pack in os.listdir(SHADERS_PACK_FOLDER):
    if not os.path.isdir(os.path.join(SHADERS_PACK_FOLDER, shader_pack)):
        continue
    shader_pack_path = os.path.join(SHADERS_PACK_FOLDER, shader_pack)
    print(f"Parsing shader pack: {shader_pack}")
    glob_patterns = [
        os.path.join(shader_pack_path, pattern) for pattern in SOURCES_TO_PARSE
    ]
    for pattern in glob_patterns:
        for file in glob.glob(pattern, recursive=True):
            if os.path.isfile(file):
                scanned_files += 1
                with open(file, "r") as f:
                    content = f.read()
                print(f"  Parsed file: {file}")
                if shader_pack not in SHADERPACKS:
                    SHADERPACKS[shader_pack] = []
                SHADERPACKS[shader_pack].append(
                    {
                        "name": file,
                        "files": {file: content},
                        "base_dir": shader_pack_path,
                    }
                )

if __name__ == "__main__":
    for shader_pack, files in SHADERPACKS.items():
        for file in files:
            render_shader(
                compile_shader(
                    shader_pack=file["name"],
                    files=file["files"],
                    base_dir=file["base_dir"],
                )
            )
