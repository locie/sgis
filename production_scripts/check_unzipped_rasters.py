import subprocess, zlib
from pathlib import Path, PurePosixPath
from collections import defaultdict
ARCHIVE = "/home/thebaulm/temporary_LaCie/rasters/downloads.temp/31/2025/BDORTHO_2-0_RVB-0M20_JP2-E080_LAMB93_D031_2025-01-01.7z.001"
FOLDER = Path("/home/thebaulm/temporary_LaCie/rasters/only_tiles/31/2025/31-2025-0M20-RGB")   # folder that corresponds to the archive root
SEVENZ = "7z"
EXTS = (".jp2", ".aux.xml", ".tab")
CHECK_CRC = True   # False = names + sizes only (much faster)

def wanted(name):
    return name.lower().endswith(EXTS)

def list_archive(archive):
    out = subprocess.run([SEVENZ, "l", "-slt", "-sccUTF-8", archive],
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    body = out.split("----------", 1)[1]
    files = defaultdict(list)
    for block in body.strip().split("\n\n"):
        info = dict(line.split(" = ", 1) for line in block.splitlines() if " = " in line)
        if info.get("Folder") == "+" or "Path" not in info:
            continue
        name = PurePosixPath(info["Path"].replace("\\", "/")).name
        if wanted(name):
            files[name].append((int(info["Size"]), info.get("CRC", "").upper()))
    return files

def crc32_of(path, chunk=1 << 22):
    crc = 0
    with open(path, "rb") as f:
        while block := f.read(chunk):
            crc = zlib.crc32(block, crc)
    return f"{crc & 0xFFFFFFFF:08X}"

arch = list_archive(ARCHIVE)

disk = defaultdict(list)
for p in FOLDER.rglob("*"):
    if p.is_file() and wanted(p.name):
        disk[p.name].append(p)

# duplicate file names (same name in several subfolders)
dup_arch = sorted(n for n, v in arch.items() if len(v) > 1)
dup_disk = sorted(n for n, v in disk.items() if len(v) > 1)

only_in_archive = sorted(arch.keys() - disk.keys())
only_on_disk = sorted(disk.keys() - arch.keys())

different = []
common = sorted((arch.keys() & disk.keys()) - set(dup_arch) - set(dup_disk))
for i, name in enumerate(common, 1):
    size, crc = arch[name][0]
    path = disk[name][0]
    if path.stat().st_size != size:
        different.append((name, "size"))
    elif CHECK_CRC and crc and crc32_of(path) != crc:
        different.append((name, "crc"))
    if i % 100 == 0:
        print(f"{i}/{len(common)} checked", flush=True)

print("Files in archive:", len(arch), "| on disk:", len(disk))
print("Only in archive:", len(only_in_archive), only_in_archive[:20])
print("Only on disk:   ", len(only_on_disk), only_on_disk[:20])
print("Different:      ", len(different), different[:20])
print("Duplicate names (skipped):", len(dup_arch), "in archive,", len(dup_disk), "on disk")