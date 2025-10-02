from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from app.inference import FaceSwapPipeline, load_image_from_any
from app.video import swap_video_stream
from pathlib import Path
from uuid import uuid4
import io, os, tempfile, shutil
from PIL import Image

app = FastAPI(title="FaceSwap API", version="1.0")

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def promote(src: str, dst: str):
    os.replace(src, dst)

pipeline = FaceSwapPipeline(
    gen_ckpt="checkpoints/simswap/netG_step13000.pth",
    arc_ckpt="arcface_model/arcface_checkpoint.tar",
    device="cpu"
)

@app.post("/swap/image")
async def swap_image(
    background_tasks: BackgroundTasks,
    source: UploadFile = File(None),
    target: UploadFile = File(None),
    source_url: str = Form(None),
    target_url: str = Form(None),
):
    src_img = await load_image_from_any(source, source_url)
    tgt_img = await load_image_from_any(target, target_url)
    out_img = pipeline.swap_image(src_img, tgt_img)
    tmp_path = OUTPUT_DIR / f"out_{uuid4().hex}.png"
    Image.fromarray(out_img).save(tmp_path, format="PNG")
    final_path = OUTPUT_DIR / "out.png"
    background_tasks.add_task(promote, str(tmp_path), str(final_path))
    return FileResponse(str(tmp_path), media_type="image/png", filename="out.png")

@app.post("/swap/video")
async def swap_video(
    background_tasks: BackgroundTasks,
    source: UploadFile = File(None),
    target: UploadFile = File(None),
    source_url: str = Form(None),
    target_url: str = Form(None),
):
    src_img = await load_image_from_any(source, source_url)
    if target is None and not target_url:
        return {"error": "target video required"}

    tmpdir = tempfile.mkdtemp(prefix="faceswap_")
    in_path = os.path.join(tmpdir, "in.mp4")
    if target is not None:
        data = await target.read()
        with open(in_path, "wb") as f:
            f.write(data)
    else:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return {"error":"URL download disabled here"}

    uniq_out = OUTPUT_DIR / f"out_{uuid4().hex}.mp4"
    swap_video_stream(pipeline, src_img, in_path, str(uniq_out))
    shutil.rmtree(tmpdir, ignore_errors=True)

    final_out = OUTPUT_DIR / "out.mp4"
    background_tasks.add_task(promote, str(uniq_out), str(final_out))
    return FileResponse(str(uniq_out), media_type="video/mp4", filename="out.mp4")