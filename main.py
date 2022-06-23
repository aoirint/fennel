from asyncio import create_subprocess_exec
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sqlite3
from tempfile import NamedTemporaryFile, TemporaryDirectory
import time
from typing import List, Optional
from urllib.parse import urljoin
from uuid import UUID, uuid4
from fastapi import Depends, FastAPI, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from pydantic import BaseModel, parse_obj_as
import magic
import os
from dotenv import load_dotenv
load_dotenv()

JWT_SECRET = os.environ['JWT_SECRET']

UPLOAD_DIR = Path(os.environ['UPLOAD_DIR'])
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_ROOT_URL = os.environ['UPLOAD_ROOT_URL']

DB_PATH = Path(os.environ['DB_PATH'])
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

version = '0.0.0'

app = FastAPI(
  version=version,
)
db: sqlite3.Connection = None

class Token(BaseModel):
  sub: UUID

class User(BaseModel):
  id: UUID

auth_scheme = HTTPBearer()

async def validate_token(cred: HTTPAuthorizationCredentials = Depends(auth_scheme)) -> User:
  assert cred is not None
  token_obj = jwt.decode(jwt=cred.credentials, key=JWT_SECRET, algorithms=['HS256'])
  token = parse_obj_as(Token, token_obj)

  user_id = token.sub
  return User(
    id=user_id,
  )

@app.on_event('startup')
def init_db():
  global db
  db = sqlite3.connect(DB_PATH)

  db.execute('''
CREATE TABLE IF NOT EXISTS files (
  id TEXT PRIMARY KEY NOT NULL,
  mimetype TEXT NOT NULL,
  original_filename TEXT NOT NULL,
  upload_user_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
)
''')

@app.on_event('shutdown')
def deinit_db():
  db.close()

async def __ffmpeg_to_vod(
  id: UUID,
  source_file: Path,
  hls_segment_path: Path,
  hls_playlist_path: Path,
):
  report_tempfile = NamedTemporaryFile(mode='w+', encoding='utf-8')
  report_loglevel = 32 # 32: info, 48: debug
  report = f'file={report_tempfile.name}:level={report_loglevel}'

  vcodec = 'libx264'
  acodec = 'aac'
  hls_time = 9

  command = [
    'ffmpeg',
    '-nostdin',
    '-i',
    str(source_file),
    '-vcodec',
    vcodec,
    '-acodec',
    acodec,
    '-f',
    'hls',
    '-hls_time',
    str(hls_time),
    '-hls_playlist_type',
    'vod',
    '-hls_segment_filename',
    str(hls_segment_path),
    '-start_number',
    '1',
    '-report',
    str(hls_playlist_path),
  ]

  proc = await create_subprocess_exec(
    command[0],
    *command[1:],
    env={
      'FFREPORT': report,
    },
  )
  
  loop = asyncio.get_event_loop()
  executor = ThreadPoolExecutor()

  report_lines = []
  def read_report(report_file):
    report_file.seek(0)
    while True:
        line = report_file.readline()
        if len(line) == 0: # EOF
          if proc.returncode is not None: # process closed and EOF
            break
          time.sleep(0.1)
          continue # for next line written
        if line.endswith('\n'):
          line = line[:-1] # strip linebreak
        report_lines.append(line)
        print(f'[{id}] REPORT: {line}', flush=True)
    print(f'[{id}] report closed') # closed when process exited

  loop.run_in_executor(executor, read_report, report_tempfile)

  returncode = await proc.wait()
  # stdout, stderr may be not closed
  print(f'[{id}] ffmpeg exited with {returncode}')

  if returncode != 0:
    raise Exception(f'[{id}] Failed to convert video')


class FileUploadResponse(BaseModel):
  id: UUID
  mimetype: str
  image_url: Optional[str] = None
  video_url: Optional[str] = None
  hls_playlist_url: Optional[str] = None

async def __upload_video_mp4(
  id: UUID,
  file: UploadFile,
  dest_dir: Path,
  user: User,
):
  file.file.seek(0)

  mime = 'video/mp4'
  extension = 'mp4'
  original_filename = file.filename
  upload_user_id = user.id
  now = datetime.now(timezone.utc).isoformat()

  with TemporaryDirectory() as tempdirname:
    tempdir = Path(tempdirname)

    video_filename = f'video.{extension}'
    video_path = tempdir / video_filename
    with open(video_path, 'wb') as fp:
      shutil.copyfileobj(file.file, fp)

    vod_dir = tempdir / 'vod'
    vod_dir.mkdir()

    hls_segment_path = vod_dir / '%d.ts'
    hls_playlist_path = vod_dir / 'playlist.m3u8'

    await __ffmpeg_to_vod(
      id=id,
      source_file=video_path,
      hls_segment_path=hls_segment_path,
      hls_playlist_path=hls_playlist_path,
    )

    video_relpath = dest_dir.relative_to(UPLOAD_DIR) / video_path.relative_to(tempdir)
    hls_playlist_relpath = dest_dir.relative_to(UPLOAD_DIR) / hls_playlist_path.relative_to(tempdir)

    video_url = urljoin(UPLOAD_ROOT_URL, str(video_relpath))
    hls_playlist_url = urljoin(UPLOAD_ROOT_URL, str(hls_playlist_relpath))

    cur = db.cursor()
    cur.execute('''
INSERT INTO files(
  id,mimetype,original_filename,upload_user_id,created_at,updated_at
) VALUES(
  ?,?,?,?,?,?
)
''', (str(id), mime, original_filename, str(upload_user_id), now, now))

    shutil.copytree(tempdir, dest_dir)
    os.chmod(dest_dir, 0o755)

    db.commit()

  return FileUploadResponse(
    id=id,
    mimetype=mime,
    video_url=video_url,
    hls_playlist_url=hls_playlist_url,
  )

async def __upload_image_png(
  id: UUID,
  file: UploadFile,
  dest_dir: Path,
  user: User,
):
  file.file.seek(0)

  mime = 'image/png'
  extension = 'png'
  original_filename = file.filename
  upload_user_id = user.id
  now = datetime.now(timezone.utc).isoformat()

  with TemporaryDirectory() as tempdirname:
    tempdir = Path(tempdirname)

    image_filename = f'image.{extension}'
    image_path = tempdir / image_filename
    with open(image_path, 'wb') as fp:
      shutil.copyfileobj(file.file, fp)

    image_relpath = dest_dir.relative_to(UPLOAD_DIR) / image_path.relative_to(tempdir)
    image_url = urljoin(UPLOAD_ROOT_URL, str(image_relpath))

    cur = db.cursor()
    cur.execute('''
INSERT INTO files(
  id,mimetype,original_filename,upload_user_id,created_at,updated_at
) VALUES(
  ?,?,?,?,?,?
)
''', (str(id), mime, original_filename, str(upload_user_id), now, now))

    shutil.copytree(tempdir, dest_dir)
    os.chmod(dest_dir, 0o755)

    db.commit()

  return FileUploadResponse(
    id=id,
    mimetype=mime,
    image_url=image_url,
  )

async def __upload_image_jpeg(
  id: UUID,
  file: UploadFile,
  dest_dir: Path,
  user: User,
):
  file.file.seek(0)

  mime = 'image/jpeg'
  extension = 'jpg'
  original_filename = file.filename
  upload_user_id = user.id
  now = datetime.now(timezone.utc).isoformat()

  with TemporaryDirectory() as tempdirname:
    tempdir = Path(tempdirname)

    image_filename = f'image.{extension}'
    image_path = tempdir / image_filename
    with open(image_path, 'wb') as fp:
      shutil.copyfileobj(file.file, fp)

    image_relpath = dest_dir.relative_to(UPLOAD_DIR) / image_path.relative_to(tempdir)
    image_url = urljoin(UPLOAD_ROOT_URL, str(image_relpath))

    cur = db.cursor()
    cur.execute('''
INSERT INTO files(
  id,mimetype,original_filename,upload_user_id,created_at,updated_at
) VALUES(
  ?,?,?,?,?,?
)
''', (str(id), mime, original_filename, str(upload_user_id), now, now))

    shutil.copytree(tempdir, dest_dir)
    os.chmod(dest_dir, 0o755)

    db.commit()

  return FileUploadResponse(
    id=id,
    mimetype=mime,
    image_url=image_url,
  )

@app.post('/files', response_model=FileUploadResponse)
async def post_files(file: UploadFile, user: User = Depends(validate_token)):
  id = uuid4()

  dest_dir = UPLOAD_DIR / str(id)

  with NamedTemporaryFile() as tf:
    shutil.copyfileobj(file.file, tf)
    tf.flush()

    mime = magic.from_file(tf.name, mime=True)

  file.file.seek(0)

  if mime == 'video/mp4':
    return await __upload_video_mp4(
      id=id,
      file=file,
      dest_dir=dest_dir,
      user=user,
    )
  elif mime == 'image/png':
    return await __upload_image_png(
      id=id,
      file=file,
      dest_dir=dest_dir,
      user=user,
    )
  elif mime == 'image/jpeg':
    return await __upload_image_jpeg(
      id=id,
      file=file,
      dest_dir=dest_dir,
      user=user,
    )
  else:
    raise JSONResponse(
      status_code=400,
      content={
        'message': 'Not supported file type',
      },
    )

class GetFilesResponseFile(BaseModel):
  id: UUID
  mimetype: str
  original_filename: str
  image_url: Optional[str]
  video_url: Optional[str]
  hls_playlist_url: Optional[str]
  created_at: datetime

class GetFilesResponse(BaseModel):
  files: List[GetFilesResponseFile]

@app.get('/files', response_model=GetFilesResponse)
async def get_files(
  offset: int = 0,
  limit: int = 20,
  user: User = Depends(validate_token),
):
  user_id = user.id

  cur = db.cursor()

  files: List[GetFilesResponseFile] = []
  for row in cur.execute('SELECT id,mimetype,original_filename,created_at FROM files WHERE upload_user_id=? LIMIT ? OFFSET ?', (str(user_id), limit, offset, )):
    id, mimetype, original_filename, created_at = row

    image_url = None
    video_url = None
    hls_playlist_url = None
    if mimetype == 'video/mp4':
      video_url = urljoin(urljoin(UPLOAD_ROOT_URL, id), 'video.mp4')
      hls_playlist_url = urljoin(urljoin(UPLOAD_ROOT_URL, id), 'playlist.m3u8')
    elif mimetype == 'image/png':
      image_url = urljoin(urljoin(UPLOAD_ROOT_URL, id), 'image.png')
    elif mimetype == 'image/jpeg':
      image_url = urljoin(urljoin(UPLOAD_ROOT_URL, id), 'image.jpg')

    files.append(GetFilesResponseFile(
      id=id,
      mimetype=mimetype,
      original_filename=original_filename,
      image_url=image_url,
      video_url=video_url,
      hls_playlist_url=hls_playlist_url,
      created_at=datetime.fromisoformat(created_at),
    ))

  return GetFilesResponse(
    files=files,
  )

@app.get('/version', response_class=PlainTextResponse)
async def get_version() -> str:
  return version
